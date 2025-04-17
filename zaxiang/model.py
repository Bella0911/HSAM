import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.modeling.mask_decoder import MaskDecoder


# 1. 加载 SAM 的 ViT 和 MaskDecoder
sam = sam_model_registry["vit_b"](checkpoint="/home/heyan/project/HSAM/sam_vit_b.pth")
sam_vit = sam.image_encoder
sam_mask_decoder = sam.mask_decoder
sam_prompt_encoder = sam.prompt_encoder

import torch
from pynvml import *


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    def __init__(self, num_pos_feats: int = 64, scale: float = None):
        super().__init__()
        if scale is None or scale <= .0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(CBR(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18), channel_attention(out_c))
        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc + xs)
        x = self.sa(x)
        return x


"""Decouple Layer"""


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=256, out_c=128):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            CBR(256, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            CBR(256, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            CBR(256, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc


"""Auxiliary Head"""


class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()

        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)

        # 添加调试输出
        #print(f"FG mask mean: {mask_fg.mean().item():.4f}, max: {mask_fg.max().item():.4f}")
        #print(f"BG mask mean: {mask_bg.mean().item():.4f}, max: {mask_bg.max().item():.4f}")
        #print(f"UC mask mean: {mask_uc.mean().item():.4f}, max: {mask_uc.max().item():.4f}")

        return mask_fg, mask_bg, mask_uc


class BBoxGenerator(nn.Module):
    def __init__(self, min_box_size=0.05):
        super().__init__()
        self.min_box_size = min_box_size  # 最小框尺寸，避免过小的框

    def forward(self, mask_fg):
        """
        改进的边界框生成器，确保生成有效的边界框
        Args:
            mask_fg: (B, 1, H, W) 前景mask
        Returns:
            boxes: (B, 4) 归一化的边界框坐标[x0,y0,x1,y1]
        """
        B, _, H, W = mask_fg.shape
        boxes = torch.zeros((B, 4), device=mask_fg.device)

        for i in range(B):
            # 获取非零像素坐标
            nonzero = torch.nonzero(mask_fg[i, 0] > 0.5)  # 使用0.5作为阈值

            if len(nonzero) == 0:
                # 如果没有前景，使用中心区域作为默认框
                boxes[i] = torch.tensor([0.25, 0.25, 0.75, 0.75], device=mask_fg.device)
                continue

            # 计算边界框坐标
            y_min = nonzero[:, 0].min().float() / H
            y_max = nonzero[:, 0].max().float() / H
            x_min = nonzero[:, 1].min().float() / W
            x_max = nonzero[:, 1].max().float() / W

            # 确保框的最小尺寸
            if (x_max - x_min) < self.min_box_size:
                x_center = (x_min + x_max) / 2
                x_min = max(0, x_center - self.min_box_size / 2)
                x_max = min(1, x_center + self.min_box_size / 2)

            if (y_max - y_min) < self.min_box_size:
                y_center = (y_min + y_max) / 2
                y_min = max(0, y_center - self.min_box_size / 2)
                y_max = min(1, y_center + self.min_box_size / 2)

            boxes[i] = torch.tensor([x_min, y_min, x_max, y_max], device=mask_fg.device)

        return boxes


class ConDSegWithSAM(nn.Module):
    def __init__(self, mask_decoder, prompt_encoder):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 核心组件
        self.decouple_layer = DecoupleLayer(in_c=256, out_c=128)
        self.aux_head = AuxiliaryHead(in_c=128)
        self.bbox_generator = BBoxGenerator()

        # SAM组件
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.sam_vit = sam_vit

        # 上采样层
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.to(self.device)

    def forward(self, image):
        # Step 1: 通过SAM ViT编码图像
        x_vit = self.sam_vit(image)

        # Step 2: 解耦特征并生成mask
        f_fg, f_bg, f_uc = self.decouple_layer(x_vit)
        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)

        # Step 3: 从前景mask生成边界框
        boxes = self.bbox_generator(mask_fg)  # (B,4)
        # 调试输出
        #print(f"Generated boxes: {boxes}")
        # Step 4: 编码边界框提示
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        # Step 5: 生成位置编码
        b, c, h, w = x_vit.shape
        image_pe = self.prompt_encoder.get_dense_pe()  # 使用SAM内置的位置编码

        # Step 6: 调用SAM MaskDecoder
        masks, _ = self.mask_decoder(
            image_embeddings=x_vit,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 上采样到原始尺寸
        masks = self.upsample(masks)
        return masks, mask_fg, mask_bg, mask_uc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 初始化模型时需要传入prompt_encoder
model = ConDSegWithSAM(sam.mask_decoder, sam.prompt_encoder)
model.to(device)

# 前向传播
input_image = torch.randn(1, 3, 1024, 1024).to(device)
output_mask = model(input_image)

def print_gpu_usage(prefix=""):
    """打印当前GPU显存使用情况"""
    torch.cuda.synchronize()
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"{prefix} GPU Memory Used: {info.used/1024**2:.2f} MB / {info.total/1024**2:.2f} MB")

def print_model_memory(model, name="Model"):
    """打印模型参数和缓存的显存"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"{name} memory: Parameters {param_size/1024**2:.2f} MB | Buffers {buffer_size/1024**2:.2f} MB")

class MemoryMonitorConDSeg(ConDSegWithSAM):
    def __init__(self, mask_decoder):
        super().__init__(mask_decoder)
        print("\n===== Initial Memory Status =====")
        print_gpu_usage("After model init")
        print_model_memory(self, "Main Model")
        print_model_memory(self.sam_vit, "SAM ViT")

    def forward(self, image):
        print("\n===== Forward Pass Memory =====")
        print_gpu_usage("Before forward")

        # 各组件显存监控
        with torch.no_grad():
            print("\n[Before SAM ViT]")
            print_gpu_usage()
            x_vit = self.sam_vit(image)
            print("[After SAM ViT]")
            print_gpu_usage()

        print("\n[Before Decouple Layer]")
        print_gpu_usage()
        f_fg, f_bg, f_uc = self.decouple_layer(x_vit)

        print("\n[After Decouple Layer]")
        print_gpu_usage()
        masks = self.mask_decoder(...)  # 原有逻辑

        print("\n[Final Output]")
        print_gpu_usage()
        return masks


