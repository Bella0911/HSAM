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
        return mask_fg, mask_bg, mask_uc



class PromptGenerator(nn.Module):
    def __init__(self, in_c=128, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_fg):
        # 生成空间注意力
        x = self.conv1(f_fg)
        attention = self.sigmoid(x)
        x = x * attention

        # 通道注意力
        channel_att = torch.mean(x, dim=(2, 3), keepdim=True)
        x = x * channel_att

        # 生成最终的prompt embedding
        x = self.conv2(x)
        embed = torch.mean(x, dim=(2, 3))  # (B,256)
        embed = self.norm(embed)
        return embed

class ConDSegWithSAM(nn.Module):
    def __init__(self, mask_decoder):
        super().__init__()
        # 初始化设备属性
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 确保输入通道数与SAM输出匹配
        self.decouple_layer = DecoupleLayer(in_c=256, out_c=128)  # SAM输出256通道
        self.aux_head = AuxiliaryHead(in_c=128)
        self.prompt_generator = PromptGenerator()
        self.mask_decoder = mask_decoder

        # 添加sparse和dense embedding转换层
        self.sparse_embed = nn.Linear(256, 256)  # 将prompt转换为sparse embedding
        self.dense_embed = nn.Conv2d(128, 256, kernel_size=1)  # 将特征转换为dense embedding

        # 添加位置编码生成器
        self.pe_layer = PositionEmbeddingRandom(128)  # 128是位置编码的通道数
        # 存储SAM组件引用
        self.sam_vit = sam_vit  # 将全局变量转为实例变量
        self.to(self.device)  # 移动到设备

        # 添加上采样层
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    @property
    def device(self):
        """获取模型所在设备的属性"""
        return self._device
    def to(self, device, *args, **kwargs):
        # 重写to方法确保所有组件同步移动
        super().to(device, *args, **kwargs)
        self.decouple_layer = self.decouple_layer.to(device)
        self.aux_head = self.aux_head.to(device)
        self.prompt_generator = self.prompt_generator.to(device)
        self.mask_decoder = self.mask_decoder.to(device)
        self.sparse_embed = self.sparse_embed.to(device)
        self.dense_embed = self.dense_embed.to(device)
        self.pe_layer = self.pe_layer.to(device)
        self.sam_vit = self.sam_vit.to(device)  # 移动SAM ViT
        return self

    def forward(self, image):
        image = image.to(self.device)
        if image.device != next(self.parameters()).device:
            raise RuntimeError(
                f"Input device ({image.device}) and model device "
                f"({next(self.parameters()).device}) must match!"
            )

        # Step 1: 通过SAM ViT编码图像
        x_vit = sam_vit(image)
        #print(f"1. x_vit shape: {x_vit.shape if x_vit is not None else 'None'}")  # 调试

        # Step 2: 解耦特征
        f_fg, f_bg, f_uc = self.decouple_layer(x_vit)
        #print(f"2. f_fg shape: {f_fg.shape if f_fg is not None else 'None'}")  # 调试

        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)

        # Step 3: 生成prompt
        prompt_embed = self.prompt_generator(f_fg)
        #print(f"3. prompt_embed: {'Exists' if prompt_embed is not None else 'None'}")  # 调试

        if prompt_embed is None:
            raise ValueError("Prompt embedding is None!")

        # 转换为SAM需要的格式
        sparse_embeddings = self.sparse_embed(prompt_embed.unsqueeze(1))
        dense_embeddings = self.dense_embed(f_fg)

        # 生成位置编码
        b, c, h, w = x_vit.shape
        image_pe = self.pe_layer((h, w)).unsqueeze(0)  # 添加batch维度

        # Step 4: 调用SAM MaskDecoder
        masks, _ = self.mask_decoder(
            image_embeddings=x_vit,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )


        masks = self.upsample(masks)
        binary_masks = (masks > 0.5).float()
        return binary_masks, mask_fg, mask_bg, mask_uc



# 3. 初始化模型
model = ConDSegWithSAM(sam_mask_decoder)
print(f"model device: {model.device}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_image = torch.randn(1, 3, 1024, 1024).to(device)  # SAM 默认输入尺寸
output_mask = model(input_image)  # 输出分割掩码




