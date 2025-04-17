import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from .utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics, mask_to_bbox
from tqdm import tqdm
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def load_names(path, file_path):
    with open(file_path, "r") as f:
        data = [line.strip() for line in f if line.strip()]

    images, masks = [], []
    for name in data:
        # 去除现有扩展名（如果有）
        stem = Path(name).stem  # 例如 "trainval_1132.jpg" -> "trainval_1132"
        img_path = os.path.join(path, "images", f"{stem}.jpg")
        mask_path = os.path.join(path, "masks", f"{stem}.jpg")
        images.append(img_path)
        masks.append(mask_path)
    return images, masks

def load_data(path,val_name=None):
    train_names_path = f"{path}/train.txt"
    # valid_names_path = f"{path}/val.txt"
    if val_name is None:
        valid_names_path = f"{path}/val.txt"
    else:
        valid_names_path = f"{path}/val_{val_name}.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        # 保持原始图像比例的同时resize到1024x1024
        h, w = image.shape[:2]
        scale = min(1024 / h, 1024 / w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h))

        # 填充到1024x1024
        pad_h = 1024 - new_h
        pad_w = 1024 - new_w
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)


        background = mask.copy()
        background = 255 - background


        """ Image """
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Background """
        background = cv2.resize(background, self.size)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return image, (mask, background)

    def __len__(self):
        return self.n_samples

def complementary_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
    normalized_loss = loss / num_pixels
    return normalized_loss

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    scaler = GradScaler()  # 初始化梯度缩放器
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        #print(f"\n===== Batch {i} =====")
        #print_gpu_usage("Before data transfer")
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)
        #print_gpu_usage("After data transfer")
        optimizer.zero_grad()
        #print_gpu_usage("After zero_grad")
        # 使用 autocast 包装前向传播
        with autocast(device_type='cuda', dtype=torch.float16):
            mask_pred, fg_pred, bg_pred, uc_pred = model(x)

            loss_mask = loss_fn(mask_pred, y1)
            loss_fg = loss_fn(fg_pred, y1)
            loss_bg = loss_fn(bg_pred, y2)

            beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
            beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
            beta1 = beta1.to(device)
            beta2 = beta2.to(device)

            preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
            loss_comp = loss_comp.to(device)

            loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp

        # 使用 scaler 缩放梯度并反向传播
        scaler.scale(loss).backward()
        #print_gpu_usage("After backward")

        accumulation_steps = 4
        loss = loss / accumulation_steps  # 缩放损失

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # 使用 scaler.step() 代替 optimizer.step()
            scaler.update()  # 更新缩放器
            optimizer.zero_grad()

        #print_gpu_usage("After optimizer step")
        epoch_loss += loss.item()

        torch.cuda.empty_cache()
        #print_gpu_usage("After cache cleanup")

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y1, mask_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss / len(loader)
    epoch_jac = epoch_jac / len(loader)
    epoch_f1 = epoch_f1 / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def visualize_masks(images, gt_masks, pred_masks, mask_fg, mask_bg ):
    """
    可视化输入图像、真实mask、预测mask和前景mask
    Args:
        images: (B,3,H,W) 输入图像
        gt_masks: (B,1,H,W) 真实mask
        pred_masks: (B,1,H,W) 预测mask
        mask_fg: (B,1,H,W) 前景mask
        mask_bg: (B,1,H,W) 背景mask
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 处理输入图像 (需要从CHW转为HWC)
    img = images[0].permute(1, 2, 0).cpu().numpy()
    if img.shape[-1] == 3:  # RGB归一化
        img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # 处理真实mask (去除通道维度)
    gt = gt_masks[0, 0].cpu().numpy()  # 从(1,H,W)变为(H,W)
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    # 处理预测mask
    pred = pred_masks[0, 0].cpu().detach().numpy()  # (H,W)
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    # 处理前景mask
    fg = mask_fg[0, 0].cpu().detach().numpy()  # (H,W)
    axes[3].imshow(fg, cmap='gray')
    axes[3].set_title("Foreground Mask")
    axes[3].axis('off')

    # 新增背景mask
    bg = mask_bg[0, 0].cpu().detach().numpy()
    axes[4].imshow(bg, cmap='gray')
    axes[4].set_title("Background Mask")
    axes[4].axis('off')


    plt.tight_layout()
    plt.show()



def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            mask_pred, fg_pred, bg_pred, uc_pred = model(x)

            # 只在最后一个批次进行可视化
            if (i == len(loader) - 1 or i == 0):
                visualize_masks(
                    images=x,          # 输入图像 [B,C,H,W]
                    gt_masks=y1,       # 真实掩码 [B,1,H,W]
                    pred_masks=mask_pred,  # 预测掩码 [B,1,H,W]
                    mask_fg=fg_pred,    # 前景预测 [B,1,H,W]
                    mask_bg=bg_pred    # 背景预测 [B,1,H,W]
                )
            loss_mask = loss_fn(mask_pred, y1)
            loss_fg = loss_fn(fg_pred, y1)
            loss_bg = loss_fn(bg_pred, y2)

            beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
            beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
            beta1 = beta1.to(device)
            beta2 = beta2.to(device)

            preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
            loss_comp = loss_comp.to(device)

            loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp

            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss / len(loader)
        epoch_jac = epoch_jac / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_precision = epoch_precision / len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]
