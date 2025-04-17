import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from network.model import ConDSegWithSAM, sam_vit, sam_mask_decoder, sam_prompt_encoder
from utils.run_engine import DATASET, load_data, train, evaluate
import argparse
import os
from datetime import datetime
import albumentations as A
import os



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--size", type=int, nargs=2, default=[1024, 1024], help="Image size (height, width)")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save models")
    parser.add_argument("--train_mask_decoder", action="store_true", help="Whether to train SAM mask decoder")
    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"condsam_{timestamp}.pth")

    # 加载数据
    (train_x, train_y), (valid_x, valid_y) = load_data(args.data_path)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5)
    ])

    train_dataset = DATASET(train_x, train_y, args.size, transform)
    valid_dataset = DATASET(valid_x, valid_y, args.size, None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = MemoryMonitorConDSeg(sam_mask_decoder).to(device)
    model = ConDSegWithSAM(sam_mask_decoder).to(device)

    # 冻结SAM image encoder
    for param in sam_vit.parameters():
        param.requires_grad = False

    # 可选是否冻结mask decoder
    if not args.train_mask_decoder:
        for param in model.mask_decoder.parameters():
            param.requires_grad = False
        print("SAM mask decoder is frozen")
    else:
        print("SAM mask decoder is trainable")

    # 优化器和损失函数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_score = 0.0

    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练阶段
        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f} | Jaccard: {train_metrics[0]:.4f} | F1: {train_metrics[1]:.4f}")

        # 验证阶段
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        print(
            f"Valid Loss: {valid_loss:.4f} | Jaccard: {valid_metrics[0]:.4f} | F1: {valid_metrics[1]:.4f} | Recall: {valid_metrics[2]:.4f} | Precision: {valid_metrics[3]:.4f}")

        # 保存最佳模型
        current_score = valid_metrics[0]  # 使用Jaccard作为评估指标
        if current_score > best_score:
            best_score = current_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'metrics': valid_metrics
            }, save_path)
            print(f"Model saved at {save_path} with Jaccard: {best_score:.4f}")


if __name__ == "__main__":
    main()

# python train.py --data_path /home/heyan/ConDSeg-main/data/TN3K2021 --batch_size 2 --lr 1e-4 --epochs 100 --size 1024 1024