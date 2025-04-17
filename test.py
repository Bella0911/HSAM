import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
import os
from network.model import ConDSegWithSAM, sam_mask_decoder
from utils.run_engine import evaluate, DATASET  # 确保从原项目中导入必要模块


def load_test_data(data_path):
    """加载测试数据集路径"""
    image_dir = os.path.join(data_path, 'images')
    mask_dir = os.path.join(data_path, 'masks')

    # 获取排序后的图像和掩膜路径列表
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    return images, masks


def main():
    parser = argparse.ArgumentParser(description="Test ConDSegWithSAM model")
    parser.add_argument("--data_path", type=str, required=True, help="测试数据集路径（需包含images/masks子目录）")
    parser.add_argument("--model_path", type=str, required=True, help="保存的模型路径(.pth文件)")
    parser.add_argument("--batch_size", type=int, default=4, help="测试批次大小")
    parser.add_argument("--size", type=int, nargs=2, default=[1024, 1024], help="输入图像尺寸（高宽）")
    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")

    # 加载测试数据集
    test_x, test_y = load_test_data(args.data_path)
    test_dataset = DATASET(test_x, test_y, args.size, transform=None)  # 与验证集一致无数据增强
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # 初始化模型并加载权重
    model = ConDSegWithSAM(sam_mask_decoder).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 损失函数
    loss_fn = nn.BCEWithLogitsLoss()

    # 执行评估
    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device)

    # 打印结果
    print("\n测试结果:")
    print(f"损失: {test_loss:.4f} | Jaccard: {test_metrics[0]:.4f}")
    print(f"F1分数: {test_metrics[1]:.4f} | 召回率: {test_metrics[2]:.4f} | 精确率: {test_metrics[3]:.4f}")


if __name__ == "__main__":
    main()

# python test.py \
#   --data_path /home/heyan/project/HSAM/dataset/polyp/TestDataset/TestDataset/Kvasir \  # 需包含images和masks子目录
#   --model_path /home/heyan/project/HSAM/result/polyp/base/condsam_20250329_213434.pth \
#   --batch_size 1 \
#   --size 1024 1024