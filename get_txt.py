import os


def generate_tif_list(img_dir='/home/heyan/project/HSAM/dataset/polyp/TestDataset/TestDataset/test/images', output_file='test.txt'):
    # 获取目录下所有文件并筛选.tif文件
    tif_files = []
    for filename in os.listdir(img_dir):
        file_path = os.path.join(img_dir, filename)
        # 检查是否为文件且扩展名为.tif（不区分大小写）
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() == '.png':
                tif_files.append(filename)

    # 对文件名进行排序（按字母顺序）
    tif_files.sort()

    # 将结果写入输出文件
    with open(output_file, 'w') as f:
        for filename in tif_files:
            f.write(f"{filename}\n")

    print(f"生成成功！共找到 {len(tif_files)} 个TIFF文件。已保存至：{output_file}")


if __name__ == '__main__':
    generate_tif_list()