"""
准备ImageNet数据集的脚本

使用方法:
1. 从 https://image-net.org/ 下载ImageNet数据集
2. 将下载的tar文件放在指定目录
3. 运行此脚本来解压和整理数据

python prepare_imagenet.py --train-tar /path/to/ILSVRC2012_img_train.tar \
                           --val-tar /path/to/ILSVRC2012_img_val.tar \
                           --output-dir ./data/imagenet
"""

import os
import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

def extract_train_data(train_tar_path, output_dir):
    """
    解压ImageNet训练集
    训练集包含1000个tar文件，每个对应一个类别
    """
    print(f"正在解压训练集: {train_tar_path}")
    train_dir = Path(output_dir) / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # 首先解压主tar文件
    with tarfile.open(train_tar_path, 'r') as tar:
        tar.extractall(train_dir)
    
    # 然后解压每个类别的tar文件
    tar_files = list(train_dir.glob('*.tar'))
    print(f"找到 {len(tar_files)} 个类别tar文件")
    
    for tar_file in tqdm(tar_files, desc="解压类别"):
        # 类别文件夹名就是tar文件名（不含.tar）
        class_name = tar_file.stem
        class_dir = train_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # 解压到对应的类别文件夹
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(class_dir)
        
        # 删除tar文件以节省空间
        tar_file.unlink()
    
    print(f"训练集解压完成: {train_dir}")
    return train_dir

def extract_val_data(val_tar_path, output_dir, val_labels_file=None):
    """
    解压ImageNet验证集
    验证集需要根据标签文件重新组织到类别文件夹中
    """
    print(f"正在解压验证集: {val_tar_path}")
    val_dir = Path(output_dir) / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 解压验证集
    with tarfile.open(val_tar_path, 'r') as tar:
        tar.extractall(val_dir)
    
    # 如果提供了标签文件，重新组织文件结构
    if val_labels_file and os.path.exists(val_labels_file):
        print("正在根据标签重组验证集...")
        organize_val_data(val_dir, val_labels_file)
    else:
        print("警告: 未提供验证集标签文件，验证集未按类别组织")
        print("请下载 https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh")
        print("或者提供标签文件来组织验证集")
    
    print(f"验证集解压完成: {val_dir}")
    return val_dir

def organize_val_data(val_dir, labels_file):
    """
    根据标签文件重组验证集数据
    """
    # 这里需要标签映射文件
    # 可以使用官方的valprep.sh脚本，或者手动创建映射
    pass

def download_val_script(output_dir):
    """
    下载并保存验证集整理脚本
    """
    script_content = """#!/bin/bash
# 此脚本用于整理ImageNet验证集
# 来源: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
"""
    
    script_path = Path(output_dir) / 'valprep.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"验证集整理脚本已保存到: {script_path}")
    return script_path

def main():
    parser = argparse.ArgumentParser(description='准备ImageNet数据集')
    parser.add_argument('--train-tar', type=str, help='训练集tar文件路径')
    parser.add_argument('--val-tar', type=str, help='验证集tar文件路径')
    parser.add_argument('--output-dir', type=str, default='./data/imagenet',
                       help='输出目录 (默认: ./data/imagenet)')
    parser.add_argument('--val-labels', type=str, help='验证集标签文件路径')
    
    args = parser.parse_args()
    
    if not args.train_tar and not args.val_tar:
        print("错误: 请至少提供 --train-tar 或 --val-tar 参数")
        print("\n使用说明:")
        print("1. 从 https://image-net.org/ 注册并下载ImageNet ILSVRC2012数据集")
        print("2. 下载文件:")
        print("   - ILSVRC2012_img_train.tar (训练集, ~138GB)")
        print("   - ILSVRC2012_img_val.tar (验证集, ~6.3GB)")
        print("3. 运行此脚本:")
        print("   python prepare_imagenet.py --train-tar /path/to/ILSVRC2012_img_train.tar \\")
        print("                               --val-tar /path/to/ILSVRC2012_img_val.tar \\")
        print("                               --output-dir ./data/imagenet")
        print("\n或者使用Kaggle数据集:")
        print("   kaggle competitions download -c imagenet-object-localization-challenge")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理训练集
    if args.train_tar:
        if not os.path.exists(args.train_tar):
            print(f"错误: 训练集文件不存在: {args.train_tar}")
        else:
            extract_train_data(args.train_tar, args.output_dir)
    
    # 处理验证集
    if args.val_tar:
        if not os.path.exists(args.val_tar):
            print(f"错误: 验证集文件不存在: {args.val_tar}")
        else:
            extract_val_data(args.val_tar, args.output_dir, args.val_labels)
    
    print("\n完成! 数据集已准备好，可以用于训练。")
    print(f"训练数据路径: {os.path.join(args.output_dir, 'train')}")
    print(f"验证数据路径: {os.path.join(args.output_dir, 'val')}")
    print("\n使用以下命令开始训练:")
    print(f"torchrun --nnodes=1 --nproc_per_node=1 train.py \\")
    print(f"    --model EqM-XL/2 \\")
    print(f"    --data-path {os.path.join(args.output_dir, 'train')} \\")
    print(f"    --num-classes 1000 \\")
    print(f"    --image-size 256 \\")
    print(f"    --global-batch-size 256 \\")
    print(f"    --epochs 80")

if __name__ == '__main__':
    main()


