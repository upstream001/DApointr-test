import torch
from utils.config import cfg_from_yaml_file
from torch.utils.data import DataLoader
from datasets.CRN_Dataset import CRNShapeNet
import sys
import os
sys.path.append('/home/tianqi/DAPoinTr')


def test_source_dataset():
    """测试源域数据集"""
    print("测试源域数据集...")

    # 创建一个简单的配置
    config = type('Config', (), {})()
    config.DATA_PATH = "/home/tianqi/DAPoinTr/data/organized_source_dataset"
    config.CLASS_CHOICE = "custom"
    config.SPLIT = "train"

    try:
        dataset = CRNShapeNet(config)
        print(f"源域数据集加载成功，数据集大小: {len(dataset)}")

        # 测试加载一个样本
        if len(dataset) > 0:
            gt, partial, idx = dataset[0]
            print(
                f"样本形状 - GT: {gt.shape}, Partial: {partial.shape}, Index: {idx}")
            print("源域数据集测试通过！")
        else:
            print("警告: 数据集为空")
    except Exception as e:
        print(f"源域数据集加载失败: {e}")


def test_target_dataset():
    """测试目标域数据集"""
    print("\n测试目标域数据集...")

    # 需要创建一个简单的目标域数据集类
    from datasets.ply_dataset import RealDataset
    import argparse

    # 创建参数对象
    args = argparse.Namespace()
    args.realdataset = "CustomTarget"
    args.class_choice = "custom"
    args.split = "train"

    # 创建配置对象
    config = type('Config', (), {})()
    config.real_dataset = "CustomTarget"
    config.class_choice = "custom"
    config.split = "train"

    # 这里需要根据实际目标域数据集的格式调整
    # 由于我们只是简单地将PLY文件放在一个目录中，我们需要一个简单的加载方式
    print("目标域数据集结构:")
    target_dir = "/home/tianqi/DAPoinTr/data/organized_target_dataset"
    files = [f for f in os.listdir(target_dir) if f.endswith('.ply')]
    print(f"目标域数据集包含 {len(files)} 个PLY文件")
    print(f"前5个文件: {files[:5]}")


if __name__ == "__main__":
    test_source_dataset()
    test_target_dataset()
    print("\n数据集验证完成！")
