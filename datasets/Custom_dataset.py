# 创建自定义数据集类
import os
import torch.utils.data as data
import numpy as np
from utils.io import read_ply_xyz
from utils.pc_transform import swap_axis
import glob
import torch
from datasets.build import DATASETS


def pc_normalize(pc):
    """将点云归一化到单位球内 (Unit Sphere)"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m < 1e-6:
        m = 1.0
    pc /= m
    return pc, centroid, m


@DATASETS.register_module()
class CustomSourceDataset(data.Dataset):
    """
    自定义源域数据集，用于加载您组织好的源域数据
    """

    def __init__(self, config):
        self.dataset_path = config.DATA_PATH
        self.class_choice = config.CLASS_CHOICE
        self.split = config.SPLIT

        # 获取所有样本目录
        self.sample_dirs = sorted([d for d in os.listdir(self.dataset_path)
                                  if os.path.isdir(os.path.join(self.dataset_path, d))])

        print(f"找到 {len(self.sample_dirs)} 个样本目录")

        self.input_ls = []
        self.gt_ls = []

        for sample_dir in self.sample_dirs:
            sample_path = os.path.join(self.dataset_path, sample_dir)

            # 读取partial点云 (raw.ply)
            partial_path = os.path.join(sample_path, 'raw.ply')
            if os.path.exists(partial_path):
                partial_pcd = read_ply_xyz(partial_path)
                # 确保点云数量为2048（根据模型要求）
                if partial_pcd.shape[0] != 2048:
                    if partial_pcd.shape[0] > 2048:
                        choice = np.random.choice(
                            partial_pcd.shape[0], 2048, replace=False)
                        partial_pcd = partial_pcd[choice, :]
                    else:
                        # 填充不足的点
                        num_pad = 2048 - partial_pcd.shape[0]
                        pad_points = partial_pcd[np.random.choice(
                            partial_pcd.shape[0], num_pad), :]
                        partial_pcd = np.concatenate(
                            [partial_pcd, pad_points], axis=0)
                self.input_ls.append(partial_pcd.astype(np.float32))
            else:
                print(f"警告: 找不到 {partial_path}")
                continue

            # 读取complete点云 (complete.ply) - 修改这里以处理PLY格式
            complete_path = os.path.join(sample_path, 'complete.ply')
            if os.path.exists(complete_path):
                # 由于complete.ply也是PLY格式，使用read_ply_xyz读取
                complete_pcd = read_ply_xyz(complete_path)
                # 确保点云数量为2048（根据模型要求）
                if complete_pcd.shape[0] != 2048:
                    if complete_pcd.shape[0] > 2048:
                        choice = np.random.choice(
                            complete_pcd.shape[0], 2048, replace=False)
                        complete_pcd = complete_pcd[choice, :]
                    else:
                        # 填充不足的点
                        num_pad = 2048 - complete_pcd.shape[0]
                        pad_points = complete_pcd[np.random.choice(
                            complete_pcd.shape[0], num_pad), :]
                        complete_pcd = np.concatenate(
                            [complete_pcd, pad_points], axis=0)
                self.gt_ls.append(complete_pcd.astype(np.float32))
            else:
                print(f"警告: 找不到 {complete_path}")
                continue

        # 确保输入和GT数据数量一致
        assert len(self.input_ls) == len(self.gt_ls), "输入数据和GT数据数量不一致"

        # 归一化 (核心修改: 使用 GT 的中心和尺度对 Partial 和 GT 进行统一缩放)
        # normalized_input = []
        # normalized_gt = []
        # for p_pcd, c_pcd in zip(self.input_ls, self.gt_ls):
        #     # 以完整的 GT 为基准计算归一化参数
        #     _, centroid, m = pc_normalize(c_pcd)
        #     # 应用到两者
        #     p_pcd = (p_pcd - centroid) / m
        #     c_pcd = (c_pcd - centroid) / m
        #     normalized_input.append(p_pcd)
        #     normalized_gt.append(c_pcd)
        
        # self.input_ls = normalized_input
        # self.gt_ls = normalized_gt

        # 对点云进行轴交换，统一坐标系
        # self.input_ls = [swap_axis(itm, swap_mode='210')
        #                  for itm in self.input_ls]
        # self.gt_ls = [swap_axis(itm, swap_mode='210') for itm in self.gt_ls]

        print(f"成功加载 {len(self.input_ls)} 个数据样本")

    def __getitem__(self, index):
        stem = index
        input_pcd = self.input_ls[index]
        gt_pcd = self.gt_ls[index]
        return torch.from_numpy(gt_pcd), torch.from_numpy(input_pcd), torch.tensor(stem)

    def __len__(self):
        return len(self.input_ls)


@DATASETS.register_module()
class CustomTargetDataset(data.Dataset):
    """
    自定义目标域数据集
    """

    def __init__(self, config):
        self.dataset_path = config.DATA_PATH
        self.class_choice = config.CLASS_CHOICE
        self.split = config.SPLIT

        # 获取所有PLY文件
        self.file_paths = sorted(
            glob.glob(os.path.join(self.dataset_path, '*.ply')))
        self.stems = [os.path.splitext(os.path.basename(path))[
            0] for path in self.file_paths]

        # 读取所有PLY文件
        self.input_ls = []
        for path in self.file_paths:
            pcd = read_ply_xyz(path)
            # 确保点云数量为2048（根据模型要求）
            if pcd.shape[0] != 2048:
                # 如果点数不足或过多，进行采样或填充
                if pcd.shape[0] > 2048:
                    choice = np.random.choice(
                        pcd.shape[0], 2048, replace=False)
                    pcd = pcd[choice, :]
                else:
                    # 填充不足的点
                    num_pad = 2048 - pcd.shape[0]
                    pad_points = pcd[np.random.choice(
                        pcd.shape[0], num_pad), :]
                    pcd = np.concatenate([pcd, pad_points], axis=0)
            self.input_ls.append(pcd.astype(np.float32))

        # 归一化 (目标域没有 GT，则对自身进行归一化)
        # normalized_input = []
        # for pcd in self.input_ls:
        #     pcd, _, _ = pc_normalize(pcd)
        #     normalized_input.append(pcd)
        # self.input_ls = normalized_input

        # 对点云进行轴交换，统一坐标系
        # input_ls_swapped = [swap_axis(itm, swap_mode='210')
        #                     for itm in self.input_ls]
        # self.input_ls = input_ls_swapped

        print(f"目标域数据集加载了 {len(self.input_ls)} 个样本")

    def __getitem__(self, index):
        stem = self.stems[index]
        input_pcd = self.input_ls[index]
        return torch.from_numpy(input_pcd), torch.tensor(index)

    def __len__(self):
        return len(self.input_ls)
