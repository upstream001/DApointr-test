import torch
import open3d as o3d
import numpy as np
import argparse
import os
import sys

# 将项目根目录添加到系统路径以确保能导入 models, tools 等
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from tools import builder
from utils.config import get_config
from utils.io import read_ply_xyz
from utils.pc_transform import swap_axis

def get_args():
    parser = argparse.ArgumentParser(description='DAPoinTr 单个点云推理脚本')
    parser.add_argument('--config', type=str, default='cfgs/CRN_models/CustomDAPoinTr_SourceOnly.yaml', help='配置文件路径')
    parser.add_argument('--ckpt', type=str, default='/home/tianqi/DAPoinTr/experiments/CustomDAPoinTr_SourceOnly/CRN_models/default/ckpt_source_best.pth', help='模型权重路径')
    parser.add_argument('--input', type=str, default='/home/tianqi/DAPoinTr/data/target_dataset/partial/000.ply', help='输入的 .ply 点云文件路径')
    parser.add_argument('--output_dir', type=str, default='experiments/test_pointcloud', help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42)
    # 模拟 main.py 中需要的 args
    parser.add_argument('--experiment_path', type=str, default='./experiments')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--start_ckpts', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test', action='store_true', default=False)
    
    args = parser.parse_args()
    return args

def pc_normalize(pc):
    """将点云归一化到单位球内 (Unit Sphere)"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m < 1e-6:
        m = 1.0
    pc /= m
    return pc, centroid, m

def preprocess_point_cloud(path):
    """读取点云并处理为 2048 点，同时进行归一化和坐标轴转换"""
    # 1. 读取坐标
    points = read_ply_xyz(path)
    print(f"原始点云数量: {points.shape[0]}")
    
    # 保存一份原始点云用于最终对比的可视化（可选）
    original_points = points.copy()
    
    # 2. 采样/填充到 2048 点
    if points.shape[0] != 2048:
        if points.shape[0] > 2048:
            choice = np.random.choice(points.shape[0], 2048, replace=False)
            points = points[choice, :]
        else:
            num_pad = 2048 - points.shape[0]
            pad_points = points[np.random.choice(points.shape[0], num_pad), :]
            points = np.concatenate([points, pad_points], axis=0)
    
    # 3. 归一化 (核心新增)
    # points, centroid, m = pc_normalize(points)
    
    # 4. 轴变换 (已取消)
    points = swap_axis(points, swap_mode='210')
    
    # 5. 转换为 Tensor
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)
    return points_tensor, original_points

def visualize_result(input_pc, output_pc):
    """可视化输入和输出点云"""
    # 转换为 Open3D 格式
    pcd_in = o3d.geometry.PointCloud()
    pcd_in.points = o3d.utility.Vector3dVector(input_pc)
    pcd_in.paint_uniform_color([1, 0, 0]) # 输入设为红色 (与 vis_ply 一致)
    
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(output_pc)
    pcd_out.paint_uniform_color([0, 0, 1]) # 输出设为蓝色
    
    # 平移以便并排观察
    bbox = pcd_in.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    pcd_out.translate([extent[0] * 1.5, 0, 0])
    
    print("正在启动可视化窗口... (红色: 原始输入, 蓝色: 并排展示的还原后结果)")
    o3d.visualization.draw_geometries([pcd_in, pcd_out], window_name="DAPoinTr Inference Result (Normalized & Restored)")

def main():
    args = get_args()
    
    # 获取配置
    config = get_config(args, logger=None)
    
    # 构建模型
    print("正在构建模型...")
    model = builder.model_builder(config.model)
    
    # 加载权重
    print(f"正在加载预训练权重: {args.ckpt}")
    builder.load_model(model, args.ckpt)
    
    model.cuda()
    model.eval()
    
    # 处理输入数据
    print(f"正在处理输入文件: {args.input}")
    input_tensor, original_points = preprocess_point_cloud(args.input)
    input_tensor = input_tensor.cuda()
    
    # 推理
    print("开始推理...")
    with torch.no_grad():
        coarse_points, densest_points, _ = model(input_tensor)
        
        # 取得推理结果 [1, 2048, 3] -> [2048, 3]
        prediction = densest_points[-1].squeeze(0).cpu().numpy()
        
        # --- 还原操作 (已全部取消) ---
        # 1. 轴还原
        prediction = swap_axis(prediction, swap_mode='210')
        # 2. 缩放还原
        # prediction = prediction * m
        # 3. 平移还原
        # prediction = prediction + centroid
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "reconstruction.ply")
    
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(prediction)
    o3d.io.write_point_cloud(out_path, pcd_final)
    print(f"结果已保存至: {out_path} (已还原至原始空间坐标)")
    
    # 可视化 (展示真实的原始点云 vs 真实的还原后结果)
    visualize_result(original_points, prediction)

if __name__ == '__main__':
    main()
