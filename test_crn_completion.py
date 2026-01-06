
import os
import sys
import torch
import numpy as np
import h5py
import open3d as o3d
import argparse

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc

def visualize_results(partial, completion, window_name="Point Cloud Completion"):
    """
    使用 Open3D 可视化残缺点云和补全后的点云
    """
    # 处理补全后的点云 (通常 completion 是 [N, 3])
    pcd_complete = o3d.geometry.PointCloud()
    pcd_complete.points = o3d.utility.Vector3dVector(completion)
    # 补全点云设为蓝色
    pcd_complete.paint_uniform_color([0, 0, 1])
    
    # 处理残缺点云
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial)
    # 残缺点云设为红色
    pcd_partial.paint_uniform_color([1, 0, 0])
    
    # 平移补全后的点云以便并排对比（可选）
    # pcd_complete.translate([0.6, 0, 0])
    
    print(f"显示窗口: {window_name} (红色: 残缺, 蓝色: 补全)")
    print("操作提示: 鼠标拖动旋转, 滚轮缩放, 按 'q' 退出当前视图并看下一个")
    
    # 同时显示两者（重叠显示可以看补全效果）
    # 创建一个窗口并添加两个点云，但为了对比清晰，我们可以稍微偏移一下补全后的点云
    # 或者分两个窗口？这里选择显示在同一个窗口，但补全点云稍微平移一点
    
    # 方案一：重叠显示 (补全包含残缺)
    o3d.visualization.draw_geometries([pcd_partial, pcd_complete.translate([0.6, 0, 0])], 
                                      window_name=window_name)

def main():
    # 路径配置
    config_path = '/home/tianqi/DAPoinTr/experiments/DAPoinTr/CRN_models/crn_source_only/config.yaml'
    ckpt_path = '/home/tianqi/DAPoinTr/experiments/DAPoinTr/CRN_models/crn_source_only/ckpt_source_best.pth'
    data_path = '/home/tianqi/DAPoinTr/data/our_data/test_data.h5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载配置
    config = cfg_from_yaml_file(config_path)
    
    # 2. 构建模型
    print("正在构建模型...")
    model = builder.model_builder(config.model)
    
    # 3. 加载权重
    print(f"正在加载权重: {ckpt_path}")
    builder.load_model(model, ckpt_path)
    model.to(device)
    model.eval()
    
    # 4. 加载测试数据
    print(f"正在从 H5 加载数据: {data_path}")
    with h5py.File(data_path, 'r') as f:
        # 注意: 如果你的 model 之前是在特定类别上跑的，这里可以筛选
        # 假设我们直接取前几个样本
        partial_pcds = f['incomplete_pcds'][()] # [N, 2048, 3]
        gt_pcds = f['complete_pcds'][()]
        
    num_samples = min(5, len(partial_pcds))
    print(f"准备测试前 {num_samples} 个样本...")
    
    with torch.no_grad():
        for i in range(num_samples):
            partial_np = partial_pcds[i].astype(np.float32)
            
            # 转换为 tensor 并送入模型
            partial_ts = torch.from_numpy(partial_np).unsqueeze(0).to(device)
            
            # 模型推理
            # DAPoinTr 返回: coarse_point, relative_xyz, out
            coarse_points, relative_xyz, out = model(partial_ts)
            
            # 获取最终补全结果 (取精细层的最后一个输出)
            completion_np = relative_xyz[-1].squeeze(0).cpu().numpy()
            
            print(f"\n--- 样本 {i} ---")
            print(f"残缺点云点数: {partial_np.shape[0]}")
            print(f"补全点云点数: {completion_np.shape[0]}")
            
            # 可视化
            visualize_results(partial_np, completion_np, window_name=f"Sample {i} Analysis")

if __name__ == "__main__":
    main()
