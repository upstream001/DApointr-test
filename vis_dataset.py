import os
import numpy as np
from utils.io import read_ply_xyz
import open3d as o3d


def visualize_dataset(dataset_path):
    """
    可视化 organized_source_dataset 数据集中的点云，并打印每个点云的点数
    """
    # 获取所有样本目录
    sample_dirs = sorted([d for d in os.listdir(dataset_path)
                         if os.path.isdir(os.path.join(dataset_path, d))])

    print(f"数据集包含 {len(sample_dirs)} 个样本")

    for i, sample_dir in enumerate(sample_dirs):
        sample_path = os.path.join(dataset_path, sample_dir)

        # 检查raw.ply和complete.ply是否存在
        raw_path = os.path.join(sample_path, 'raw.ply')
        complete_path = os.path.join(sample_path, 'complete.ply')

        print(f"\n样本 {i+1}: {sample_dir}")

        # 读取并显示partial点云 (raw.ply)
        if os.path.exists(raw_path):
            raw_pcd = read_ply_xyz(raw_path)
            print(f"  Raw点云: {raw_pcd.shape[0]} 个点")

            # 创建Open3D点云对象
            raw_o3d_pcd = o3d.geometry.PointCloud()
            raw_o3d_pcd.points = o3d.utility.Vector3dVector(raw_pcd)
            raw_o3d_pcd.paint_uniform_color([1, 0, 0])  # 红色表示partial点云
        else:
            print(f"  警告: 找不到 {raw_path}")
            raw_o3d_pcd = None

        # 读取并显示complete点云 (complete.ply)
        if os.path.exists(complete_path):
            complete_pcd = read_ply_xyz(complete_path)
            print(f"  Complete点云: {complete_pcd.shape[0]} 个点")

            # 创建Open3D点云对象
            complete_o3d_pcd = o3d.geometry.PointCloud()
            complete_o3d_pcd.points = o3d.utility.Vector3dVector(complete_pcd)
            complete_o3d_pcd.paint_uniform_color([0, 0, 1])  # 蓝色表示complete点云
        else:
            print(f"  警告: 找不到 {complete_path}")
            complete_o3d_pcd = None

        # 可视化点云
        geometries = []
        if raw_o3d_pcd is not None:
            # 将raw点云平移以方便观察
            raw_o3d_pcd = raw_o3d_pcd.translate([-3, 0, 0])
            geometries.append(raw_o3d_pcd)

        if complete_o3d_pcd is not None:
            geometries.append(complete_o3d_pcd)

        if geometries:
            print(f"  显示点云... (红色为partial，蓝色为complete，可能重叠)")
            o3d.visualization.draw_geometries(geometries,
                                              window_name=f"Sample {i+1}: {sample_dir}")
        else:
            print("  没有可显示的点云")

        # 询问是否继续
        if i < len(sample_dirs) - 1:
            user_input = input("按 Enter 继续下一个样本，输入 'q' 退出: ")
            if user_input.lower() == 'q':
                break


if __name__ == "__main__":
    dataset_path = "/home/tianqi/DAPoinTr/data/organized_source_dataset"
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
    else:
        visualize_dataset(dataset_path)
