import open3d as o3d
import numpy as np
import argparse
import os

def measure_point_cloud(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在。")
        return

    # 读取点云文件
    print(f"正在读取点云: {file_path} ...")
    pcd = o3d.io.read_point_cloud(file_path)
    
    if pcd.is_empty():
        print("错误: 点云为空或无法读取。")
        return

    # 将点云转换为 numpy 数组
    points = np.asarray(pcd.points)
    
    # 计算点云中心 (质心)
    center = np.mean(points, axis=0)
    print(f"点云中心坐标: {center}")

    # 计算所有点到中心的距离
    # np.linalg.norm 计算欧几里得距离
    distances = np.linalg.norm(points - center, axis=1)

    # 计算平均距离
    avg_distance = np.mean(distances)
    
    # 计算最远点距离
    max_distance = np.max(distances)

    # 计算最远点索引
    max_idx = np.argmax(distances)
    farthest_point = points[max_idx]

    print("-" * 30)
    print(f"计算结果:")
    print(f"平均距离 (Average Distance to Center): {avg_distance:.6f}")
    print(f"最远距离 (Max Distance to Center):     {max_distance:.6f}")
    print(f"最远点坐标: {farthest_point}")
    print(f"总点数 (Total Points):                {len(points)}")
    print("-" * 30)

    # --- 可视化部分 ---
    print("正在准备可视化...")
    
    # 1. 为点云着色
    # 先给所有点涂上中灰色
    colors = np.full((len(points), 3), [0.7, 0.7, 0.7]) 
    # 将最远点涂成鲜艳的红色
    colors[max_idx] = [1, 0, 0] 
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 2. 为了更显眼，在中心和最远点各放一个小球
    # 计算一个小球的合适半径 (例如最远距离的 2%)
    sphere_radius = max_distance * 0.02
    
    # 中心点球体 (蓝色)
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    center_sphere.paint_uniform_color([0, 0, 1]) # 蓝色
    center_sphere.translate(center)
    
    # 最远点球体 (红色)
    farthest_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)
    farthest_sphere.paint_uniform_color([1, 0, 0]) # 红色
    farthest_sphere.translate(farthest_point)

    # 3. 添加一个坐标轴参考
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max_distance * 0.2, origin=center)

    print("启动窗口: 蓝色小球为中心，红色大球为最远点。")
    o3d.visualization.draw_geometries([pcd, center_sphere, farthest_sphere, coord_frame],
                                      window_name="Point Cloud Measurement Visualization",
                                      width=1024, height=768)

    # --- 保存部分 ---
    save_path = file_path.replace(".ply", "_removed_farthest.ply")
    print(f"\n正在处理删除操作...")
    
    # 使用索引选择，invert=True 表示删除这些索引的点
    cleaned_pcd = pcd.select_by_index([max_idx], invert=True)
    
    print(f"原始点数: {len(pcd.points)}")
    print(f"删除后点数: {len(cleaned_pcd.points)}")
    
    # 保存新点云
    # success = o3d.io.write_point_cloud(save_path, cleaned_pcd)
    # if success:
    #     print(f"成功保存处理后的点云至: {save_path}")
    # else:
    #     print("保存失败。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算点云中点到中心的平均和最大距离、可视化并删除最远点")
    parser.add_argument("--input", default="/home/tianqi/DAPoinTr/data/organized_source_dataset/sample_000/complete_removed_farthest_removed_farthest_removed_farthest_removed_farthest.ply", help="输入的 .ply 文件路径")
    
    args = parser.parse_args()
    
    measure_point_cloud(args.input)
