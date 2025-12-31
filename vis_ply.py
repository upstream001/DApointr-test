import open3d as o3d
import argparse
import numpy as np
import os

def load_point_cloud(file_path):
    """
    加载点云文件，支持 .ply, .pcd, .npy 格式
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension in ['.pcd', '.ply']:
        pcd = o3d.io.read_point_cloud(file_path)
    elif file_extension == '.npy':
        pts = np.load(file_path)
        # 如果是 (B, N, 3) 形状，取第一个 batch
        if pts.ndim == 3:
            pts = pts[0]
        # 如果是 (N, 3) 形状
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")
    return pcd

def visualize(file1, file2, offset=False):
    try:
        pcd1 = load_point_cloud(file1)
        pcd2 = load_point_cloud(file2)
    except Exception as e:
        print(f"读取点云时发生错误: {e}")
        return

    if pcd1.is_empty() and pcd2.is_empty():
        print("错误: 两个点云都为空。")
        return

    # 为区分两个点云，分别着色
    # 第一个点云着红色 (Red)
    pcd1.paint_uniform_color([1, 0, 0])
    # 第二个点云着蓝色 (Blue)
    pcd2.paint_uniform_color([0, 0, 1])

    if offset:
        # 计算偏移，使点云并排显示
        bbox = pcd1.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        # 沿着 X 轴平移 1.2 倍宽度
        pcd2.translate([extent[0] * 1.2, 0, 0])
        print("已启用偏移: 并排显示。")
    else:
        print("未启用偏移: 重叠显示（方便对比形状差异）。")

    print(f"正在显示:")
    print(f"  红色 (Red):  {file1}")
    print(f"  蓝色 (Blue): {file2}")
    
    # 打印点数信息
    print(f"点云1点数: {len(pcd1.points)}")
    print(f"点云2点数: {len(pcd2.points)}")
    print("\n操作提示: 使用鼠标左键旋转，右键平移，滚轮缩放。按 'q' 键退出。")

    # 启动可视化窗口
    o3d.visualization.draw_geometries([pcd1, pcd2], 
                                      window_name="PoinTr 点云对比可视化",
                                      width=1280, 
                                      height=720,
                                      left=50, 
                                      top=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoinTr 项目点云对比可视化脚本")
    parser.add_argument("--file1", default="/home/tianqi/DAPoinTr/complete.ply", type=str, help="第一个点云文件路径 (.ply, .pcd, .npy)")
    parser.add_argument("--file2", default="/home/tianqi/DAPoinTr/data/organized_source_dataset/sample_000/raw.ply", type=str, help="第二个点云文件路径 (.ply, .pcd, .npy)")
    parser.add_argument("--offset", action="store_true", help="是否并排显示点云 (默认重叠显示)")
    
    args = parser.parse_args()
    visualize(args.file1, args.file2, args.offset)
