
import h5py
import numpy as np
import os

def check_normalization(file_path):
    print(f"\n正在检查文件: {file_path}")
    if not os.path.exists(file_path):
        print("错误: 文件不存在")
        return

    with h5py.File(file_path, 'r') as f:
        # 查看文件中的 key
        print(f"Keys in H5: {list(f.keys())}")
        
        # 获取点云数据 (通常是 'complete_pcds' 或 'incomplete_pcds')
        if 'complete_pcds' in f:
            data = f['complete_pcds'][()]
            name = 'Complete PCDs'
        elif 'incomplete_pcds' in f:
            data = f['incomplete_pcds'][()]
            name = 'Incomplete PCDs'
        else:
            print("错误: 找不到点云数据 key")
            return

        print(f"数据形状: {data.shape}")
        
        # 计算每个样本的质心
        centroids = np.mean(data, axis=1) # [N, 3]
        avg_centroid = np.mean(centroids, axis=0)
        
        # 计算每个样本点到原点的距离
        dist_from_origin = np.sqrt(np.sum(data**2, axis=2)) # [N, 2048]
        max_dists = np.max(dist_from_origin, axis=1) # 每个样本的最远点距离
        
        print(f"--- 统计结果 ({name}) ---")
        print(f"平均质心 (Average Centroid): {avg_centroid}")
        print(f"质心偏差最大值: {np.max(np.abs(centroids))}")
        print(f"所有样本中点到原点的最大距离: {np.max(max_dists):.6f}")
        print(f"所有样本中点到原点的最小最大距离: {np.min(max_dists):.6f}")
        print(f"平均最大距离 (Average Max Distance): {np.mean(max_dists):.6f}")
        
        # 判断标准：
        # 如果质心接近 [0,0,0] 且 平均最大距离接近 1.0，则说明是单位球归一化
        if np.all(np.abs(avg_centroid) < 1e-2) and np.abs(np.mean(max_dists) - 1.0) < 0.1:
            print(">>> 结论: 该数据集【已归一化】到单位球 (Unit Sphere)。")
        elif np.max(np.abs(data)) <= 1.0001:
            print(">>> 结论: 该数据集【已归一化】到 [-1, 1] 范围内。")
        else:
            print(">>> 结论: 该数据集【未归一化】到标准范围或使用了非标准缩放。")

if __name__ == "__main__":
    dataset_dir = "/home/tianqi/DAPoinTr/data/our_data"
    files_to_check = ["train_data.h5", "test_data.h5"]
    
    for filename in files_to_check:
        check_normalization(os.path.join(dataset_dir, filename))
