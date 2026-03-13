import numpy as np
import os

# 定义你的 .npy 文件路径
npy_file = r"your_path_to_files"

try:
    data = np.load(npy_file, allow_pickle=True)
    
    # 情况 A: 数据是一个字典 (通常包含 'root_pos', 'joint_pos' 等)
    if isinstance(data.item(), dict):
        content = data.item()
        print("--- 专家数据包含以下 Keys ---")
        for key in content.keys():
            # 顺便打印每个 key 对应的维度，方便对齐
            val = content[key]
            shape = val.shape if hasattr(val, 'shape') else "N/A"
            print(f"Key: {key:15} | Shape: {shape}")
            
    # 情况 B: 数据直接是一个 ndarray
    else:
        print("--- 专家数据是一个直接的 ndarray ---")
        print(f"Shape: {data.shape}")

except Exception as e:
    print(f"读取失败: {e}")

def analyze_amp_15d_features(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    # 加载数据字典
    data = np.load(file_path, allow_pickle=True).item()
    
    # 原始组件
    root_pos = np.array(data['root_position'])    # (N, 3)
    root_quat = np.array(data['root_quaternion']) # (N, 4)
    joint_pos = np.array(data['joint_positions']) # (N, 12)

    print(f"--- 原始维度分析 ---")
    print(f"Root Pos: {root_pos.shape}, Root Quat: {root_quat.shape}, Joint Pos: {joint_pos.shape}")

    # --- 模拟 15 维特征拼接逻辑 ---
    # 逻辑猜测：1 (Root Z) + 4 (Root Quat) + 10 (Joints?) = 15
    # 或者 3 (Root Pos) + 0 (Quat?) + 12 (Joints) = 15
    
    # 我们直接取前几帧，打印可能的组合维度
    print(f"\n--- 尝试寻找那 15 维的组合 ---")
    
    # 组合 1: Root Z (1) + Root Quat (4) + 前 10 个关节 (10) = 15
    feat_1 = np.concatenate([root_pos[:1, 2:], root_quat[:1], joint_pos[:1, :10]], axis=-1)
    
    # 组合 2: Root Pos (3) + Joint Pos (12) = 15 (没有旋转)
    feat_2 = np.concatenate([root_pos[:1], joint_pos[:1]], axis=-1)
    
    # 组合 3: Root Quat (4) + Joint Pos (11?) = 15
    # ...
    
    print(f"组合 A (Root Z + Quat + 10 Joints) 维度: {feat_1.shape}")
    print(f"组合 B (Root Pos + 12 Joints)      维度: {feat_2.shape}")

    # 打印 15 维数据的第一帧 (假设是组合 B)
    if feat_2.shape[1] == 15:
        print("\n✅ 匹配成功！那 15 维很可能是：Root Position (3) + Joint Positions (12)")
        print(f"第一帧数据内容:\n{feat_2[0]}")
    elif feat_1.shape[1] == 15:
        print("\n✅ 匹配成功！那 15 维很可能是：Root Z (1) + Root Quat (4) + 10 Joints")
        print(f"第一帧数据内容:\n{feat_1[0]}")
    else:
        print("\n❌ 未能直接匹配 15 维，正在打印各部分值以供检查：")
        print(f"Root Position (前3维): {root_pos[0]}")
        print(f"Root Quaternion (前4维): {root_quat[0]}")
        print(f"Joint Positions (前12维): {joint_pos[0]}")

if __name__ == "__main__":
    analyze_amp_15d_features(npy_file)
