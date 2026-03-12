import numpy as np
import pickle
import os
import glob

pkl_dir = r"D:\比赛与大项目\2026寒假集训\魔法原子\1212_bvh_yyh_wide_foot"
output_dir = r"D:\比赛与大项目\2026寒假集训\魔法原子\converted_npy"

os.makedirs(output_dir, exist_ok=True)

# 正确的腿关节 index
LEG_JOINT_IDS = [
    1,2,3,4,5,6,
    8,9,10,11,12,13
]

for pkl_file in glob.glob(os.path.join(pkl_dir, "*.pkl")):

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    root_pos = np.array(data["root_pos"])
    root_rot = np.array(data["root_rot"])
    joint_pos = np.array(data["dof_pos"])

    fps = data["fps"]

    frames = joint_pos.shape[0]

    # 只截取腿部关节
    joint_pos_leg = joint_pos[:, LEG_JOINT_IDS]

    # debug检查
    print("Processing:", os.path.basename(pkl_file))
    print("frames:", frames)
    print("original joint_dim:", joint_pos.shape[1])
    print("leg joint_dim:", joint_pos_leg.shape[1])

    motion = {
        "joints_list": LEG_JOINT_IDS,
        "joint_positions": [joint_pos_leg[i].astype(np.float32) for i in range(frames)],
        "root_position": [root_pos[i].astype(np.float32) for i in range(frames)],
        "root_quaternion": [root_rot[i].astype(np.float32) for i in range(frames)],
        "fps": float(fps)
    }

    save_path = os.path.join(
        output_dir,
        os.path.basename(pkl_file).replace(".pkl", ".npy")
    )

    np.save(save_path, motion)

    print("Converted:", save_path)
    print("-"*50)