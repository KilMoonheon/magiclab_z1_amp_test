# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys
import os
import numpy as np
import torch
SIM_ORDER_INDICES = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]

def patch_motion_files(motion_dir):
    """
    既然找不到代码在哪里加载，我们直接在源头上把所有 .npy 的内容顺序改了，
    并保存为 '_ready.npy' 后缀的文件，然后让 cfg 指向新文件。
    这是最暴力但也最有效的办法。
    """
    ready_dir = os.path.join(motion_dir, "reordered")
    os.makedirs(ready_dir, exist_ok=True)
    
    for file in os.listdir(motion_dir):
        if file.endswith(".npy") and not file.endswith("_ready.npy"):
            path = os.path.join(motion_dir, file)
            data = np.load(path, allow_pickle=True).item()
            
            # 执行重映射
            joint_pos = np.array(data['joint_positions'])
            if joint_pos.shape[1] == 12:
                data['joint_positions'] = joint_pos[:, SIM_ORDER_INDICES]
                # 同时也更新一下 joints_list 字符串（如果有的话），防止判别器报错
                if 'joints_list' in data:
                    new_list = [data['joints_list'][i] for i in SIM_ORDER_INDICES]
                    data['joints_list'] = new_list
                
                new_path = os.path.join(ready_dir, file.replace(".npy", "_ready.npy"))
                np.save(new_path, data)
                print(f"[Patch] 已重排并保存: {new_path}")
    
    return ready_dir

# --- 1. 终极路径修复补丁 ---
base_site_packages = "/home/oper/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages"
# 确保定义这两个核心根目录
exts_root = os.path.join(base_site_packages, "isaacsim/exts") 
cache_root = os.path.join(base_site_packages, "isaacsim/extscache")

if os.path.exists(cache_root):
    import subprocess
    try:
        # 深度搜索所有含有 .so 的目录
        cmd = f"find {cache_root} -name '*.so' -exec dirname {{}} \; | sort -u"
        lib_dirs = subprocess.check_output(cmd, shell=True).decode().splitlines()
        
        # 注入 LD_LIBRARY_PATH
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + current_ld
        
        # 将扩展包根目录加入 sys.path
        cache_folders = [os.path.join(cache_root, f) for f in os.listdir(cache_root) 
                         if os.path.isdir(os.path.join(cache_root, f))]
        for folder in cache_folders:
            if folder not in sys.path:
                sys.path.insert(0, folder)
        
        print(f"已自动扫描并挂载 {len(lib_dirs)} 个库目录。")
    except Exception as e:
        print(f"路径扫描失败: {e}")

# --- 接下来是原有的 URDF 和 Magicbot 逻辑 ---
# 此时 exts_root 已经定义好了
urdf_ext_path = os.path.join(exts_root, "isaacsim.asset.importer.urdf")
if os.path.exists(urdf_ext_path):
    if urdf_ext_path not in sys.path:
        sys.path.insert(0, urdf_ext_path)
    os.environ["LD_LIBRARY_PATH"] = urdf_ext_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 补上 Magicbot 路径
magicbot_path = "/opt/magic_robotics/magicbot_z1_sdk/lib"
if os.path.exists(magicbot_path):
    os.environ["PYTHONPATH"] = magicbot_path + ":" + os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = magicbot_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 添加核心 URDF 路径
urdf_ext_path = os.path.join(exts_root, "isaacsim.asset.importer.urdf")
if os.path.exists(urdf_ext_path):
    if urdf_ext_path not in sys.path:
        sys.path.insert(0, urdf_ext_path)
    os.environ["LD_LIBRARY_PATH"] = urdf_ext_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 动态扩展命名空间
try:
    import isaacsim
    import omni
    def extend_ns(module, paths):
        for p in paths:
            target_dir = os.path.join(p, module.__name__)
            if os.path.exists(target_dir) and target_dir not in module.__path__:
                module.__path__.append(target_dir)
    
    all_search_paths = cache_folders + [urdf_ext_path]
    extend_ns(isaacsim, all_search_paths)
    extend_ns(omni, all_search_paths)
    print(f"已挂载 {len(cache_folders)} 个扩展包并同步加载库路径。")
except ImportError:
    pass

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if ("Magic" in task_spec.id) and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

import importlib

def get_runner_class(runner_class_name):
    """根据字符串动态加载 Runner 类"""
    try:
        if "." in runner_class_name:
            module_path, class_name = runner_class_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        else:
            # 如果只是短名称 "OnPolicyRunner"，则从标准库导入
            from rsl_rl.runners import OnPolicyRunner
            return OnPolicyRunner
    except (ImportError, AttributeError) as e:
        print(f"无法加载 Runner 类: {runner_class_name}，错误: {e}")
        # 备选方案：尝试直接导入标准 Runner
        from rsl_rl.runners import OnPolicyRunner
        return OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import magiclab_rl_lab.tasks  # noqa: F401
from magiclab_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if hasattr(agent_cfg, 'amp'):
        raw_path = os.path.expanduser(agent_cfg.amp["motion_files_path"])
        # 转换并获取新目录
        ready_path = patch_motion_files(raw_path)
        # 将配置指向转换后的目录
        agent_cfg.amp["motion_files_path"] = ready_path
        print(f"[INFO] AMP 专家数据已自动重映射并指向: {ready_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        # 尝试从 unwrapped 环境中获取 robot 资产
        # 在 Isaac Lab 中，资产通常存储在 scene 字典中，键名通常为 "robot" 或 "articulation"
        # 我们可以遍历 scene 中的所有 Articulation 资产
        print("-" * 50)
        print("[INFO] 正在扫描场景中的关节信息...")
        scene_entities = env.unwrapped.scene.articulations
        if scene_entities:
            for entity_name, entity_obj in scene_entities.items():
                joint_names = entity_obj.joint_names
                print(f"资产名称: {entity_name}")
                print(f"关节数量: {len(joint_names)}")
                print(f"关节列表: {joint_names}")
        else:
            print("[WARN] 未在 scene.articulations 中找到任何机器人资产。")
        print("-" * 50)
    except Exception as e:
        print(f"[WARN] 打印关节列表失败: {e}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = get_runner_class(agent_cfg.runner_class_name)
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
