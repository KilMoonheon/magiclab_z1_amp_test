# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg
)

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 核心：必须指定动态加载的路径
    runner_class_name = "amp_rsl_rl.runners.AMPOnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False

    # 策略配置：注意 class_name 必须匹配源码里的字典
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # 算法配置：必须改为 AMP_PPO
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMP_PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # 判别器配置 (对应你源码中的 self.discriminator_cfg)
    discriminator = {
        "hidden_dims": [1024, 512],
        "reward_scale": 2.0,
        "loss_type": "BCEWithLogits",
        "empirical_normalization": True,
    }

    # 专家数据集配置 (对应你源码中的 self.dataset_cfg)
    dataset = {
        "amp_data_path": "/home/oper/magiclab_rl_lab/source/magiclab_rl_lab/magiclab_rl_lab/data/robots/converted_npy",
        "datasets": {
            "run_ahead_001": 1.0,
            "walk_ahead_002": 1.0,
            "walk_back_002": 1.0,
            "walk_left_001": 1.0
        },
        "slow_down_factor": 1.0,
    }
