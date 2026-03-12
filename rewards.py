from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from magiclab_rl_lab.tasks.locomotion import mdp
try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils
"""
Joint penalties.
"""

def compute_task_reward(self):

    # velocity tracking
    lin_vel_error = torch.sum(
        torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
        dim=1,
    )

    r_vel = torch.exp(-lin_vel_error / 0.25)

    # alive reward
    r_alive = 1.0

    # torque penalty
    r_torque = torch.sum(torch.square(self.torques), dim=1)

    reward = (
        2.0 * r_vel
        + 0.2 * r_alive
        - 0.0002 * r_torque
    )

    return reward

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_threshold: float, command_name: str = "base_velocity",  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < command_threshold)

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float, command_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward

def amp_reward(env: ManagerBasedRLEnv, resource_name: str) -> torch.Tensor:
    """从 RSL-RL 的 AMP 处理器中提取奖励。
    
    该函数会从环境注册的资源中寻找 AMP 管理器，并获取当前步的判别器得分。
    """
    # 这里的 resource_name 对应配置里的 "amp_handler"
    # Isaac Lab 在集成 RSL-RL AMP 时，会自动将处理后的奖励存放在特定的资源器中
    amp_handler = env.common_step_index.get(resource_name, None)
    
    if amp_handler is not None:
        # 返回判别器给出的风格奖励
        return amp_handler.get_reward()
    else:
        # 如果还没初始化好，返回全 0
        return torch.zeros(env.num_envs, device=env.device)
    

def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def foot_clearance_reward(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
# ) -> torch.Tensor:
#     """Reward the swinging feet for clearing a specified height off the ground"""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
#     foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
#     reward = foot_z_target_error * foot_velocity_tanh
#     return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data._w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty_decay(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    decay_std: float = 1.0, # 离开球多远后恢复“强制对称”
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")
) -> torch.Tensor:
    """
    动态对称惩罚：
    离球越近，惩罚越小（允许不对称步态去控球/踢球）；
    离球越远，惩罚越大（强制恢复标准步态跑位）。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    
    # 原始方差惩罚（越不对称，值越大）
    variance_error = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + \
                     torch.var(torch.clip(last_contact_time, max=0.5), dim=1)

    # 计算人球距离，生成“反向衰减”
    # 当 dist=0, mask=0 (不惩罚)；当 dist 变大, mask 趋向 1 (严厉惩罚)
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w[:, :2]
    dist = torch.norm(ball_pos - robot_pos, dim=-1)
    
    # 使用 1 - exp(-dist) 逻辑
    symmetry_mask = 1.0 - torch.exp(-dist / decay_std)

    return variance_error * symmetry_mask

"""
Feet Gait rewards.
"""


'''
def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward
'''

def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward



def feet_contact_number(
    env: ManagerBasedRLEnv,
    period: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Isaac Lab 风格的步态接触数奖励。"""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    
    # return float mask 1 is stance, 0 is swing
    phase = global_phase
    sin_pos = torch.sin(2 * torch.pi * phase)
    # Add double support phase
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    # left foot stance
    stance_mask[:, 0] = sin_pos >= 0
    # right foot stance
    stance_mask[:, 1] = sin_pos < 0

    cmd_norm = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)

    stance_mask[:, 0][cmd_norm < 0.02] = 1
    stance_mask[:, 1][cmd_norm < 0.02] = 1

    contact = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]) > 5

    reward = torch.where(contact == stance_mask, 1.0, -0.3)

    return torch.mean(reward, dim=1)


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


"""
Kick rewards.
"""

def ball_distance_decay_factor(
    env: ManagerBasedRLEnv, 
    std: float = 0.5, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")
) -> torch.Tensor:
    """计算距离衰减因子，作为其他奖励的乘数内核。"""
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w[:, :2]
    dist = torch.norm(ball_pos - robot_pos, dim=-1)
    # 使用高斯/指数衰减：离球越近，值越接近 1.0；离球越远，趋近于 0
    return torch.exp(-dist / std)


def ball_velocity_towards_goal_decay(
    env: ManagerBasedRLEnv,
    goal_pos: list[float],
    decay_std: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励球向球门移动的速度，但只有在机器人靠近球时该奖励才生效。"""
    ball_data = env.scene[ball_cfg.name].data
    ball_vel = ball_data.root_lin_vel_w[:, :2]
    ball_pos = ball_data.root_pos_w[:, :2]
    target_pos = torch.tensor(goal_pos, device=env.device, dtype=torch.float)[:2]

    # 方向向量
    to_goal_vec = target_pos - ball_pos
    to_goal_vec = to_goal_vec / (torch.norm(to_goal_vec, dim=1, keepdim=True) + 1e-6)
    
    # 基础速度奖励
    vel_towards_goal = torch.sum(ball_vel * to_goal_vec, dim=1)
    vel_reward = torch.clamp(vel_towards_goal, min=0.0)

    # 距离衰减集成
    decay = ball_distance_decay_factor(env, std=decay_std, asset_cfg=asset_cfg, ball_cfg=ball_cfg)
    
    return vel_reward * decay


def kick_alignment_decay(
    env: ManagerBasedRLEnv,
    goal_pos: list[float],
    decay_std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")
) -> torch.Tensor:
    """奖励“机器人-球-球门”三点一线对齐。距离越近，对齐的重要性越高。"""
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w[:, :2]
    target_pos = torch.tensor(goal_pos, device=env.device, dtype=torch.float)[:2]

    v_rb = ball_pos - robot_pos
    v_rb = v_rb / (torch.norm(v_rb, dim=1, keepdim=True) + 1e-6)

    v_bg = target_pos - ball_pos
    v_bg = v_bg / (torch.norm(v_bg, dim=1, keepdim=True) + 1e-6)

    alignment = torch.clamp(torch.sum(v_rb * v_bg, dim=1), min=0.0)
    
    # 距离衰减集成：防止机器人在极远处盲目追求对齐而忽略了跑位
    decay = ball_distance_decay_factor(env, std=decay_std, asset_cfg=asset_cfg, ball_cfg=ball_cfg)
    
    return alignment * decay

def dominant_leg_shooting_swing_decay(
    env: ManagerBasedRLEnv,
    goal_pos: list[float],
    dominant_leg_idx: 1,          # 指定惯用脚的索引（例如 14 是左，15 是右）
    robot_ball_std: float = 0.3,
    ball_goal_std: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")
) -> torch.Tensor:
    """强化惯用腿射门：只奖励指定腿向球的冲击速度，培养‘黄金右脚’。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w[:, :2]
    target_pos = torch.tensor(goal_pos, device=env.device, dtype=torch.float)[:2]

    # 获取指定惯用腿的速度和位置
    # body_ids[dominant_leg_idx] 选取特定索引的 link
    leg_id = [asset_cfg.body_ids[dominant_leg_idx]]
    foot_vel = asset.data.body_lin_vel_w[:, leg_id, :2]
    foot_pos = asset.data.body_pos_w[:, leg_id, :2]

    # 逻辑同前：计算冲击速度
    to_ball_vec = ball_pos.unsqueeze(1) - foot_pos
    to_ball_vec = to_ball_vec / (torch.norm(to_ball_vec, dim=-1, keepdim=True) + 1e-6)
    swing_vel = torch.clamp(torch.sum(foot_vel * to_ball_vec, dim=-1).squeeze(1), min=0.0)

    # 双重衰减因子
    rb_decay = ball_distance_decay_factor(env, std=robot_ball_std, asset_cfg=asset_cfg, ball_cfg=ball_cfg)
    bg_decay = torch.exp(-torch.norm(target_pos - ball_pos, dim=-1) / ball_goal_std)

    return swing_vel * rb_decay * bg_decay

def dribble_towards_goal_decay(
    env: ManagerBasedRLEnv,
    goal_pos: list[float],
    decay_std: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    带球奖励：奖励机器人和足球作为一个整体向球门靠近的速度。
    只有当机器人靠近球时（Decay因子高），且两者都在向目标移动时，奖励才成立。
    """
    # 获取数据
    robot_vel = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2]
    ball_data = env.scene[ball_cfg.name].data
    ball_vel = ball_data.root_lin_vel_w[:, :2]
    ball_pos = ball_data.root_pos_w[:, :2]
    target_pos = torch.tensor(goal_pos, device=env.device, dtype=torch.float)[:2]

    # 计算球到球门的方向向量
    to_goal_vec = target_pos - ball_pos
    to_goal_vec = to_goal_vec / (torch.norm(to_goal_vec, dim=1, keepdim=True) + 1e-6)

    # 1. 计算机器人向球门的速度投影
    robot_vel_towards_goal = torch.sum(robot_vel * to_goal_vec, dim=1)
    # 2. 计算足球向球门的速度投影
    ball_vel_towards_goal = torch.sum(ball_vel * to_goal_vec, dim=1)

    # 协同速度：取两者的共同进步部分（最小值），防止人跑得快球不动，或球踢飞了人不动
    # 使用 torch.min 强制要求两者同步移动
    combined_vel = torch.min(robot_vel_towards_goal, ball_vel_towards_goal)
    reward = torch.clamp(combined_vel, min=0.0)

    # 距离衰减：只有在“控球范围内”才叫带球
    decay = ball_distance_decay_factor(env, std=decay_std, asset_cfg=asset_cfg, ball_cfg=ball_cfg)

    return reward * decay

def ball_relative_velocity_penalty(
    env: ManagerBasedRLEnv,
    decay_std: float,               # 新增
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")
) -> torch.Tensor:
    robot_vel = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2]
    ball_vel = env.scene[ball_cfg.name].data.root_lin_vel_w[:, :2]
    relative_vel_norm = torch.norm(ball_vel - robot_vel, dim=-1)
    
    # 使用传入的 decay_std
    decay = ball_distance_decay_factor(env, std=decay_std, asset_cfg=asset_cfg, ball_cfg=ball_cfg)
    return relative_vel_norm * decay

def goal_score_reward(
    env: ManagerBasedRLEnv,
    goal_box: list[float],
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")
) -> torch.Tensor:
    """如果球进入球门区域，给予巨大的奖励。
    goal_box: [x_min, x_max, y_min, y_max]
    """
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w

    # 检查坐标是否在球门矩形框内
    in_goal = (
        (ball_pos[:, 0] > goal_box[0]) &
        (ball_pos[:, 0] < goal_box[1]) &
        (ball_pos[:, 1] > goal_box[2]) &
        (ball_pos[:, 1] < goal_box[3]) &
        (ball_pos[:, 2] < 1.2) # 高度限制
    )

    return in_goal.float()

def ball_boundary_penalty(
    env: ManagerBasedRLEnv,
    x_limit: float,
    y_limit: float,
    threshold: float = 0.5, # 距离边界多远时开始惩罚
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")
) -> torch.Tensor:
    """当球靠近边界时给予平滑的负奖励"""
    ball: RigidObject = env.scene[ball_cfg.name]
    relative_ball_pos = ball.data.root_pos_w - env.scene.env_origins
    
    # 计算距离边界的距离
    dist_x = x_limit - torch.abs(relative_ball_pos[:, 0])
    dist_y = y_limit - torch.abs(relative_ball_pos[:, 1])
    
    # 如果距离小于 threshold，则产生惩罚 (使用 exp 让靠近边缘时惩罚剧增)
    penalty_x = torch.where(dist_x < threshold, torch.exp(threshold - dist_x) - 1.0, 0.0)
    penalty_y = torch.where(dist_y < threshold, torch.exp(threshold - dist_y) - 1.0, 0.0)
    
    return -(penalty_x + penalty_y)

def heading_align_with_ball(
    env: ManagerBasedRLEnv, 
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """回归基础：只要求机器人的‘正脸’盯着球看"""
    
    asset = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    
    # 1. 计算球相对于机器人的位置向量 (XY平面)
    relative_pos_w = ball.data.root_pos_w - asset.data.root_pos_w
    ball_direction_w = relative_pos_w[:, :2]
    
    # 2. 获取机器人当前的朝向 (Forward向量)
    forward_l = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, forward_l)
    heading_direction_w = forward_w[:, :2]
    
    # 3. 归一化并计算相似度
    heading_dir_norm = torch.nn.functional.normalize(heading_direction_w, dim=-1)
    ball_dir_norm = torch.nn.functional.normalize(ball_direction_w, dim=-1)
    
    cosine_sim = torch.sum(heading_dir_norm * ball_dir_norm, dim=-1)
    
    # 使用平方项：引导更明确，但比 4 次方平稳
    return torch.pow(torch.clamp(cosine_sim, min=0.0), 1)

def get_soccer_phase_weights(
    env: ManagerBasedRLEnv, 
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> dict[str, torch.Tensor]:
    """
    计算三个阶段的平滑权重。
    - Phase 1: Seek (寻球) -> 距离球远时权重高
    - Phase 2: Dribble (带球) -> 距离球近且离球门远时高
    - Phase 3: Shoot (射门) -> 距离球近且离球门近时高
    """
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w[:, :2]
    goal_pos = torch.tensor([4.5, 0.0], device=env.device) # 假设球门在x=4.5

    dist_rb = torch.norm(ball_pos - robot_pos, dim=-1)
    dist_bg = torch.norm(goal_pos - ball_pos, dim=-1)

    # 1. 寻球权重：使用 Sigmoid 逻辑，当距离 > 0.6m 时主要权重在寻球
    # 距离球越近，seek_weight 越趋近于 0
    seek_weight = torch.sigmoid((dist_rb - 0.5) * 10.0)
    
    # 2. 交互权重（带球+射门）：1 - 寻球权重
    interact_weight = 1.0 - seek_weight
    
    # 3. 射门 vs 带球：在 interact 激活的前提下，根据球到球门的距离分配
    # 距离球门越近（< 2.0m），shoot_weight 越高
    shoot_weight = interact_weight * (1.0 - torch.sigmoid((dist_bg - 2.0) * 5.0))
    dribble_weight = interact_weight - shoot_weight

    return {
        "seek": seek_weight,
        "dribble": dribble_weight,
        "shoot": shoot_weight
    }

def seek_ball_reward(env, ball_cfg):
    """重构后的寻球：包含对齐球、跑位到球后、面向球门三个逻辑"""
    asset = env.scene["robot"]
    ball = env.scene[ball_cfg.name]
    
    # 1. 基础对齐：脸对着球 (确保视野不丢)
    align_ball = heading_align_with_ball(env, ball_cfg)
    
    # 2. 跑位：靠近球后方的 Setup Point (这是最关键的绕后逻辑)
    setup_reward = position_behind_ball_reward(env, offset=0.3) 
    
    # 3. 朝向对齐：身体看着球门 (确保踢球瞬间姿势正确)
    align_goal = body_align_with_goal(env)
    
    # 4. 动态权重
    weights = get_soccer_phase_weights(env, ball_cfg)
    
    # 组合奖励：跑位权重最高，对齐次之
    # 只有在 seek 阶段，这些奖励才生效
    total_seek = (setup_reward * 1.0 + align_ball * 0.5 + align_goal * 1.5)
    
    return total_seek * weights["seek"]

def dribble_phase_reward(
    env: ManagerBasedRLEnv,
    goal_pos: list[float],
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    # 获取球当前的 X 轴速度
    ball_vel_x = ball.data.root_lin_vel_w[:, 0]
    
    # 论文思路：奖励球向目标前进的速度。
    # 只要球向 x=4.5 方向移动，就给分，速度越快分越高
    progression_reward = torch.clamp(ball_vel_x, min=0.0) * 2.0 
    
    weights = get_soccer_phase_weights(env, ball_cfg, asset_cfg)
    return progression_reward * weights["dribble"]

def shoot_phase_reward(
    env: ManagerBasedRLEnv,
    goal_pos: list[float],           # 依然保留作为基础参考
    goal_box: list[float],           # [x_min, x_max, y_min, y_max]
    dominant_leg_idx: int,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    修改后的射门阶段奖励：
    1. 引入球门区域投影逻辑，不再死磕中心点。
    2. 结合论文的 Goal Progress 思路，奖励球向球门方向的冲刺速度。
    """
    # 1. 获取阶段权重
    weights = get_soccer_phase_weights(env, ball_cfg, asset_cfg)
    
    # 2. 获取球和机器人的状态
    ball = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_pos_w[:, :2]
    ball_vel = ball.data.root_lin_vel_w[:, :2]
    
    # 目标球门线的位置 (取 goal_box 的 x_min)
    target_x = goal_box[0] 
    goal_y_min = goal_box[2]
    goal_y_max = goal_box[3]

    # --- 核心修改：区域射门奖励 (Towards Goal Area) ---
    # 计算球到达球门线所需的预计时间: t = (target_x - current_x) / vel_x
    # 避免除以 0，且只考虑向前的速度 (vel_x > 0)
    safe_vel_x = torch.clamp(ball_vel[:, 0], min=1e-3)
    time_to_goal = (target_x - ball_pos[:, 0]) / safe_vel_x
    
    # 预测球到达目标线时的 Y 坐标: y_future = y_now + vel_y * t
    predicted_y = ball_pos[:, 1] + ball_vel[:, 1] * time_to_goal
    
    # 判定 1: 球是否正朝着球门框架内滚动 (预测 Y 在 y_min 和 y_max 之间)
    # 同时要求球在机器人前方 (time_to_goal > 0)
    is_towards_goal_area = (time_to_goal > 0) & (predicted_y > goal_y_min) & (predicted_y < goal_y_max)
    
    # 判定 2: 球的水平速度奖励 (参考论文 Goal Progress 权重 500)
    # 只要球向球门跑，就给分；如果指向区域内，给双倍奖励
    ball_speed_forward = torch.clamp(ball_vel[:, 0], min=0.0)
    area_shoot_reward = ball_speed_forward * (1.0 + is_towards_goal_area.float() * 2.0)

    # 3. 惯用脚冲击球的速度奖励 (原有逻辑，鼓励发力)
    swing_reward = dominant_leg_shooting_swing_decay(env, goal_pos, dominant_leg_idx, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
    
    # 4. 进球瞬间的大额奖励 (最终结果奖励)
    is_goal = goal_score_reward(env, goal_box, ball_cfg)
    
    # 5. 综合计算
    # 论文思路：奖励 = 动作质量(swing) + 过程导向(area_shoot) + 结果导向(is_goal)
    total_shoot_reward = (
        swing_reward * 1.0 +          # 踢球动作分
        area_shoot_reward * 5.0 +     # 射门质量分 (新增)
        is_goal * 100.0               # 进球终点分
    )
    
    return total_shoot_reward * weights["shoot"]

def feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.3) -> torch.Tensor:
    """奖励腾空时间，这是跑步步态的核心逻辑。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 提取最后一次腾空的时长
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # 判断是否刚着地
    first_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    # 逻辑：(腾空时间 - 阈值) * 刚落地瞬间
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # 只在有速度指令时奖励
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm > 0.1)

def base_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚横滚和俯仰角速度，增加整体稳定性。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)

def ball_towards_goal_area_reward(
    env: ManagerBasedRLEnv,
    goal_x: float = 4.5,
    goal_width: float = 3.0, # 球门总宽度
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")
) -> torch.Tensor:
    """奖励球的速度矢量指向球门区域（扇形区域）"""
    ball = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_pos_w[:, :2]
    ball_vel = ball.data.root_lin_vel_w[:, :2]
    
    # 1. 计算球到球门两侧的临界角度
    # 球门左柱和右柱的坐标
    goal_y_top = goal_width / 2.0
    goal_y_bottom = -goal_width / 2.0
    
    # 计算球到两个门柱的向量
    vec_to_top = torch.tensor([goal_x, goal_y_top], device=env.device) - ball_pos
    vec_to_bottom = torch.tensor([goal_x, goal_y_bottom], device=env.device) - ball_pos
    
    # 计算这两个向量的角度（弧度）
    angle_top = torch.atan2(vec_to_top[:, 1], vec_to_top[:, 0])
    angle_bottom = torch.atan2(vec_to_bottom[:, 1], vec_to_bottom[:, 0])
    
    # 2. 计算球当前的速度方向角度
    ball_vel_angle = torch.atan2(ball_vel[:, 1], ball_vel[:, 0])
    
    # 3. 检查速度角度是否在门柱角度范围内
    # 使用 torch.max/min 确保处理好正负号
    min_angle = torch.min(angle_top, angle_bottom)
    max_angle = torch.max(angle_top, angle_bottom)
    
    in_sector = (ball_vel_angle >= min_angle) & (ball_vel_angle <= max_angle)
    
    # 4. 结合球的速度大小，速度越快且在范围内，奖励越高
    speed = torch.norm(ball_vel, dim=-1)
    return in_sector.float() * speed

def stagnation_penalty(
    env: ManagerBasedRLEnv,
    velocity_threshold: float = 0.15,  # 稍微调高一点，防止原地蠕动蹭分
    angular_threshold: float = 0.15,
    grace_period_steps: int = 50,      # 关键：给机器人约 1 秒(50步)的起步/站稳时间
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    带倒计时的静止惩罚：
    1. 利用 env.episode_length_buf 确保机器人在 grace_period_steps 之后才开始计分。
    2. 只有当持续静止时，权重才会显现。
    """
    asset = env.scene[asset_cfg.name]
    
    # 获取当前速度模长
    lin_vel_norm = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
    ang_vel_norm = torch.norm(asset.data.root_ang_vel_w[:, 2:], dim=-1)
    
    # 判定当前是否静止
    is_stagnant = (lin_vel_norm < velocity_threshold) & (ang_vel_norm < angular_threshold)
    
    # 判定是否过了‘宽限期’ (等效于论文中的计时 1s)
    # episode_length_buf 记录了每个 env 当前运行了多少 step
    past_grace_period = env.episode_length_buf > grace_period_steps
    
    # 只有过了宽限期且依然静止的 env，才返回 1.0 触发惩罚
    reward = is_stagnant & past_grace_period
    
    return reward.float()

def position_behind_ball_reward(
    env: ManagerBasedRLEnv, 
    offset: float = 0.5, 
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励机器人移动到球与球门连线的后方（预备踢球点）"""
    # 1. 正确获取物理对象 (Asset)
    ball = env.scene[ball_cfg.name]
    robot = env.scene[asset_cfg.name]
    
    ball_pos = ball.data.root_pos_w[:, :2]
    robot_pos = robot.data.root_pos_w[:, :2]
    
    # 假设球门中心在 [4.5, 0.0]
    goal_pos = torch.tensor([4.5, 0.0], device=env.device)
    
    # 2. 计算从球门到球的单位向量 (指向球后方)
    goal_to_ball = ball_pos - goal_pos
    unit_vec = goal_to_ball / (torch.norm(goal_to_ball, dim=-1, keepdim=True) + 1e-6)
    
    # 3. 目标点 = 球的位置 + offset * 方向向量
    setup_point = ball_pos + unit_vec * offset
    
    # 4. 计算距离
    dist_to_setup = torch.norm(robot_pos - setup_point, dim=-1)
    
    # 5. 计算稳定性奖励 (修正 AttributeError 的位置)
    # 注意：此处必须使用 robot.data 而不是 asset_cfg.data
    robot_vel_lin = robot.data.root_lin_vel_b[:, :2]
    stability_bonus = torch.exp(-2.0 * dist_to_setup) * torch.exp(-torch.norm(robot_vel_lin, dim=-1))
    
    # 返回 跑位奖励 + 稳定性加成
    return torch.exp(-2.0 * dist_to_setup) + stability_bonus

def body_align_with_goal(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励机器人的朝向（Heading）与球门中心对齐"""
    asset = env.scene[asset_cfg.name]
    goal_pos = torch.tensor([4.5, 0.0], device=env.device)
    
    # 1. 机器人到球门的方向
    to_goal_vec = goal_pos - asset.data.root_pos_w[:, :2]
    to_goal_dir = torch.nn.functional.normalize(to_goal_vec, dim=-1)
    
    # 2. 机器人当前的 Heading (Forward 向量)
    forward_l = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, forward_l)[:, :2]
    forward_dir = torch.nn.functional.normalize(forward_w, dim=-1)
    
    # 3. 余弦相似度
    cosine_sim = torch.sum(forward_dir * to_goal_dir, dim=-1)
    
    return torch.clamp(cosine_sim, min=0.0)
