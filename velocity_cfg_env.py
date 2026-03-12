import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from magiclab_rl_lab.assets.robots.magiclab import MAGICLAB_Z1_23DOF_CFG as ROBOT_CFG
from magiclab_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    # 1. 地面：改为平地 (9x6球场)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.1)),  # 绿色草地感
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 3. 足球：标准5号球
    soccer_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            # 刚体基础属性
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # 质量属性
            mass_props=sim_utils.MassPropertiesCfg(mass=0.45),
            # 碰撞几何属性（保持默认即可）
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 物理材质属性：在这里设置弹性(restitution)和摩擦力(friction)
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.7,  # <--- 弹性系数在这里设置
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.11)),
    )


    # 4. 球门可视化（分为三个部分：左门柱、右门柱、横梁）
    goal_left_post = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/left_post",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 1.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.5, 1.3, 0.6)),
    )


    goal_right_post = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/right_post",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 1.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.5, -1.3, 0.6)),
    )


    goal_crossbar = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/crossbar",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 2.6, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.5, 0.0, 1.2)),
    )


    # 5. 球场可视化
    visual_field = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/VisualField",
        spawn=sim_utils.CuboidCfg(
            size=(9.0, 6.0, 0.01),  # X=9m, Y=6m, Z=0.01m (1厘米厚)
            collision_props=None,  # 关掉碰撞属性，使其仅作为视觉元素存在
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.0)),  # 深绿色
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005)),  # 稍微抬高 0.005m 防止与地面重叠闪烁
    )


    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """事件配置：处理随机初始化"""

    # 机器人位置随机化：
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 3.5), "y": (-2.75, 2.75), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-1.0, 3.5), "y": (-0.75, 0.75), "z": (-0.75, 0.75)},
        },
    )

    # 足球位置随机化：
    reset_ball = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("soccer_ball"),
            "pose_range": {"x": (-1.0, 4.0), "y": (-2.75, 2.75), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )

    # 关节随机化
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5), # 在初始姿态基础上做 50%-150% 的缩放偏移
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformLevelVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=False,
    #     debug_vis=True,
    #     ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.0, 0.0),
    #         lin_vel_y=(0.0, 0.0),
    #         ang_vel_z=(0.0, 0.0),
    #     ),
    #     limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.0, 0.0),
    #         lin_vel_y=(0.0, 0.0),
    #         ang_vel_z=(0.0, 0.0),
    #     ),
    # )

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.15, 1.0), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.25, 0.25)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.25, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        scale=0.25, use_default_offset=True, 
        preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class AmpCfg(ObsGroup):
        """AMP observations for discriminator"""

        # root height
        root_height = ObsTerm(
            func=mdp.base_pos_z
        )

        # root orientation
        root_rot = ObsTerm(
            func=mdp.root_quat_w
        )

        # joint pos (only legs)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_hip_yaw_joint",
                        "left_knee_joint",
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",
                        "right_hip_pitch_joint",
                        "right_hip_roll_joint",
                        "right_hip_yaw_joint",
                        "right_knee_joint",
                        "right_ankle_pitch_joint",
                        "right_ankle_roll_joint",
                    ],
                    preserve_order=True
                )
            }
        )

        # joint vel
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_hip_yaw_joint",
                        "left_knee_joint",
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",
                        "right_hip_pitch_joint",
                        "right_hip_roll_joint",
                        "right_hip_yaw_joint",
                        "right_knee_joint",
                        "right_ankle_pitch_joint",
                        "right_ankle_roll_joint",
                    ],
                    preserve_order=True
                )
            }
        )

        def __post_init__(self):
            self.history_length = 2
            self.concatenate_terms = True

    amp: AmpCfg = AmpCfg()


    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                joint_names=[
                                    "left_hip_pitch_joint",
                                    "left_hip_roll_joint",
                                    "left_hip_yaw_joint",
                                    "left_knee_joint",
                                    "left_ankle_pitch_joint",
                                    "left_ankle_roll_joint",
                                    "right_hip_pitch_joint",
                                    "right_hip_roll_joint",
                                    "right_hip_yaw_joint",
                                    "right_knee_joint",
                                    "right_ankle_pitch_joint",
                                    "right_ankle_roll_joint",
                                ], 
                                preserve_order=True)},
                                noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel,
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                joint_names=[
                                    "left_hip_pitch_joint",
                                    "left_hip_roll_joint",
                                    "left_hip_yaw_joint",
                                    "left_knee_joint",
                                    "left_ankle_pitch_joint",
                                    "left_ankle_roll_joint",
                                    "right_hip_pitch_joint",
                                    "right_hip_roll_joint",
                                    "right_hip_yaw_joint",
                                    "right_knee_joint",
                                    "right_ankle_pitch_joint",
                                    "right_ankle_roll_joint",
                                ], 
                                preserve_order=True)},
                                scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action,
                                clip=(-100.0, 100.0),
                                scale=1.0,
                              )
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.4})

        # --- 足球相关感知 (新增) ---
        # 1. 足球相对于机器人的位置 (x, y, z) - 让机器人知道球在哪
        ball_pos_rel = ObsTerm(
            func=mdp.ball_pos_rel,
            params={"ball_cfg": SceneEntityCfg("soccer_ball")}
        )
        # 2. 足球的速度 - 让机器人感知球是否被踢走了
        ball_vel = ObsTerm(
            func=mdp.ball_velocity,
            params={"ball_cfg": SceneEntityCfg("soccer_ball")},
            scale=0.5
        )
        # 3. 球门相对于足球的位置 - 建立“球-球门”连线感，帮助机器人找准踢球角度
        # 球门中心假设在 (4.5, 0.0)
        ball_to_goal_rel = ObsTerm(
            func=mdp.ball_to_goal_rel,
            params={"ball_cfg": SceneEntityCfg("soccer_ball"), "goal_pos": [4.5, 0.0, 0.0]}
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,
                                params={"asset_cfg": SceneEntityCfg("robot",
                                joint_names=[
                                    "left_hip_pitch_joint",
                                    "left_hip_roll_joint",
                                    "left_hip_yaw_joint",
                                    "left_knee_joint",
                                    "left_ankle_pitch_joint",
                                    "left_ankle_roll_joint",
                                    "right_hip_pitch_joint",
                                    "right_hip_roll_joint",
                                    "right_hip_yaw_joint",
                                    "right_knee_joint",
                                    "right_ankle_pitch_joint",
                                    "right_ankle_roll_joint",
                                ],
                                preserve_order=True)},
                                )
        
        
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, 
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                joint_names=[
                                    "left_hip_pitch_joint",
                                    "left_hip_roll_joint",
                                    "left_hip_yaw_joint",
                                    "left_knee_joint",
                                    "left_ankle_pitch_joint",
                                    "left_ankle_roll_joint",
                                    "right_hip_pitch_joint",
                                    "right_hip_roll_joint",
                                    "right_hip_yaw_joint",
                                    "right_knee_joint",
                                    "right_ankle_pitch_joint",
                                    "right_ankle_roll_joint",
                                ],
                                preserve_order=True)},
                                scale=0.05)
        last_action = ObsTerm(func=mdp.last_action,
                              clip=(-100.0, 100.0),
                              scale=1.0,
                              )
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.4})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        # 足球相关
        ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel, params={"ball_cfg": SceneEntityCfg("soccer_ball")})
        ball_vel_world = ObsTerm(
            func=mdp.ball_velocity,
            params={"ball_cfg": SceneEntityCfg("soccer_ball")}
        )

        contact_mask = ObsTerm(func=mdp.contact_mask, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")})
        
        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """针对 AMP 优化的奖励配置"""

    # --- 1. 风格奖励 (核心) ---
    amp_reward = RewTerm(func=mdp.amp_reward, weight=0, params={"resource_name": "amp_handler"})

    # --- 2. 任务目标奖励 (保留) ---
    # 引导机器人寻找球
    seek_ball = RewTerm(func=mdp.seek_ball_reward, weight=0.0, params={"ball_cfg": SceneEntityCfg("soccer_ball")})
    # 进球大奖
    goal_event = RewTerm(func=mdp.goal_score_reward, weight=0.0, params={"goal_box": [4.3, 5.0, -1.3, 1.3], "ball_cfg": SceneEntityCfg("soccer_ball")})
    # 带球和射门阶段奖励
    dribble_progression = RewTerm(func=mdp.dribble_phase_reward, weight=0, params={"goal_pos": [4.5, 0.0], "ball_cfg": SceneEntityCfg("soccer_ball"), "asset_cfg": SceneEntityCfg("robot")})
    shooting_dominant = RewTerm(func=mdp.shoot_phase_reward, weight=0.0, params={"dominant_leg_idx": 1, "goal_pos": [4.5, 0.0], "goal_box": [4.3, 5.0, -1.3, 1.3], "ball_cfg": SceneEntityCfg("soccer_ball"), "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link", ".*knee_link"])})

    # 足球出界惩罚
    ball_out_of_bounds = RewTerm(func=mdp.ball_boundary_penalty, weight=0.0, params={"x_limit": 4.5, "y_limit": 3.0})
    # 控球稳定性
    ball_stability = RewTerm(func=mdp.ball_relative_velocity_penalty, weight=0.0, params={"decay_std": 0.5})

     # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )


    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.7})

    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-3.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )



    feet_contact_number = RewTerm(
        func=mdp.feet_contact_number,
        weight=0.5,#1.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "period": 0.6,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )




'''
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
'''

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

    # 足球出界终止
    ball_out = DoneTerm(
        func=mdp.ball_out_of_bounds,  # 这里现在可以正确找到函数了
        params={
            "x_limit": 4.5,
            "y_limit": 3.0,
            "ball_cfg": SceneEntityCfg("soccer_ball")
        }
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)

'''
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
'''

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """针对足球射门任务的训练环境配置"""

    # 1. 场景设置
    # env_spacing 必须加大！球场长9米，加上缓冲区，建议设为 12-15米
    # 否则 A 场地的球被踢飞后会滚进 B 场地的球门，干扰训练数据
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=12.0)

    # 基础组件
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP 逻辑
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """初始化后的自定义设置"""

        # --- 核心频率设置 ---
        # decimation = 4 意味着控制频率为 1/0.002/4 = 125Hz
        # 踢球是瞬时爆发动作，125Hz 才能保证脚接触球的物理模拟准确，50Hz(10)太慢了
        self.decimation = 5

        # 射门回合长度，固定点位练习 5-10s 足够，长途奔跑建议 15s
        self.episode_length_s = 30.0

        # --- 仿真设置 ---
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # 提高 GPU 物理计算的 Patch 数量，防止并行环境过多时溢出
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15

        # 更新传感器频率，使其每个物理步都更新
        self.scene.contact_forces.update_period = self.sim.dt

        # --- 地形逻辑清理 ---
        # 既然我们现在用的是 Plane（平地），不再需要检查 terrain_generator
        # 这一段可以安全移除或保持默认，因为它不会被触发
        if hasattr(self.scene.terrain, "terrain_generator") and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = getattr(self.curriculum, "terrain_levels",
                                                                      None) is not None


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """用于可视化播放（Testing/Play）的配置"""

    def __post_init__(self):
        super().__post_init__()

        # 播放时不需要 4096 个环境，16-32 个足以观察
        self.scene.num_envs = 16

        # 移除原有的地形生成器行/列设置，因为现在是平地
        # 确保指令范围使用的是训练后的极限范围（Limit Ranges）
        if hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
