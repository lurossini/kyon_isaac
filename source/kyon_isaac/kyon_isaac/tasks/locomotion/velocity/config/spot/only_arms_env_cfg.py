# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ActionTermCfg as ActTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, ImuCfg, CameraCfg, TiledCameraCfg

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# arms stuff
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp

import kyon_isaac.tasks.locomotion.velocity.mdp as kyon_mdp
from kyon_isaac.tasks.locomotion.velocity.config.spot.flat_env_cfg import KyonFlatEnvCfg, KyonFullFlatEnvCfg
from kyon_isaac.assets.kyon_train import KYON_FULL_BODY_CFG_TRAIN
from kyon_isaac.assets.kyon_play import KYON_FULL_BODY_CFG_PLAY
# from kyon_isaac.sensors import ActionHistorySensorCfg

import torch
from isaaclab.assets import Articulation

import kyon_isaac
from pathlib import Path
KYON_ISAAC_BASE_DIR = Path(kyon_isaac.__file__).resolve().parent
KYON_LOWER_BODY_ENV_CFG = KyonFlatEnvCfg()
KYON_FULL_BODY_ENV_CFG = KyonFullFlatEnvCfg()

##
# Pre-defined configs
##
from kyon_isaac.assets.kyon_train import *
from kyon_isaac.assets.kyon_play import *

import numpy as np
np.set_printoptions(precision=3)

@configclass
class KyonEventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "pose_range": {"x": (-0., 0.), "y": (-0., 0.), "yaw": (-0., 0.)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_box = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (2.0, 2.0), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
            "pose_range": {"x": (-4., 4.), "y": (-4.0, 4.0), "z": (0., 0.5)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("box")
        }
    )

@configclass
class KyonCommandsCfg:
    left_ee_pose = manipulation_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="dagana_1_base",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=manipulation_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(-3.14, 3.14),
        ),
    )

    right_ee_pose = manipulation_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="dagana_2_base",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=manipulation_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(-0.25, 0.35),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(-3.14, 3.14),
        ),
    )

@configclass
class KyonActionsCfg:
    """Action specifications for the MDP."""
    # Lower-body navigation actions from previous trained policy
    pre_trained_policy_action: kyon_mdp.PreTrainedPolicyActionCfg = kyon_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        # policy_path=f"{KYON_ISAAC_BASE_DIR}/../../../scripts/rsl_rl/logs/rsl_rl/kyon_flat/new_locomotion_roll_015_no_arms/exported/policy.pt",
        policy_path=f"{KYON_ISAAC_BASE_DIR}/../../../scripts/rsl_rl/logs/rsl_rl/kyon_flat/2025-12-16_14-08-27/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=KYON_FULL_BODY_ENV_CFG.actions,
        low_level_observations=KYON_FULL_BODY_ENV_CFG.observations.policy,
    )

    left_arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_yaw_1", "shoulder_pitch_1", "elbow_pitch_1", "wrist_pitch_1", "wrist_yaw_1"],
        body_name="dagana_1_base",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

    right_arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_yaw_2", "shoulder_pitch_2", "elbow_pitch_2", "wrist_pitch_2", "wrist_yaw_2"],
        body_name="dagana_2_base",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

@configclass
class KyonObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)
        left_arm_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        right_arm_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # @configclass
    # class CriticCameraCfg(PolicyCfg):
    #     """Observations for critic group."""

    #     # `` observation terms (order preserved)
    #     distance_from_box = ObsTerm(
    #         func=kyon_mdp.relative_position,
    #         params={
    #             "source_asset_cfg": SceneEntityCfg("robot"),
    #             "target_asset_cfg": SceneEntityCfg("box")
    #         }
    #     )
    #     is_box_in_fov = ObsTerm(
    #         func=kyon_mdp.asset_in_fov,
    #         params={
    #             "camera_name": "front_up_camera",
    #             "asset_name": "box"
    #         }
    #     )
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True

    class CriticArmsCfg(PolicyCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # @configclass
    # class RGBCameraCfg(ObsGroup):
    #     """Observations for policy group with RGB images."""

    #     front_up_cam_image = ObsTerm(
    #         func=mdp.image,
    #         params={
    #             "sensor_cfg": SceneEntityCfg("front_up_camera"), 
    #             "data_type": "rgb", 
    #             "normalize": False}
    #     )

    #     front_up_cam_depth = ObsTerm(
    #         func=mdp.image,
    #         params={
    #             "sensor_cfg": SceneEntityCfg("front_up_camera"), 
    #             "data_type": "distance_to_image_plane", 
    #             "normalize": True}
    #     )

    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # critic: CriticCameraCfg = CriticCameraCfg()
    critic_arms: CriticArmsCfg = CriticArmsCfg()
    # rgb: RGBCameraCfg = RGBCameraCfg()

@configclass 
class KyonRewardsCfg():
    """Reward terms for the MDP."""

    # task terms
    left_end_effector_position_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="dagana_1_base"), "command_name": "left_ee_pose"},
    )
    left_end_effector_position_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="dagana_1_base"), "std": 0.1, "command_name": "left_ee_pose"},
    )
    left_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="dagana_1_base"), "command_name": "left_ee_pose"},
    )


    right_end_effector_position_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="dagana_2_base"), "command_name": "right_ee_pose"},
    )
    right_end_effector_position_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="dagana_2_base"), "std": 0.1, "command_name": "right_ee_pose"},
    )
    right_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="dagana_2_base"), "command_name": "right_ee_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


    termination_reward = RewTerm(func=mdp.is_terminated, weight=-400.0)
    # goal_reached = RewTerm(
    #     func=kyon_mdp.goal_reached,
    #     weight=5.,
    #     params={
    #         "source_asset_cfg": SceneEntityCfg("robot"),
    #         "target_asset_cfg": SceneEntityCfg("box"),
    #         "threshold": 0.7
    #     }
    # )
    # ori_towards_goal = RewTerm(
    #     func=kyon_mdp.orient_towards_goal,
    #     weight=1.,
    #     params={
    #         "source_asset_cfg": SceneEntityCfg("robot"),
    #         "target_asset_cfg": SceneEntityCfg("box"),
    #     }
    # )
    # hierarchy = RewTerm(
    #     func=kyon_mdp.test_hierarchy,
    #     weight=1.,
    #     params={
    #         "rewards": {
    #             "rew_1": RewTerm(
    #                 func=kyon_mdp.orient_towards_goal,
    #                 weight=1.,
    #                 params={
    #                     "source_asset_cfg": SceneEntityCfg("robot"),
    #                     "target_asset_cfg": SceneEntityCfg("box"),
    #                 }
    #             ),
    #             "rew2": RewTerm(
    #                     func=kyon_mdp.goal_reached,
    #                     weight=5.,
    #                     params={
    #                         "source_asset_cfg": SceneEntityCfg("robot"),
    #                         "target_asset_cfg": SceneEntityCfg("box"),
    #                         "threshold": 0.7
    #                     }
    #             ),  
    #         }
    #     }
    # )

@configclass
class KyonTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"), "threshold": 1.0},
    )
    # goal_reached = DoneTerm(
    #     func=kyon_mdp.goal_reached_termination,
    #     params={
    #         "source_asset_cfg": SceneEntityCfg("robot"),
    #         "target_asset_cfg": SceneEntityCfg("box"),
    #         "threshold": 0.7,
    #     },
    # )

@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = KYON_LOWER_BODY_ENV_CFG.scene
    actions: KyonActionsCfg = KyonActionsCfg()
    observations: KyonObservationsCfg = KyonObservationsCfg()
    events: KyonEventCfg = KyonEventCfg()
    # mdp settings
    commands: KyonCommandsCfg = KyonCommandsCfg()
    rewards: KyonRewardsCfg = KyonRewardsCfg()
    terminations: KyonTerminationsCfg = KyonTerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = KYON_LOWER_BODY_ENV_CFG.sim.dt
        self.sim.render_interval = KYON_LOWER_BODY_ENV_CFG.decimation
        self.decimation = KYON_LOWER_BODY_ENV_CFG.decimation * 10
        self.episode_length_s = 2.0

        self.scene.num_envs = 512

        # Switch to full KYON configurationw
        self.scene.robot = KYON_FULL_BODY_CFG_TRAIN.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
        )

        self.scene.box: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/box",  # Spawns a box in every environment
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 0.8, 0.8),  # Example size
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # Green box
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8,
                    max_linear_velocity=0.0, # Optionally make it immobile
                    max_angular_velocity=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(2.0, 0.0, 0.7), # Initial position (e.g., slightly above ground)
                # You'll randomize the position in the reset function
            ),
        )
        # self.scene.front_up_camera = TiledCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/zed_front_up_mount_link/front_up_camera",
        #     update_period=0.0333,
        #     height=120,
        #     width=192,
        #     data_types=["rgb", "distance_to_image_plane"],
        #     debug_vis=True,
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        #     ),
        #     offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        #     depth_clipping_behavior="max",
        # )
        # self.image_obs_list = ["front_up_camera"]

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
