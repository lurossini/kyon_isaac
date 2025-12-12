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

import kyon_isaac.tasks.locomotion.velocity.mdp as kyon_mdp
from kyon_isaac.tasks.locomotion.velocity.config.spot.flat_env_cfg import KyonFlatEnvCfg
# from kyon_isaac.sensors import ActionHistorySensorCfg

import torch
from isaaclab.assets import Articulation

import kyon_isaac
from pathlib import Path
KYON_ISAAC_BASE_DIR = Path(kyon_isaac.__file__).resolve().parent
KYON_LOWER_BODY_ENV_CFG = KyonFlatEnvCfg()

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

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "pose_range": {"x": (-0., 0.), "y": (-0., 0.), "yaw": (-0., 0.)},
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #             "roll": (-0.0, 0.0),
    #             "pitch": (-0.0, 0.0),
    #             "yaw": (-0.0, 0.0),
    #         },
    #     },
    # )

    reset_box = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (2.0, 2.0), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
            "pose_range": {"x": (1., 2.), "y": (-1.0, 1.0), "z": (0., 0.5)},
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
class KyonActionsCfg:
    """Action specifications for the MDP."""
    # Lower-body navigation actions from previous trained policy
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{KYON_ISAAC_BASE_DIR}/../../../scripts/rsl_rl/logs/rsl_rl/kyon_flat/new_locomotion_roll_015_no_arms/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=KYON_LOWER_BODY_ENV_CFG.actions.joint_pos,
        low_level_observations=KYON_LOWER_BODY_ENV_CFG.observations.policy,
    )

@configclass
class KyonObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

        # `` observation terms (order preserved)
        distance_from_box = ObsTerm(
            func=kyon_mdp.relative_position,
            params={
                "source_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_cfg": SceneEntityCfg("box")
            }
        )
        is_box_in_fov = ObsTerm(
            func=kyon_mdp.asset_in_fov,
            params={
                "camera_name": "front_up_camera",
                "asset_name": "box"
            }
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RGBCameraCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        front_up_cam_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_up_camera"), 
                "data_type": "rgb", 
                "normalize": False}
        )

        front_up_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_up_camera"), 
                "data_type": "distance_to_image_plane", 
                "normalize": True}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    rgb: RGBCameraCfg = RGBCameraCfg()

@configclass 
class KyonRewardsCfg():
    """Reward terms for the MDP."""

    # termination_reward = RewTerm(func=mdp.is_terminated, weight=150.0)
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
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"), "threshold": 1.0},
    # )
    goal_reached = DoneTerm(
        func=kyon_mdp.goal_reached_termination,
        params={
            "source_asset_cfg": SceneEntityCfg("robot"),
            "target_asset_cfg": SceneEntityCfg("box"),
            "threshold": 0.7,
        },
    )

@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = KYON_LOWER_BODY_ENV_CFG.scene
    actions: KyonActionsCfg = KyonActionsCfg()
    observations: KyonObservationsCfg = KyonObservationsCfg()
    events: KyonEventCfg = KyonEventCfg()
    # mdp settings
    # commands: CommandsCfg = CommandsCfg()
    rewards: KyonRewardsCfg = KyonRewardsCfg()
    terminations: KyonTerminationsCfg = KyonTerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = KYON_LOWER_BODY_ENV_CFG.sim.dt
        self.sim.render_interval = KYON_LOWER_BODY_ENV_CFG.decimation
        self.decimation = KYON_LOWER_BODY_ENV_CFG.decimation * 10
        self.episode_length_s = 2.0

        self.scene.num_envs = 512

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
        self.scene.front_up_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/zed_front_up_mount_link/front_up_camera",
            update_period=0.0333,
            height=120,
            width=192,
            data_types=["rgb", "distance_to_image_plane"],
            debug_vis=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
            depth_clipping_behavior="max",
        )
        self.image_obs_list = ["front_up_camera"]

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
