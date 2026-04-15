import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import kyon_isaac.tasks.locomotion.velocity.mdp as kyon_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.navigation.mdp as navigation_mdp
from isaaclab.sensors import ContactSensorCfg, ImuCfg, CameraCfg, TiledCameraCfg

from kyon_isaac.tasks.locomotion.velocity.config.kyon.flat_env_cfg import KyonFullFlatEnvCfg, KyonTerminationsCfg

from kyon_isaac.assets.kyon_train import KYON_LOWER_BODY_CFG_TRAIN, KYON_FULL_BODY_CFG_TRAIN

import kyon_isaac
from pathlib import Path
KYON_FULL_BODY_ENV_CFG = KyonFullFlatEnvCfg()
KYON_ISAAC_BASE_DIR = Path(kyon_isaac.__file__).resolve().parent

from kyon_isaac.tasks.locomotion.velocity.mdp.rewards import goal_reached
import torch
from collections.abc import Sequence
from isaaclab.managers.reward_manager import RewardManager
def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
    """Returns the episodic sum of individual reward terms.

    Args:
        env_ids: The environment ids for which the episodic sum of
            individual reward terms is to be returned. Defaults to all the environment ids.

    Returns:
        Dictionary of episodic sum of individual reward terms.
    """
    # resolve environment ids
    if env_ids is None:
        env_ids = slice(None)   
    # store information
    extras = {}
    for key in self._episode_sums.keys():
        # store information
        # r_1 + r_2 + ... + r_n
        episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
        extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
        # reset episodic sum
        self._episode_sums[key][env_ids] = 0.0
    # reset all the reward terms
    for term_cfg in self._class_term_cfgs:
        term_extras = term_cfg.func.reset(env_ids=env_ids)
        if term_extras is not None:
            extras.update(term_extras)
    # return logged information
    return extras

RewardManager.reset = reset

@configclass
class CommandsCfg:
    
    left_ee_pose = kyon_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="wrist_yaw_1_link",
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=kyon_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-3, 3),
            pos_y=(-3, 3),
            pos_z=(0.3, 0.7),
            roll=(0.0, 0.0),
            pitch=(3.14, 3.14),
            yaw=(-3.14, -3.14),
        ),
    )

@configclass
class CommandsVLMCfg:
    """For inference only, with commands provided by the user in scripts/rsl_rl/play.py"""

    left_ee_pose = kyon_mdp.PoseCommandCfg(
        asset_name="robot",
        camera_frame="zed_front_up_mount_link",
        body_name="wrist_yaw_1_link",
        debug_vis=True,
        resampling_time_range=(8.0, 8.0),
    )

@configclass
class ActionsCfg:
    # Lower-body locomotion frozen policy
    pre_trained_policy_action: kyon_mdp.PreTrainedPolicyActionCfg = kyon_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{KYON_ISAAC_BASE_DIR}/../../../scripts/rsl_rl/logs/rsl_rl/kyon_flat/legged_locomotion/exported/policy.pt",
        low_level_decimation=1,
        low_level_actions=KYON_FULL_BODY_ENV_CFG.actions.joint_pos,
        low_level_observations=KYON_FULL_BODY_ENV_CFG.observations.policy,
    )

    # Left arm Cartesian action
    # left_arm_action = DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=["shoulder_yaw_1", "shoulder_pitch_1", "elbow_pitch_1", "wrist_pitch_1", "wrist_yaw_1"],
    #     body_name="wrist_yaw_1_link",
    #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    # )

    # Left arm joint space action
    upper_body_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["shoulder_yaw_1", "shoulder_pitch_1", "elbow_pitch_1", "wrist_pitch_1", "wrist_yaw_1"], 
        scale=1.5, 
        use_default_offset=True
    )

@configclass
class RewardsCfg:

    # Hierarchical reward
    # hierarchy = RewTerm(
    #     func=kyon_mdp.Hierarchy,
    #     weight=1.0,
    #     params={
    #         "rewards": {
    #             # "goal_reached": RewTerm(
    #             #     func=kyon_mdp.goal_reached_command_base_new,
    #             #     weight=1.0,
    #             #     params={
    #             #         "asset_cfg": SceneEntityCfg("robot"),
    #             #         "command_name": "left_ee_pose",
    #             #         "threshold": 0.5,
    #             #         "std": 0.5,
    #             #     }
    #             # ),
    #             "reg_manipulation": {
    #                 "left_arm_joint_pos": RewTerm(
    #                     func=kyon_mdp.joint_pos_hierarchy,
    #                     weight=1.0,
    #                     params={
    #                         "lb_action_name": "pre_trained_policy_action",
    #                         "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_yaw_1", "shoulder_pitch_1", "elbow_pitch_1", "wrist_pitch_1", "wrist_yaw_1"]),
    #                         "std": 0.5
    #                     }
    #                 )
    #             },
    #             "left_ee_pos_tracking": {
    #                 "left_ee_pos_tracking": RewTerm(
    #                     func=kyon_mdp.position_command_error_gauss,
    #                     weight=1.0,
    #                     params={
    #                         "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_1_link"),
    #                         "std": 0.5,
    #                         "command_name": "left_ee_pose",
    #                     },
    #                 ),
    #                 "left_ee_pos_tracking_fine_grained": RewTerm(
    #                     func=kyon_mdp.position_command_error_gauss,
    #                     weight=5.0,
    #                     params={
    #                         "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_1_link"),
    #                         "std": 0.1,
    #                         "command_name": "left_ee_pose",
    #                     },
    #                 )
    #             },
    #             # "left_end_effector_orientation_tracking": RewTerm(
    #             #     func=kyon_mdp.orientation_command_error,
    #             #     weight=-1,
    #             #     params={
    #             #         "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_1_link"), 
    #             #         "command_name": "left_ee_pose"
    #             #     },
    #             # ) 
    #         }
    #     }
    # )

    # goal_reached = RewTerm(
    #     func=kyon_mdp.goal_reached_command_base_new,
    #                 weight=1.,
    #                 params={
    #                     "asset_cfg": SceneEntityCfg("robot"),
    #                     "command_name": "left_ee_pose",
    #                     "threshold": 0.5,
    #                 }
    # )

    reg_manipulation = RewTerm(
        func=kyon_mdp.arm_nominal_until_close,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_yaw_1", "shoulder_pitch_1", "elbow_pitch_1", "wrist_pitch_1", "wrist_yaw_1"]),
            "command_name": "left_ee_pose",
            "release_dist": 0.25,
            "slope": 0.05,
            "std": 0.5
        }
    )

    lb_action_regularization = RewTerm(
        func=kyon_mdp.action_regularization,
        weight=-0.1,
        params={
            "action_name": "pre_trained_policy_action"
        }
    )


    left_ee_pos_tracking = RewTerm(
        func=kyon_mdp.position_command_error_gauss,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_1_link"),
            "std": 0.5,
            "command_name": "left_ee_pose",
        },
    )
    left_ee_pos_tracking_fine_grained = RewTerm(
        func=kyon_mdp.position_command_error_gauss,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_1_link"),
            "std": 0.1,
            "command_name": "left_ee_pose",
        },
    )

    
    action_smoothness = RewTerm(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)

    # joint space rewards

    joint_vel = RewTerm(
        func=spot_mdp.joint_velocity_penalty,
        weight=-5.0e-2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_yaw_1", "shoulder_pitch_1", "elbow_pitch_1", "wrist_pitch_1", "wrist_yaw_1"])
        },
    )
    joint_acc = RewTerm(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_yaw_1", "shoulder_pitch_1", "elbow_pitch_1", "wrist_pitch_1", "wrist_yaw_1"])
        },
    )

@configclass
class ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot" , joint_names=['shoulder_yaw_1', 'shoulder_pitch_1', 'knee_pitch_1', 'wrist_pitch_1', 'wrist_yaw_1'])}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos_error_history = ObsTerm(
            func=kyon_mdp.joint_pos_error, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['shoulder_yaw_1', 'shoulder_pitch_1', 'knee_pitch_1', 'wrist_pitch_1', 'wrist_yaw_1'])}, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=3
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['shoulder_yaw_1', 'shoulder_pitch_1', 'knee_pitch_1', 'wrist_pitch_1', 'wrist_yaw_1'])}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        left_arm_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        left_arm_pose =ObsTerm(func=kyon_mdp.get_relative_pose, params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_yaw_1_link"])})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    

@configclass
class ObservationWithRGBDCfg(ObservationCfg):
    @configclass
    class RGBDCameraCfg(ObsGroup):
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
    rgbd: RGBDCameraCfg = RGBDCameraCfg()


@configclass
class LocomanipulationKyonSceneCfg(KyonFullFlatEnvCfg):
    # Basic settings
    observations: ObservationCfg = ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP setting
    rewards: RewardsCfg = RewardsCfg()
    terminations: KyonTerminationsCfg = KyonTerminationsCfg()

    curriculum = None

    def __post_init__(self):
        super().__post_init__()

        self.sim.dt = KYON_FULL_BODY_ENV_CFG.sim.dt
        self.sim.render_interval = KYON_FULL_BODY_ENV_CFG.decimation
        self.decimation = KYON_FULL_BODY_ENV_CFG.decimation

        # Add termination for arms collisions
        self.terminations.arms_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["shoulder_pitch_1_link", "elbow_pitch_1_link", "wrist_pitch_1_link"]), "threshold": 1.0},
        )
        
        # Reset uncontrolled arm joint target to default to avoid moving it to zero
        self.events.reset_arms = EventTerm(
            func=kyon_mdp.reset_joint_target_to_default, 
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_yaw_.*", "shoulder_pitch_.*", "elbow_pitch_.*", "wrist_pitch_.*", "wrist_yaw_.*", "dagana_.*"])
            },
        )

@configclass
class LocomanipulationKyonSceneCfg_PLAY(LocomanipulationKyonSceneCfg):

    # Use custom observations and commands during inference with camera
    # observations: ObservationWithRGBDCfg = ObservationWithRGBDCfg()
    # commands: CommandsVLMCfg = CommandsVLMCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.episode_length_s = 100000

        # Add camera
        # self.scene.front_up_camera = TiledCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/zed_front_up_mount_link/front_up_camera",
        #     update_period=0.0333,
        #     height=600,
        #     width=860,
        #     data_types=["rgb", "distance_to_image_plane"],
        #     debug_vis=True,
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        #     ),
        #     offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        #     depth_clipping_behavior="max",
        # )
        # self.image_obs_list = ["front_up_camera"]

        # Table
        self.scene.packing_table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/PackingTable",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
                scale=(0.75, 0.75, 0.75),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
        )

        self.scene.mug = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Mug",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 1.], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_D1.usd",
                # scale=(0.75, 0.75, 0.75),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        )

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
