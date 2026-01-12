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
from isaaclab.managers import RewardTermCfg as RewTerm
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

from kyon_isaac.tasks.locomotion.velocity.config.spot.flat_env_cfg import KyonFullFlatEnvCfg, KyonFullFlatEnvCfg_PLAY, KyonCommandsCfg, KyonActionsCfg, KyonRewardsCfg, KyonTerminationsCfg, KyonObservationsMjxCfg

import kyon_isaac
from pathlib import Path
KYON_FULL_BODY_ENV_CFG = KyonFullFlatEnvCfg()
KYON_ISAAC_BASE_DIR = Path(kyon_isaac.__file__).resolve().parent

@configclass
class CommandsCfg(KyonCommandsCfg):
    
    left_ee_pose = kyon_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="wrist_yaw_1_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        # ranges=manipulation_mdp.UniformPoseCommandCfg.Ranges(
        #     pos_x=(0.6, 0.8),
        #     pos_y=(0.2, 0.4),
        #     pos_z=(0.0, 0.4),
        #     roll=(-1.6, -1.4),
        #     pitch=(0.0, 0.0),
        #     yaw=(-1.6, -1.4),
        # ),
        ranges=kyon_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-2, 2),
            pos_y=(-2, 2),
            pos_z=(0.3, 0.7),
            roll=(-1.6, -1.4),
            pitch=(0.0, 0.0),
            yaw=(-1.6, -1.4),
        ),
    )

    # right_ee_pose = manipulation_mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="wrist_yaw_2_link",
    #     resampling_time_range=(4.0, 4.0),
    #     debug_vis=True,
    #     ranges=manipulation_mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.6, 0.8),
    #         pos_y=(-0.4, -0.2),
    #         pos_z=(0.0, 0.4),
    #         roll=(-1.6, -1.4),
    #         pitch=(0.0, 0.0),
    #         yaw=(-1.6, -1.4),
    #     ),
    # )

@configclass
class ActionsCfg:
    # lower_body_joint_pos = kyon_mdp.AgileBasedLowerBodyActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "hip_roll_.*",
    #         "hip_pitch_.*",
    #         "knee_pitch_.*",
    #     ],
    #     policy_output_scale=0.5,
    #     obs_group_name="lower_body_policy",  # need to be the same name as the on in ObservationCfg
    #     policy_path=f"/workspace/kyon_isaac/scripts/rsl_rl/logs/rsl_rl/kyon_flat/2026-01-07_15-04-08/exported/policy.pt",
    # )

    pre_trained_policy_action: kyon_mdp.PreTrainedPolicyActionCfg = kyon_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        # policy_path=f"{KYON_ISAAC_BASE_DIR}/../../../scripts/rsl_rl/logs/rsl_rl/kyon_flat/new_locomotion_roll_015_no_arms/exported/policy.pt",
        policy_path=f"{KYON_ISAAC_BASE_DIR}/../../../scripts/rsl_rl/logs/rsl_rl/kyon_flat/2026-01-07_15-04-08/exported/policy.pt",
        low_level_actions=KYON_FULL_BODY_ENV_CFG.actions.joint_pos,
        low_level_observations=KYON_FULL_BODY_ENV_CFG.observations.policy,
    )


    # left_arm_action = DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=["shoulder_yaw_1", "shoulder_pitch_1", "elbow_pitch_1", "wrist_pitch_1", "wrist_yaw_1"],
    #     body_name="wrist_yaw_1_link",
    #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    # )

    upper_body_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["shoulder_yaw_.*", "shoulder_pitch_.*", "elbow_pitch_.*", "wrist_pitch_.*"], 
        scale=1., 
        use_default_offset=True
    )

    # right_arm_action = DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",.
    #     joint_names=["shoulder_yaw_2", "shoulder_pitch_2", "elbow_pitch_2", "wrist_pitch_2", "wrist_yaw_2"],
    #     body_name="wrist_yaw_2_link",
    #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    # )

@configclass
class RewardsCfg:
    left_ee_pos_tracking = RewTerm(
        func=kyon_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_1_link"),
            "command_name": "left_ee_pose",
        },
    )

    left_ee_pos_tracking_fine_grained = RewTerm(
        func=kyon_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_1_link"),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )
    # left_end_effector_orientation_tracking = RewTerm(
    #     func=manipulation_mdp.orientation_command_error,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_1_link"), "command_name": "right_ee_pose"
    #         },
    # )
    # right_ee_pos_tracking = RewTerm(
    #     func=manipulation_mdp.position_command_error,
    #     weight=-2.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_2_link"),
    #         "command_name": "right_ee_pose",
    #     },
    # )

    # right_ee_pos_tracking_fine_grained = RewTerm(
    #     func=manipulation_mdp.position_command_error_tanh,
    #     weight=2.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_2_link"),
    #         "std": 0.05,
    #         "command_name": "right_ee_pose",
    #     },
    # )

    # right_end_effector_orientation_tracking = RewTerm(
    #     func=manipulation_mdp.orientation_command_error,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="wrist_yaw_2_link"), "command_name": "right_ee_pose"
    #         },
    # )

@configclass
class ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos_error_history = ObsTerm(
            func=kyon_mdp.joint_pos_error, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=3
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        left_arm_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        # right_arm_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_sensor")}
        )

        contact_forces = ObsTerm(
            func=kyon_mdp.contact_forces, 
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*")
            }, 
        )

        joint_effort = ObsTerm(
            func=mdp.joint_effort, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    # lower_body_policy: KyonObservationsMjxCfg.PolicyCfg = KyonObservationsMjxCfg.PolicyCfg()
    # lower_body_policy.actions = ObsTerm(func=mdp.last_action, params={"action_name":"lower_body_joint_pos"})


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

        self.events.reset_arms = None
        self.episode_length_s = 2.0

        # # Table
        # self.scene.packing_table = AssetBaseCfg(
        #     prim_path="/World/envs/env_.*/PackingTable",
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #     ),
        # )

        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
        #         scale=(0.75, 0.75, 0.75),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        #     ),
        # )

@configclass
class LocomanipulationKyonSceneCfg_PLAY(LocomanipulationKyonSceneCfg):
    def __post_init__(self):
        super().__post_init__()