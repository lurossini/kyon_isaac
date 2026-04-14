# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, ImuCfg, CameraCfg, TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import kyon_isaac.tasks.locomotion.velocity.mdp as kyon_mdp


##
# Pre-defined configs
##
from kyon_isaac.assets.kyon_train import KYON_LOWER_BODY_CFG_TRAIN, KYON_FULL_BODY_CFG_TRAIN
from kyon_isaac.assets.kyon_play import KYON_LOWER_BODY_CFG_PLAY, KYON_FULL_BODY_CFG_PLAY


# PYRAMID_STEPS_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=9,
#     num_cols=21,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     sub_terrains={
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
#         ),
#     },
# )


@configclass
class KyonActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*"], scale=0.5, use_default_offset=True)

@configclass
class KyonCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1., 1.), ang_vel_z=(-1.5, 1.5)
        ),
    )

@configclass 
class KyonCommandsPLAYCfg:
    """Command specifications for the MDP in play mode."""

    base_velocity = kyon_mdp.VelocityCommandCfg(
        asset_name="robot",
        debug_vis=True,
        ranges=kyon_mdp.VelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
    )

@configclass
class KyonObservationsCfg:
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
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*", "knee_pitch_.*"])}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos_error_history = ObsTerm(
            func=kyon_mdp.joint_pos_error, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*", "knee_pitch_.*"])}, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=3
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*", "knee_pitch_.*"])}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

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
            func=mdp.joint_effort, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*", "knee_pitch_.*"])}
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class KyonEventCfg:
    """Configuration for randomization."""

    # startup
    reset_arms = None

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
        }
    )

    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "friction_distribution_params": (0.8, 1.0),
            "armature_distribution_params": (0.8, 1.2),
        }
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=kyon_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*", "knee_pitch_.*"]),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

@configclass
class KyonRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=1.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=1.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # foot_clearance = RewardTermCfg(
    #     func=spot_mdp.foot_clearance_reward,
    #     weight=0.5,
    #     # weight=4.,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.2,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="contact_.*"),
    #     },
    # )
    gait = RewardTermCfg(
        func=kyon_mdp.GaitReward,
        weight=2.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("contact_1", "contact_4"), ("contact_2", "contact_3")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty, weight=-0.4, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # base_orientation = RewardTermCfg(
        # func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    # )
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="knee_pitch_.*"), "threshold": 1.0},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="contact_.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_pos = RewardTermCfg(
        func=kyon_mdp.joint_position_penalty,
        # weight=-0.7,
        weight=-1.4,
        params={
            "asset_cfg": SceneEntityCfg("robot"), # , joint_names="hip_roll_.*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-5.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

@configclass
class KyonTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "knee_pitch_.*"]), "threshold": 1.0},
    )
    arms_contact = None
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class KyonRoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings
    observations: KyonObservationsCfg = KyonObservationsCfg()
    actions: KyonActionsCfg = KyonActionsCfg()
    commands: KyonCommandsCfg = KyonCommandsCfg()

    # MDP setting
    rewards: KyonRewardsCfg = KyonRewardsCfg()
    terminations: KyonTerminationsCfg = KyonTerminationsCfg()
    events: KyonEventCfg = KyonEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(-1.5, -4.5, 0.3), origin_type="asset_root", env_index=0, asset_name="robot")

    # Imu
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        self.scene.num_envs = 8192
        
        # switch robot to Kyon
        self.scene.robot = KYON_LOWER_BODY_CFG_TRAIN.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # imu
        self.scene.imu_sensor = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/imu_link")

        # terrain
        # self.scene.terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="generator",
        #     terrain_generator=COBBLESTONE_ROAD_CFG,
        #     max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        #     collision_group=-1,
        #     physics_material=sim_utils.RigidBodyMaterialCfg(
        #         friction_combine_mode="multiply",
        #         restitution_combine_mode="multiply",
        #         static_friction=1.0,
        #         dynamic_friction=1.0,
        #     ),
        #     visual_material=sim_utils.MdlFileCfg(
        #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        #         project_uvw=True,
        #         texture_scale=(0.25, 0.25),
        #     ),
        #     debug_vis=True,
        # )

        # no height scan
        self.scene.height_scanner = None


class KyonRoughEnvCfg_PLAY(KyonRoughEnvCfg):

    commands: KyonCommandsPLAYCfg = KyonCommandsPLAYCfg()

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        self.scene.robot = KYON_LOWER_BODY_CFG_PLAY.replace(prim_path="{ENV_REGEX_NS}/Robot")


        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.commands = KyonCommandsPLAYCfg()

       
        # remove random pushing event

class KyonFullRoughEnvCfg(KyonRoughEnvCfg):
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = KYON_FULL_BODY_CFG_TRAIN.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.events.reset_arms = EventTerm(
            func=kyon_mdp.reset_joint_target_to_default, 
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_pitch_.*", "wrist_.*", "dagana_.*"])
            },
        )

        # self.events.reset_arms = EventTerm(
        #     func=kyon_mdp.random_joint_position_velocity, 
        #     mode="interval",
        #     interval_range_s=(0.5, 0.5),
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*", "dagana_.*"]),
        #         "pos_lims": (-2, 2),
        #         "vel_lims": (-10, 10)
        #     },
        # )

class KyonFullRoughEnvCfg_PLAY(KyonRoughEnvCfg_PLAY):

    commands: KyonCommandsPLAYCfg = KyonCommandsPLAYCfg()

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.scene.robot = KYON_FULL_BODY_CFG_PLAY.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 100

        self.events.reset_arms = EventTerm(
            func=kyon_mdp.reset_joint_target_to_default, 
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*", "dagana_.*"])
            },
        )