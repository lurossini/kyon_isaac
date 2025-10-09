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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, ImuCfg

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import kyon_isaac.tasks.locomotion.velocity.mdp as kyon_mdp
# from kyon_isaac.sensors import ActionHistorySensorCfg


##
# Pre-defined configs
##
from kyon_isaac.assets.kyon import KYON_LOWER_BODY_CFG  # isort: skip


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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class KyonActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.4, use_default_offset=True)


@configclass
class KyonCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1., 1.), ang_vel_z=(-1.5, 1.5)
        ),
    )

@configclass
class KyonObservationsMjxCfg:
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
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos_error_history = ObsTerm(
            func=kyon_mdp.joint_pos_error, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=3
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        # joint_effort = ObsTerm(
        #     func=mdp.joint_effort, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        # )
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
            func=mdp.joint_effort, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class KyonObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
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
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
                #
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    


@configclass
class KyonEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "stiffness_distribution_params": ()
            "damping_distribution_params" ()
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
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
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
class KyonRewardsMjxCfg:
    air_time = RewardTermCfg(
        func=kyon_mdp.reward_feet_air_time,
              weight=0.1,
              params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*"),
              },
    )
    # air_time = RewardTermCfg(
    #     func=spot_mdp.air_time_reward,
    #     weight=5.0,
    #     params={
    #         "mode_time": 0.3,
    #         "velocity_threshold": 0.5,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*"),
    #     },
    # )
    base_angular_velocity = RewardTermCfg(
        func=kyon_mdp.reward_tracking_ang_vel,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            },
    )
    base_linear_velocity = RewardTermCfg(
        func=kyon_mdp.reward_tracking_lin_vel,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    foot_clearance = RewardTermCfg(
        func=kyon_mdp.cost_feet_clearance,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="contact_.*"), 
            "target_height": 0.1,
        }
    )
    # foot_clearance = RewardTermCfg(
    #     func=spot_mdp.foot_clearance_reward,
    #     weight=0.5,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="contact_.*"),
    #     },
    # )
    lin_vel_z = RewardTermCfg(
        func=kyon_mdp.cost_lin_vel_z,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    ang_vel_xy = RewardTermCfg(
        func=kyon_mdp.cost_ang_vel_xy,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    orientation = RewardTermCfg(
        func=kyon_mdp.cost_orientation,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    posture = RewardTermCfg(
        func=kyon_mdp.reward_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    torques = RewardTermCfg(
        func=kyon_mdp.cost_torques,
        weight=-0.0002,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # action_rate = RewardTermCfg(
    #     func=kyon_mdp.cost_action_rate,
    #     weight=-0.01,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("action_history")
    #     }
    # )
    energy = RewardTermCfg(
        func=kyon_mdp.cost_energy,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    feet_slip = RewardTermCfg(
        func=kyon_mdp.cost_feet_slip,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="contact_.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*"),
            "threshold": 1.0,
        }
    )
    # foot_slip = RewardTermCfg(
    #     func=spot_mdp.foot_slip_penalty,
    #     weight=-0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="contact_.*"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*"),
    #         "threshold": 1.0,
    #     },
    # )


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
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="contact_.*"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=10.0,
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
        func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
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
    # joint_pos = RewardTermCfg(
    #     func=spot_mdp.joint_position_penalty,
    #     weight=-0.7,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="hip_roll_.*"),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.5,
    #     },
    # )
    joint_torques_hip_roll = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="hip_roll_.*")},
    )
    joint_torques_hip_pitch = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="hip_pitch_.*")},
    )
    joint_torques_knee_pitch = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="knee_pitch_.*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-5.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    # contact_forces = RewardTermCfg(
    #     func=kyon_mdp.min_contact_forces,
    #     weight=-5.0e-4,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="contact_.*")}
    # )


@configclass
class KyonTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "knee_pitch_.*"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class KyonFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings
    observations: KyonObservationsCfg = KyonObservationsMjxCfg()
    actions: KyonActionsCfg = KyonActionsCfg()
    commands: KyonCommandsCfg = KyonCommandsCfg()

    # MDP setting
    rewards: KyonRewardsCfg = KyonRewardsCfg()
    terminations: KyonTerminationsCfg = KyonTerminationsCfg()
    events: KyonEventCfg = KyonEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

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

        
        # switch robot to Kyon-d
        self.scene.robot = KYON_LOWER_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # imu
        self.scene.imu_sensor = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/imu_link")

        # self.scene.action_history = ActionHistorySensorCfg(prim_path="{ENV_REGEX_NS}/Robot", history_length=3)



        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        # no height scan
        self.scene.height_scanner = None


class KyonFlatEnvCfg_PLAY(KyonFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
