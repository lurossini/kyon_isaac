"""Configuration for the Centauro upper robot
The following configurations are available:

* :obj:`KYON_LOWER_BODY_CFG`: centauro upper body with torso and daganas on both hands
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os 

##
# Configuration of Centauro Upper body with daganas
##

KYON_LOWER_BODY_CFG_PLAY = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.abspath(os.path.dirname(__file__))}/kyon.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
        #TODO what these do?
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        #copy_from_source=False,

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.806), #height from floor when in homing
        #default centauro homing
        joint_pos={
            "hip_pitch_1": -0.7,
            "hip_pitch_2": 0.7,
            "hip_pitch_3": -0.7,
            "hip_pitch_4": 0.7,
            "hip_roll_1": 0.15,
            "hip_roll_2": -0.15,
            "hip_roll_3": -0.15,
            "hip_roll_4": 0.15,
            "knee_pitch_1": 1.4,
            "knee_pitch_2": -1.4,
            "knee_pitch_3": 1.4,
            "knee_pitch_4": -1.4,
        }
    ),
    actuators={
        "motorA": DCMotorCfg(
            joint_names_expr=["hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*"],
            saturation_effort=185,
            effort_limit=185,
            velocity_limit=7.6,
            stiffness=350, #8000.0,
            damping=10,
            armature=0.234,
            friction=4.68,
            dynamic_friction=4.68,
            viscous_friction=1.7,
        ),
    },
    #TODO what these do?
    #soft_joint_pos_limit_factor=1.0,
)

KYON_FULL_BODY_NO_ARMS_COLLISION_CFG_PLAY = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.abspath(os.path.dirname(__file__))}/kyon_full_no_arms_collision.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
        #TODO what these do?
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        #copy_from_source=False,

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.806), #height from floor when in homing
        #default centauro homing
        joint_pos={
            "hip_pitch_1": -0.7,
            "hip_pitch_2": 0.7,
            "hip_pitch_3": -0.7,
            "hip_pitch_4": 0.7,
            "hip_roll_1": 0.15,
            "hip_roll_2": -0.15,
            "hip_roll_3": -0.15,
            "hip_roll_4": 0.15,
            "knee_pitch_1": 1.4,
            "knee_pitch_2": -1.4,
            "knee_pitch_3": 1.4,
            "knee_pitch_4": -1.4,
            "shoulder_yaw_1": 0.0,
            "shoulder_yaw_2": 0.0,
            "shoulder_pitch_1": 1.5,
            "shoulder_pitch_2": -1.5,
            "elbow_pitch_1": 2.7,
            "elbow_pitch_2": -2.7,
            "wrist_pitch_1": 0.7,
            "wrist_pitch_2": -0.7,
            "wrist_yaw_1": 0.0,
            "wrist_yaw_2": 0.0,
            "dagana_1_clamp_joint": 0.0,
            "dagana_2_clamp_joint": 0.0,
        }
    ),
    actuators={
        "motorA": DCMotorCfg(
            joint_names_expr=["hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*"],
            saturation_effort=185,
            effort_limit=185,
            velocity_limit=7.6,
            stiffness=600, #8000.0,
            damping=10,
            armature=0.234,
            friction=4.68,
            dynamic_friction=4.68,
            viscous_friction=1.7,
        ),
        "motorB": DCMotorCfg(
            joint_names_expr=["shoulder_pitch_.*"],
            saturation_effort=146,
            effort_limit=146,
            velocity_limit=3.8,
            stiffness=300, #4000,
            damping=10,
            armature=0.472,
            friction=2.75,
            dynamic_friction=2.75,
            viscous_friction=5.1,
        ),
        "motorC": DCMotorCfg(
            joint_names_expr=["shoulder_yaw_.*", "elbow_pitch_.*"],
            saturation_effort=122,
            effort_limit=122,
            velocity_limit=4.6,
            stiffness=100, #4000,
            damping=5,
            armature=0.382,
            friction=3.2,
            dynamic_friction=3.2,
            viscous_friction=6.75,
        ),
        "motorD": DCMotorCfg(
            joint_names_expr=["wrist_.*"],
            saturation_effort=25,
            effort_limit=25,
            velocity_limit=14,
            stiffness=100, #4000,
            damping=5,
            armature=0.078,
            friction=1.,
            dynamic_friction=1.,
            viscous_friction=0.7,
        ),
    },
    #TODO what these do?
    #soft_joint_pos_limit_factor=1.0,
)


KYON_WHEEL_BODY_CFG_PLAY = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.abspath(os.path.dirname(__file__))}/kyon_full_wheels.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
        #TODO what these do?
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        #copy_from_source=False,

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.806), #height from floor when in homing
        #default centauro homing
        joint_pos={
            "hip_pitch_1": -0.7,
            "hip_pitch_2": 0.7,
            "hip_pitch_3": -0.7,
            "hip_pitch_4": 0.7,
            "hip_roll_1": 0.15,
            "hip_roll_2": -0.15,
            "hip_roll_3": -0.15,
            "hip_roll_4": 0.15,
            "knee_pitch_1": 1.4,
            "knee_pitch_2": -1.4,
            "knee_pitch_3": 1.4,
            "knee_pitch_4": -1.4,
            "ankle_yaw_1": 0.0,
            "ankle_yaw_2": 0.0,
            "ankle_yaw_3": 0.0,
            "ankle_yaw_4": 0.0,
            "wheel_joint_1": 0.0,
            "wheel_joint_2": 0.0,
            "wheel_joint_3": 0.0,
            "wheel_joint_4": 0.0,
            "shoulder_yaw_1": 0.0,
            "shoulder_yaw_2": 0.0,
            "shoulder_pitch_1": 1.,
            "shoulder_pitch_2": -1.,
            "elbow_pitch_1": 2.1,
            "elbow_pitch_2": -2.1,
            "wrist_pitch_1": 0.7,
            "wrist_pitch_2": -0.7,
            "wrist_yaw_1": 0.0,
            "wrist_yaw_2": 0.0,
            "dagana_1_clamp_joint": 0.0,
            "dagana_2_clamp_joint": 0.0,
        }
    ),
    actuators={
        "motorA": DCMotorCfg(
            joint_names_expr=["hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*"],
            saturation_effort=185,
            effort_limit=185,
            velocity_limit=7.6,
            stiffness=600, #8000.0,
            damping=10,
            armature=0.234,
            friction=4.68,
            dynamic_friction=4.68,
            viscous_friction=1.7,
        ),
        "motorB": DCMotorCfg(
            joint_names_expr=["shoulder_pitch_.*"],
            saturation_effort=146,
            effort_limit=146,
            velocity_limit=3.8,
            stiffness=300, #4000,
            damping=10,
            armature=0.472,
            friction=2.75,
            dynamic_friction=2.75,
            viscous_friction=5.1,
        ),
        "motorC": DCMotorCfg(
            joint_names_expr=["shoulder_yaw_.*", "elbow_pitch_.*"],
            saturation_effort=122,
            effort_limit=122,
            velocity_limit=4.6,
            stiffness=100, #4000,
            damping=5,
            armature=0.382,
            friction=3.2,
            dynamic_friction=3.2,
            viscous_friction=6.75,
        ),
        "motorD": DCMotorCfg(
            joint_names_expr=["wrist_.*"],
            saturation_effort=22,
            effort_limit=25,
            velocity_limit=14,
            stiffness=100, #4000,
            damping=5,
            armature=0.078,
            friction=1.,
            dynamic_friction=1.,
            viscous_friction=0.7,
        ),
        "motorWheel": DCMotorCfg(
            joint_names_expr=["wheel_.*"],
            saturation_effort=25,
            effort_limit=25,
            velocity_limit=14,
            stiffness=0, #4000,
            damping=10,
            armature=0.078,
            friction=1.,
            dynamic_friction=1.,
            viscous_friction=0.7,
        ),
    },
    #TODO what these do?
    #soft_joint_pos_limit_factor=1.0,
)


KYON_FULL_BODY_CFG_PLAY = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.abspath(os.path.dirname(__file__))}/kyon_full.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
        #TODO what these do?
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        #copy_from_source=False,

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.806), #height from floor when in homing
        #default centauro homing
        joint_pos={
            "hip_pitch_1": -0.7,
            "hip_pitch_2": 0.7,
            "hip_pitch_3": -0.7,
            "hip_pitch_4": 0.7,
            "hip_roll_1": 0.15,
            "hip_roll_2": -0.15,
            "hip_roll_3": -0.15,
            "hip_roll_4": 0.15,
            "knee_pitch_1": 1.4,
            "knee_pitch_2": -1.4,
            "knee_pitch_3": 1.4,
            "knee_pitch_4": -1.4,
            "shoulder_yaw_1": 0.0,
            "shoulder_yaw_2": 0.0,
            "shoulder_pitch_1": 1.5,
            "shoulder_pitch_2": -1.5,
            "elbow_pitch_1": 2.7,
            "elbow_pitch_2": -2.7,
            "wrist_pitch_1": 0.7,
            "wrist_pitch_2": -0.7,
            "wrist_yaw_1": 0.0,
            "wrist_yaw_2": 0.0,
            "dagana_1_clamp_joint": 0.0,
            "dagana_2_clamp_joint": 0.0,
        }
    ),
    actuators={
        "motorA": DCMotorCfg(
            joint_names_expr=["hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*"],
            saturation_effort=185,
            effort_limit=185,
            velocity_limit=7.6,
            stiffness=600, #8000.0,
            damping=10,
            armature=0.234,
            friction=4.68,
            dynamic_friction=4.68,
            viscous_friction=1.7,
        ),
        "motorB": DCMotorCfg(
            joint_names_expr=["shoulder_pitch_.*"],
            saturation_effort=146,
            effort_limit=146,
            velocity_limit=3.8,
            stiffness=300, #4000,
            damping=10,
            armature=0.472,
            friction=2.75,
            dynamic_friction=2.75,
            viscous_friction=5.1,
        ),
        "motorC": DCMotorCfg(
            joint_names_expr=["shoulder_yaw_.*", "elbow_pitch_.*"],
            saturation_effort=122,
            effort_limit=122,
            velocity_limit=4.6,
            stiffness=100, #4000,
            damping=5,
            armature=0.382,
            friction=3.2,
            dynamic_friction=3.2,
            viscous_friction=6.75,
        ),
        "motorD": DCMotorCfg(
            joint_names_expr=["wrist_.*"],
            saturation_effort=25,
            effort_limit=25,
            velocity_limit=14,
            stiffness=100, #4000,
            damping=5,
            armature=0.078,
            friction=1.,
            dynamic_friction=1.,
            viscous_friction=0.7,
        ),
    },
    #TODO what these do?
    #soft_joint_pos_limit_factor=1.0,
)

KYON_ONLY_ARMS_CFG_PLAY = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.abspath(os.path.dirname(__file__))}/kyon_only_arms.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
        #TODO what these do?
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        #copy_from_source=False,

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.806), #height from floor when in homing
        #default centauro homing
        joint_pos={
            "shoulder_yaw_1": 0.0,
            "shoulder_yaw_2": 0.0,
            "shoulder_pitch_1": 1.5,
            "shoulder_pitch_2": -1.5,
            "elbow_pitch_1": 2.7,
            "elbow_pitch_2": -2.7,
            "wrist_pitch_1": 0.7,
            "wrist_pitch_2": -0.7,
            "wrist_yaw_1": 0.0,
            "wrist_yaw_2": 0.0,
            "dagana_1_clamp_joint": 0.0,
            "dagana_2_clamp_joint": 0.0,
        }
    ),
    actuators={
        "motorB": DCMotorCfg(
            joint_names_expr=["shoulder_pitch_.*"],
            saturation_effort=146,
            effort_limit=146,
            velocity_limit=3.8,
            stiffness=100, #4000,
            damping=5,
            armature=0.472,
            friction=2.75,
            dynamic_friction=2.75,
            viscous_friction=5.1,
        ),
        "motorC": DCMotorCfg(
            joint_names_expr=["shoulder_yaw_.*", "elbow_pitch_.*"],
            saturation_effort=122,
            effort_limit=122,
            velocity_limit=4.6,
            stiffness=100, #4000,
            damping=5,
            armature=0.382,
            friction=3.2,
            dynamic_friction=3.2,
            viscous_friction=6.75,
        ),
        "motorD": DCMotorCfg(
            joint_names_expr=["wrist_.*"],
            saturation_effort=25,
            effort_limit=25,
            velocity_limit=14,
            stiffness=100, #4000,
            damping=5,
            armature=0.078,
            friction=1.,
            dynamic_friction=1.,
            viscous_friction=0.7,
        ),
    },
    #TODO what these do?
    #soft_joint_pos_limit_factor=1.0,
)