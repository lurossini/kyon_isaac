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

KYON_LOWER_BODY_CFG_TRAIN = ArticulationCfg(
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
        "hip_roll": DelayedPDActuatorCfg(
            joint_names_expr=["hip_roll_[1-2-3-4]"],
            # saturation_effort=185,
            effort_limit=185,
            velocity_limit=7.6,
            stiffness=350, #8000.0,
            damping=10,
            armature=0.234,
            friction=4.68,
            dynamic_friction=1.7,
            min_delay=2,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=14,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "hip_pitch": DelayedPDActuatorCfg(
            joint_names_expr=["hip_pitch_[1-2-3-4]"],
            # saturation_effort=185,
            effort_limit=185,
            velocity_limit=7.6,
            stiffness=350.0,
            damping=10,
            armature=0.234,
            friction=4.68,
            dynamic_friction=1.7,
            min_delay=2,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=14,  # physics time steps (max: 2.0*4=8.0ms)
            # dynamic_friction=
        ),
        "knee_pitch": DelayedPDActuatorCfg(
            joint_names_expr=["knee_pitch_[1-2-3-4]"],
            # saturation_effort=185,
            effort_limit=185,
            velocity_limit=7.6,
            stiffness=350, #4000,
            damping=10,
            armature=0.234,
            friction=4.68,
            dynamic_friction=1.7,
            min_delay=2,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=14,  # physics time steps (max: 2.0*4=8.0ms)
        ),
    },
    #TODO what these do?
    #soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Centauro Upper body with daganas"""


# Other configs are defined by copying the base config and modifying the necessary fields
#CENTAURO_UPPER_DAGANA_XXX_CFG = CENTAURO_UPPER_DAGANA_CFG.copy()
"""Configuration of Centauro Upper body with daganas in and XXX"""