# Copyright (c) 2025, Ioannis Dadiotis.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the IIT Kyon robot.

Reference:

* https://github.com/ADVRHumanoids/iit-kyon-ros-pkg

"""

from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import os

##
# Configuration - Actuators.
##

"""Configuration for KYON DC actuator models."""
KYON_ACTUATOR_A_CFG = DCMotorCfg(
    joint_names_expr=["hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*"],
    saturation_effort=185.0,
    effort_limit=100.0,
    velocity_limit=7.6, # peak velocity 8.5
    stiffness=250,
    damping=10.0,
    armature=0.234,
    friction=4.68e-1    # random
)

KYON_ACTUATOR_B_CFG = DCMotorCfg(
    joint_names_expr=["shoulder_yaw.*"],
    saturation_effort=146.0,
    effort_limit=49.0,
    velocity_limit=2.4, # peak velocity 3.8
    stiffness=1000,
    damping=10.0,
    armature=0.472
)

KYON_ACTUATOR_C_CFG = DCMotorCfg(
    joint_names_expr=["shoulder_pitch.*", "elbow_pitch.*"],
    saturation_effort=122.0,
    effort_limit=39.0,
    velocity_limit=2.9, # peak velocity 4.6
    stiffness=1000,
    damping=10.0,
    armature=0.382
)

KYON_ACTUATOR_D_CFG = DCMotorCfg(
    joint_names_expr=["wrist_.*"],
    saturation_effort=25.0,
    effort_limit=7.7,
    velocity_limit=14.0,    # peak velocity 18
    stiffness=500,
    damping=5.0,
    friction=0.1,
    armature=0.1    # random, not tested yet
)

KYON_ACTUATOR_GRIPPER_CFG = DCMotorCfg(
    joint_names_expr=["dagana_.*"],
    saturation_effort=8.0,
    effort_limit=4.0,
    velocity_limit=16.8,
    stiffness=100,
    damping=5.0,
    friction=4.68e-1,
    armature=3.53e-3
)

##
# Configuration - Articulation.
##

"""Configuration of Kyon robot using DC Motor."""
KYON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.abspath(os.path.dirname(__file__))}/kyon.usd",
#        usd_path=f"/home/idadiotis/isaaclab_ws/src/iit-kyon-ros-pkg/kyon_isaac/usd/kyon_minimal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "hip_roll_.*": 0.0,
            "hip_pitch_1": 0.7,
            "hip_pitch_2": -0.7,
            "hip_pitch_3": 0.7,
            "hip_pitch_4": -0.7,
            "knee_pitch_1": -1.4,
            "knee_pitch_2": 1.4,
            "knee_pitch_3": -1.4,
            "knee_pitch_4": 1.4,
            # "hip_pitch_1": -0.5,      # spider-like
            # "hip_pitch_2": 0.5,
            # "hip_pitch_3": 0.5,
            # "hip_pitch_4": -0.5,
            # "knee_pitch_1": 1.0,
            # "knee_pitch_2": -1.0,
            # "knee_pitch_3": -1.0,
            # "knee_pitch_4": 1.0,
            # "shoulder_yaw_1": 0.0,    
            # "shoulder_yaw_2": 0.0, 
            # "shoulder_pitch_1": 1.0,   
            # "shoulder_pitch_2": -1.0,
            # "elbow_pitch_1": 2.5,   
            # "elbow_pitch_2": -2.5,        
            # "wrist_pitch_1": -0.75,
            # "wrist_pitch_2": 0.75,
            # "wrist_yaw_1": 0.0,     
            # "wrist_yaw_2": 0.0,
            # "dagana_1_clamp_joint": 0.0,
            # "dagana_2_clamp_joint": 0.0,
        },
    ),
    actuators={
        "legs": KYON_ACTUATOR_A_CFG,
        "arms_sh_yaw": KYON_ACTUATOR_B_CFG,
        "arms_middle": KYON_ACTUATOR_C_CFG,
        "arms_final": KYON_ACTUATOR_D_CFG,
        "grippers": KYON_ACTUATOR_GRIPPER_CFG
        },
    soft_joint_pos_limit_factor=0.95,
)

