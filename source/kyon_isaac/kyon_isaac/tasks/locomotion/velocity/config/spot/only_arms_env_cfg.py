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
from isaaclab.sensors import ContactSensorCfg, ImuCfg, CameraCfg

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import kyon_isaac.tasks.locomotion.velocity.mdp as kyon_mdp
# from kyon_isaac.sensors import ActionHistorySensorCfg


##
# Pre-defined configs
##
from kyon_isaac.assets.kyon_train import KYON_ONLY_ARMS_CFG_TRAIN
from kyon_isaac.assets.kyon_play import KYON_ONLY_ARMS_CFG_PLAY


@configclass
class KyonActionsCfg:
    """Action specifications for the MDP."""
    # base_vel = kyon_mdp.VelocityBaseAction()
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*"], scale=0.4, use_default_offset=True)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.4, use_default_offset=True)