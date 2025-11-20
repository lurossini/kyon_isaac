# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

# from kyon_isaac.sensors import ActionHistorySensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].flatten(start_dim=1)
    return net_contact_forces

def last_last_action(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    action_history_sensor: ActionHistorySensor = env.scene.sensors[sensor_cfg.name]
    if action_history_sensor.data.action_matrix.shape[2] == 0:
        action_history_sensor.set_action_size(env.action_space.shape[1])

    action_matrix = action_history_sensor.data.action_matrix.roll(1, dims=1)
    action_matrix[:, 0] = mdp.last_action(env)
    return action_matrix

def joint_pos_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the target joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.joint_pos_target[:, asset_cfg.joint_ids]

def relative_distance(env: ManagerBasedRLEnv, 
                      source_asset_cfg: SceneEntityCfg,
                      target_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute the relative distance between two assets (the source is the robot)"""
    source_asset: Articulation = env.scene[source_asset_cfg.name]
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]

    source_pos = source_asset.data.root_pos_w
    target_pos = target_asset.data.root_pos_w

    return torch.norm(target_pos - source_pos, dim=1).unsqueeze(1)
