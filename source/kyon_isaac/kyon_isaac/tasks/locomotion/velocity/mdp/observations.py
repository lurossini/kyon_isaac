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
from isaaclab.sensors import ContactSensor, TiledCamera, TiledCameraCfg
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
import isaaclab.utils.math as math

import numpy as np

from kyon_isaac.sensors.ray_caster import KyonRayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].flatten(start_dim=1)
    return net_contact_forces

def joint_pos_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the target joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.joint_pos_target[:, asset_cfg.joint_ids]

def joint_pos_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos_target[:, asset_cfg.joint_ids]

def relative_position(env: ManagerBasedRLEnv, 
                      source_asset_cfg: SceneEntityCfg,
                      target_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute the relative distance between two assets (the source is the robot)"""
    source_asset: Articulation = env.scene[source_asset_cfg.name]
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]

    source_pos = source_asset.data.root_pos_w
    target_pos = target_asset.data.root_pos_w
    relative_pos_w = target_pos - source_pos

    q_source = source_asset.data.root_link_quat_w
    relative_pos_s = math.quat_apply_inverse(q_source, relative_pos_w)

    # print(f'from obs: {relative_pos_s.tolist()}')

    return relative_pos_s

def heading_direction(env: ManagerBasedRLEnv, 
                      source_asset_cfg: SceneEntityCfg,
                      target_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute the relative distance between two assets (the source is the robot)"""
    source_asset: Articulation = env.scene[source_asset_cfg.name]
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]

    source_pos = source_asset.data.root_pos_w
    target_pos = target_asset.data.root_pos_w
    relative_pos_w = target_pos - source_pos

    q_source = source_asset.data.root_link_quat_w
    relative_pos_s = math.quat_apply_inverse(q_source, relative_pos_w)
    angle = torch.atan2(relative_pos_s[:, 1], relative_pos_s[:, 0])

    return angle

def get_absolute_pose(env: ManagerBasedRLEnv,
                      asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_link_pose_w[:, asset_cfg.body_ids]
    return body_pos_w.reshape(env.scene.num_envs, -1)

def get_relative_pose(env: ManagerBasedRLEnv,
                      asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pose_w = asset.data.body_link_pose_w[:, asset_cfg.body_ids].squeeze(1)
    body_pose_b = torch.zeros_like(body_pose_w)
    root_quat_inv = math.quat_conjugate(asset.data.root_quat_w)
    root_pos_inv = -math.quat_apply(root_quat_inv, asset.data.root_pos_w)
    body_pose_b[:, :3], body_pose_b[:, 3:] = math.combine_frame_transforms(
        root_pos_inv,
        root_quat_inv,
        body_pose_w[:, :3],
        body_pose_w[:, 3:],
    )
    return body_pose_b


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: KyonRayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    
    ret = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    ret[sensor.occlusion_mask] = 0.0

    return ret