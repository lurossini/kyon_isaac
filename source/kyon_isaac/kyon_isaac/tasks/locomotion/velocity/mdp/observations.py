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

    # print(f'distance: {relative_pos_s.tolist()}')

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

# def asset_in_fov(env: ManagerBasedRLEnv,
#                  camera_cfg: TiledCameraCfg,
#                  asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     camera: TiledCamera = env.scene[camera_cfg.name]
#     asset: RigidObject = env.scene[asset_cfg.name]
#     robot: Articulation = env.scene["robot"]

#     asset_pos = asset.data.root_pos_w
#     camera_pos = camera.data.pos_w
#     relative_pos_w = asset_pos - camera_pos
#     q_base = robot.data.root_quat_w
#     q_camera = camera.data.quat_w_world
#     relative_pos_b = math.quat_apply_inverse(q_base, relative_pos_w)
#     relative_pos_c = math.quat_apply_inverse(q_camera, relative_pos_b)
#     distance = torch.norm(relative_pos_c, dim=1)

#     hfov = np.arctan(camera.cfg.spawn.horizontal_aperture / (2 * camera.cfg.spawn.focal_length))
#     vfov = np.arctan(camera.cfg.spawn.vertical_aperture / (2 * camera.cfg.spawn.focal_length))

#     if isinstance(asset, RigidObject):
#         w_min, w_max, corners_w = get_asset_world_bbox(asset)
#     else:
#         w_min, w_max, corners_w = get_asset_world_aabb(asset)

#     # yaw = torch.abs(torch.atan2(relative_pos_c[:, 1], relative_pos_c[:, 0]))
#     # pitch = torch.abs(torch.atan2(relative_pos_c[:, 2], relative_pos_c[:, 0]))
#     yaw = torch.atan2(corners_w[..., 1], corners_w[..., 0])
#     pitch = torch.atan2(corners_w[..., 2], corners_w[..., 0])
#     # check = torch.logical_and(yaw < hfov, pitch < vfov).unsqueeze(1)
#     in_fov = ((yaw.abs() < hfov) & (pitch.abs() < vfov)).any(dim=1)
#     print(in_fov.float())
#     return in_fov.float()

def asset_in_fov(env, camera_name: str, asset_name: str) -> torch.Tensor:
    """
    Check if any part of a RigidObject asset is inside the camera FOV.

    Returns:
        torch.Tensor: [num_envs, 1] float tensor where 1.0 = any corner in FOV
    """

    # ----------------------------------------------------------------------
    # 1. Runtime objects
    # ----------------------------------------------------------------------
    camera = env.scene[camera_name]   # TiledCamera
    asset  = env.scene[asset_name]    # RigidObject
    robot  = env.scene["robot"]       # Articulation / robot base

    # ----------------------------------------------------------------------
    # 2. Configuration objects
    # ----------------------------------------------------------------------
    asset_cfg  = getattr(env.scene.cfg, asset_name)   # RigidObjectCfg
    camera_cfg = getattr(env.scene.cfg, camera_name)  # TiledCameraCfg

    # ----------------------------------------------------------------------
    # 3. Local bounding box corners (Cuboid only)
    # ----------------------------------------------------------------------
    size = torch.tensor(asset_cfg.spawn.size, device=env.device, dtype=torch.float32)
    half_extents = 0.5 * size

    signs = torch.tensor([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], device=env.device, dtype=torch.float32)

    corners_local = signs * half_extents   # [8, 3]

    # ----------------------------------------------------------------------
    # 4. Transform corners to world space
    # ----------------------------------------------------------------------
    corners_local_exp = corners_local.unsqueeze(0)   # [1, 8, 3]
    rotated = math.quat_apply(asset.data.root_quat_w.unsqueeze(1), corners_local_exp)  # [num_envs, 8, 3]
    corners_world = rotated + asset.data.root_pos_w.unsqueeze(1)                        # [num_envs, 8, 3]

    # ----------------------------------------------------------------------
    # 5. Transform from world → robot base → camera frame
    # ----------------------------------------------------------------------
    # world → robot base
    corners_base = math.quat_apply_inverse(
        robot.data.root_quat_w.unsqueeze(1),
        corners_world - robot.data.root_pos_w.unsqueeze(1)
    )

    # robot base → camera frame
    corners_cam = math.quat_apply_inverse(
        camera.data.quat_w_world.unsqueeze(1),
        corners_base
    )

    # ----------------------------------------------------------------------
    # 6. Compute camera FOV
    # ----------------------------------------------------------------------
    hfov = camera.cfg.spawn.horizontal_aperture / (2 * camera.cfg.spawn.focal_length)
    vfov = camera.cfg.spawn.vertical_aperture / (2 * camera.cfg.spawn.focal_length)

    hfov = torch.tensor(hfov, device=env.device, dtype=torch.float32)
    vfov = torch.tensor(vfov, device=env.device, dtype=torch.float32)

    # ----------------------------------------------------------------------
    # 7. Check yaw/pitch for each corner
    # ----------------------------------------------------------------------
    x = corners_cam[..., 0]
    y = corners_cam[..., 1]
    z = corners_cam[..., 2]

    yaw   = torch.abs(torch.atan2(y, x))
    pitch = torch.abs(torch.atan2(z, x))

    in_fov_per_corner = (yaw < hfov) & (pitch < vfov)

    # True if any corner is inside FOV
    is_any_corner_in_fov = torch.any(in_fov_per_corner, dim=1)
    print(is_any_corner_in_fov.unsqueeze(1).float())
    return is_any_corner_in_fov.unsqueeze(1).float()   # [num_envs, 1]

