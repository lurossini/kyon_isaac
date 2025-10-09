from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import kyon_isaac.tasks.locomotion.velocity.mdp as kyon_mdp


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_tracking_lin_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
   
    # Tracking of linear velocity commands (xy axes).
    sigma = np.sqrt(0.25)
    asset: RigidObject = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command("base_velocity")[:, :2]
    local_vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = commands - local_vel
    lin_vel_error = lin_vel_error / sigma
    return torch.exp(-torch.sum(torch.square(lin_vel_error), dim=1))

def reward_tracking_ang_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        )  -> torch.Tensor:
    # Tracking of angular velocity commands (yaw).
    sigma = np.sqrt(0.25)
    asset: RigidObject = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel = asset.data.root_ang_vel_b[:, 2]
    ang_vel_error = commands - ang_vel
    ang_vel_error = ang_vel_error / sigma 
    return torch.exp(-torch.square(ang_vel_error))

def reward_posture(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_angles = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    commands = env.command_manager.get_command("base_velocity")[:, :2]

    cost = torch.sum(torch.square(joint_angles - default_pos), dim=1) #* self._weights)
    cmd_norm = torch.norm(commands, dim=1)
    weight = torch.where(
        cmd_norm < 0.01,
        -10.0,
        0.0,
    )
    return torch.exp(weight * cost)

def cost_lin_vel_z(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
    # Penalize z axis base linear velocity.
    asset: RigidObject = env.scene[asset_cfg.name]
    global_linvel = asset.data.root_lin_vel_w[:, 2]
    return torch.square(global_linvel)

def cost_ang_vel_xy(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    # Penalize xy axes base angular velocity.
    asset: RigidObject = env.scene[asset_cfg.name]
    global_angvel = asset.data.root_ang_vel_w[:, :2]
    return torch.sum(torch.square(global_angvel), dim=1)

def cost_orientation(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
    # Penalize non flat base orientation.
    asset: RigidObject = env.scene[asset_cfg.name]
    torso_zaxis = asset.data.projected_gravity_b[:, :2]
    return torch.sum(torch.square(torso_zaxis), dim=1)

# Energy related rewards.

def cost_torques(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
    # Penalize torques.
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque
    return torch.sqrt(torch.sum(torch.square(torques), dim=1)) + torch.sum(torch.abs(torques), dim=1)

def cost_energy(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    # Penalize energy consumption.
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel
    torques = asset.data.applied_torque
    return torch.sum((torch.abs(qvel) * torch.abs(torques)), dim=1)

def cost_action_rate(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    env.action_manager.get_term
    c1 = mdp.last_action(env)
    c2 = kyon_mdp.last_last_action(env, sensor_cfg)
    return c1 + c2

def cost_feet_slip(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg, 
        sensor_cfg: SceneEntityCfg,
        threshold: float
        ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w
    is_contact = torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > threshold

    vel_xy_norm_sq = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)   
    return torch.sum(vel_xy_norm_sq * is_contact, dim=1)

def cost_feet_clearance(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg, 
        target_height: float,
        ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_xy = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    vel_norm = torch.sqrt(torch.linalg.norm(vel_xy, axis=-1))
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # TODO(kevin): Desired foot height should be proportional to the command.
    # desired_z = 0.05 + torch.linalg.norm(command[:2]) * 0.1
    delta = torch.abs(foot_z - target_height)
    return torch.sum(delta, dim=1)

# def cost_feet_height(
#         env: ManagerBasedRLEnv, 
#         asset_cfg: SceneEntityCfg, 
#         swing_peak: torch.Tensor,
#         first_contact: torch.Tensor,
#         fault_mask: torch.Tensor,
#         ) -> torch.Tensor:
#     cmd_norm = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2])
#     error = swing_peak / self._config.reward_config.max_foot_height - 1.0
#     cost = torch.sum(torch.square(error) * first_contact * fault_mask)
#     cost *= cmd_norm >= 0.01  # No penalty for zero commands.
#     return cost

def reward_feet_air_time(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    # Reward air time.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]


    cmd_norm = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :], dim=1)
    rew_air_time = torch.sum((current_air_time - 0.1), dim=1)
    return torch.where(
        cmd_norm > 0.01,
        rew_air_time * cmd_norm,
        0.0,
    )