from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab.utils.math import sample_uniform

# from kyon_isaac.sensors import ActionHistorySensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv

def reset_joint_target_to_default(env: ManagerBasedRLEnv,  env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    
    articulation_asset: Articulation = env.scene[asset_cfg.name]

    # obtain default joint positions
    print(articulation_asset.data.default_joint_pos.shape)
    print(env_ids)
    print(asset_cfg.joint_ids)
    default_joint_pos = articulation_asset.data.default_joint_pos[:, asset_cfg.joint_ids].clone()
    default_joint_vel = articulation_asset.data.default_joint_vel[:, asset_cfg.joint_ids].clone()
    # reset joint targets if required
    articulation_asset.set_joint_position_target(default_joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    articulation_asset.set_joint_velocity_target(default_joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_joints_around_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the interval around the default position and velocity by the given ranges.

    This function samples random values from the given ranges around the default joint positions and velocities.
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    # print("DEBUG")
    # print(asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids].shape)
    # print(asset_cfg.joint_ids)
    joint_min_pos = asset.data.default_joint_pos[env_ids[:, None], asset_cfg.joint_ids] + position_range[0]
    joint_max_pos = asset.data.default_joint_pos[env_ids[:, None], asset_cfg.joint_ids] + position_range[1]
    joint_min_vel = asset.data.default_joint_vel[env_ids[:, None], asset_cfg.joint_ids] + velocity_range[0]
    joint_max_vel = asset.data.default_joint_vel[env_ids[:, None], asset_cfg.joint_ids] + velocity_range[1]
    # clip pos to range
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids[:, None], asset_cfg.joint_ids, ...]
    joint_min_pos = torch.clamp(joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_max_pos = torch.clamp(joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    # clip vel to range
    joint_vel_abs_limits = asset.data.soft_joint_vel_limits[env_ids[:, None], asset_cfg.joint_ids]
    joint_min_vel = torch.clamp(joint_min_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    joint_max_vel = torch.clamp(joint_max_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    # sample these values randomly
    joint_pos = sample_uniform(joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device)
    joint_vel = sample_uniform(joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=asset_cfg.joint_ids)
