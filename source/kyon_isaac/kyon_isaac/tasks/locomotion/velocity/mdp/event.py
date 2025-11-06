from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

# from kyon_isaac.sensors import ActionHistorySensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

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
