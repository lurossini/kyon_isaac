from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def goal_reached_termination(
    env: ManagerBasedRLEnv,
    source_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    threshold: float
) -> torch.Tensor:
    source_asset: Articulation = env.scene[source_asset_cfg.name]
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]

    source_pos = source_asset.data.root_pos_w
    target_pos = target_asset.data.root_pos_w
    distance = torch.norm(target_pos - source_pos, dim=1)

    return torch.where(distance < threshold, True, False)