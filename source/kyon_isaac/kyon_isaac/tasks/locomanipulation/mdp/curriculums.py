from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

def goal_distance_range(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int],
    command: str,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.1,
    step_size: float = 0.1,
) -> dict:
    """ Curriculum term based on the distance between the robot and the goal. """
    
    """ 
    The x-y range of the goal is increased for every environment in which the robot reaches the goal 
    within the episode length. Otherwise, the range is decreased. 
    """

    # Get the command term and the reference pose
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    cmd_values = env.command_manager.get_command(command)

    # Get the asset and the current body pose
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice) or body_ids is None:
        body_id = 0
    else:
        body_id = body_ids[0]
    curr_pos_w = asset.data.body_pos_w[env_ids_t, body_id]

    # Fetch the command term object to update per-environment ranges.
    cmd_obj = getattr(env.command_manager, "_terms", {}).get(command, None)
    if cmd_obj is None or not hasattr(cmd_obj, "get_ranges") or not hasattr(cmd_obj, "update_ranges"):
        raise RuntimeError(
            f"Command term '{command}' does not support per-environment range updates. "
            "Expected get_ranges(env_ids) and update_ranges(env_ids, new_ranges)."
        )

    # Prefer world-frame desired position if available on the command object.
    # Fallback to command manager output if the term does not expose pose_command_w.
    if hasattr(cmd_obj, "pose_command_w"):
        des_pos_w = cmd_obj.pose_command_w[env_ids_t, :3]
    else:
        des_pos_w = cmd_values[env_ids_t, :3]

    # Compute the distance
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    # Update environements based on the distance to the goal
    move_up = distance <= threshold
    move_down = distance > threshold

    curr_ranges = cmd_obj.get_ranges(env_ids_t)

    def _update_xy_range(range_values: torch.Tensor) -> torch.Tensor:
        # Keep range center fixed while scaling half-width per env.
        center = 0.5 * (range_values[:, 0] + range_values[:, 1])
        half = 0.5 * (range_values[:, 1] - range_values[:, 0])
        zero_half = half == 0

        expand_factor = 1.0 + step_size
        shrink_factor = 1.0 - step_size
        scale = torch.ones_like(half)
        scale = torch.where(move_up, torch.full_like(scale, expand_factor), scale)
        scale = torch.where(move_down, torch.full_like(scale, shrink_factor), scale)
        half = half * scale
        half = torch.where(zero_half & move_up, torch.full_like(half, step_size), half)
        return torch.stack((center - half, center + half), dim=1)

    updated_ranges = {
        "pos_x": _update_xy_range(curr_ranges["pos_x"]),
        "pos_y": _update_xy_range(curr_ranges["pos_y"]),
    }
    cmd_obj.update_ranges(env_ids_t, updated_ranges)

    # Report the mean upper half-range across all environments ([:, 1] = upper bound).
    return {
        "pos_x_mean_upper_range": cmd_obj._ranges["pos_x"][:, 1].mean(),
        "pos_y_mean_upper_range": cmd_obj._ranges["pos_y"][:, 1].mean(),
        "reset_distance_mean": distance.mean(),
        "reset_success_rate": move_up.float().mean(),
    }

