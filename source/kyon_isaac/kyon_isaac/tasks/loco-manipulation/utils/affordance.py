from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class Affordance:
    def __init__(self, env: ManagerBasedRLEnv, asset_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

        self.env = env
        self.robot = self.env.scene[robot_cfg.name]
        self.asset = self.env.scene[asset_name]
        self.asset_cfg  = getattr(self.env.scene.cfg, asset_name)   # RigidObjectCfg

    def get_bounding_box(self) -> torch.Tensor:

        size = torch.tensor(self.asset_cfg.spawn.size, device=self.env.device, dtype=torch.float32)
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
        ], device=self.env.device, dtype=torch.float32)

        corners_local = signs * half_extents  # [8,3]

        # Expand to batch and flatten for quat_apply
        num_corners = corners_local.shape[0]
        corners_local_exp = corners_local.unsqueeze(0).expand(self.env.scene.num_envs, -1, -1)  # [num_envs,8,3]
        corners_local_flat = corners_local_exp.reshape(-1, 3)                      # [num_envs*8,3]

        # Repeat quaternions for each corner
        asset_quats_flat = self.asset.data.root_quat_w.unsqueeze(1).expand(-1, num_corners, -1).reshape(-1, 4)  # [num_envs*8,4]

        # Rotate corners
        rotated_flat = math.quat_apply(asset_quats_flat, corners_local_flat)  # [num_envs*8,3]

        # Reshape and add position
        rotated = rotated_flat.view(self.env.scene.num_envs, num_corners, 3)
        corners_world = rotated + self.asset.data.root_pos_w.unsqueeze(1)

        # Transform to robot base
        robot_quats_flat = self.robot.data.root_quat_w.unsqueeze(1).expand(-1, num_corners, -1).reshape(-1, 4)
        corners_base_flat = (corners_world - self.robot.data.root_pos_w.unsqueeze(1)).reshape(-1, 3)
        corners_base = math.quat_apply_inverse(robot_quats_flat, corners_base_flat).view(self.env.scene.num_envs, num_corners, 3)

        return corners_base
    
class BoxBimanualGrasping(Affordance):
    def __init__(self, env: ManagerBasedRLEnv, asset_name: str):

        super().__init__(env, asset_name)

    def __call__(self):

        """
        Penalize tracking of the position error using L2-norm.
        The function computes the position error between the desired position (from the afforance) and the
        current position of the asset's body (in world frame). The position error is computed as the L2-norm
        of the difference between the desired and current positions.
        """
        # extract the asset (to enable type hinting)
        asset: RigidObject = self.env.scene[self.asset_cfg.name]
        command = self.env.command_manager.get_command(self.command_name)
        # obtain the desired and current positions
        des_pos_b = command[:, :3]
        des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
        return torch.norm(curr_pos_w - des_pos_w, dim=1)