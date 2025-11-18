from dataclasses import MISSING
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass, math

import numpy as np
import torch

# if TYPE_CHECKING:
from isaaclab.envs import ManagerBasedEnv

class VelocityBaseActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] # It can refer to any ActionTerm but we want it referred to VelocityBaseAction only!
    scale: float


class VelocityBaseAction(ActionTerm):
    cfg: VelocityBaseActionCfg
    _asset: Articulation
    _scale: torch.Tensor | float

    def __init__(self, cfg: VelocityBaseActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # initialize raw action
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)   
        self._processed_actions = torch.zeros_like(self.raw_actions)

        self.root_state = self._asset.data.root_link_state_w.clone()

        # parse scale
        if isinstance(cfg.scale, int):
            self._scale = float(cfg.scale)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self):
        pass
        # get base orientation
        # quat = self.root_state[:, 3:7]
        # print(type(quat[0]))
        # delta_xy = torch.tensor(np.array(kio.get_key()[0:2] + [0]) * sim_dt).float()
        # delta_xy = math.quat_apply_inverse(math.quat_inv(quat), delta_xy)
        
        # delta_yaw = kio.get_key()[2] * sim_dt
        # half = torch.tensor(delta_yaw * 0.5)

        # q_delta = torch.stack([
        #     torch.cos(half),      # w
        #     torch.tensor(0.),     # x
        #     torch.tensor(0.),     # y
        #     torch.sin(half)       # z
        # ]).to(root_state.device)

        # root_state[:, :3] += delta_xy
        # root_state[:, 3:7] = quat_mul(q_delta, root_state[:, 3:7])
        # robot.write_root_state_to_sim(root_state)
        # scene.write_data_to_sim()