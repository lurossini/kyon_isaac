from __future__ import annotations

from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
import torch

class KyonRayCaster(RayCaster):
    """
    A ray-cast sensor that can be attached to an entity in the scene.

    This class extends the base :class:`isaaclab.sensors.RayCaster` with
    additional functionalities, such as a mask that creates holes in the height scan to simulate occlusions.
    """

    cfg: KyonRayCasterCfg

    def __init__(self, cfg: KyonRayCasterCfg):
        super().__init__(cfg)

        self.failure_rate = cfg.failure_rate
        self.interval_range_s = cfg.interval_range_s

        if cfg.update_period > cfg.update_occlusion_period:
            raise ValueError(f"update_period ({cfg.update_period}) must be lower than or equal to update_occlusion_period ({cfg.update_occlusion_period})")

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        
        # Reset occlusion mask to no occlusion for the reset environments
        self._is_occlusion_outdated[env_ids] = False
        self.occlusion_mask[env_ids] = False

    def _initialize_impl(self):
        super()._initialize_impl()
        self._is_occlusion_outdated = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._timestamp_last_occlusion_update = torch.full((self._num_envs,), -torch.inf, device=self._device)

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self.occlusion_mask = torch.zeros(self._view.count, self.num_rays, device=self.device, dtype=torch.bool)

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        env_ids_flat = env_ids.flatten()

        # Apply occlusion mask to ray hits
        self._is_occlusion_outdated |= (
            (self._timestamp > self.interval_range_s[0])
            & (self._timestamp < self.interval_range_s[1])
            & (self._timestamp - self._timestamp_last_occlusion_update + 1e-6 >= self.cfg.update_occlusion_period)
        )
        outdated_occlusion_env_ids = self._is_occlusion_outdated.nonzero().squeeze(-1)
        if len(outdated_occlusion_env_ids) > 0:
            r = torch.empty(len(outdated_occlusion_env_ids), self.num_rays, device=self.device)
            self.occlusion_mask[outdated_occlusion_env_ids] = r.uniform_(0, 1) < self.failure_rate
            self._timestamp_last_occlusion_update[outdated_occlusion_env_ids] = self._timestamp[outdated_occlusion_env_ids]
            self._is_occlusion_outdated[outdated_occlusion_env_ids] = False
        else:
            self.occlusion_mask[env_ids_flat] = False
        self._data.ray_hits_w[self.occlusion_mask] = 0.0

        # --- debug ---
        # active = self.occlusion_mask.nonzero(as_tuple=False)  # (N, 2): [[env, ray], ...]
        # print(f"[RayCaster] occluded rays: {active.tolist()}", flush=True)
        # --- end debug ---

@configclass
class KyonRayCasterCfg(RayCasterCfg):

    '''
    Force to KyonRayCaster class type
    '''
    class_type: type = KyonRayCaster

    '''
    Interval of time (in seconds) in which the occlusion mask is active
    default = (0.0, 0.0) means the occlusion mask is always inactive (no holes in height scan).
    '''
    interval_range_s: tuple[float, float] = (0.0, 0.0)

    '''
    Period (in seconds) at which the occlusion mask is updated
    '''
    update_occlusion_period: float = 0.1

    '''
    Failure rate (between 0 and 1) for the occlusion mask, 
    i.e., the probability of each ray being occluded (set to 0 in height scan).
    '''
    failure_rate: float = 0.05

