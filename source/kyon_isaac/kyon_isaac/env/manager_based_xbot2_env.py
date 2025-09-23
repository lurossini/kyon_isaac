from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationManager

class ManagerBasedXBot2Env(ManagerBasedRLEnv):

    def __init__(self, cfg: ManagerBasedRLEnvCfg):
        super().__init__(cfg)