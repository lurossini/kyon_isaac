from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationManager, ActionManager, CommandManager
from isaaclab.sensors.contact_sensor import ContactSensorCfg, ContactSensorData
from isaaclab.sensors.imu import ImuCfg, ImuData
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
import torch
from typing import Sequence
import isaaclab.utils.string as string_utils

class XBot2RobotData:
    def __init__(self):
        self.root_lin_vel_b = torch.zeros((1, 3))
        self.root_ang_vel_b = torch.zeros((1, 3))
        self.projected_gravity_b = torch.zeros((1, 3))
        self.joint_pos = torch.zeros((1, 12))
        self.default_joint_pos = torch.zeros((1, 12))
        self.joint_vel = torch.zeros((1, 12))
        self.default_joint_vel = torch.zeros((1, 12))
        self.applied_torque = torch.zeros((1, 12))

class XBot2Robot:
    def __init__(self, cfg: ArticulationCfg):
        self.cfg = cfg
        self.data: XBot2RobotData = XBot2RobotData()
        self.joint_names: list[str] = [
            'hip_roll_1', 'hip_pitch_1', 'knee_pitch_1',
            'hip_roll_2', 'hip_pitch_2', 'knee_pitch_2',
            'hip_roll_3', 'hip_pitch_3', 'knee_pitch_3',
            'hip_roll_4', 'hip_pitch_4', 'knee_pitch_4'
            ]
        self.num_joints: int = len(self.joint_names)
    def find_joints(self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        if joint_subset is None:
            joint_subset = self.joint_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    
class XBot2ImuSensor:
    def __init__(self, cfg: ImuCfg):
        self.cfg = cfg
        self.data = ImuData()
        self.data.lin_acc_b = torch.zeros((1, 3))
        self.data.ang_vel_b = torch.zeros((1, 3))

    
class XBot2ContactSensor:
    
    def __init__(self, cfg: ContactSensorCfg):
        self.cfg = cfg
        self.data = ContactSensorData()
        self.body_names: list[str] = ['contact_1', 'contact_2', 'contact_3', 'contact_4']
        self.num_bodies: int = len(self.body_names)
        self.data.net_forces_w_history = torch.zeros((1, self.cfg.history_length, self.num_bodies, 3))
    
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)
    
    
class XBot2Scene:
    
    def __init__(self, cfg: InteractiveSceneCfg):
        self.cfg = cfg
        self._assets = dict()
        self.sensors = dict()
        self.num_envs = 1
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            
            print(f'Loading asset: {asset_name} type {type(asset_cfg)}')
            if isinstance(asset_cfg, ArticulationCfg):
                self._assets[asset_name] = XBot2Robot(asset_cfg)
            elif isinstance(asset_cfg, ImuCfg):
                self._assets[asset_name] = XBot2ImuSensor(asset_cfg)
                self.sensors[asset_name] = self._assets[asset_name]
            elif isinstance(asset_cfg, ContactSensorCfg):
                self._assets[asset_name] = XBot2ContactSensor(asset_cfg)
                self.sensors[asset_name] = self._assets[asset_name]
            
    def __getitem__(self, key: str):
        return self._assets[key]
    def keys(self):
        return self._assets.keys()

class SimMockup:
    def __init__(self):
        self.device = "cuda:0"
    def is_playing(self):
        return True
    

class ManagerBasedXBot2Env(ManagerBasedEnv):

    def __init__(self, cfg: ManagerBasedEnvCfg):
        
        self.sim = SimMockup()  # Placeholder for actual simulation initialization
        self.scene = XBot2Scene(cfg.scene)  # Placeholder for actual scene initialization
        self.action_manager = ActionManager(cfg=cfg.actions, env=self)
        self.command_manager = CommandManager(cfg=cfg.commands, env=self) 
        self.observation_manager = ObservationManager(cfg=cfg.observations, env=self)
        super().__init__(cfg)