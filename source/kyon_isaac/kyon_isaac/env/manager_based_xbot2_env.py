from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationManager, ActionManager, CommandManager
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
    data: XBot2RobotData = XBot2RobotData() 
    joint_names: list[str] = [
        'hip_roll_1', 'hip_pitch_1', 'knee_pitch_1',
        'hip_roll_2', 'hip_pitch_2', 'knee_pitch_2',
        'hip_roll_3', 'hip_pitch_3', 'knee_pitch_3',
        'hip_roll_4', 'hip_pitch_4', 'knee_pitch_4'
    ]
    num_joints: int = len(joint_names)
    def find_joints(self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        if joint_subset is None:
            joint_subset = self.joint_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

@configclass
class XBot2ImuSensorData:
    ang_vel_b: torch.Tensor = torch.zeros((1, 3))
    lin_acc_b: torch.Tensor = torch.zeros((1, 3))
    
class XBot2ImuSensor:
    data: XBot2ImuSensorData = XBot2ImuSensorData()

@configclass
class XBot2Scene:
    robot: XBot2Robot = XBot2Robot()
    imu_sensor: XBot2ImuSensor = XBot2ImuSensor()
    num_envs: int = 1
    def __getitem__(self, key: str):
        return getattr(self, key)
    def keys(self):
        return self.__dict__.keys()

class SimMockup:
    def __init__(self):
        self.device = "cuda:0"
    def is_playing(self):
        return True
    

class ManagerBasedXBot2Env(ManagerBasedEnv):

    def __init__(self, cfg: ManagerBasedEnvCfg):
        
        self.sim = SimMockup()  # Placeholder for actual simulation initialization
        self.scene = XBot2Scene()  # Placeholder for actual scene initialization
        self.action_manager = ActionManager(cfg=cfg.actions, env=self)
        self.command_manager = CommandManager(cfg=cfg.commands, env=self) 
        self.observation_manager = ObservationManager(cfg=cfg.observations, env=self)
        super().__init__(cfg)