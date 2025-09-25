from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationManager, ActionManager, CommandManager
from isaaclab.sensors.contact_sensor import ContactSensorCfg, ContactSensorData
from isaaclab.sensors.imu import ImuCfg, ImuData
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
import torch
from tensordict import TensorDict
from typing import Sequence
import isaaclab.utils.string as string_utils
from xbot2_zmq_robot_interface import ZmqRobot

class XBot2RobotData:
    def __init__(self, num_joint: int):
        self.root_lin_vel_b = torch.zeros((1, 3))
        self.root_ang_vel_b = torch.zeros((1, 3))
        self.projected_gravity_b = torch.zeros((1, 3))
        self.joint_pos = torch.zeros((1, 12))
        self.default_joint_pos = torch.zeros((1, 12))
        self.joint_vel = torch.zeros((1, 12))
        self.default_joint_vel = torch.zeros((1, 12))
        self.applied_torque = torch.zeros((1, 12))
        self.joint_pos_target = torch.zeros((1, 12))

class XBot2Robot:
    
    def __init__(self, cfg: ArticulationCfg):
        self.cfg = cfg
        self.data: XBot2RobotData = XBot2RobotData(len(self.cfg.init_state.joint_pos.keys()))
        
        self.xbot_robot = ZmqRobot()
        self.joint_names: list[str] = list(self.cfg.init_state.joint_pos.keys())

        for i, jname in enumerate(self.joint_names):
            self.data.default_joint_pos[i] = self.cfg.init_state.joint_pos[jname]
            self.data.default_joint_vel[i] = self.cfg.init_state.joint_vel[jname]

        self.num_joints: int = len(self.joint_names)

    def find_joints(self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        if joint_subset is None:
            joint_subset = self.joint_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)
    
    def update(self):
        self.data.joint_pos = self.xbot_robot.getJointPosition()
        self.data.joint_vel = self.xbot_robot.getJointVelocities()
        self.data.applied_torque = self.xbot_robot.getJointEffort()
        self.data.joint_pos_target = self.xbot_robot.getPositionReference()
        
        pass
    
    def set_joint_position_target(self, target, joint_ids):
        self.data.joint_pos_target[joint_ids] = target
        
    def move(self):
        pass

    
class XBot2ImuSensor:
    
    def __init__(self, cfg: ImuCfg):
        self.cfg = cfg
        self.data = ImuData()
        self.data.lin_acc_b = torch.zeros((1, 3))
        self.data.ang_vel_b = torch.zeros((1, 3))
    
    def update(self):
        pass

    
class XBot2ContactSensor:
    
    def __init__(self, cfg: ContactSensorCfg):
        self.cfg = cfg
        self.data = ContactSensorData()
        self.body_names: list[str] = ['contact_1', 'contact_2', 'contact_3', 'contact_4']
        self.num_bodies: int = len(self.body_names)
        self.data.net_forces_w_history = torch.zeros((1, self.cfg.history_length, self.num_bodies, 3))
    
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)
    
    def update(self):
        pass
    
    
class XBot2Scene:
    
    def __init__(self, cfg: InteractiveSceneCfg):
        self.cfg = cfg
        self._assets = dict()
        self.sensors = dict()
        self.robot: XBot2Robot = None
        self.num_envs = 1
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            
            print(f'Loading asset: {asset_name} type {type(asset_cfg)}')
            if isinstance(asset_cfg, ArticulationCfg):
                self._assets[asset_name] = XBot2Robot(asset_cfg)
                self.robot = self._assets[asset_name]
            elif isinstance(asset_cfg, ImuCfg):
                self._assets[asset_name] = XBot2ImuSensor(asset_cfg)
                self.sensors[asset_name] = self._assets[asset_name]
            elif isinstance(asset_cfg, ContactSensorCfg):
                self._assets[asset_name] = XBot2ContactSensor(asset_cfg)
                self.sensors[asset_name] = self._assets[asset_name]
                
    def update(self):
        for asset in self._assets.values():
            asset.update()
            
    def write_data_to_robot(self):
        self.robot.move()
            
    def __getitem__(self, key: str):
        return self._assets[key]
    
    def keys(self):
        return self._assets.keys()

class SimMockup:
    def __init__(self):
        self.device = "cpu"
    def is_playing(self):
        return True
    

class ManagerBasedXBot2Env:

    def __init__(self, cfg: ManagerBasedEnvCfg):
        self.num_envs = 1
        self.device = 'cpu'
        self.step_dt = cfg.sim.dt * cfg.decimation
        self.sim = SimMockup()  # Placeholder for actual simulation initialization
        self.scene = XBot2Scene(cfg.scene)  # Placeholder for actual scene initialization
        self.action_manager = ActionManager(cfg=cfg.actions, env=self)
        self.command_manager = CommandManager(cfg=cfg.commands, env=self) 
        self.observation_manager = ObservationManager(cfg=cfg.observations, env=self)
        self.unwrapped = self  # Placeholder for actual unwrapping logic
        self.num_actions = self.action_manager.total_action_dim
        
    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(obs_dict, batch_size=[self.num_envs])
    
    def reset(self) -> tuple[TensorDict, dict]:
        raise NotImplementedError("Reset not implemented yet (?!?!)")
    
    def step(self, action: torch.Tensor):
        
        # process and apply actions
        self.action_manager.process_action(action.to(self.device))
        self.action_manager.apply_action()
        
        # write data to robot
        self.scene.write_data_to_robot()

        # update scene
        self.scene.update()
        
        # TODO sync dt
        print('step')