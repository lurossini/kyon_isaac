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
from .xbot2_zmq_robot_interface import ZmqRobot
import numpy as np
import time

class XBot2RobotData:
    def __init__(self, num_joint: int):
        self.root_lin_vel_b = torch.zeros((1, 3))
        self.root_ang_vel_b = torch.zeros((1, 3))
        self.projected_gravity_b = torch.zeros((1, 3))
        self.joint_pos = torch.zeros((1, num_joint))
        self.default_joint_pos = torch.zeros((1, num_joint))
        self.joint_vel = torch.zeros((1, num_joint))
        self.default_joint_vel = torch.zeros((1, num_joint))
        self.applied_torque = torch.zeros((1, num_joint))
        self.joint_pos_target = torch.zeros((1, num_joint))
        self.joint_vel_target = torch.zeros((1, num_joint))

class XBot2Robot:
    
    def __init__(self, cfg: ArticulationCfg, xbot_robot: ZmqRobot):
        self.cfg = cfg
        self.xbot_robot = xbot_robot
        self.joint_names: list[str] = ['hip_roll_1', 'hip_roll_2', 'hip_roll_3', 'hip_roll_4', 
                                       'hip_pitch_1', 'hip_pitch_2', 'hip_pitch_3', 'hip_pitch_4', 
                                       'knee_pitch_1', 'knee_pitch_2', 'knee_pitch_3', 'knee_pitch_4']
                                       
        
        arms = False
        if arms:
            self.joint_names = ['hip_roll_1', 'hip_roll_2', 'hip_roll_3', 'hip_roll_4', 
                                'shoulder_yaw_1', 'shoulder_yaw_2', 
                                'hip_pitch_1', 'hip_pitch_2', 'hip_pitch_3', 'hip_pitch_4', 
                                'shoulder_pitch_1', 'shoulder_pitch_2', 'knee_pitch_1', 
                                'knee_pitch_2', 'knee_pitch_3', 'knee_pitch_4', 
                                'elbow_pitch_1', 'elbow_pitch_2',
                                'wrist_pitch_1', 'wrist_pitch_2', 
                                'wrist_yaw_1', 'wrist_yaw_2', 
                                'dagana_1_clamp_joint', 'dagana_2_clamp_joint']
            
            self.fixed_joints = ['shoulder_yaw_1', 'shoulder_pitch_1', 'elbow_pitch_1', 'wrist_pitch_1', 'wrist_yaw_1', 'dagana_1_clamp_joint',
                                 'shoulder_yaw_2', 'shoulder_pitch_2', 'elbow_pitch_2', 'wrist_pitch_2', 'wrist_yaw_2', 'dagana_2_clamp_joint']
            ctrl_mode = [0 if j in self.fixed_joints else 25 for j in self.joint_names]

        wheels = True
        if wheels:
            self.joint_names = ['hip_roll_1', 'hip_roll_2', 'hip_roll_3', 'hip_roll_4', 
                                'shoulder_yaw_1', 'shoulder_yaw_2', 
                                'hip_pitch_1', 'hip_pitch_2', 'hip_pitch_3', 'hip_pitch_4', 
                                'shoulder_pitch_1', 'shoulder_pitch_2', 'knee_pitch_1', 
                                'knee_pitch_2', 'knee_pitch_3', 'knee_pitch_4', 
                                'elbow_pitch_1', 'elbow_pitch_2',
                                # 'ankle_yaw_1', 'ankle_yaw_2', 'ankle_yaw_3', 'ankle_yaw_4',
                                'wrist_pitch_1', 'wrist_pitch_2', 
                                'wheel_joint_1', 'wheel_joint_2', 'wheel_joint_3', 'wheel_joint_4',
                                'wrist_yaw_1', 'wrist_yaw_2', 
                                'dagana_1_clamp_joint', 'dagana_2_clamp_joint']
            self.fixed_joints = ['shoulder_yaw_1', 'shoulder_pitch_1', 'elbow_pitch_1', 'wrist_pitch_1', 'wrist_yaw_1', 'dagana_1_clamp_joint',
                                 'shoulder_yaw_2', 'shoulder_pitch_2', 'elbow_pitch_2', 'wrist_pitch_2', 'wrist_yaw_2', 'dagana_2_clamp_joint']
                                 #  'ankle_yaw_1', 'ankle_yaw_2', 'ankle_yaw_3', 'ankle_yaw_4']
            self.wheel_joints = [f'wheel_joint_{i}' for i in range(1, 5)]   
            ctrl_mode = [0 if j in self.fixed_joints else (26 if j in self.wheel_joints else 25) for j in self.joint_names]

        self.idx_xbot_to_isaac = []
        for jn in self.joint_names:
            self.idx_xbot_to_isaac.append(self.xbot_robot.joint_names.index(jn))

        self.num_joints: int = len(self.joint_names)
        self.data: XBot2RobotData = XBot2RobotData(self.num_joints)
        
        self.stiffness = np.zeros(len(self.joint_names))
        self.damping = np.zeros(len(self.joint_names))

        for act in self.cfg.actuators.values():
            _, joints = self.find_joints(act.joint_names_expr, self.joint_names, preserve_order=True)
            for j in joints:
                idx = self.joint_names.index(j)
                self.stiffness[idx] = act.stiffness
                self.damping[idx] = act.damping

        self.stiffness = np.array(self.stiffness)
        self.damping = np.array(self.damping)
        
        self.xbot_robot.enableJoints(self.joint_names)  
        self.xbot_robot.set_filter_frequency_hz(40.0, False)
        self.xbot_robot.setVelocityReference(np.zeros(self.num_joints))
        self.xbot_robot.setEffortReference(np.zeros(self.num_joints))
        self.xbot_robot.setStiffness(self.stiffness)
        self.xbot_robot.setDamping(self.damping)
        # self.xbot_robot.setCtrlMode(np.ones(self.num_joints, dtype=int) * 25)
        self.xbot_robot.setCtrlMode(np.array(ctrl_mode))
        self.time = 0

        for i, jname in enumerate(self.joint_names):
            self.data.default_joint_pos[0, i] = self.cfg.init_state.joint_pos[jname]

        for i in range(len(self.joint_names)):
            print(f'Joint {i} idx {self.idx_xbot_to_isaac[i]} name {self.joint_names[i]} default pos {self.data.default_joint_pos[0, i]} stiffness {self.stiffness[i]} damping {self.damping[i]}')

    def find_joints(self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        if joint_subset is None:
            joint_subset = self.joint_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)
              
    def update(self):
        self.data.joint_pos[0, :] = torch.tensor(self.xbot_robot.getJointPosition())[self.idx_xbot_to_isaac]
        self.data.joint_vel[0, :] = torch.tensor(self.xbot_robot.getMotorVelocities())[self.idx_xbot_to_isaac]
        self.data.applied_torque[0, :] = torch.tensor(self.xbot_robot.getJointEffort())[self.idx_xbot_to_isaac]
        self.data.joint_pos_target[0, :] = torch.tensor(self.xbot_robot.getPositionReference())[self.idx_xbot_to_isaac]
        self.data.joint_vel_target[0, :] = torch.tensor(self.xbot_robot.getVelocityReference())[self.idx_xbot_to_isaac]
        self.data.projected_gravity_b[0, :] = -torch.tensor(self.xbot_robot.getImuOrientation()[2, :]) * torch.tensor([1, -1, -1])
        self.data.root_ang_vel_b[0, :] = torch.tensor(self.xbot_robot.getImuAngularVelocity()) * torch.tensor([1, -1, -1])
        self.time += 0.02
    
    def set_joint_position_target(self, target, joint_ids):
        self.data.joint_pos_target[0, joint_ids] = target

    def set_joint_velocity_target(self, target, joint_ids):
        self.data.joint_vel_target[0, joint_ids] = target
        
    def move(self):
        self.xbot_robot.setPositionReference(self.data.joint_pos_target[0, :].numpy())
        self.xbot_robot.setVelocityReference(self.data.joint_vel_target[0, :].numpy())
        self.xbot_robot.move()

    
class XBot2ImuSensor:
    
    def __init__(self, cfg: ImuCfg, xbot_robot: ZmqRobot):
        self.cfg = cfg
        self.data = ImuData()
        self.xbot_robot = xbot_robot
        self.data.lin_acc_b = torch.zeros((1, 3))
        self.data.ang_vel_b = torch.zeros((1, 3))
    
    def update(self):
        self.data.ang_vel_b[0, :] = torch.tensor(self.xbot_robot.getImuAngularVelocity())

    
class XBot2ContactSensor:
    
    def __init__(self, cfg: ContactSensorCfg):
        self.cfg = cfg
        self.data = ContactSensorData()
        wheels = True
        if wheels:  
            self.body_names:list[str] = ['wheel_1', 'wheel_2', 'wheel_3', 'wheel_4']
        else:
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
        self.xbot_robot = ZmqRobot()
        self.num_envs = 1
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            
            print(f'Loading asset: {asset_name} type {type(asset_cfg)}')
            if isinstance(asset_cfg, ArticulationCfg):
                self._assets[asset_name] = XBot2Robot(asset_cfg, self.xbot_robot)
                self.robot = self._assets[asset_name]
            elif isinstance(asset_cfg, ImuCfg):
                self._assets[asset_name] = XBot2ImuSensor(asset_cfg, self.xbot_robot)
                self.sensors[asset_name] = self._assets[asset_name]
            elif isinstance(asset_cfg, ContactSensorCfg):
                self._assets[asset_name] = XBot2ContactSensor(asset_cfg)
                self.sensors[asset_name] = self._assets[asset_name]
                
    def update(self):
        self.xbot_robot.sense()
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
        self.t_last = time.time()
        
    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(obs_dict, batch_size=[self.num_envs])
    
    def reset(self) -> tuple[TensorDict, dict]:
        raise NotImplementedError("Reset not implemented yet (?!?!)")
    
    def step(self, action: torch.Tensor):

        time.sleep(max(0, self.step_dt - (time.time() - self.t_last)))
        self.t_last = time.time()

        self.command_manager.compute(self.step_dt)
        
        # process and apply actions
        self.action_manager.process_action(action.to(self.device))
        self.action_manager.apply_action()
        
        # write data to robot
        self.scene.write_data_to_robot()

        # update scene
        self.scene.update()

        obs =  self.get_observations()

        # obsvec = obs['policy'].flatten()
        # print('---')
        # print('ang vel', obsvec[0:3])
        # print('proj grav', obsvec[3:6])
        # print('vel cmd', obsvec[6:9])
        # print('joint pos', obsvec[9:21])
        # print('joint vel', obsvec[21:33])
        # print('action', obsvec[33:45])
        # print('---')

        return obs
        