# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher
from isaaclab.utils import math

import numpy as np
import time

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from kyon_isaac.assets.kyon_play import KYON_ONLY_ARMS_CFG_PLAY   # isort:skip
import keyboard_input

torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)

kio = keyboard_input.KeyboardIO()
ref = [0, 0, 0]


def ori_error_euler_quat(euler: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Computes the orientation error between an orientation expressed in Euler and quaternion respectively. 
       Returns the orientation error in Euler form"""
    eul_to_quat = math.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

    # Make q_target consistent with q_current hemisphere
    dot = torch.sum(eul_to_quat * quat, dim=1, keepdim=True)
    eul_to_quat = torch.where(dot < 0, -eul_to_quat, eul_to_quat)

    # siciliano's method
    # print(dot)
    # print(f'act: {quat}')
    # print(f'ref: {eul_to_quat}')
    rot_err = torch.zeros_like(euler)
    rot_err = eul_to_quat[:, 3] * quat[:, :3] - quat[:, 3] * eul_to_quat[:, :3] - torch.cross(quat[:, :3], eul_to_quat[:, :3], dim=1)
    return rot_err

    ori_error_quat = math.quat_mul(eul_to_quat, math.quat_inv(quat))

    # extract axis-angle
    w = ori_error_quat[:, 0]
    xyz = ori_error_quat[:, 1:]
    angle = 2 * torch.atan2(xyz.norm(dim=1), w)       # (N,)
    axis = xyz / (xyz.norm(dim=1, keepdim=True) + 1e-8)

    rotvec = axis * angle.unsqueeze(-1)
    return rotvec       # (N, 3)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = KYON_ONLY_ARMS_CFG_PLAY.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    total_mass = torch.sum(robot.data.default_mass, dim=-1) #.expand(scene.num_envs, -1).unsqueeze(1)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    init_root_state = robot.data.root_link_state_w.clone()
    old_xy_ref = init_root_state[:, :3]
    old_ori_ref = torch.stack(math.euler_xyz_from_quat(init_root_state[:, 3:7]), dim=1)

    # Simulation loop
    while simulation_app.is_running():
        # get the current root state
        root_state = robot.data.root_link_state_w.clone()
        quat = root_state[:, 3:7]

        # velocity reference from keyboard
        vel_xy_ref = kio.get_key()[:2] + [0.0]
        vel_xy_ref = torch.tensor(vel_xy_ref, device=args_cli.device).float().expand(scene.num_envs, -1)
        vel_xy_ref[:] = math.quat_apply_inverse(math.quat_inv(quat), vel_xy_ref[:])

        # linear position reference
        delta_xy_ref = vel_xy_ref * sim_dt
        xy_ref = old_xy_ref + delta_xy_ref
        old_xy_ref = xy_ref
         
        # angular velocity reference
        omega = kio.get_key()[2]
        ang_vel_ref = torch.tensor([0, 0, omega], device=args_cli.device).float().expand(scene.num_envs, -1)

        # orientation reference
        # ori_ref = torch.stack(math.euler_xyz_from_quat(root_state[:, 3:7]), dim=1) + ang_vel_ref * sim_dt
        ori_ref = old_ori_ref + ang_vel_ref * sim_dt
        old_ori_ref = ori_ref
        ori_ref_quat = math.quat_from_euler_xyz(ori_ref[:, 0], ori_ref[:, 1], ori_ref[:, 2])

        print(f'pose_ref: {torch.hstack((xy_ref, ori_ref_quat))}')
        print(f'pose: {root_state[:, :7]}')
        print(f'vel_ref: {torch.hstack((vel_xy_ref, ang_vel_ref))}')
        print(f'vel    : {root_state[:, 7:]}')

        pose_error = math.compute_pose_error(root_state[:, :3], root_state[:, 3:7], xy_ref, ori_ref_quat)
        F = (2.5*torch.tensor([800, 800, 800, 200, 200, 200], device=args_cli.device) * torch.hstack(pose_error) + 
             torch.tensor([400, 400, 400, 100, 100, 100], device=args_cli.device) * (torch.hstack((vel_xy_ref, ang_vel_ref)) - root_state[:, 7:])).unsqueeze(1)
        # F[:, :, 2] += total_mass * 9.81
        # F[:, :] = math.quat_apply_inverse(math.quat_inv(quat), F[:, :])
        print(f'F: {F}')
        robot.set_external_force_and_torque(forces=F[:, :, :3], torques=F[:, :, 3:], body_ids=robot.find_bodies("pelvis")[0])

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    # sim.set_camera_view([2.5, 0.0, 2.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Keyboard IO
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
