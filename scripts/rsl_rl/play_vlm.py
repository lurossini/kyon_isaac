# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import threading
import zmq
from collections import OrderedDict

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

class ZMQIO:    
    """ZMQ communication class to send image and prompt to another process running the Cosmos VLM"""
    REMOTE_IP: str = "localhost"

    def __init__(self):
        context = zmq.Context()
        self.__socket_pull = context.socket(zmq.PULL)
        self.__socket_pull.connect(f"tcp://{self.REMOTE_IP}:5555")
        self.__socket_pull.setsockopt(zmq.CONFLATE, 1)
        self.__socket_push = context.socket(zmq.PUSH)
        self.__socket_push.connect(f"tcp://{self.REMOTE_IP}:5556")

        self.__latest_image_sent = None
        self.__get_new_image = True
        self.__latest_detection = None
        self.latest_detection_lock = threading.Lock()

        self.__img_buffer = OrderedDict()

        receiver_thread = threading.Thread(
            target=self.zmq_receiver,
            daemon=True
        )
        receiver_thread.start()

    def zmq_receiver(self):
        while True:
            try:
                # Drain queue and keep only the newest message
                msg = None
                while True:
                    msg = self.__socket_pull.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                self.__get_new_image = True

            if msg is not None:
                try:
                    response_text = msg
                    response_dict = response_text

                    # Sometimes VLM may reply with a list(dict). In this case, take the first element
                    if isinstance(response_dict, list):
                        if len(response_dict) > 0:
                            response_dict = response_dict[0]
                        else:
                            response_dict = None

                    with self.latest_detection_lock:
                        self.__latest_detection = response_dict

                except Exception as e:
                    print("Error parsing response:", e)

            threading.Event().wait(0.001)

    def send_image(self, obs, counter):
        # send camera views
        frame_depth = obs['rgbd'][0, :, :, :]
        frame_depth = frame_depth.detach().cpu().contiguous().numpy()
        frame_depth[:, :, :3] /= 255

        self.__img_buffer[counter] = frame_depth
        if len(self.__img_buffer) > 100:
            self.__img_buffer.popitem(last=False)

        if self.__get_new_image:
            self.__latest_image_sent = frame_depth
            self.__get_new_image = False
        
        meta = {
            "seq": counter,
            "type": "rgbd",
            "dtype": str(frame_depth.dtype),
            "shape": frame_depth.shape,
            "note": "Detect the mug in the image.\n"+
                    "Return the results in JSON format with this MANDATORY information:\n"+
                    "- detected (true/false)\n"+
                    "- confidence (0–1)\n"+
                    "- box_2d in [x1, y1, x2, y2] in normalized coordinates.\n"
        }

        self.__socket_push.send_multipart([json.dumps(meta).encode("utf-8"), frame_depth.tobytes()])


    def get_sent_image(self, seq: int):
        return self.__img_buffer[seq]
    
    @property
    def get_latest_detection(self):
        return self.__latest_detection

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--interactive", action="store_true", default=False, help="Run in real-time, if possible.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# joy
if args_cli.interactive:
    import pygame
    import zmq
    from proto import joy_msg_pb2

    REMOTE_IP = '*'
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.bind(f"tcp://{REMOTE_IP}:5050")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

zmq_io = ZMQIO()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import kyon_isaac.tasks  # noqa: F401


# PLACEHOLDER: Extension template (do not remove this comment)

if args_cli.interactive:
    rx_msg = joy_msg_pb2.JoyMsg()

import zmq
import cv2
import json

import numpy as np 
np.set_printoptions(suppress=True, precision=3)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None) 

    ### Get camera properties
    height = env_cfg.scene.front_up_camera.height 
    width = env_cfg.scene.front_up_camera.width
    focal_length_cm = env_cfg.scene.front_up_camera.spawn.focal_length
    h_aperture = env_cfg.scene.front_up_camera.spawn.horizontal_aperture 
    v_aperture = env_cfg.scene.front_up_camera.spawn.vertical_aperture

    if v_aperture is None:
        v_aperture = h_aperture * (height / width)

    print(f'width: {width}')
    print(f'height: {height}')
    print(f'h_aperture: {h_aperture}')
    print(f'v_aperture: {v_aperture}')

    # Pixel size and focal length: if the pixel is square, sx = sy and fx = fy
    sx = h_aperture / width         # horizontal pixel size
    sy = v_aperture / height        # vertical pixel size
    fx = focal_length_cm / sx       # horizontal focal length in pixels
    fy = focal_length_cm / sy       # vertical focal length in pixels

    print(f'sx: {sx}')
    print(f'sy: {sy}')

    # Compute intrinsic matrix
    K = torch.tensor([[fx, 0, width/2], [0, fy, height/2], [0, 0, 1]], device=args_cli.device)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    ref = [0., 0., 0.]
    counter = 0
    detected = False

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # send image from camera to VLM
            zmq_io.send_image(obs, counter)

            # get latest processed image
            with zmq_io.latest_detection_lock:
                latest_detection = zmq_io.get_latest_detection

            # check that the first response has been received 
            if latest_detection is not None:
                if latest_detection['detected'] == True:
                    
                    # compute center of the bounding box
                    x_center = int((latest_detection['bbox_pixels'][0] + latest_detection['bbox_pixels'][2]) / 2)
                    y_center = int((latest_detection['bbox_pixels'][1] + latest_detection['bbox_pixels'][3]) / 2)
                    
                    # get depth of the bbox center pixel
                    depth = zmq_io.get_sent_image(latest_detection['seq'])[y_center, x_center, 3]

                    # convert from image frame to camera frame
                    x_obj = (x_center - width/2) * depth / fx
                    y_obj = (y_center - height/2) * depth / fy
                    y_obj_camera = -x_obj
                    z_obj_camera = -y_obj
                    x_obj_camera = depth

                    # convert to PoseCommand                  
                    print(f'Estimated position: {x_obj_camera}, {y_obj_camera}, {z_obj_camera}')
                    cmd = torch.tensor([x_obj_camera, y_obj_camera, z_obj_camera, 0, 0, 0, 1])   # ignore orientation tracking
                    
                    if torch.norm(cmd[:3]) < 4:         # ignore too far targets that may come from hallucination or detection errors
                        env.unwrapped.command_manager.get_term('left_ee_pose').set_command(cmd.unsqueeze(0).repeat(env.unwrapped.num_envs, 1).float())
                        detected = True
                else:
                    # env.unwrapped.command_manager.get_term('left_ee_pose').reset_command()
                    detected = False

            # agent stepping
            actions = policy(obs)

            # explore the environment moving the robot with the joystick
            if args_cli.interactive:
                while True:
                    try:
                        msg = socket.recv(flags=zmq.NOBLOCK)
                        rx_msg.ParseFromString(msg)
                        ref = [-rx_msg.axes[1], -rx_msg.axes[0], -rx_msg.axes[3]]
                    except zmq.Again:
                        break    
                if not detected:                # base velocity references from joystick can be sent as soon as the object is not detected
                    torch_ref = torch.tensor(ref)
                    actions[:, :3] = torch_ref.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        counter += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
