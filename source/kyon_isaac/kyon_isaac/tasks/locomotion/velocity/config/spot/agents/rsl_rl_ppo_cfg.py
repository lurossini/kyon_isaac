# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from dataclasses import MISSING

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlDistillationStudentTeacherCfg, RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg
from kyon_isaac.rl.algorithms import DistillationWithCNN
from kyon_isaac.rl.modules import ActorCriticWithCNN, StudentTeacherWithCNN

@configclass
class SpotFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "kyon_flat"
    store_code_state = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True, 
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256, 256], 
        critic_hidden_dims=[256, 256, 256, 256], 
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=32, 
        learning_rate=3.0e-4,  
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

from rsl_rl.runners import on_policy_runner
on_policy_runner.ActorCriticWithCNN = ActorCriticWithCNN

@configclass
class VisionRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    resolution: tuple[int] = MISSING
    in_channels: int = MISSING

@configclass
class KyonVisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 8
    max_iterations = 1500
    save_interval = 50
    experiment_name = "kyon_navigation"
    store_code_state = False
    obs_groups = {"policy": ["critic"], "critic": ["critic"]} #, "rgb": ["rgb"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_obs_normalization=False, 
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        # in_channels=4,
        # resolution=(192, 120),   # resolution zedx mini 1920 x 1200
        activation="elu",
    )
    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=0.5,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.005,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=0.5,
    # )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=32, 
        learning_rate=3.0e-4,  
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

from rsl_rl.runners import distillation_runner
distillation_runner.StudentTeacherWithCNN = StudentTeacherWithCNN

@configclass
class VisionRslRlDistillationStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    resolution: tuple[int] = MISSING
    in_channels: int = MISSING

from rsl_rl.runners import distillation_runner
distillation_runner.DistillationWithCNN = DistillationWithCNN

@configclass
class KyonVisionStudentPPORunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 8
    max_iterations = 1500
    save_interval = 50
    experiment_name = "kyon_navigation"
    store_code_state = False
    obs_groups = {"policy": ["policy"], "teacher": ["critic"], "rgb": ["rgb"]}
    policy = VisionRslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacherWithCNN",
        init_noise_std=0.5,
        student_obs_normalization=False, 
        teacher_obs_normalization=False,
        student_hidden_dims=[128, 128],
        teacher_hidden_dims=[128, 128],
        in_channels=4,
        resolution=(192, 120),   # resolution zedx mini 1920 x 1200
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="DistillationWithCNN",
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )
