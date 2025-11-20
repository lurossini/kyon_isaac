# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from kyon_isaac.rl.modules import ActorCriticWithCNN

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
class KyonVisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "kyon_navigation"
    store_code_state = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "rgb": ["rgb"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticWithCNN",
        init_noise_std=1.0,
        actor_obs_normalization=True, 
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256, 256], 
        critic_hidden_dims=[256, 256, 256, 256], 
        resolution=(64, 64),
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4, 
        learning_rate=3.0e-4,  
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )