# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationStudentTeacherRecurrentCfg
)

@configclass
class KyonRoughDistillationRunnerMLPCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 1000
    save_interval = 50
    experiment_name = "kyon_rough"
    obs_groups = {"policy": ["student"], "teacher": ["policy"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[256, 256, 256, 256],
        teacher_hidden_dims=[256, 256, 256, 256],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class KyonRoughDistillationRunnerRecurrentCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 1000
    save_interval = 50
    experiment_name = "kyon_rough"
    obs_groups = {"policy": ["policy"], "teacher": ["critic"]}
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[256, 256, 256, 256],
        teacher_hidden_dims=[512, 256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=2.0e-4,
        gradient_length=15,
    )