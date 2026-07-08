# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SpherePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 250
    experiment_name = "uhas_control"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, #! Exploration noise for actions
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, # 1.0
        use_clipped_value_loss=True,
        clip_param=0.2, # 0.1 - 0.3 Impact: Lower values make updates more conservative (safer but slower); your value is standard, but too low might allow destructive updates leading to collapse.
        entropy_coef=0.005, # 0.015 # 0.0 - 0.01 Typical range: 3–30. Impact: More epochs refine the policy better but risk overfitting
        num_learning_epochs=5, #8 # 3-30 Impact: Smaller batches (like yours) add noise for better generalization but can cause unstable gradients; larger might smooth updates.
        num_mini_batches=4,# 8 # 4 - 4096 Impact: Too high (yours is moderate-high) can cause large updates that overshoot and collapse performance; common culprit for plummeting after good start.
        learning_rate=5.0e-4, # Try smaller # 1.0 e -4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0, # 0-5 - 5 Impact: Prevents exploding gradients; yours is reasonable, but lowering could add safety.
    )

