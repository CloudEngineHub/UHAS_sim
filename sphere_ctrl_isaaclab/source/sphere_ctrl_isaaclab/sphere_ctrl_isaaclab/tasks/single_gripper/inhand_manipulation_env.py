# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate,
    matrix_from_quat, quat_from_matrix, matrix_from_euler
)

from sphere_ctrl_isaaclab.tasks.utils.sphere_torch_utils import *
from .hand_env_cfg import SingleHandEnvCfg

class InHandManipulationEnv(DirectRLEnv):
    cfg: SingleHandEnvCfg

    def __init__(self, cfg: SingleHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Curriculum / stage vars - direct access (will raise AttributeError if missing/misspelled)
        self.curriculm_length = self.cfg.curriculm_length
        self.n_iter_per_stage = self.cfg.n_iter_per_stage
        self.iteration_bias = self.cfg.iteration_bias
        self.learning_iter = 0

        # Flags - direct access
        self.training = self.cfg.training

        # Hand info
        self.num_hand_dofs = self.hand.num_joints

        # Action buffers
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # All joints from the asset are considered actuated (no need for actuated_joint_names)
        self.actuated_dof_indices = list(range(self.num_hand_dofs))
        self.joint_index_dict = {name: idx for idx, name in enumerate(self.hand.joint_names)}

        # Joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # Goal
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.18], device=self.device)
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # Success tracking
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.total_resets = torch.tensor(-self.num_envs, dtype=torch.float, device=self.device)
        self.reset_counts = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.all_successes = []
        self.running_sum_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.avg_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        # New evaluation metric: target pose success rate over first 10 tries per env (skipping the initial reset call)
        self.eval_target_reset_counts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.eval_num_successes = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.eval_num_tries = torch.tensor(0.0, dtype=torch.float, device=self.device)


        # Unit tensors
        self.x_unit_tensor = torch.tensor([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0., 1., 0.], device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0., 0., 1.], device=self.device).repeat((self.num_envs, 1))
        self.PI = torch.tensor(np.pi, dtype=torch.float, device=self.device)

        # === Sphere + Kinematics setup (minimal, NO ft / tip_rot observations) ===
        kin_chains = self.cfg.cik_info["kin_chains"]
        self.finger_chains = [
            chain for chain in kin_chains
            if "palm_normal" not in chain and "sphere_frame" not in chain
        ]

        self.joint_info = self.cfg.cik_info["joint_info"]
        self.sphere_radius = float(self.joint_info["sphere_frame"][6])
        self.joint_type_info = self.cfg.cik_info["joint_type_info"]

        # q_0 / q_sphere / q_ref
        q_open_dict = self.cfg.cik_info.get("q_open_palm", {})
        q_sphere_dict = self.cfg.cik_info.get("q_sphere", {})
        joint_names = self.hand.joint_names
        self.q_0 = torch.tensor([q_open_dict.get(j, 0.0) for j in joint_names],
                                dtype=torch.float, device=self.device)
        self.q_sphere = torch.tensor([q_sphere_dict.get(j, 0.0) for j in joint_names],
                                     dtype=torch.float, device=self.device)
        self.q_ref = self.q_sphere.clone()

        # Build minimal structures for q_ref solver
        self.joint_ls = {}
        self.joint_limits = {}
        for chain in self.finger_chains:
            for joint in chain[1:-1]:
                if joint in self.joint_type_info:
                    self.joint_ls[joint] = torch.norm(
                        torch.tensor(self.joint_type_info[joint]["nj"][:2], dtype=torch.float, device=self.device))
                    self.joint_limits[joint] = (
                        torch.tensor(self.joint_info[joint][4], dtype=torch.float, device=self.device),
                        torch.tensor(self.joint_info[joint][5], dtype=torch.float, device=self.device),
                    )

        # Main / non-main indices
        self.main_joints = []
        for chain in self.finger_chains:
            for joint in chain[1:-1]:
                if self.joint_type_info.get(joint, {}).get("type") == "A":
                    self.main_joints.append(joint)
                    break
        self.main_indices = torch.tensor([self.joint_index_dict[j] for j in self.main_joints],
                                         dtype=torch.int, device=self.device)
        all_idx = torch.arange(self.num_hand_dofs, device=self.device)
        self.non_main_indices = all_idx[~torch.isin(all_idx, self.main_indices)]

        # === Follow multi_env logic exactly for transforms ===
        # 1. Build T_world_to_sphere (flip + inclination)
        T_flip = torch.eye(4, device=self.device)
        T_flip[1, 1] = -1
        T_flip[2, 2] = -1

        inc = torch.deg2rad(torch.tensor(self.cfg.inclination, device=self.device))
        T_incl = torch.eye(4, device=self.device)
        T_incl[0, 0] = torch.cos(inc)
        T_incl[0, 2] = torch.sin(inc)
        T_incl[2, 0] = -torch.sin(inc)
        T_incl[2, 2] = torch.cos(inc)

        self.T_world_to_sphere = T_flip @ T_incl

        # 2. Get T_sphere_to_base from cik_info
        quat_s = torch.tensor(self.joint_info["sphere_frame"][1], device=self.device)[[3, 0, 1, 2]]
        self.T_base_to_sphere = torch.eye(4, device=self.device)
        self.T_base_to_sphere[:3, :3] = matrix_from_quat(quat_s)
        self.T_base_to_sphere[:3, 3] = torch.tensor(self.joint_info["sphere_frame"][0], device=self.device)
        self.T_sphere_to_base = torch.inverse(self.T_base_to_sphere)

        # 3. T_world_to_base = T_world_to_sphere @ T_sphere_to_base
        self.T_world_to_base = self.T_world_to_sphere @ self.T_sphere_to_base

        # Write initial hand pose
        hand_state = torch.zeros_like(self.hand.data.default_root_state)
        hand_state[:, :3] = self.T_world_to_base[:3, 3] + self.scene.env_origins
        hand_state[:, 3:7] = quat_from_matrix(self.T_world_to_base[:3, :3])
        self.hand.write_root_state_to_sim(hand_state)

        # === Config-driven init_pos and inhand_position (direct access) ===
        sphere_radius_t = torch.tensor(self.sphere_radius, device=self.device)
        init_pos_cfg = torch.tensor(self.cfg.init_pos, dtype=torch.float, device=self.device)
        inhand_pos_cfg = torch.tensor(self.cfg.inhand_position, dtype=torch.float, device=self.device)

        self.sphere_init_pos_h = torch.cat([init_pos_cfg * sphere_radius_t, torch.ones(1, device=self.device)])
        self.sphere_in_hand_pos_h = torch.cat([inhand_pos_cfg * sphere_radius_t, torch.ones(1, device=self.device)])

        self.init_pos_in_sphere = self.sphere_init_pos_h[:3].repeat(self.num_envs, 1)
        self.in_hand_pos = (self.T_world_to_sphere @ self.sphere_in_hand_pos_h)[:3].repeat(self.num_envs, 1)

        print(f"Using configurable init_pos (radii): {self.cfg.init_pos}")
        print(f"Using inhand_position (radii): {self.cfg.inhand_position}")

        # Joint transforms
        self.T_joints = {}
        for chain in self.finger_chains:
            for joint in chain[1:]:
                xyz = torch.tensor(self.joint_info[joint][0], device=self.device)
                qj = torch.tensor(self.joint_info[joint][1], device=self.device)[[3, 0, 1, 2]]
                Tj = torch.eye(4, device=self.device)
                Tj[:3, :3] = matrix_from_quat(qj)
                Tj[:3, 3] = xyz
                self.T_joints[joint] = Tj.clone()

        # === Compute q_ref (will raise clear error on failure - no silent fallback) ===
        print("Computing q_ref ...")
        ref = import_reference_sphere()
        self.ref_sphere_xyz = torch.tensor(ref["xyz"], device=self.device, dtype=torch.float32)
        self.ref_sphere_r = torch.tensor(ref["r"], device=self.device, dtype=torch.float32)

        self.q_ref = self.q_sphere.clone()
        radius_mask = self.ref_sphere_r >= 0

        for chain in self.finger_chains:
            if len(chain) < 2: continue
            chain_i = 0
            points_scaled = self.ref_sphere_xyz * self.ref_sphere_r[chain_i].unsqueeze(-1) * sphere_radius_t
            ones = torch.ones((points_scaled.shape[0], 1), device=self.device)
            torch_points_h = torch.cat([points_scaled, ones], dim=1).permute(1, 0)
            radii_mask = radius_mask[chain_i]

            T_cum = self.T_sphere_to_base.clone()
            for joint in chain[1:-1]:
                Tj = self.T_joints[joint]
                qv = self.q_ref[self.joint_index_dict[joint]]
                c, s = torch.cos(qv), torch.sin(qv)
                Trot = torch.zeros((4, 4), device=self.device)
                Trot[0, 0] = c; Trot[0, 1] = -s; Trot[1, 0] = s; Trot[1, 1] = c
                Trot[2, 2] = Trot[3, 3] = 1.0

                Tjs = torch.inverse(T_cum @ Tj @ Trot)
                tpoints = (Tjs @ torch_points_h).permute(1, 0)[:, :3]

                jtype = self.joint_type_info.get(joint, {}).get("type", "A")
                if jtype == "A":
                    T_cum = T_cum @ Tj @ Trot
                    continue
                if jtype in ("B", "D"):
                    l = self.joint_ls.get(joint, torch.tensor(0.01, device=self.device))
                    pn = torch.norm(tpoints[:, :2], dim=1)
                    rmask = (pn >= l).T
                    pmask = (rmask & radii_mask).T

                    new_q = torch_solve_for_B_joint(
                        joint, self.q_ref.unsqueeze(0), self.joint_index_dict,
                        tpoints.unsqueeze(1), self.joint_limits.get(joint, (torch.tensor(-1.), torch.tensor(1.))),
                        pmask.unsqueeze(1), self.joint_type_info
                    )
                    if self.joint_type_info.get(joint, {}).get("og_type") == "A":
                        new_q = self.q_ref[self.joint_index_dict[joint]]
                    self.q_ref[self.joint_index_dict[joint]] = new_q.squeeze()

                    qv2 = self.q_ref[self.joint_index_dict[joint]]
                    c2, s2 = torch.cos(qv2), torch.sin(qv2)
                    Trot2 = torch.zeros((4, 4), device=self.device)
                    Trot2[0, 0] = c2; Trot2[0, 1] = -s2; Trot2[1, 0] = s2; Trot2[1, 1] = c2
                    Trot2[2, 2] = Trot2[3, 3] = 1.0
                    T_cum = T_cum @ Tj @ Trot2

        print("q_ref ready. Main joints for lateral penalty:", self.main_joints)

        # Action history (use num_dofs)
        self.actions = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        self.last_actions = self.actions.clone()
        self.last_last_actions = self.actions.clone()

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=[0, 0, -0.6])

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if hasattr(self, "event_manager") and "prestartup" in getattr(self.event_manager, "available_modes", []):
            self.event_manager.apply(mode="prestartup")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        
        self.actions = torch.clamp( 
            self.actions,
            -torch.ones_like(self.actions),
            torch.ones_like(self.actions)
        )
        self.last_last_actions = self.last_actions.clone()
        self.last_actions = self.actions.clone()
        self.actions = actions.clone()

        if self.cfg.action_noise > 0.0 and self.training:
            self.actions = self.actions + get_noise(self.actions, self.cfg.action_noise)

        self.processed_actions = self.actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.processed_actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices] +
            (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        )
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        return self.compute_full_observations()

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _get_rewards(self) -> torch.Tensor:
        hand_dof_pos = self.hand.data.joint_pos[:, self.actuated_dof_indices]
        lateral_diffs = [hand_dof_pos[:, self.main_indices] - self.q_ref[self.main_indices][None, :]]
        radii_diffs = [hand_dof_pos[:, self.non_main_indices] - self.q_ref[self.non_main_indices][None, :]]
        q_vels = [self.hand.data.joint_vel[:, self.actuated_dof_indices]]
        q_effs = [self.hand.data.applied_torque[:, self.actuated_dof_indices]]

        total_reward, self.reset_goal_buf, self.successes[:], self.consecutive_successes[:], \
        lat_pen, rad_pen, dist_rew, rot_rew, vel_pen, en_pen, act_pen = compute_rewards(
            self.reset_buf, self.reset_goal_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.in_hand_pos, self.goal_rot,
            self.cfg.dist_reward_scale, self.cfg.rot_reward_scale, self.cfg.rot_eps,
            self.actions, self.cfg.lateral_position_penalty_scale,
            lateral_diffs, self.cfg.radii_position_penalty_scale,
            radii_diffs, self.cfg.success_tolerance, self.cfg.reach_goal_bonus,
            self.cfg.fall_dist, self.cfg.fall_penalty, self.cfg.av_factor,
            q_vels, self.cfg.velocity_scale,
            self.cfg.max_velocity, self.cfg.velocity_tolerance,
            self.cfg.energy_scale, q_effs,
            self.cfg.action_rate,
            self.last_actions, self.last_last_actions
        )

        if "log" not in self.extras: self.extras["log"] = {}
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        goal_envs = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_envs) > 0:
            self._reset_target_pose(goal_envs)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance, torch.zeros_like(self.episode_length_buf), self.episode_length_buf)
            max_succ = self.successes >= self.cfg.max_consecutive_success
            time_out = (self.episode_length_buf >= self.max_episode_length - 1) | max_succ
        else:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        self.learning_iter = (self.common_step_counter // max(1, self.cfg.num_steps_per_env)) + self.iteration_bias

        # Reset hand root
        hand_state = torch.zeros_like(self.hand.data.default_root_state[env_ids])
        hand_state[:, :3] = self.T_world_to_base[:3, 3] + self.scene.env_origins[env_ids]
        hand_state[:, 3:7] = quat_from_matrix(self.T_world_to_base[:3, :3])
        self.hand.write_root_state_to_sim(hand_state, env_ids=env_ids)

        self.in_hand_pos[env_ids] = (self.T_world_to_sphere @ self.sphere_in_hand_pos_h)[:3].repeat(len(env_ids), 1)

        # Object reset - multi style
        num_r = len(env_ids)
        pos_noise = sample_uniform(-1., 1., (num_r, 3), device=self.device); pos_noise[:, 2] = 0.
        pos_sphere = self.init_pos_in_sphere[env_ids] + self.cfg.reset_position_noise * pos_noise * self.sphere_radius

        angles = torch.tensor([0., np.pi/2, np.pi, 3*np.pi/2], device=self.device)
        roll = angles[torch.randint(0, 4, (num_r,), device=self.device)]
        pitch = angles[torch.randint(0, 4, (num_r,), device=self.device)]
        yaw = torch.rand(num_r, device=self.device) * 2 * np.pi

        R_face = matrix_from_euler(torch.stack([roll, pitch, torch.zeros_like(yaw)], dim=-1), convention="XYZ")
        R_yaw = matrix_from_euler(torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1), convention="XYZ")
        R = R_yaw @ R_face

        T_so = torch.eye(4, device=self.device).unsqueeze(0).repeat(num_r, 1, 1)
        T_so[:, :3, :3] = R
        T_so[:, :3, 3] = pos_sphere

        T_wo = self.T_world_to_sphere @ T_so
        pos_w = T_wo[:, :3, 3] + self.scene.env_origins[env_ids]
        quat_w = quat_from_matrix(T_wo[:, :3, :3])

        obj_state = torch.zeros_like(self.object.data.default_root_state[env_ids])
        obj_state[:, :3] = pos_w
        obj_state[:, 3:7] = quat_w
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros_like(obj_state[:, 7:]), env_ids=env_ids)

        # Hand DOF reset
        dof_pos = self.q_0.unsqueeze(0).repeat(num_r, 1) if self.q_0.numel() > 0 else self.hand.data.default_joint_pos[env_ids].clone()
        dof_pos = dof_pos + sample_uniform(-1., 1., (num_r, self.num_hand_dofs), device=self.device) * self.cfg.reset_dof_pos_noise
        dof_vel = self.hand.data.default_joint_vel[env_ids].clone() + sample_uniform(-1., 1., (num_r, self.num_hand_dofs), device=self.device) * self.cfg.reset_dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids, joint_ids=self.actuated_dof_indices)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids, joint_ids=self.actuated_dof_indices)

        

        # Evaluation Code
        if self.cfg.evaluation:
            # Accumulate successes and update average one environment at a time
            # to ensure exact multiples of 100 are hit for printing.
            # NOTE: Logic preserved exactly (reset_counts < 2) to remove effect of first resets.
            for id in env_ids:
                if self.reset_counts[id] < 2:
                    # Add the consecutive successes for this specific environment
                    self.running_sum_successes += self.successes[id]
                    # Append success value to list for min, max, std and histogram
                    if self.reset_counts[id] == 1:
                        self.all_successes.append(self.successes[id].item())
                    # Increment the total resets count
                    self.total_resets += 1

                    # Update average
                    if self.total_resets > 0:
                        self.avg_successes = self.running_sum_successes / self.total_resets

                    # Print every 100 samples
                    if self.total_resets % 100 == 0 and self.total_resets > 0:
                        successes_tensor = torch.tensor(self.all_successes, dtype=torch.float, device=self.device)

                        mean_val = self.avg_successes.item()
                        std_val  = torch.std(successes_tensor, unbiased=False).item()
                        min_val  = successes_tensor.min().item()
                        max_val  = successes_tensor.max().item()
                        n_samples = int(self.total_resets.item())

                        # === Histogram (0 to 10) ===
                        from collections import Counter
                        counts = Counter(self.all_successes)
                        hist = [counts.get(i, 0) for i in range(11)]  # counts for 0, 1, ..., 10
                        print(f"Average consecutive successes after {n_samples} samples: "
                            f"Mean = {mean_val:.3f}, Std = {std_val:.3f}, "
                            f"Min = {min_val:.1f}, Max = {max_val:.1f}")
                        print(f"Histogram (0-10): {hist}")
                        print("-" * 60)

                # Always increment the reset count for this environment
                self.reset_counts[id] += 1

        self._reset_target_pose(env_ids)
        self._compute_intermediate_values()
        self.successes[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

    def _reset_target_pose(self, env_ids):
        # New target-pose success rate evaluation (first 10 tries per env after init reset)
        if self.cfg.evaluation:
            current_counts = self.eval_target_reset_counts[env_ids]
            # Count tries #2 to #11 (skip the very first init call per env, which always has reset_goal_buf=0)
            count_mask = (current_counts >= 1) & (current_counts < 11)
            # count_mask = (current_counts >= 1)
            if count_mask.any():
                relevant_envs = env_ids[count_mask]
                # reset_goal_buf==1 means the just-completed try for this target pose was successful (goal reached)
                successes_this = self.reset_goal_buf[relevant_envs].to(torch.float)
                self.eval_num_successes += successes_this.sum()
                num_new = relevant_envs.numel()
                self.eval_num_tries += float(num_new)
                current_total = int(self.eval_num_tries.item())
                prev_total = current_total - num_new
                # Print whenever we cross or land on a multiple of 100 (handles batch adds of >1 envs at once)
                if (current_total // 100) > (prev_total // 100):
                    rate = (self.eval_num_successes / self.eval_num_tries * 100).item()
                    print(f"Target pose success rate after {current_total} tries: {rate:.2f}%")

            # Always increment (even for the ignored first call and beyond 10)
            self.eval_target_reset_counts[env_ids] += 1

        prev_rot = self.goal_rot[env_ids].clone()
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        candidate_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1],
            self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        rel_q = quat_mul(quat_conjugate(prev_rot), candidate_rot)
        angle = 2 * torch.acos(torch.clamp(rel_q[:, 0].abs(), max=1.0))
        too_close = angle < torch.pi / 2
        flip = torch.zeros_like(candidate_rot)
        flip[:, 3] = 1.0
        candidate_rot[too_close] = quat_mul(candidate_rot[too_close], flip[too_close])
        self.goal_rot[env_ids] = candidate_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)
        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        self.hand_dof_pos = self.hand.data.joint_pos[:, self.actuated_dof_indices]
        self.hand_dof_vel = self.hand.data.joint_vel[:, self.actuated_dof_indices]

        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        self.learning_iter = (self.common_step_counter // max(1, self.cfg.num_steps_per_env)) + self.iteration_bias

    def compute_full_observations(self):
        # Progressive noise scale
        cur_iter = self.learning_iter
        stage3_start = 2 * self.n_iter_per_stage
        noise_scale = 0.0 if cur_iter < stage3_start else min(1.0, max(0.0, (cur_iter - stage3_start) / self.n_iter_per_stage))

        # === CRITIC (clean) ===
        # Object information is always added to the critic
        critic_list = [
            self.object_pos,
            self.object_rot,
            self.object_linvel,
            self.cfg.vel_obs_scale * self.object_angvel,
        ]

        # c_linvel_obs controls whether hand DOF velocity is included in critic
        if self.cfg.c_handvel_obs:
            critic_list.append(self.cfg.vel_obs_scale * self.hand_dof_vel)

        critic_list.extend([
            self.goal_rot,
            quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
            unscale(self.hand_dof_pos, self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                    self.hand_dof_upper_limits[:, self.actuated_dof_indices]),
            self.actions,
        ])
        critic_obs = torch.cat(critic_list, dim=-1)

        # === ACTOR / POLICY (noisy + hand DOF noise) ===
        # Hand DOF position noise (actor only)
        hand_dof_pos_n = unscale(self.hand_dof_pos, self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                                 self.hand_dof_upper_limits[:, self.actuated_dof_indices])
        if self.training and self.cfg.hand_dof_pos_noise > 0.0:
            hand_dof_pos_n = hand_dof_pos_n + get_noise(hand_dof_pos_n, self.cfg.hand_dof_pos_noise * noise_scale)

        # Hand DOF velocity noise (actor only)
        hand_dof_vel_n = self.cfg.vel_obs_scale * self.hand_dof_vel
        if self.training and self.cfg.hand_dof_vel_noise > 0.0:
            hand_dof_vel_n = hand_dof_vel_n + get_noise(hand_dof_vel_n, self.cfg.hand_dof_vel_noise * noise_scale)

        obj_pos_n = self.object_pos.clone()
        if self.training and self.cfg.obj_pos_noise > 0.0:
            obj_pos_n = obj_pos_n + get_noise(obj_pos_n, self.cfg.obj_pos_noise * noise_scale)

        obj_rot_n = self.object_rot.clone()
        if self.training and self.cfg.obj_rot_noise > 0.0:
            obj_rot_n = quat_mul(random_small_quats(len(obj_rot_n), self.cfg.obj_rot_noise * noise_scale, self.device), obj_rot_n)

        policy_list = [
            hand_dof_pos_n,
            obj_pos_n,
            obj_rot_n,
        ]

        # linvel_obs controls whether we include hand DOF velocity in actor observations
        if self.cfg.handvel_obs:
            policy_list.append(hand_dof_vel_n)

        if self.cfg.obj_linvel_obs:
            oln = self.object_linvel.clone()
            if self.training and self.cfg.obj_linvel_noise > 0.0:
                oln = oln + get_noise(oln, self.cfg.obj_linvel_noise * noise_scale)
            policy_list.append(oln)

        if self.cfg.obj_angvel_obs:
            oan = self.object_angvel.clone()
            if self.training and self.cfg.obj_angvel_noise > 0.0:
                oan = oan + get_noise(oan, self.cfg.obj_angvel_noise * noise_scale)
            policy_list.append(self.cfg.vel_obs_scale * oan)

        policy_list.extend([
            self.goal_rot,
            quat_mul(obj_rot_n, quat_conjugate(self.goal_rot)),
            self.actions,
        ])
        policy_obs = torch.cat(policy_list, dim=-1)

        return {"policy": policy_obs, "critic": critic_obs}


# ==================== Helpers & Reward ====================

def get_noise(value, std):
    return torch.zeros_like(value) if std <= 0 else torch.randn_like(value) * std


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def random_small_quats(num_envs: int, max_angle_rad: float, device: torch.device) -> torch.Tensor:
    angles = torch.rand(num_envs, device=device) * 2 * max_angle_rad - max_angle_rad
    axes = torch.randn(num_envs, 3, device=device)
    axes = axes / torch.norm(axes, dim=-1, keepdim=True)
    half = angles / 2.0
    return torch.cat([torch.cos(half).unsqueeze(-1), torch.sin(half).unsqueeze(-1) * axes], dim=-1)


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    qd = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(qd[:, 1:4], p=2, dim=-1), max=1.0))


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    lateral_position_penalty_scale: float,
    lateral_position_diff: list[torch.Tensor],
    radii_position_penalty_scale: float,
    radii_position_diff: list[torch.Tensor],
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
    q_vel: list[torch.Tensor],
    velocity_scale: float,
    max_velocity: float,
    velocity_tolerance: float,
    energy_scale: float,
    q_eff: list[torch.Tensor],
    action_rate: float,
    last_actions: torch.Tensor,
    last_last_actions: torch.Tensor,
):
    goal_dist = torch.clamp(torch.norm(object_pos - target_pos, p=2, dim=-1), 0.0, fall_dist)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps)

    lat_pen = torch.cat([torch.sum(d**2, dim=-1) for d in lateral_position_diff])
    rad_pen = torch.cat([torch.sum(d**2, dim=-1) for d in radii_position_diff])

    reward = (dist_rew * dist_reward_scale + rot_rew * rot_reward_scale +
              lat_pen * lateral_position_penalty_scale + rad_pen * radii_position_penalty_scale)

    vel_pen = torch.cat([torch.sum((v / max(1e-6, max_velocity - velocity_tolerance))**2, dim=-1) for v in q_vel])
    en_pen = torch.cat([torch.sum(torch.abs(v) * torch.abs(e), dim=-1) for v, e in zip(q_vel, q_eff)])
    act_pen = torch.sum((actions - last_actions)**2, dim=-1) + torch.sum((actions - 2*last_actions + last_last_actions)**2, dim=-1)

    reward = reward + vel_pen * velocity_scale + en_pen * energy_scale + act_pen * action_rate

    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    num_resets = torch.sum(resets)
    finished = torch.sum(successes * resets.float())
    cons_succ = torch.where(num_resets > 0,
                            av_factor * finished / num_resets + (1.0 - av_factor) * consecutive_successes,
                            consecutive_successes)
    return reward, goal_resets, successes, cons_succ, lat_pen, rad_pen, dist_rew, rot_rew, vel_pen, en_pen, act_pen
