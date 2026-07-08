from __future__ import annotations
import numpy as np
import torch
import time
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, matrix_from_quat, quat_from_matrix, quat_from_euler_xyz, matrix_from_euler
from isaaclab.managers import SceneEntityCfg, EventManager
# import tf.transformations as tf
from .multi_env_cfg import SphereEnvCfg
from sphere_ctrl_isaaclab.tasks.utils.sphere_torch_utils import *

class InHandManipulationEnv(DirectRLEnv):
    cfg: SphereEnvCfg
    def __init__(self, cfg: SphereEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)# Runs setup_scene

        # Curriculum Learning Vars
        self.n_iter_per_stage = self.cfg.curriculm_length / 4
        self.learning_iter = 0

        # Configuration Flags, for easy access
        self.training = self.cfg.training
        self.randomize_vectors_phi = self.cfg.randomize_vectors_phi & self.training
        self.n_fp_samples = self.cfg.fp_sample_size
        self.max_fingers = self.cfg.max_fingers
        self.merge_fingers = self.cfg.merge_fingers

        # Initialize dictionaries
        self.num_hand_dofs = dict() 
        self.hand_dof_targets = dict()
        self.prev_targets = dict()
        self.cur_targets = dict()
        self.finger_chains = dict()
        self.finger_chain_order = dict()
        self.chain_indices = dict()
        self.n_fingers = dict()
        self.n_joints = dict()
        self.joints = dict()
        self.joint_index_dict = dict()
        self.actuated_dof_indices = dict()
        self.joint_info = dict()
        self.joint_type_info = dict()
        self.sphere_radius = dict()
        self.main_joints = dict()
        self.main_indices = dict()
        self.non_main_indices = dict()
        self.chain_joint_indices = dict()
        self.hand_dof_lower_limits = dict()
        self.hand_dof_upper_limits = dict()
        self.m_joint_res = dict()
        self.m_joint_zero_idx = dict()
        self.m_joint_max = dict()
        self.m_joint_anchor_dist = dict()
        self.m_ft_sphere_h = dict()
        self.m_ft_joint_h = dict()
        self.joint_ls = dict()
        self.next_joint_dict = dict() 
        self.joint_phis = dict()
        self.m_joint_q_lists = dict()
        self.boxes_dict = dict()
        self.joint_limits = dict()
        
        self.q_0 = dict()
        self.q_sphere = dict()
        self.q_ref = dict()
        self.q = dict()
        self.past_q = dict()

        self.T_base_to_sphere = dict()
        self.T_base_to_sphere_all = dict()
        self.T_sphere_to_base = dict()
        self.T_sphere_to_base_all = dict()
        self.T_sphere_to_joints = dict()
        self.T_joints_to_sphere = dict()
        self.T_world_to_base = dict()
        self.T_sphere_to_world = dict()

        # Obs
        self.fp_pos = dict()
        self.fp_linvel = dict()
        self.fp_angvel = dict()
        self.fp_spherical_coords = dict()
        self.past_fp_pos = dict()
        self.past_fp_linvel = dict()
        self.past_fp_angvel = dict()
        self.past_fp_spherical_coords = dict()

        self.T_perfect_sphere_to_joints = dict()
        self.T_joints_to_perfect_sphere = dict()
        self.tip_vec_in_joints = dict()
        self.T_joints = dict()
        self.T_joints_inv = dict()
        self.fp_point_tensors = dict()
        self.tip_vec = dict()
        self.past_tip_vec = dict()

        self.root_offsets = torch.zeros((self.num_envs, self.max_fingers, 1), dtype=torch.float, device=self.device)
        self.finger_lateral_min = torch.zeros((self.num_envs, self.max_fingers), dtype=torch.float, device=self.device)
        self.finger_lateral_max = torch.zeros((self.num_envs, self.max_fingers), dtype=torch.float, device=self.device)
        self.init_finger_lateral_min = torch.zeros((self.num_envs, self.max_fingers), dtype=torch.float, device=self.device)
        self.init_finger_lateral_max = torch.zeros((self.num_envs, self.max_fingers), dtype=torch.float, device=self.device)
        self.in_hand_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.init_pos_in_sphere = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device) 
        self.object_pos_in_sphere = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device) 
        self.object_rot_in_sphere = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_linvel_in_sphere = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device) 
        self.object_angvel_in_sphere = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device) 
        self.goal_rot_in_sphere = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot_in_sphere[:, 0] = 1.0
        self.T_world_to_obj = torch.zeros((self.num_envs, 4, 4), dtype=torch.float, device=self.device) 
        self.T_sphere_to_obj = torch.zeros((self.num_envs, 4, 4), dtype=torch.float, device=self.device)
        self.past_object_pos_in_sphere = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.past_object_rot_in_sphere = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device) 
        self.past_object_linvel_in_sphere = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.past_object_angvel_in_sphere =torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ref_sphere_xyz = torch.tensor(self.cfg.ref_sphere["xyz"], dtype=torch.float32, device=self.device)
        self.ref_sphere_r = torch.tensor(self.cfg.ref_sphere["r"], dtype=torch.float32, device=self.device)
        self.current_dofs = dict() 
        self.current_vel = dict()

        # Vector information
        self.init_vector_phis = torch.tensor(self.cfg.vector_phis, dtype = torch.float, device =self.device)
        self.init_vector_phis = torch.deg2rad(self.init_vector_phis)
        self.max_vector_offsets = torch.tensor(self.cfg.max_vector_offsets, dtype=torch.float, device=self.device)
        self.min_vector_offsets = torch.tensor(self.cfg.min_vector_offsets, dtype=torch.float, device=self.device)
        self.vector_phis = self.init_vector_phis.repeat(self.num_envs, self.max_fingers, 1)
        self.obs_indices = torch.tensor(self.cfg.obs_idx, dtype = torch.int, device =self.device)

        # Surface points
        self.n_sphere_surface_samples = self.cfg.n_sphere_surface_samples
        self.sphere_fibonacci = torch_sample_fibonacci_points_sphere(self.n_sphere_surface_samples , self.device)
        theta, phi = self.sphere_fibonacci[:, 0], self.sphere_fibonacci[:, 1] # [n_points] 
        x = torch.sin(phi) * torch.cos(theta) # [n_points]
        y = torch.sin(phi) * torch.sin(theta) # [n_points]
        z = torch.cos(phi) # [n_points]
        self.xyz_sphere_points = torch.stack([x, y, z], dim=1) # [n_points, 3]
        self.sphere_points_phi = self.sphere_fibonacci[:, 1].repeat(self.num_envs, 1)
        print(f"Generating Spheres Surface with {self.n_sphere_surface_samples} samples")
        print()

        # Transformations to Sim World
        self.init_T_world_to_sphere = torch.eye(4, dtype=torch.float, device=self.device)
        self.init_T_world_to_sphere[1,1] = -1
        self.init_T_world_to_sphere[2,2] = -1

        # Compute cos and sin
        inclination = torch.deg2rad(torch.tensor(self.cfg.inclination, dtype=torch.float, device=self.device))
        cos = torch.cos(inclination)
        sin = torch.sin(inclination)
        
        # Create the transformation matrices T
        T = torch.eye(4, device=self.device)
        T[0, 0] = cos
        T[0, 2] = sin
        T[2, 0] = -sin
        T[2, 2] = cos
        self.init_T_world_to_sphere = self.init_T_world_to_sphere @ T

        self.T_world_to_sphere = self.init_T_world_to_sphere.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        
        # World to Sphere Transforms
        self.T_sphere_to_world = torch.inverse((self.T_world_to_sphere)) 

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.all_env_indices = torch.arange(self.num_envs, device=self.device)[:, None] 

        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device) 
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.18], device=self.device) # Used for goal markers

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # M_joint random
        self.rand_lateral_anchor_offset = torch.zeros((self.num_envs, self.max_fingers), dtype=torch.float, device=self.device)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.avg_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.running_sum_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.total_resets = torch.tensor(-self.num_envs, dtype=torch.float, device=self.device)
        self.reset_counts = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.all_successes = []
        # New evaluation metric: target pose success rate over first 10 tries per env (skipping the initial reset call)
        self.eval_target_reset_counts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.eval_num_successes = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.eval_num_tries = torch.tensor(0.0, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.PI = torch.tensor(np.pi, dtype=torch.float, device=self.device)

        # Actions
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.last_last_actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)

        # Max joint change per step
        self.max_delta_q = torch.deg2rad(torch.tensor(self.cfg.max_delta_q, dtype = torch.float, device = self.device))
        
        # Merge fingers vars
        self.env_merge_flags = torch.zeros(self.num_envs, dtype = torch.bool, device = self.device) 
        self.merge_indices = torch.tensor([0, 1], dtype = torch.int, device = self.device)
        self.merge_frequency = torch.tensor(self.cfg.merge_frequency, dtype = torch.float, device = self.device)

        for robot in self.cfg.robots:
            # Hand Information
            self.num_hand_dofs[robot] = self.hands[robot].num_joints

            # buffers for position targets
            self.hand_dof_targets[robot] = torch.zeros((self.num_robots[robot], self.num_hand_dofs[robot]), dtype=torch.float, device=self.device)
            self.prev_targets[robot] = torch.zeros((self.num_robots[robot], self.num_hand_dofs[robot]), dtype=torch.float, device=self.device)
            self.cur_targets[robot] = torch.zeros((self.num_robots[robot], self.num_hand_dofs[robot]), dtype=torch.float, device=self.device)
            
            # Sphere Information
            print("--------------------------------------")
            print()
            print(f"Initializing {robot} Hand")
            print()
            kin_chains = self.cfg.cik_infos[robot]["kin_chains"]
            self.finger_chains[robot] = [ # Finger chains 
                chain for chain in kin_chains
                if "palm_normal" not in chain and "sphere_frame" not in chain
            ]
            self.finger_chain_order[robot] = self.cfg.cik_infos[robot]["chain_order"]
            self.chain_indices[robot] = []

            # Finger chain order and indices for controller
            for idx in self.cfg.cik_infos[robot]["chain_indices"]: # Merge actions of missing fingers
                chain_idx = [idx]
                tensor_idx = torch.tensor(chain_idx, dtype=torch.int, device=self.device).squeeze()
                # Keep it 1-D even for single element
                if tensor_idx.ndim == 0:
                    tensor_idx = tensor_idx.unsqueeze(0)
                self.chain_indices[robot].append(tensor_idx)
            
            for i in range(self.cfg.max_fingers): 
                if i not in self.finger_chain_order[robot]: 
                    self.finger_chain_order[robot].append(i)

            print(f"Finger Chains: {self.finger_chains[robot]}")
            print()
            self.n_fingers[robot] = len(self.finger_chains[robot])

            # Initial indices from the original finger chain lists
            finger_unorganize_indices = torch.tensor(self.finger_chain_order[robot], dtype = torch.int, device = self.device) 
            finger_unorganize_indices = finger_unorganize_indices
            finger_organize_indices = torch.argsort(finger_unorganize_indices)

            # Joint list
            self.joints[robot] = []
            for chain in self.finger_chains[robot]:
                for j in chain[1:-1]: 
                    self.joints[robot].append(j)
            self.n_joints[robot] = len(self.joints[robot]) 
            self.joint_index_dict[robot] = {element: index for index, element in enumerate(self.joints[robot])}

            # indices of self.joints in simulated hand
            self.actuated_dof_indices[robot] = list()
            for joint_name in self.joints[robot]:
                self.actuated_dof_indices[robot].append(self.hands[robot].joint_names.index(joint_name))
            # Joint info
            self.joint_info[robot] = self.cfg.cik_infos[robot]["joint_info"]
            self.joint_type_info[robot] = self.cfg.cik_infos[robot]["joint_type_info"]
            
            # Sphere radius
            self.sphere_radius[robot] = torch.tensor(self.joint_info[robot]["sphere_frame"][6], dtype=torch.float, device=self.device)
            
            # Main Finger Joints 
            self.main_joints[robot] = []
            repeat_main_joints = []
            for c, chain in enumerate(self.finger_chains[robot]):
                chain_idx  = self.chain_indices[robot][c]
                m_join_exists = False
                for joint in chain[1:-1]:
                    if self.joint_type_info[robot][joint]["type"]=="A":
                        self.main_joints[robot].append(joint)
                        m_join_exists = True
                        if chain_idx.numel()>1:
                            repeat_main_joints.append(joint)
                        break
                if m_join_exists == False:
                    raise ValueError(f"Main Joint not found in finger {chain}. Fingers without Main Joint are not implemented yet.")
            
            for joint in repeat_main_joints:
                self.main_joints[robot].append(joint)

            self.main_joints[robot] = [self.main_joints[robot][i] for i in finger_organize_indices.tolist()]

            # Get chain joint indices
            self.chain_joint_indices[robot] = []
            for chain in self.finger_chains[robot]:
                joints = chain[1:-1]
                joint_indices = torch.tensor([self.joint_index_dict[robot][j] for j in joints],
                                                    dtype= torch.int, device = self.device)
                self.chain_joint_indices[robot].append(joint_indices)

            self.main_indices[robot] = torch.tensor([self.joint_index_dict[robot][j] for j in self.main_joints[robot]], dtype= torch.int, device = self.device)

            # Assuming self.main_indices is a list or tensor of indices
            all_indices = torch.arange(self.hands[robot].data.joint_pos.shape[1], device=self.hands[robot].data.joint_pos.device)  # All possible indices for the sliced tensor

            # Compute complementary indices
            self.non_main_indices[robot] = all_indices[~torch.isin(all_indices, self.main_indices[robot])]

            # joint limits
            joint_pos_limits = self.hands[robot].root_physx_view.get_dof_limits().to(self.device)
            self.hand_dof_lower_limits[robot] = joint_pos_limits[..., 0]
            self.hand_dof_upper_limits[robot] = joint_pos_limits[..., 1]

            # M joint lookup information
            self.m_joint_res[robot]= torch.tensor([self.joint_type_info[robot][q_A]["resolution"] for q_A in self.main_joints[robot]], dtype= torch.float, device=self.device)
            self.m_joint_zero_idx[robot] = torch.tensor([self.joint_type_info[robot][q_A]["zero_idx"] for q_A in self.main_joints[robot]], dtype= torch.int, device=self.device)
            self.m_joint_max[robot] = torch.tensor([len(self.joint_type_info[robot][q_A]["q_list"]) - 1 for q_A in self.main_joints[robot]], dtype= torch.int, device=self.device)
            self.m_joint_anchor_dist[robot] = torch.tensor([self.joint_type_info[robot][q_A]["anchor_dist"] for q_A in self.main_joints[robot]], dtype= torch.float, device=self.device)
            
            # M joint no lookup info
            self.m_ft_sphere_h[robot] = dict()
            self.m_ft_joint_h[robot] = dict()
            one = torch.tensor([1], dtype= torch.float, device=self.device)
            for joint in self.main_joints[robot]:
                self.m_ft_sphere_h[robot][joint] = torch.cat((torch.tensor(self.joint_type_info[robot][joint]["ft_sphere"], dtype= torch.float, device=self.device), one))
                self.m_ft_joint_h[robot][joint] = torch.cat((torch.tensor(self.joint_type_info[robot][joint]["ft_joint"], dtype= torch.float, device=self.device) , one))

            # Load and Preprocess Sphere dictionary
            sphere_control = self.cfg.cik_infos[robot]["sphere_control"]
            self.joint_ls[robot] = dict()
            self.next_joint_dict[robot] = dict() 
            self.joint_phis[robot] = dict()
            self.m_joint_q_lists[robot] = dict()
            self.boxes_dict[robot] = dict()
            self.joint_limits[robot] = dict()
            
            for chain in self.finger_chains[robot]:
                for joint in chain[1:-1]:
                    self.joint_ls[robot][joint] = torch.norm(torch.tensor(self.joint_type_info[robot][joint]["nj"][:2], dtype = torch.float, device =self.device))
                    self.boxes_dict[robot][joint] = (
                        torch.tensor(self.joint_type_info[robot][joint]["box_min"], dtype = torch.float, device =self.device),
                        torch.tensor(self.joint_type_info[robot][joint]["box_max"], dtype = torch.float, device =self.device)
                        )
                    self.joint_limits[robot][joint] = (
                        torch.tensor(self.joint_info[robot][joint][4], dtype = torch.float, device =self.device),
                        torch.tensor(self.joint_info[robot][joint][5], dtype = torch.float, device =self.device)
                        )
                    if joint in self.main_joints[robot]:
                        self.next_joint_dict[robot].update(sphere_control[joint]["next_joint"])
                        self.m_joint_q_lists[robot][joint] = torch.tensor(self.joint_type_info[robot][joint]["q_list"], dtype= torch.float, device=self.device)
                        for sj in sphere_control[joint]["joint_phis"].keys():
                            self.joint_phis[robot][sj] = torch.tensor(sphere_control[joint]["joint_phis"][sj], dtype= torch.float, device=self.device)

            finger_lateral_min = self.cfg.cik_infos[robot]["min_offsets"]
            finger_lateral_max = self.cfg.cik_infos[robot]["max_offsets"]
            self.finger_lateral_min[self.robot_indices[robot]] = torch.tensor(finger_lateral_min, dtype=torch.float, device=self.device).repeat(self.num_robots[robot], 1)
            self.finger_lateral_max[self.robot_indices[robot]] = torch.tensor(finger_lateral_max, dtype=torch.float, device=self.device).repeat(self.num_robots[robot], 1)
            self.init_finger_lateral_min[self.robot_indices[robot]] = self.finger_lateral_min[self.robot_indices[robot]].clone()
            self.init_finger_lateral_max[self.robot_indices[robot]] = self.finger_lateral_max[self.robot_indices[robot]].clone()

            # Reference Hand Poses
            q_open_dict = self.cfg.cik_infos[robot]["q_open_palm"]
            q_sphere_dict = self.cfg.cik_infos[robot]["q_sphere"]

            # Get init q_0 for all spheres
            self.q_0[robot] = torch.tensor([q_open_dict[j] for j in self.joints[robot]], dtype = torch.float, device =self.device)
            self.q_sphere[robot] = torch.tensor([q_sphere_dict[j] for j in self.joints[robot]], dtype = torch.float, device =self.device)
            self.q[robot] = self.q_sphere[robot].repeat(self.num_robots[robot], 1)
            self.past_q[robot] = self.q_sphere[robot].repeat(self.num_robots[robot], 1)
            
            # Sphere Ts
            quat = torch.tensor(self.joint_info[robot]["sphere_frame"][1], dtype=torch.float, device = self.device)[[3, 0, 1, 2]]
            T_quat = torch.eye(4, dtype = torch.float, device = self.device)            
            T_quat[:3,:3] = matrix_from_quat(quat)
            self.T_base_to_sphere[robot] = torch.tensor(T_quat, dtype=torch.float, device=self.device)
            self.T_base_to_sphere[robot][:3, 3] = torch.tensor(self.joint_info[robot]["sphere_frame"][0], device=self.device)
            self.T_base_to_sphere_all[robot] = self.T_base_to_sphere[robot].repeat(self.num_robots[robot], 1, 1)
            self.T_sphere_to_base[robot] = torch.inverse(self.T_base_to_sphere[robot])
            self.T_sphere_to_base_all[robot] = self.T_sphere_to_base[robot].repeat(self.num_robots[robot], 1, 1)
            self.T_sphere_to_joints[robot] = dict() 
            self.T_joints_to_sphere[robot] = dict() 

            # Obs
            self.fp_pos[robot] = torch.zeros((self.num_robots[robot], self.max_fingers, self.n_fp_samples, 3), dtype = torch.float, device= self.device)
            self.fp_linvel[robot] = torch.zeros((self.num_robots[robot],self.max_fingers, self.n_fp_samples, 3), dtype=torch.float, device = self.device ) 
            self.fp_angvel[robot] = torch.zeros((self.num_robots[robot], self.max_fingers, self.n_fp_samples,3), dtype=torch.float, device = self.device ) 
            self.fp_spherical_coords[robot] =  torch.zeros((self.num_robots[robot], self.max_fingers, self.n_fp_samples, 3), dtype=torch.float, device = self.device ) 
            self.past_fp_pos[robot] = torch.zeros((self.num_robots[robot], self.max_fingers, self.n_fp_samples, 3), dtype = torch.float, device= self.device)
            self.past_fp_linvel[robot] = torch.zeros((self.num_robots[robot],self.max_fingers, self.n_fp_samples, 3), dtype=torch.float, device = self.device ) 
            self.past_fp_angvel[robot] = torch.zeros((self.num_robots[robot], self.max_fingers, self.n_fp_samples,3), dtype=torch.float, device = self.device ) 
            self.past_fp_spherical_coords[robot] =  torch.zeros((self.num_robots[robot], self.max_fingers, self.n_fp_samples, 3), dtype=torch.float, device = self.device ) 
            
            # Joint Ts 
            self.T_perfect_sphere_to_joints[robot] = dict()
            self.T_joints_to_perfect_sphere[robot] = dict()
            self.T_joints[robot] = dict()
            self.T_joints_inv[robot] = dict()

            fp_point_sample = self.cfg.cik_infos[robot][self.cfg.fp_sample] # Finger print point samples 
            self.fp_point_tensors[robot] = {}
            for c, chain in enumerate(self.finger_chains[robot]):
                for joint in chain[1:]:
                    points = fp_point_sample.get(joint, [])
                    if len(points) == 0:
                        continue
                    self.fp_point_tensors[robot][self.joint_index_dict[robot][joint]] = torch.tensor(points, dtype=torch.float, device=self.device)  # [num_samples, 3]

            for c, chain in enumerate(self.finger_chains[robot]):
                points_pos = []
                T_cumulative = self.T_sphere_to_base[robot].clone() # [4, 4]
                for joint in chain[1:-1]:
                    # T_previous joint to next
                    xyz = torch.tensor(self.joint_info[robot][joint][0],dtype=torch.float, device=self.device)
                    quat = torch.tensor(self.joint_info[robot][joint][1], dtype=torch.float, device = self.device)[[3, 0, 1, 2]]
                    T_quat = torch.eye(4, dtype = torch.float, device = self.device)            
                    T_quat[:3,:3] = matrix_from_quat(quat)
                    T_joint = T_quat
                    T_joint[:3, 3] = xyz
                    self.T_joints[robot][joint] = T_joint.clone()
                    self.T_joints_inv[robot][joint] = torch.inverse(T_joint)
                    T_cumulative = T_cumulative @ T_joint

                    # Place joint in perfect sphere Positions
                    q_values = self.q_sphere[robot][self.joint_index_dict[robot][joint]] 
                    cos_q = torch.cos(q_values)
                    sin_q = torch.sin(q_values)
                    R_z = torch.tensor([
                        [cos_q, -sin_q, torch.zeros_like(cos_q)],
                        [sin_q, cos_q, torch.zeros_like(cos_q)],
                        [torch.zeros_like(cos_q), torch.zeros_like(cos_q), torch.ones_like(cos_q)]
                    ], dtype= torch.float, device = self.device)

                    # Update
                    T_rot = torch.eye(4, device=self.device)
                    T_rot[:3, :3] = R_z
                    T_cumulative = T_cumulative @ T_rot # [4, 4]

                    # Save Transforms
                    self.T_perfect_sphere_to_joints[robot][joint] = T_cumulative.clone() # [4, 4]
                    self.T_joints_to_perfect_sphere[robot][joint] = torch.inverse(T_cumulative)

                    og_points = self.fp_point_tensors[robot].get(self.joint_index_dict[robot][joint], torch.empty((0, 3), dtype=torch.float, device=self.device))
                    if og_points.numel() != 0:
                        # Transform points to sphere frame
                        ones = torch.ones_like(og_points[..., :1])  # [num_samples, 1]
                        points_h = torch.cat((og_points, ones), dim=-1)  # [num_samples, 4]
                        points = (T_cumulative @ points_h.T).T[:, :3]  # [num_samples, 3]
                        points_pos.append(points)

                xyz = torch.tensor(self.joint_info[robot][chain[-1]][0],dtype=torch.float, device=self.device)
                quat = torch.tensor(self.joint_info[robot][chain[-1]][1], dtype=torch.float, device = self.device)[[3, 0, 1, 2]]
                T_quat = torch.eye(4, dtype = torch.float, device = self.device)            
                T_quat[:3,:3] = matrix_from_quat(quat)
                T_joint = T_quat
                T_joint[:3, 3] = xyz
                self.T_joints[robot][chain[-1]] = T_joint.clone()

            self.tip_vec[robot] = torch.zeros((self.num_robots[robot], self.max_fingers, 3), dtype=torch.float, device=self.device)
            self.past_tip_vec[robot] = torch.zeros((self.num_robots[robot], self.max_fingers, 3), dtype=torch.float, device=self.device)

            # Calculate q_ref
            self.q_ref[robot] = self.q_sphere[robot].clone()
            radius_mask = self.ref_sphere_r>=0
            for c, chain in enumerate(self.finger_chains[robot]): 
                chain_idxs = self.chain_indices[robot][c]
                chain_i = chain_idxs[0]
                points_scaled = self.ref_sphere_xyz * self.ref_sphere_r[chain_i].unsqueeze(-1) * self.sphere_radius[robot]     # [n_points, 3]
                ones = torch.ones((points_scaled.shape[0], 1), device=self.device)
                torch_points_h = torch.cat([points_scaled, ones], dim=1)     # [n_points, 4]
                torch_points_h = torch_points_h.permute(1, 0) # Adjust indexing and permute to [4, n_points]
                radii_mask = radius_mask[chain_i]  # [n_points]

                T_cumulative = self.T_sphere_to_base[robot]  # [4, 4]
                for joint in chain[1:-1]: 
                    T_joint = self.T_joints[robot][joint]
                    q_values = self.q_ref[robot][self.joint_index_dict[robot][joint]]
                    cos = torch.cos(q_values)
                    sin = torch.sin(q_values)

                    T_rot = torch.zeros((4, 4), dtype=q_values.dtype, device=self.device)
                    T_rot[0, 0] = cos
                    T_rot[0, 1] = -sin
                    T_rot[1, 0] = sin
                    T_rot[1, 1] = cos
                    T_rot[2, 2] = 1.0
                    T_rot[3, 3] = 1.0

                    T_joint_to_sphere = torch.inverse(T_cumulative @ T_joint @ T_rot)

                    torch_transformed_points_h = (T_joint_to_sphere @ torch_points_h).permute(1,0) # [n_points, 4] # unsqueeze [1]
                    joint_points = torch_transformed_points_h[:, :3] # [n_points, 3] 

                    if (self.joint_type_info[robot][joint]["type"] == "A"):
                        T_cumulative = T_cumulative @ T_joint @ T_rot
                        continue
                    elif self.joint_type_info[robot][joint]["type"] == "C":
                        raise ValueError("C Joints not implemented")
                    elif (self.joint_type_info[robot][joint]["type"] == "B") | (self.joint_type_info[robot][joint]["type"] == "D"):
                        l = self.joint_ls[robot][joint]

                    # Range Filter
                    points_norm = torch.norm(joint_points[:, :2], dim=1)
                    range_mask = (points_norm >= l).T
                    point_mask = (range_mask & radii_mask).T # [n_points]
                    
                    new_q = torch_solve_for_B_joint(
                        joint, 
                        self.q_ref[robot].unsqueeze(0),
                        self.joint_index_dict[robot],
                        joint_points.unsqueeze(1),
                        self.joint_limits[robot][joint], 
                        point_mask.unsqueeze(1), 
                        self.joint_type_info[robot]
                    )
                    
                    if ("og_type" in self.joint_type_info[robot][joint].keys()):
                            if self.joint_type_info[robot][joint]["og_type"] == "A":
                                new_q = self.q_ref[robot][self.joint_index_dict[robot][joint]] 
                    self.q_ref[robot][self.joint_index_dict[robot][joint]] = new_q.squeeze()

                    # Get current joint position
                    q_values = self.q_ref[robot][self.joint_index_dict[robot][joint]]
                    cos = torch.cos(q_values)
                    sin = torch.sin(q_values)
                    T_rot = torch.zeros((4, 4), dtype=q_values.dtype, device=self.device)
                    T_rot[0, 0] = cos
                    T_rot[0, 1] = -sin
                    T_rot[1, 0] = sin
                    T_rot[1, 1] = cos
                    T_rot[2, 2] = 1.0
                    T_rot[3, 3] = 1.0
                    
                    T_cumulative = T_cumulative @ T_joint @ T_rot
            self.T_world_to_base[robot] = self.T_world_to_sphere[self.robot_indices[robot]] @ self.T_sphere_to_base[robot] 
            if self.n_fingers[robot] < 5:
                self.merge_fingers = True
        self.T_rot_template = torch.eye(4, dtype = torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        # Computing Inhand pos and initial object pos is wrt sphere frame
        print("--------------------------------------")
        print()
        print("Setting Up Hand and Object Init Positions")
        print()
        sphere_in_hand_pos = torch.tensor(self.cfg.inhand_position, dtype=torch.float, device=self.device)
        sphere_init_pos = torch.tensor(self.cfg.init_pos, dtype=torch.float, device=self.device)
        self.sphere_in_hand_pos_h = torch.cat([sphere_in_hand_pos, torch.tensor([1.0], dtype=torch.float, device=self.device)], dim = 0)
        self.sphere_init_pos_h = torch.cat([sphere_init_pos, torch.tensor([1.0], dtype=torch.float, device=self.device)], dim = 0)
        print("Sphere Object inhand Position: ", sphere_in_hand_pos)
        print("Sphere Object init Position: ", sphere_init_pos)
        print()

        print("--------------------------------------")
        print(" Initialization Complete")
        print("--------------------------------------")
        print()

    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=[0,0,-0.6])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.hands = dict()
        self.objects = dict() 
        self.custom_event_managers = dict()

        # Calculate robot numbers and indices
        self.num_robots = dict()
        self.robot_indices = dict()

        # Calculate number of robot types
        num_robot_types = len(self.cfg.robots)

        # Base number of environments per robot type
        base = self.num_envs // num_robot_types

        # Remainder to distribute
        remainder = self.num_envs % num_robot_types

        # Populate self.num_robots
        self.num_robots = self.cfg.num_robots
        start = 0
        
        for i, robot in enumerate(self.cfg.robots):
            num = base + 1 if i < remainder else base
            self.num_robots[robot] = num

            # Populate self.robot_indices with tensor of env indices for this robot
            self.robot_indices[robot] = torch.arange(start, start + num, device=self.device)
            local_env_ids = self.robot_indices[robot] - self.robot_indices[robot][0]
            start += num

            # add hand, in-hand object, and goal object
            self.hands[robot] = Articulation(self.cfg.robot_cfgs[robot])
            self.objects[robot] = RigidObject(self.cfg.object_cfgs[robot])
            self.scene.articulations[robot] = self.hands[robot]
            self.scene.rigid_objects[f"{robot}_object"] = self.objects[robot]

            self.custom_event_managers[robot] = EventManager(self.cfg.custom_events[robot], self)

            # add articulation to scene - we must register to scene to randomize with EventManager
            if "prestartup" in self.custom_event_managers[robot].available_modes:
                self.custom_event_managers[robot].apply(mode="prestartup", env_ids = local_env_ids)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_last_actions = self.last_actions.clone()
        self.last_actions = self.actions.clone()
        self.actions = actions.clone()
        self.actions = torch.clamp( 
            self.actions,
            -torch.ones_like(self.actions),
            torch.ones_like(self.actions)
        )
        
        self.processed_actions = actions.clone()

        if (self.cfg.action_noise > 0.0) & self.training:
            noise = get_noise(self.processed_actions, self.cfg.action_noise)
            self.processed_actions += noise

        if self.cfg.debug_fingers:
            env_times = (self.episode_length_buf * self.step_dt) * self.cfg.debug_speed_factor
            perfect_sphere_action = unscale(
                torch.tensor(1.0, dtype=torch.float, device = self.device), 
                self.min_vector_offsets[0],
                self.max_vector_offsets[0]
            )
            processed_actions = get_debug_finger_actions(
                env_times[0], 
                self.processed_actions, 
                self.max_fingers, 
                self.vector_phis.size(2), 
                perfect_sphere_action, 
                self.device
            )
            self.processed_actions = processed_actions.clone()
            

        # Get Lateral joints
        main_joint_offsets = self.processed_actions[:, :self.max_fingers]

        # merge fingers ids
        merge_env_ids = torch.nonzero(self.env_merge_flags).squeeze(-1)
        
        # Lateral Movements Map piecewise
        main_joint_offsets = torch.where(
            main_joint_offsets < 0,
            main_joint_offsets * -self.finger_lateral_min, 
            main_joint_offsets * self.finger_lateral_max
        )

        if self.cfg.randomize_theta_anchors & self.training:
            main_joint_offsets += self.rand_lateral_anchor_offset

        # Merged joint preprocess
        if self.merge_fingers:
            if self.merge_indices.numel()>1: 
                if merge_env_ids.numel() > 0:
                    mean_offsets = main_joint_offsets[merge_env_ids[:, None], self.merge_indices].mean(dim=-1, keepdim=True)
                    main_joint_offsets[merge_env_ids[:, None], self.merge_indices] = mean_offsets.expand(-1, self.merge_indices.numel())

        main_joint_offsets = torch.clamp( 
            main_joint_offsets,
            self.finger_lateral_min,
            self.finger_lateral_max
        )

        # Vector offset actions
        torch_vector_offsets = self.processed_actions[:, self.max_fingers:].view(self.num_envs, self.max_fingers,  self.vector_phis.size(2))

        # Merged Fingers 
        if self.merge_fingers:
            if self.merge_indices.numel()>1: 
                if merge_env_ids.numel() > 0:
                    mean_offsets = torch_vector_offsets[merge_env_ids][:, self.merge_indices].mean(dim=-2, keepdim=True)
                    torch_vector_offsets[merge_env_ids[:, None], self.merge_indices, :] = mean_offsets.expand(-1, self.merge_indices.numel(), torch_vector_offsets.size(2))

        torch_vector_offsets = scale(
            torch_vector_offsets, 
            self.min_vector_offsets[None,:],
            self.max_vector_offsets[None,:]
        )
        
        torch_vector_offsets = torch.clamp(
            torch_vector_offsets, 
            self.min_vector_offsets[None,:],
            self.max_vector_offsets[None,:]
        )

        # Compute deformed sphere points with offsets
        r = torch_compute_deformed_sphere_points_detached_fingers_dynamic_anchors(
            self.vector_phis[:, :self.max_fingers, :], 
            torch_vector_offsets[:, :self.max_fingers, :], 
            self.sphere_fibonacci, 
            torch.tensor(1.0, dtype=torch.float, device=self.device),
            self.device
        )
        all_points = r.unsqueeze(-1) * self.xyz_sphere_points
        ones = torch.ones((all_points.size(0), all_points.size(1),all_points.size(2), 1), device=self.device) 
        all_points_h = torch.cat([all_points, ones], dim=3) # [num_envs, num_fingers, n_points, 4] 
        radius_mask = r >= 0 # [n_envs, num_fingers, n_points]

        for robot in self.cfg.robots:
            sub_main_joint_offsets = main_joint_offsets[self.robot_indices[robot]]

            # debug object in hand
            if self.cfg.debug_object_inhand:
                object_default_state = torch.zeros_like(self.objects[robot].data.default_root_state)
                object_default_state[:, 0:3] = (
                    self.in_hand_pos[self.robot_indices[robot]] + self.scene.env_origins[self.robot_indices[robot]]
                )

                object_default_state[:, 3] = 1.0
                self.objects[robot].write_root_pose_to_sim(object_default_state[:, :7])
                self.objects[robot].write_root_velocity_to_sim(object_default_state[:, 7:])

            if (self.cfg.max_delta_q > 0.0) & (self.cfg.max_delta_q < 360.0):
                self.past_q[robot] = self.q[robot].clone()
            self.q[robot] = self.q_sphere[robot].clone().repeat(self.num_robots[robot], 1)
            all_points_h[self.robot_indices[robot], :,:, :3] = all_points_h[self.robot_indices[robot], :,:,:3] * self.sphere_radius[robot]

            main_joint_targets = torch_solve_for_A_joints_ohne_interpolation( 
                self.main_joints[robot], 
                sub_main_joint_offsets,
                self.m_joint_res[robot],
                self.m_joint_zero_idx[robot],
                self.m_joint_max[robot],
                self.m_joint_anchor_dist[robot],
                self.m_joint_q_lists[robot]
            )
            self.q[robot][:, self.main_indices[robot]] = main_joint_targets

            # Solve IK for Hand in Sim
            T_rot = self.T_rot_template[self.robot_indices[robot]]
            for c, chain in enumerate(self.finger_chains[robot]): 
                chain_idxs = self.chain_indices[robot][c]
                chain_i = torch.full((self.num_robots[robot],), chain_idxs[0], dtype = torch.int, device=self.device)
                torch_points_h = all_points_h[self.robot_indices[robot], chain_i, :, :].squeeze(1)
                torch_points_h = torch_points_h.permute(0, 2, 1) 
                radii_mask = radius_mask[self.robot_indices[robot], chain_i, :].squeeze(1)  
                T_cumulative = self.T_sphere_to_base_all[robot] # [num_envs, 4, 4]  
                T_cumulative_inv = self.T_base_to_sphere_all[robot]  # [num_envs, 4, 4]  

                for joint in chain[1:-1]: 
                    T_joint = self.T_joints[robot][joint]
                    T_joint_inv = self.T_joints_inv[robot][joint]
                    q_values = self.q[robot][:, self.joint_index_dict[robot][joint]]
                    cos = torch.cos(q_values)
                    sin = torch.sin(q_values)

                    T_rot[:, 0, 0] = cos
                    T_rot[:, 0, 1] = -sin
                    T_rot[:, 1, 0] = sin
                    T_rot[:, 1, 1] = cos

                    T_joint_to_sphere = T_rot.permute(0,2,1) @ T_joint_inv @ T_cumulative_inv

                    torch_transformed_points_h = (T_joint_to_sphere @ torch_points_h).permute(2,0,1) # [n_points, num_envs, 4]
                    joint_points = torch_transformed_points_h[:, :, :3]

                    if (self.joint_type_info[robot][joint]["type"] == "A"):
                        # Get current joint position
                        q_values = self.q_sphere[robot][self.joint_index_dict[robot][joint]].clone().repeat(self.num_robots[robot])
                        cos = torch.cos(q_values)
                        sin = torch.sin(q_values)
                        T_rot[:, 0, 0] = cos
                        T_rot[:, 0, 1] = -sin
                        T_rot[:, 1, 0] = sin
                        T_rot[:, 1, 1] = cos
                        
                        T_cumulative = T_cumulative @ T_joint @ T_rot
                        T_cumulative_inv = T_rot.permute(0,2,1) @ T_joint_inv @ T_cumulative_inv
                        continue

                    elif self.joint_type_info[robot][joint]["type"] == "C":
                        raise ValueError("C Joints not implemented")
                    elif (self.joint_type_info[robot][joint]["type"] == "B") | (self.joint_type_info[robot][joint]["type"] == "D"):
                        l = self.joint_ls[robot][joint]

                    # Range Filter
                    points_norm = torch.norm(joint_points[:,:, :2], dim=2)
                    range_mask = (points_norm >= l).T
                    
                    # finger_width mask 
                    z_values = joint_points[:, :, 2] 
                    z_mask = ((torch.abs(z_values) <= 0.1*self.sphere_radius[robot])).T
                    point_mask = (range_mask & radii_mask & z_mask).T # [n_points, n_envs]

                    new_q = torch_solve_for_B_joint(
                        joint, 
                        self.q[robot],
                        self.joint_index_dict[robot],
                        joint_points,
                        self.joint_limits[robot][joint], 
                        point_mask, 
                        self.joint_type_info[robot]
                    )
                    
                    if ("og_type" in self.joint_type_info[robot][joint].keys()):
                            if self.joint_type_info[robot][joint]["og_type"] == "A":
                                new_q = self.q[robot][:, self.joint_index_dict[robot][joint]] 
                    self.q[robot][:, self.joint_index_dict[robot][joint]] = new_q

                    # Get current joint position
                    q_values = self.q[robot][:, self.joint_index_dict[robot][joint]]
                    cos = torch.cos(q_values)
                    sin = torch.sin(q_values)
                    T_rot[:, 0, 0] = cos
                    T_rot[:, 0, 1] = -sin
                    T_rot[:, 1, 0] = sin
                    T_rot[:, 1, 1] = cos
                    T_cumulative = T_cumulative @ T_joint @ T_rot
                    T_cumulative_inv = T_rot.permute(0,2,1) @ T_joint_inv @ T_cumulative_inv

            # Max q change
            if (self.cfg.max_delta_q > 0.0) & (self.cfg.max_delta_q < 360.0):
                diff = self.q[robot] - self.past_q[robot]
                clamped_diff = torch.clamp(diff, -self.max_delta_q, self.max_delta_q)
                self.q[robot] = self.past_q[robot] + clamped_diff

    def _apply_action(self) -> None: # Only function to go with sim_dt

        for robot in self.cfg.robots:
            # Send joint values to robot
            self.cur_targets[robot][:, self.actuated_dof_indices[robot]] = self.q[robot].clone()

            # Debug
            if self.cfg.debug_object_inhand:
                self.cur_targets[robot][:, self.actuated_dof_indices[robot]] = self.q_0[robot].repeat(self.num_robots[robot], 1).clone()

            # Acting moving average
            self.cur_targets[robot] = ( 
                self.cfg.act_moving_average * self.cur_targets[robot]
                + (1.0 - self.cfg.act_moving_average) * self.prev_targets[robot]
            )

            self.cur_targets[robot]= saturate(
                self.cur_targets[robot],
                self.hand_dof_lower_limits[robot],
                self.hand_dof_upper_limits[robot],
            )
            self.prev_targets[robot]= self.cur_targets[robot]

            self.hands[robot].set_joint_position_target( 
                self.cur_targets[robot][:, self.actuated_dof_indices[robot]], joint_ids=self.actuated_dof_indices[robot]
            )

    def _get_observations(self) -> dict:
        return self.compute_full_observations()
    
    def reset(self, seed=None, options=None):
        # Resets the environment, computes initial observations, and applies reset-mode events.
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _get_rewards(self) -> torch.Tensor:
        lateral_position_diffs = []
        hand_dof_vels = []
        hand_dof_effs = []
        radii_position_diffs = []

        for robot in self.cfg.robots:
            lateral_position_diffs.append(self.hands[robot].data.joint_pos[:, self.actuated_dof_indices[robot]][:, self.main_indices[robot]] - self.q_ref[robot][self.main_indices[robot]][None, :])
            hand_dof_vels.append(self.hands[robot].data.joint_vel[:, self.actuated_dof_indices[robot]].clone())
            hand_dof_effs.append(self.hands[robot].data.applied_torque[:, self.actuated_dof_indices[robot]].clone())
            radii_position_diffs.append(self.hands[robot].data.joint_pos[:, self.actuated_dof_indices[robot]][:, self.non_main_indices[robot]] - self.q_ref[robot][self.non_main_indices[robot]][None, :])
        actions = self.actions
        
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:], 
            lateral_position_penalty, radii_position_penalty, dist_rew, rot_rew, velocity_penalty, energy_penalty, action_rate_penalty
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            actions,
            self.cfg.lateral_position_penalty_scale,
            lateral_position_diffs,
            self.cfg.radii_position_penalty_scale,
            radii_position_diffs,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
            hand_dof_vels, 
            self.cfg.velocity_scale,
            self.cfg.max_velocity,
            self.cfg.velocity_tolerance,
            self.cfg.energy_scale,
            hand_dof_effs, 
            self.cfg.action_rate,
            self.last_actions,
            self.last_last_actions
        )
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        if abs(self.cfg.lateral_position_penalty_scale) > 0.0:
            self.extras["log"]["lateral_position_penalty"] = lateral_position_penalty.mean() * self.cfg.lateral_position_penalty_scale
        if abs(self.cfg.radii_position_penalty_scale) > 0.0:
            self.extras["log"]["radii_position_penalty"] = radii_position_penalty.mean() * self.cfg.radii_position_penalty_scale
        if abs(self.cfg.dist_reward_scale) > 0.0:
            self.extras["log"]["dist_rew"] = dist_rew.mean() * self.cfg.dist_reward_scale
        if abs(self.cfg.rot_reward_scale) > 0.0:
            self.extras["log"]["rot_rew"] = rot_rew.mean() * self.cfg.rot_reward_scale
        if abs(self.cfg.velocity_scale) > 0.0:
            self.extras["log"]["velocity_penalty"] = velocity_penalty.mean() * self.cfg.velocity_scale
        if abs(self.cfg.energy_scale) > 0.0:
            self.extras["log"]["energy_penalty"] = energy_penalty.mean() * self.cfg.energy_scale
        if abs(self.cfg.action_rate) > 0.0:
            self.extras["log"]["action_rate_penalty"] = action_rate_penalty.mean() * self.cfg.action_rate
        
        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        
        if self.cfg.debug_fingers:
            out_of_reach = torch.zeros_like(goal_dist, dtype=torch.bool, device=goal_dist.device)
        else:
            out_of_reach = (goal_dist >= self.cfg.fall_dist)
            
        if self.cfg.reset_timer:
            if self.cfg.max_consecutive_success > 0:
                # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
                rot_dist = rotation_distance(self.object_rot, self.goal_rot)
                self.episode_length_buf = torch.where(
                    torch.abs(rot_dist) <= self.cfg.success_tolerance,
                    torch.zeros_like(self.episode_length_buf),
                    self.episode_length_buf,
                )
                max_success_reached = self.successes >= self.cfg.max_consecutive_success
        else:
            if self.cfg.max_consecutive_success > 0:
                max_success_reached = self.successes >= self.cfg.max_consecutive_success

        if self.cfg.debug_fingers:
            time_out = torch.zeros_like(self.episode_length_buf, dtype=torch.bool, device=goal_dist.device)
        else:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            if self.cfg.max_consecutive_success > 0:
                time_out = time_out | max_success_reached
        
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids) 
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)
        
        current_iter = self.learning_iter
        stage_2_start = self.n_iter_per_stage    

        if current_iter < stage_2_start:
            stage_2_scale = 0.0
        else:
            # Linear ramp from 0.0 → 1.0 over the whole Stage 3
            stage_2_scale = min(1.0, max(0.0, (current_iter - stage_2_start) / self.n_iter_per_stage))

        if self.randomize_vectors_phi & self.cfg.training:
            self.vector_phis[env_ids, :, :] = self.init_vector_phis.repeat(len(env_ids), self.max_fingers, 1)
            offsets = (torch.rand(self.vector_phis[env_ids, :, :].size(), dtype=torch.float, device=self.device) - 0.5 )*self.cfg.vector_range * stage_2_scale
            
            merge_env_ids = torch.nonzero(self.env_merge_flags[env_ids]).squeeze(-1)
            if merge_env_ids.numel() > 0:
                mean_offsets = offsets[merge_env_ids][:, self.merge_indices].mean(dim=-2, keepdim=True)
                offsets[merge_env_ids[:, None], self.merge_indices, :] = mean_offsets.expand(-1, self.merge_indices.numel(), offsets.size(2))
            self.vector_phis[env_ids, :, :] += offsets

            self.vector_phis[env_ids, :, :] = torch.clamp(
                self.vector_phis[env_ids, :, :],
                0,
                torch.pi
            )
        
        if self.cfg.randomize_gravity & self.training:
            # World to Sphere Transforms
            self.T_world_to_sphere[env_ids] = add_random_T(
                len(env_ids),
                self.cfg.gravity_angle_range * stage_2_scale,
                self.init_T_world_to_sphere,
                self.device
            )
            self.T_sphere_to_world[env_ids] = torch.inverse((self.T_world_to_sphere[env_ids])) 


        for robot in self.cfg.robots:
            # print("robot", robot)
            robot_indices = self.robot_indices[robot]
            # Get global env_ids to reset for this robot
            mask = torch.isin(env_ids, robot_indices)
            sub_env_ids = env_ids[mask]
            if len(sub_env_ids) == 0:
                continue
            # Compute local indices (0-based for this robot's assets)
            local_env_ids = sub_env_ids - robot_indices[0]

            # Resets articulation and rigid body attributes
            self.hands[robot].reset(local_env_ids)
            self.objects[robot].reset(local_env_ids)

            # apply events such as randomization for environments that need a reset
            if self.custom_event_managers[robot]:
                if "reset" in self.custom_event_managers[robot].available_modes:
                    env_step_count = self._sim_step_counter
                    self.custom_event_managers[robot].apply(mode="reset", env_ids = local_env_ids, global_env_step_count=env_step_count)

            # Reset Hand position
            self.T_world_to_base[robot][local_env_ids] = self.T_world_to_sphere[sub_env_ids] @ self.T_sphere_to_base[robot]
            hand_init_pos = self.T_world_to_base[robot][local_env_ids, :3, 3]
            hand_init_quat = quat_from_matrix(self.T_world_to_base[robot][local_env_ids, :3, :3])
            hand_state = torch.zeros_like(self.hands[robot].data.default_root_state[local_env_ids], device = self.device)

            hand_state[:, 0:3] = (
                hand_init_pos + self.scene.env_origins[sub_env_ids]
            )
            hand_state[:, 3:7] = hand_init_quat
            self.hands[robot].write_root_state_to_sim(hand_state, env_ids = local_env_ids)

            # Update inhand_pos 
            robot_inhand_pos_h = self.sphere_in_hand_pos_h.clone()
            robot_inhand_pos_h[:3] = robot_inhand_pos_h[:3] * self.sphere_radius[robot]
            robot_init_pos_h = self.sphere_init_pos_h.clone()
            robot_init_pos_h[:3] = robot_init_pos_h[:3] * self.sphere_radius[robot]
            self.in_hand_pos[sub_env_ids] = (self.T_world_to_sphere[sub_env_ids] @ robot_inhand_pos_h)[:, :3]
            self.init_pos_in_sphere[sub_env_ids] = robot_init_pos_h[:3]

            # Reset object
            object_default_state = torch.zeros_like(self.objects[robot].data.default_root_state[local_env_ids], device = self.device)
            pos_noise = sample_uniform(-1.0, 1.0, (len(sub_env_ids), 3), device=self.device)
            pos_noise[:, 2] = 0.0

            if self.cfg.debug_fingers:
                b = torch.tensor([0.0, 0.0, -0.3 ], dtype=torch.float, device=self.device)
                object_default_state[:, 0:3] = (
                    b[None,:] + self.scene.env_origins[sub_env_ids]
                )

                rot_noise = sample_uniform(0.0, 0.0, (len(sub_env_ids), 2), device=self.device)  # noise for X and Y rotation
                object_default_state[:, 3:7] = randomize_rotation(
                    rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[sub_env_ids], self.y_unit_tensor[sub_env_ids]
                )
            else:
                num_resets = len(sub_env_ids)
                pos_sphere = (
                    self.init_pos_in_sphere[sub_env_ids]
                    + self.cfg.reset_position_noise * pos_noise * self.sphere_radius[robot]
                )

                # Random 90-degree steps for roll & pitch → brings ANY face to the top
                angles = torch.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2], device=self.device)
                roll  = angles[torch.randint(0, 4, (num_resets,), device=self.device)]
                pitch = angles[torch.randint(0, 4, (num_resets,), device=self.device)]

                # Full random yaw around the vertical (sphere Z-axis)
                yaw = torch.rand(num_resets, device=self.device) * 2 * np.pi

                # 4. Face-selection rotation only (roll + pitch, yaw = 0)
                euler_face = torch.stack([roll, pitch, torch.zeros_like(yaw)], dim=-1)          # (N, 3)
                R_face = matrix_from_euler(euler_face, convention="XYZ")

                # 5. Pure rotation around sphere Z
                euler_yaw = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1)
                R_yaw = matrix_from_euler(euler_yaw, convention="XYZ")

                # 6. Compose: apply face alignment FIRST, then yaw AROUND sphere Z
                #     → this guarantees the continuous randomization is exactly around the sphere’s local Z
                R_sphere_to_object = R_yaw @ R_face

                T_sphere_to_object = torch.eye(4, device=self.device).unsqueeze(0).repeat(num_resets, 1, 1)
                T_sphere_to_object[:, :3, :3] = R_sphere_to_object
                T_sphere_to_object[:, :3, 3] = pos_sphere

                T_world_to_object = self.T_world_to_sphere[sub_env_ids] @ T_sphere_to_object
                pos_world = (
                    T_world_to_object[:, :3, 3]
                    + self.scene.env_origins[sub_env_ids]
                )        
                quat_world = quat_from_matrix(T_world_to_object[:, :3,:3])        

                # 6. Write final world-frame pose into the state tensor
                object_default_state[:, 0:3] = pos_world
                object_default_state[:, 3:7] = quat_world

            self.objects[robot].write_root_pose_to_sim(object_default_state[:, :7], env_ids=local_env_ids)
            self.objects[robot].write_root_velocity_to_sim(object_default_state[:, 7:], env_ids=local_env_ids)

            # reset hand
            dof_pos_noise = sample_uniform(-1.0, 1.0, (len(sub_env_ids), self.q_0[robot].shape[0]), device=self.device) *  self.cfg.reset_dof_pos_noise 
            dof_pos = self.q_0[robot].unsqueeze(0).repeat(len(sub_env_ids), 1)
            dof_pos = dof_pos[None, :] + dof_pos_noise
            dof_vel_noise = sample_uniform(-1.0, 1.0, (len(local_env_ids), self.num_hand_dofs[robot]), device=self.device)
            dof_vel = self.hands[robot].data.default_joint_vel[local_env_ids].clone() + self.cfg.reset_dof_vel_noise * dof_vel_noise

            self.prev_targets[robot][local_env_ids] = dof_pos
            self.cur_targets[robot][local_env_ids] = dof_pos
            self.hand_dof_targets[robot][local_env_ids] = dof_pos

            self.hands[robot].set_joint_position_target(dof_pos, env_ids=local_env_ids, joint_ids=self.actuated_dof_indices[robot])
            self.hands[robot].write_joint_state_to_sim(dof_pos, dof_vel, env_ids=local_env_ids, joint_ids=self.actuated_dof_indices[robot])

            # Merge fingers if mode activated
            if (self.merge_fingers):
                if (self.n_fingers[robot] == 5):
                    rand_vals = torch.rand(len(local_env_ids), device=self.device)
                    self.env_merge_flags[sub_env_ids] = rand_vals < self.merge_frequency
                else:
                    self.env_merge_flags[sub_env_ids] = 1.0

            if self.cfg.randomize_theta_anchors & self.training:
                self.rand_lateral_anchor_offset[sub_env_ids, :] = (torch.rand_like(self.rand_lateral_anchor_offset[sub_env_ids, :]) - 0.5)* self.cfg.rand_theta * stage_2_scale
                self.rand_lateral_anchor_offset[sub_env_ids, :] = torch.clamp( 
                self.rand_lateral_anchor_offset[sub_env_ids, :],
                self.finger_lateral_min[sub_env_ids, :],
                self.finger_lateral_max[sub_env_ids, :]
            )
            
        # Evaluation Code
        if self.cfg.evaluation:
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

        # Reset goals for specified env_ids
        self._reset_target_pose(env_ids)
        self._compute_intermediate_values()

        # Reset success counter
        self.successes[env_ids] = 0

        # reset the episode length buffer
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
        self.learning_iter = (self.common_step_counter // self.cfg.num_steps_per_env) + self.cfg.iteration_bias
        
        for robot in self.cfg.robots:
            # Hand Data
            hand_dof_pos = self.hands[robot].data.joint_pos.clone()
            self.current_dofs[robot] = hand_dof_pos[:, self.actuated_dof_indices[robot]]
            hand_dof_vel = self.hands[robot].data.joint_vel.clone()
            self.current_vel[robot] = hand_dof_vel[:, self.actuated_dof_indices[robot]]
            T_rot = self.T_rot_template[self.robot_indices[robot]]
            
            for c, chain in enumerate(self.finger_chains[robot]):
                c_idx = self.chain_indices[robot][c]
                # Initialize
                points_pos = []
                points_linvel = []
                points_angvel = []
                influence_joints = []
                T_cumulative = self.T_sphere_to_base_all[robot]  # [num_envs, 4, 4] #!
                T_cumulative_inv = self.T_base_to_sphere_all[robot]

                # Solve for finger
                for joint in chain[1:-1]:
                    influence_joints.append(joint)

                    # Get the T_parent_to_joint and add to the T_sphere_to_joint
                    T_joint = self.T_joints[robot][joint]
                    T_joint_inv = self.T_joints_inv[robot][joint]

                    # Get current joint position
                    q_values = self.current_dofs[robot][:, self.joint_index_dict[robot][joint]]
                    cos = torch.cos(q_values)
                    sin = torch.sin(q_values)
                    T_rot[:, 0, 0] = cos                    
                    T_rot[:, 0, 1] = -sin
                    T_rot[:, 1, 0] = sin
                    T_rot[:, 1, 1] = cos

                    T_cumulative = T_cumulative @ T_joint @ T_rot  # [num_envs, 4, 4]
                    self.T_sphere_to_joints[robot][joint] = T_cumulative.clone()
                    T_cumulative_inv = T_rot.permute(0,2,1) @ T_joint_inv @ T_cumulative_inv
                    self.T_joints_to_sphere[robot][joint] = T_cumulative_inv.clone()

                    # Check for Points of fingerprint
                    og_points = self.fp_point_tensors[robot].get(self.joint_index_dict[robot][joint], torch.empty((0, 3), dtype=torch.float, device=self.device))
                    if og_points.numel() == 0:
                        continue
                    og_points = og_points.unsqueeze(0).expand(self.num_robots[robot], -1, -1)  # [num_envs, num_samples, 3]

                    # Transform points to sphere frame
                    ones = torch.ones_like(og_points[..., :1])  # [num_envs, num_samples, 1]
                    points_h = torch.cat((og_points, ones), dim=-1)  # [num_envs, num_samples, 4]
                    points = torch.einsum('eij,epj->epi', T_cumulative, points_h)[:, :, :3]  # [num_envs, num_samples, 3] 
                    points_pos.append(points)

                    # Compute point velocities
                    if self.cfg.linvel_obs:
                        indices = [self.joint_index_dict[robot][j] for j in influence_joints]
                        q_pos = torch.stack([self.T_sphere_to_joints[robot][j][:, :3, 3] for j in influence_joints], dim=1)  # [num_envs, num_infl, 3]
                        q_axis = torch.stack([self.T_sphere_to_joints[robot][j][:, :3, 2] for j in influence_joints], dim=1)  # [num_envs, num_infl, 3]
                        vel = self.current_vel[robot][:, indices]  # [num_envs, num_infl]
                        diff = points[:, :, None, :] - q_pos[:, None, :, :]  # [num_envs, num_samples, num_infl, 3]
                        cross = torch.cross(diff, q_axis[:, None, :, :], dim=-1)  # [num_envs, num_samples, num_infl, 3]
                        weighted_cross = cross * vel[:, None, :, None]  # [num_envs, num_samples, num_infl, 3]
                        point_velocities = weighted_cross.sum(dim=2)  # [num_envs, num_samples, 3]
                        points_linvel.append(point_velocities)
                    if self.cfg.angvel_obs:
                        w = (q_axis * vel[:, :, None]).sum(dim=1)  # [num_envs, 3]
                        points_angvel.append(w[:, None, :].repeat(1, points.shape[1], 1))

                # Update fp obs sample 
                points_pos = torch.cat(points_pos, dim=1)
                self.fp_pos[robot][:, c_idx, :, :] = points_pos[:, None, :, :]
                if self.cfg.linvel_obs:
                    points_linvel = torch.cat(points_linvel, dim=1)
                    self.fp_linvel[robot][:, c_idx, :, :] = points_linvel[:, None, :, :]
                if self.cfg.angvel_obs:
                    points_angvel = torch.cat(points_angvel, dim=1)
                    self.fp_angvel[robot][:, c_idx, :, :] = points_angvel[:, None, :, :]

                if self.cfg.tip_rot:
                    T_joint = self.T_joints[robot][chain[-1]]
                    T_cumulative = T_cumulative @ T_joint
                    self.tip_vec[robot][:, c_idx, :] = T_cumulative[:, None, :3, 2]

            # Object Data
            self.object_pos[self.robot_indices[robot]] = self.objects[robot].data.root_pos_w.clone() - self.scene.env_origins[self.robot_indices[robot]].clone() 
            self.object_rot[self.robot_indices[robot]] = self.objects[robot].data.root_quat_w.clone()
            self.T_world_to_obj[self.robot_indices[robot],:3,:3] = matrix_from_quat(self.object_rot[self.robot_indices[robot]]) # [num_envs, 3, 3]
            self.T_world_to_obj[self.robot_indices[robot],:3, 3] = self.object_pos[self.robot_indices[robot]]
            self.T_sphere_to_obj[self.robot_indices[robot]] = self.T_sphere_to_world[self.robot_indices[robot]] @ self.T_world_to_obj[self.robot_indices[robot]]
            self.object_pos_in_sphere[self.robot_indices[robot]] = self.T_sphere_to_obj[self.robot_indices[robot],:3, 3]
            obj_R_in_sphere = self.T_sphere_to_obj[self.robot_indices[robot], :3, :3]
            self.object_rot_in_sphere[self.robot_indices[robot]] = quat_from_matrix(obj_R_in_sphere)
            self.object_linvel[self.robot_indices[robot]] = self.objects[robot].data.root_lin_vel_w.clone()
            self.object_linvel_in_sphere[self.robot_indices[robot]] = torch.matmul(self.T_sphere_to_world[self.robot_indices[robot], :3,:3], self.object_linvel[self.robot_indices[robot]].unsqueeze(-1)).squeeze(-1)
            self.object_angvel[self.robot_indices[robot]] = self.objects[robot].data.root_ang_vel_w.clone()
            self.object_angvel_in_sphere[self.robot_indices[robot]] = torch.matmul(self.T_sphere_to_world[self.robot_indices[robot], :3,:3], self.object_angvel[self.robot_indices[robot]].unsqueeze(-1)).squeeze(-1)

    def compute_full_observations(self):
        # === Progressive noise scale during Stage 3 ===
        current_iter = self.learning_iter
        stage_3_start = 2 * self.n_iter_per_stage

        if current_iter < stage_3_start:
            noise_scale = 0.0
        else:
            # Linear ramp from 0.0 → 1.0 over the whole Stage 3
            noise_scale = min(1.0, max(0.0, (current_iter - stage_3_start) / self.n_iter_per_stage))

        # Gather all obs and stack across robots
        obs = []
        privileged_obs = []
        for robot in self.cfg.robots:
            # Compute goal_rot in Sphere
            goal_R = matrix_from_quat(self.goal_rot[self.robot_indices[robot]])
            goal_R_in_sphere = self.T_sphere_to_world[self.robot_indices[robot], :3,:3] @ goal_R
            self.goal_rot_in_sphere[self.robot_indices[robot]] = quat_from_matrix(goal_R_in_sphere)

            # Normalize positions/velocities (clean versions)
            self.object_pos_in_sphere[self.robot_indices[robot]] /= self.sphere_radius[robot]
            self.object_linvel_in_sphere[self.robot_indices[robot]] /= self.sphere_radius[robot]
            self.fp_pos[robot] /= self.sphere_radius[robot]
            if self.cfg.linvel_obs or self.cfg.c_linvel_obs:
                self.fp_linvel[robot] /= self.sphere_radius[robot]

            # === POLICY OBSERVATIONS (noisy, masked for failures) ===
            object_pos = self.object_pos_in_sphere[self.robot_indices[robot]].clone()
            object_rot = self.object_rot_in_sphere[self.robot_indices[robot]].clone()
            object_linvel = self.object_linvel_in_sphere[self.robot_indices[robot]].clone()
            object_angvel = self.object_angvel_in_sphere[self.robot_indices[robot]].clone()

            # Noisy object states (policy only)
            if self.training and self.cfg.obj_pos_noise > 0.0:
                noise = get_noise(object_pos, self.cfg.obj_pos_noise * noise_scale)
                object_pos += noise
            if self.training and self.cfg.obj_rot_noise > 0.0:
                noise_quats = random_small_quats(len(self.robot_indices[robot]), self.cfg.obj_rot_noise * noise_scale, self.device)
                object_rot = quat_mul(noise_quats, object_rot)
            if self.training and self.cfg.obj_linvel_noise > 0.0:
                noise = get_noise(object_linvel, self.cfg.obj_linvel_noise * noise_scale)
                object_linvel += noise
            if self.training and self.cfg.obj_angvel_noise > 0.0:
                noise = get_noise(object_angvel, self.cfg.obj_angvel_noise * noise_scale)
                object_angvel += noise

            sub_obs = [
                object_pos,
                object_rot
            ]
            
            if self.cfg.obj_linvel_obs:
                sub_obs.append(object_linvel)

            if self.cfg.obj_angvel_obs:
                sub_obs.append(object_angvel)

            # print(robot)
            # print(self.object_pos_in_sphere[self.robot_indices[robot]])
            sub_obs.extend(
                [
                self.goal_rot_in_sphere[self.robot_indices[robot]],
                quat_mul(object_rot, quat_conjugate(self.goal_rot_in_sphere[self.robot_indices[robot]])),
                self.actions[self.robot_indices[robot]]
                ]
            )
            
            # Hand observations
            # Noisy hand pos (policy only)
            fp_pos = self.fp_pos[robot].clone()
            if self.training and self.cfg.hand_pos_noise > 0.0:
                noise = get_noise(fp_pos, self.cfg.hand_pos_noise * noise_scale)
                fp_pos += noise
            sub_obs.append(fp_pos[:, :, self.obs_indices, :].flatten(start_dim=1))
            if self.cfg.tip_rot:
                tip_vec = self.tip_vec[robot].clone()
                if self.training and self.cfg.hand_rot_noise > 0.0:
                    noise = get_noise(tip_vec, self.cfg.hand_rot_noise * noise_scale)
                    dot = (noise * tip_vec).sum(dim=-1, keepdim=True) 
                    noise = noise - dot * tip_vec
                    tip_vec = tip_vec + noise
                    # Renormalize back to unit length (numerically stable)
                    tip_vec = torch.nn.functional.normalize(tip_vec, dim=-1, eps=1e-8)

                sub_obs.append(tip_vec.flatten(start_dim=1))

            if self.cfg.linvel_obs:
                fp_linvel = self.fp_linvel[robot].clone()
                if self.training and self.cfg.hand_linvel_noise > 0.0:
                    noise = get_noise(fp_linvel, self.cfg.hand_linvel_noise * noise_scale)
                    fp_linvel += noise
                sub_obs.append(fp_linvel[:, :, self.obs_indices, :].flatten(start_dim=1))

            if self.cfg.angvel_obs:
                fp_angvel = self.fp_angvel[robot].clone()
                if self.training and self.cfg.hand_angvel_noise > 0.0:
                    noise = get_noise(fp_angvel, self.cfg.hand_angvel_noise * noise_scale)
                    fp_angvel += noise
                sub_obs.append(self.cfg.vel_obs_scale * fp_angvel[:, :, self.obs_indices, :].flatten(start_dim=1))

            if self.cfg.merge_flag_input:
                sub_obs.append(self.env_merge_flags[self.robot_indices[robot]].to(dtype=torch.float)[:,None])

            sub_obs = torch.cat(sub_obs, dim=-1)
            obs.append(sub_obs)

            # === PRIVILEGED OBSERVATIONS FOR CRITIC (always clean, no noise) ===
            sub_priv = [
                self.object_pos_in_sphere[self.robot_indices[robot]],
                self.object_rot_in_sphere[self.robot_indices[robot]],
                self.object_linvel_in_sphere[self.robot_indices[robot]],
                self.cfg.vel_obs_scale * self.object_angvel_in_sphere[self.robot_indices[robot]],
                self.goal_rot_in_sphere[self.robot_indices[robot]],
                quat_mul(self.object_rot_in_sphere[self.robot_indices[robot]], quat_conjugate(self.goal_rot_in_sphere[self.robot_indices[robot]])),
                self.actions[self.robot_indices[robot]],
                self.fp_pos[robot][:, :, self.obs_indices, :].flatten(start_dim=1),
            ]
            # print(robot, self.fp_pos[robot][:, :, self.obs_indices, :].flatten(start_dim=1))

            if self.cfg.c_tip_rot:
                sub_priv.append(self.tip_vec[robot].flatten(start_dim=1))
            if self.cfg.c_linvel_obs:
                sub_priv.append(self.fp_linvel[robot][:, :, self.obs_indices, :].flatten(start_dim=1))
            if self.cfg.c_angvel_obs:
                sub_priv.append(self.cfg.vel_obs_scale * self.fp_angvel[robot][:, :, self.obs_indices, :].flatten(start_dim=1))
            if self.cfg.merge_flag_input:
                sub_priv.append(self.env_merge_flags[self.robot_indices[robot]].to(dtype=torch.float)[:,None])

            sub_priv = torch.cat(sub_priv, dim=-1)
            privileged_obs.append(sub_priv)

        obs = torch.cat(obs, dim=0)
        privileged_obs = torch.cat(privileged_obs, dim=0)
        return {"policy": obs, "critic": privileged_obs}
    
    def step(self, action: torch.Tensor): # Overwrite original step to handle multiple robots
        action = action.to(self.device)
        # process actions
        self._pre_physics_step(action)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()

            # Write actions/data to sim per asset
            for robot in self.cfg.robots: 
                self.hands[robot].write_data_to_sim()
                self.objects[robot].write_data_to_sim()
                self.object_rot[self.robot_indices[robot]] = self.objects[robot].data.root_quat_w.clone()

            # simulate
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        # self.reset_goal_buf = success_during_decimation

        # post-step:
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids) 
            # update articulation kinematics
            self.scene.write_data_to_sim()

            # Write actions/data to sim per asset
            for robot in self.cfg.robots:
                self.hands[robot].write_data_to_sim()
                self.objects[robot].write_data_to_sim()

            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        if self.custom_event_managers[robot]:
            if "interval" in self.custom_event_managers[robot].available_modes:
                self.custom_event_managers[robot].apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def random_small_quats(num_envs: int, max_angle_rad: float, device: torch.device) -> torch.Tensor:
    # Generates random quaternions for small rotations with angles uniform in [-max_angle_deg, max_angle_deg].
    angles = torch.rand(num_envs, device=device) * 2 * max_angle_rad - max_angle_rad
    axes = torch.randn(num_envs, 3, device=device)
    axes = axes / torch.norm(axes, dim=-1, keepdim=True)
    half_angles = angles / 2.0
    cos = torch.cos(half_angles).unsqueeze(-1)
    sin = torch.sin(half_angles).unsqueeze(-1)
    return torch.cat([cos, sin * axes], dim=-1)

@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention

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

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    goal_dist = torch.clamp(goal_dist, 0.0, fall_dist)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist 
    # dist_rew = torch.where(goal_dist >= fall_dist, 0, goal_dist * dist_reward_scale)
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) 

    # reward computation
    lateral_position_penalty = torch.cat([torch.sum(diff**2, dim=-1) for diff in lateral_position_diff])
    radii_position_penalty = torch.cat([torch.sum(diff**2, dim=-1) for diff in radii_position_diff])

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew * dist_reward_scale + rot_rew * rot_reward_scale + lateral_position_penalty * lateral_position_penalty_scale + radii_position_penalty * radii_position_penalty_scale

    # Velocity penalty reward
    velocity_penalty = torch.cat([torch.sum((vel / max(1e-6, max_velocity - velocity_tolerance))**2, dim=-1)  for vel in q_vel])

    # Energy reward
    energy_penalty = torch.cat([torch.sum(torch.abs(vel) * torch.abs(eff), dim=-1)  for vel, eff in zip(q_vel, q_eff)])

    # Action Rate Reward
    action_rate_penalty = (torch.sum((actions - last_actions)**2, dim=-1) + torch.sum((actions - 2*last_actions + last_last_actions)**2, dim=-1)) 

    # Reward
    reward = reward + velocity_penalty * velocity_scale + energy_penalty * energy_scale + action_rate_penalty * action_rate

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes, lateral_position_penalty, radii_position_penalty, dist_rew, rot_rew, velocity_penalty, energy_penalty, action_rate_penalty
