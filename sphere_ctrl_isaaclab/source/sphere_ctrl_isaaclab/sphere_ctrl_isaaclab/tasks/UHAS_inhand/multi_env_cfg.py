# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from sphere_ctrl_isaaclab.assets.grippers.shadow.shadow_right.shadow_right import SHADOW_HAND_CFG
from sphere_ctrl_isaaclab.assets.grippers.leap.leap_right.leap_right import LEAP_HAND_CFG as LEAP_HAND_RIGHT_CFG
from sphere_ctrl_isaaclab.assets.grippers.mano.mano_right.mano_right import MANO_HAND_CFG as MOD_MANO_HAND_RIGHT_CFG
from sphere_ctrl_isaaclab.assets.grippers.allegro.allegro_right.allegro_right import ALLEGRO_HAND_CFG as ALLEGRO_HAND_RIGHT_CFG
from sphere_ctrl_isaaclab.assets.grippers.wuji.wuji_right.wuji_right import WUJI_HAND_CFG as WUJI_HAND_RIGHT_CFG
from sphere_ctrl_isaaclab import IRVL_ASSET_PATH
from sphere_ctrl_isaaclab.tasks.UHAS_inhand.agents.rsl_rl_ppo_cfg import SpherePPORunnerCfg     
from sphere_ctrl_isaaclab.tasks.utils.dr_funcs import *
from sphere_ctrl_isaaclab.tasks.utils.sphere_torch_utils import import_reference_sphere
import os 
import json

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import ViewerCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import torch

# Load structured data from JSON
def load_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


@configclass
class EventCfg:
    """Configuration for randomization."""
    
    def __init__(
            self, 
            robot,
            training: bool = False, 
            n_iter_per_stage: int = 2500,
            randomize_scale: bool = False, 
            randomize_mass: bool = False, 
            randomize_efforts: bool = False, 
            randomize_vel: bool = False, 
            cube_scale: float = 1.0,
        ):
        
        if training:
            # -- object
            if randomize_scale:
                self.object_scale_size = EventTerm(
                    func=mdp.randomize_rigid_body_scale,
                    mode="prestartup",
                    params={
                        "asset_cfg": SceneEntityCfg(f"{robot}_object"),
                        "scale_range": (0.95 * cube_scale, 1.05 * cube_scale),  
                    },
                )

            self.object_physics_material = EventTerm( 
                func=curriculum_randomize_rigid_body_material,
                min_step_count_between_reset=720,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(f"{robot}_object"),
                    "static_friction_range": (0.3, 1.75), 
                    "dynamic_friction_range": (0.3, 1.75), 
                    "restitution_range": (0.0, 0.5), 
                    "num_buckets": 350,
                    "make_consistent": True,
                    "n_iter_per_stage": n_iter_per_stage,
                },
            )

            if randomize_mass:
                self.object_scale_mass = EventTerm( 
                    func=curriculum_randomize_rigid_body_mass,
                    min_step_count_between_reset=720,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(f"{robot}_object"),
                        "mass_distribution_params": (0.5, 1.5), 
                        "operation": "scale",
                        "distribution": "uniform",
                        "n_iter_per_stage": n_iter_per_stage,
                    },
                )

            # -- robot
            if randomize_efforts:
                self.robot_joint_effort = EventTerm( 
                    func=curriculum_randomize_joint_effort_limit,
                    min_step_count_between_reset=720,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(robot, joint_names=".*"),
                        "effort_limit_distribution_params": (0.75, 1.5), 
                        "homogeneous_distribution_parameters": (1.0, 1.0),  
                        "operation": "scale",  
                        "distribution": "uniform",  
                        "n_iter_per_stage": n_iter_per_stage,
                    },
                )
            
            if randomize_vel:
                self.robot_joint_velocity = EventTerm( 
                    func=curriculum_randomize_joint_velocity_limit,
                    min_step_count_between_reset=720,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(robot, joint_names=".*"),
                        "velocity_limit_distribution_params": (0.75, 1.5),  
                        "homogeneous_distribution_parameters": (1.0, 1.0),  
                        "operation": "scale",  
                        "distribution": "uniform",  
                        "n_iter_per_stage": n_iter_per_stage,
                    },
                )
                
            
            
            self.robot_physics_material = EventTerm( 
                func=curriculum_randomize_rigid_body_material,
                mode="reset",
                min_step_count_between_reset=720,
                params={
                    "asset_cfg": SceneEntityCfg(robot),
                    "static_friction_range": (1.0, 1.0),
                    "dynamic_friction_range": (1.0, 1.0), 
                    "restitution_range": (0.05, 0.4), 
                    "num_buckets": 2,
                    "make_consistent": True,
                    "n_iter_per_stage": n_iter_per_stage,
                },
            )

            self.robot_scale_mass = EventTerm( 
                    func=curriculum_randomize_rigid_body_mass,
                    min_step_count_between_reset=720,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(robot, body_names=".*"),
                        "mass_distribution_params": (0.8, 1.2), 
                        "operation": "scale",
                        "distribution": "uniform",
                        "n_iter_per_stage": n_iter_per_stage,
                    },
                )
            
            self.robot_joint_params = EventTerm( 
                    func=curriculum_randomize_joint_parameters,
                    min_step_count_between_reset=720,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(robot, joint_names=".*"),
                        "friction_distribution_params": (0.6, 1.2), 
                        "armature_distribution_params": (0.9, 1.1),
                        "operation": "scale",
                        "distribution": "uniform",
                        "n_iter_per_stage": n_iter_per_stage,
                    },
                )

            self.robot_joint_stiffness_and_damping = EventTerm( 
                func=curriculum_randomize_actuator_gains,
                min_step_count_between_reset=720,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(robot, joint_names=".*"),
                    "stiffness_distribution_params": (0.5, 1.5), 
                    "damping_distribution_params": (0.5, 1.5), 
                    "homogeneous_stiffness_distribution_params": (1.0, 1.0), 
                    "homogeneous_damping_distribution_params": (1.0, 1.0), 
                    "operation": "scale",
                    "distribution": "log_uniform",
                    "n_iter_per_stage": n_iter_per_stage,
                },
            )

        else:
            self.object_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                min_step_count_between_reset=10000,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(f"{robot}_object"),
                    "static_friction_range": (0.8, 0.8),
                    "dynamic_friction_range": (0.8, 0.8),
                    "restitution_range": (0.25, 0.25),
                    "num_buckets": 2,
                },
            )
            self.robot_physics_material = EventTerm( 
                func=mdp.randomize_rigid_body_material,
                mode="reset",
                min_step_count_between_reset=10000,
                params={
                    "asset_cfg": SceneEntityCfg(robot),
                    "static_friction_range": (1.0, 1.0),
                    "dynamic_friction_range": (1.0, 1.0),
                    "restitution_range": (0.05, 0.05),
                    "num_buckets": 2,
                },
            )

# Parse --num_envs from CLI
import sys
num_envs = 8192  

for i, arg in enumerate(sys.argv):
    if arg == "--num_envs" and i + 1 < len(sys.argv):
        try:
            num_envs = int(sys.argv[i + 1])
            print(f"[Config] Overridden num_envs from CLI: {num_envs}")
        except ValueError:
            print(f"[Config] Warning: Invalid --num_envs value, using default {num_envs}")
        break
    elif arg.startswith("--num_envs="):
        try:
            num_envs = int(arg.split("=", 1)[1])
            print(f"[Config] Overridden num_envs from CLI: {num_envs}")
        except ValueError:
            pass
        break

@configclass
class SphereEnvCfg(DirectRLEnvCfg):
    # Environment Options
    episode_length_s = 30.0 # Count down timer to reset environment
    rsl_config = SpherePPORunnerCfg()
    num_steps_per_env = rsl_config.num_steps_per_env
    max_consecutive_success = 10
    training = True    
    evaluation = False
    reset_timer = True # reset episode length timer at every repose.
    robots = ["shadow_right", "leap_right", "mano_right", "allegro_right"]
    # Possible values: "shadow_right", "leap_right", "mano_right", "allegro_right", "wuji_right"

    # Gripper options
    max_fingers = 5
    scale = 7.0 / 6.0 # 6 cm size for Isaac lab default DexCube

    # Debug 
    debug_fingers = False 
    debug_object_inhand = False
    debug_speed_factor = 1.0 # Action Hz

    ###########################################################################################
    # Training Options
    decimation = 2 # Manages model frequency
    inclination = -15.0 # Degrees - Negative -> Downward (fingers pointing down)
    base_radius = 0.09119 # Used to center object scaling around real world object and gripper 
    cube_mass = 0.104 # Mass of RW cube
    cube_size = 0.07 # Size of RW cube
    max_delta_q = 360.0 # Degrees
    
    # Joint config Options
    act_moving_average = 1.0
    kP = 3.0 # 0.55
    kD = 0.2 # < 0.05 is unstable
    effort_lim = 0.4 # 0.15
    vel_lim = 5.0 # 1.35
    act_st_friction = 0.02 # 0.01
    act_dyn_friction = 0.02 #0.01
    armature = 0.00149376

    # Vector Phis
    vector_phis = [60.0, 120.0] # 2 
    max_vector_offsets = [2.0] * len(vector_phis) # In radius #2.0
    min_vector_offsets = [-2.0] * len(vector_phis) # In radius #2.0

    # CIK Options
    n_sphere_surface_samples = 1000

    # Merge Fingers
    merge_fingers = True if training else False
    merge_flag_input = True 
    merge_frequency = 0.3

    # Observations
    obj_linvel_obs = True
    obj_angvel_obs = True
    tip_rot = True 
    linvel_obs = True
    angvel_obs = True

    # Priviledge State
    c_tip_rot = True 
    c_linvel_obs = True
    c_angvel_obs = True

    # Scale for all angular velocities
    vel_obs_scale = 0.2

    # FP Sample obs
    obs_idx = [3, -1] # Indices of obs (if fixed_phi False)

    # Stage 1 Randomization
    randomize_scale = True 
    action_noise = 0.0 # direct model output 

    # Stage 2 Randomization (friction and gains always on)
    randomize_mass = True
    randomize_effort = True
    randomize_vel = True
    randomize_gravity = True # Simulates random gravity with inclination 
    gravity_angle_range = 5.0 # degrees 
    randomize_vectors_phi = True  
    vector_range = torch.pi/12 # radians # Randomization range
    randomize_theta_anchors = True 
    rand_theta = torch.pi/36 # radians # Max theta range in anchor randomization 

    # Stage 3 Randomization 
    hand_pos_noise = 0.0 # unit radius
    hand_rot_noise = 0.0 # tip rot vector noise
    hand_linvel_noise = 0.0 # unit radius/s 
    hand_angvel_noise = 0.0 # unit radians/s 
    obj_pos_noise = 0.0 # unit radius
    obj_rot_noise = 0.0 # radians
    obj_linvel_noise = 0.0 # unit radius/s 
    obj_angvel_noise = 0.0 # unit radians/s 
    
    # Reward Info
    init_pos = [0.65, 0.0, 0.3] # unit: radii MJ= [-0.00037, 0.00556, 0.0455] unit: meters in sphere frame
    inhand_position = [0.65, 0.0, 0.3] # In sphere frame units: radii
    lateral_position_penalty_scale = -0.016
    radii_position_penalty_scale = -0.004
    energy_scale = -0.0
    action_rate = -0.0
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0 
    reach_goal_bonus = 250
    fall_penalty = -100
    velocity_scale = -0.0 
    max_velocity = 3.0
    velocity_tolerance = 1.0
    rot_eps = 0.1

    # reset
    reset_position_noise = 0.1 # unit radius 
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0

    # Curriculum 
    curriculm_length = 10000 # Iteration counts for the entire curriculum (4 equal stages, stage length =/4)
    iteration_bias = 0 * (curriculm_length / 3) # Debug Curriculum training
    n_iter_per_stage = curriculm_length / 3

    ###########################################################################################    
    # Sphere information and FingerPrint Sample to use
    sphere_cik_info = "sphere_cik.json" 
    fp_sample = "fp_sample_7" # Was using fp_projected before
    fp_sample_size = 7
    success_tolerance = 0.1
    av_factor = 0.1
    fall_dist = 0.2   
    ref_sphere = import_reference_sphere()

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution = 0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(2.5, 2.5, 2.5),           # ← change this to your desired camera position
        lookat=(0.0, 0.0, -0.4),        # ← where the camera should point (e.g. center of sphere)
        origin_type="world",           # keep "world" for absolute coords
        resolution=(1920, 1080),        # optional: higher quality video
    )
    
    cik_infos = {}
    robot_cfgs = {}
    object_cfgs = {}
    cube_scales = {}
    cube_masses = {}

    # Calculate number of robot types
    num_robot_types = len(robots)

    # Base number of environments per robot type
    base = num_envs // num_robot_types

    # Remainder to distribute
    remainder = num_envs % num_robot_types

    articulation_cfg = ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim = effort_lim,
            stiffness=kP,
            damping=kD,
            velocity_limit_sim= vel_lim,
            friction = act_st_friction,
            dynamic_friction = act_dyn_friction,
            armature = armature
        )
    

    # Populate self.num_robots
    num_robots = {}
    custom_events = {}
    start = 0
    cube_density = cube_mass/(cube_size**3)
    for i, robot in enumerate(robots):
        num = base + 1 if i < remainder else base
        num_robots[robot] = num

        # Build regex as alternation of string numbers in range
        nums = [str(j) for j in range(start, start + num)]
        regex = '|'.join(nums)
        if len(nums) > 1:
            regex = f"({regex})"
        robot_path = f"/World/envs/env_{regex}/{robot}"
        object_path = robot_path + "_object"
        if robot == "shadow_right":
            robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path=robot_path, actuators = {"fingers": articulation_cfg})
            cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "shadow", robot,sphere_cik_info)
        elif robot == "leap_right":
            robot_cfg: ArticulationCfg = LEAP_HAND_RIGHT_CFG.replace(prim_path=robot_path, actuators = {"fingers": articulation_cfg})
            cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "leap", robot,sphere_cik_info)
        elif robot == "mano_right":
            robot_cfg: ArticulationCfg = MOD_MANO_HAND_RIGHT_CFG.replace(prim_path=robot_path, actuators = {"fingers": articulation_cfg})
            cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "mano", robot,sphere_cik_info)
        elif robot == "allegro_right":
            robot_cfg: ArticulationCfg = ALLEGRO_HAND_RIGHT_CFG.replace(prim_path=robot_path, actuators = {"fingers": articulation_cfg})
            cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "allegro", robot,sphere_cik_info)
        elif robot == "wuji_right":
            robot_cfg: ArticulationCfg = WUJI_HAND_RIGHT_CFG.replace(prim_path=robot_path, actuators = {"fingers": articulation_cfg})
            cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "wuji", robot,sphere_cik_info)
        else:
            raise ValueError(f"Unable to identify Robot Tag {robot}.")
        
        
        start += num
        robot_cfgs[robot] = robot_cfg
        cik_infos[robot] = load_from_json(cik_json_path) 
        sphere_radius = cik_infos[robot]["joint_info"]["sphere_frame"][6]
        cube_scale = scale * sphere_radius/base_radius
        robot_cube_mass = cube_density * ((cube_size * sphere_radius/base_radius))**3 # Can end up in very small values for small grippers
        robot_cube_mass = cube_density * (cube_size)**3 * (sphere_radius/base_radius) # Could use linear, since effort (torque) is linear to hand size
        cube_scales[robot] = cube_scale
        cube_masses[robot] = robot_cube_mass
        print(f"Loaded sphere information from {cik_json_path}")

        # in-hand object
        object_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path=object_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False, 
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass = robot_cube_mass), # Density doesn't work in this class
                scale=(cube_scale, cube_scale, cube_scale),

            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0 , 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        object_cfgs[robot] = object_cfg

        # Randomizations
        custom_events[robot] = EventCfg(robot, training, n_iter_per_stage, randomize_scale, randomize_mass, randomize_effort, randomize_vel, cube_scale)
    

    # Model Size
    action_space = (1 + len(vector_phis)) * max_fingers 
    observation_space = 100
    state_space = 0
    
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )

    # scene
    spacing = 0.75
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=spacing, replicate_physics=False)

    # Clean for better env.yaml
    del nums
    del start
    del sphere_radius 
    del cube_scale
    del base