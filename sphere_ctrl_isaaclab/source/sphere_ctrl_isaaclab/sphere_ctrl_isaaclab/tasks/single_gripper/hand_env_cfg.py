# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause
from sphere_ctrl_isaaclab.assets.grippers.shadow.shadow_right.shadow_right import SHADOW_HAND_CFG
from sphere_ctrl_isaaclab.assets.grippers.leap.leap_right.leap_right import LEAP_HAND_CFG as LEAP_HAND_RIGHT_CFG
from sphere_ctrl_isaaclab.assets.grippers.mano.mano_right.mano_right import MANO_HAND_CFG as MOD_MANO_HAND_RIGHT_CFG
from sphere_ctrl_isaaclab.assets.grippers.wuji.wuji_right.wuji_right import WUJI_HAND_CFG as WUJI_HAND_RIGHT_CFG
from sphere_ctrl_isaaclab.assets.grippers.allegro.allegro_right.allegro_right import ALLEGRO_HAND_CFG as ALLEGRO_HAND_RIGHT_CFG
from sphere_ctrl_isaaclab import IRVL_ASSET_PATH
from sphere_ctrl_isaaclab.tasks.single_gripper.agents.rsl_rl_ppo_cfg import SingleGripperPPORunnerCfg
import os 
import json

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Curriculum-based domain randomization functions (from multi-gripper setup)
from sphere_ctrl_isaaclab.tasks.utils.dr_funcs import *
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

# Load structured data from JSON
def load_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

@configclass
class EventCfg:
    """Configuration for randomization (aligned with multi_env_cfg curriculum DR)."""
    
    def __init__(
            self, 
            robot: str = "robot",
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
                        "asset_cfg": SceneEntityCfg("object"),
                        "scale_range": (0.95 * cube_scale, 1.05 * cube_scale),  # Updated to match multi_env_cfg tighter range
                    },
                )

            self.object_physics_material = EventTerm(
                func=curriculum_randomize_rigid_body_material,
                min_step_count_between_reset=720,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
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
                        "asset_cfg": SceneEntityCfg("object"),
                        "mass_distribution_params": (0.5, 1.5), 
                        "operation": "scale",
                        "distribution": "uniform",
                        "n_iter_per_stage": n_iter_per_stage,
                    },
                )

            # -- robot (single-hand naming: "robot")
            if randomize_efforts:
                self.robot_joint_effort = EventTerm( 
                    func=curriculum_randomize_joint_effort_limit,
                    min_step_count_between_reset=720,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
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
                        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
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
                    "asset_cfg": SceneEntityCfg("robot"),
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
                        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
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
                        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
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
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
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
            # Non-training: fixed narrow randomization (aligned with multi_env_cfg)
            self.object_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                min_step_count_between_reset=10000,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
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
                    "asset_cfg": SceneEntityCfg("robot"),
                    "static_friction_range": (1.0, 1.0),
                    "dynamic_friction_range": (1.0, 1.0),
                    "restitution_range": (0.05, 0.05),
                    "num_buckets": 2,
                },
            )

@configclass
class SingleHandEnvCfg(DirectRLEnvCfg):
    # env Options
    decimation = 2 # Manages model frequency
    episode_length_s = 30.0
    max_consecutive_success = 10
    inclination = -20.0 # Degrees - Negative -> Downward (fingers pointing down)
    sphere_cik_info = "sphere_cik.json" # Used only for start up configuration

    # Dynamic num_steps_per_env (single gripper specific)
    rsl_config = SingleGripperPPORunnerCfg()
    num_steps_per_env = rsl_config.num_steps_per_env

    # Robot selection (saved in config file after training for reproducibility)
    robot: str = "wuji_right"
    # Possible values: "shadow_right", "leap_right", "mano_right", "allegro_right", "wuji_right"

    # Training Options
    inhand_position = [0.65, 0.0, 0.3] # In sphere frame units: radii
    init_pos = [0.65, 0.0, 0.3]        # Initial object position in sphere frame (radii units) - used in reset
    action_noise = 0.05 # % 
    training = True    
    evaluation = True

    # DexCube scaling parameters (matching multi_env_cfg.py logic)
    scale = 7.0 / 6.0
    base_radius = 0.09119
    cube_mass = 0.104
    cube_size = 0.07

    # Curriculum parameters (direct access - must be defined)
    curriculm_length = 6000
    iteration_bias = 0 # Used to advance the curriculum manually.

    # Reset noise parameters (direct access)
    reset_position_noise = 0.1
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0

    # Randomization flags (extended to match multi_env_cfg)
    randomize_scale = True 
    randomize_mass = True
    randomize_effort = False
    randomize_vel = True
    n_iter_per_stage = curriculm_length / 3

    # Actuator / Joint config (new: shared with multi_env_cfg for consistency)
    kP = 3.0
    kD = 0.2
    effort_lim = 0.4
    vel_lim = 5.0
    act_st_friction = 0.02
    act_dyn_friction = 0.02
    armature = 0.00149376

    # Observation flags (actor/policy)
    obj_linvel_obs = True
    obj_angvel_obs = True
    handvel_obs = True       

    # Privileged / Critic observation flags
    c_handvel_obs = True     

    # Observation noise (applied only to actor/policy observations - clean direct access, no getattr fallbacks)
    hand_dof_pos_noise = 0.0
    hand_dof_vel_noise = 0.0
    obj_pos_noise = 0.0
    obj_rot_noise = 0.0
    obj_linvel_noise = 0.0
    obj_angvel_noise = 0.0

    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    reach_goal_bonus = 250
    fall_penalty = -100
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

    # Additional reward scales used in compute_rewards (direct access)
    lateral_position_penalty_scale = -0.016
    radii_position_penalty_scale = -0.004
    velocity_scale = 0.0
    max_velocity = 3.0
    velocity_tolerance = 1.0
    energy_scale = 0.0
    action_rate = 0.0

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

    # robot dependent configuration (only right hands - left hands and mirror logic removed as deprecated)
    if robot == "shadow_right":
        robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "shadow", robot, sphere_cik_info)
        action_space = 21

    elif robot == "leap_right":
        robot_cfg: ArticulationCfg = LEAP_HAND_RIGHT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "leap", robot, sphere_cik_info)
        action_space = 16

    elif robot == "mano_right":
        robot_cfg: ArticulationCfg = MOD_MANO_HAND_RIGHT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "mano", robot, sphere_cik_info)
        action_space = 21

    elif robot == "allegro_right":
        robot_cfg: ArticulationCfg = ALLEGRO_HAND_RIGHT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "allegro", robot, sphere_cik_info)
        action_space = 16

    elif robot == "wuji_right":
        robot_cfg: ArticulationCfg = WUJI_HAND_RIGHT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "wuji", robot, sphere_cik_info)
        action_space = 20

    else:
        raise ValueError(f"Unable to identify Robot Tag {robot}. Only right hands are supported.")

    # Apply shared actuator configuration (new unified way from multi_env_cfg)
    articulation_cfg = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit_sim=effort_lim,
        stiffness=kP,
        damping=kD,
        velocity_limit_sim=vel_lim,
        friction=act_st_friction,
        dynamic_friction=act_dyn_friction,
        armature=armature
    )
    robot_cfg = robot_cfg.replace(actuators={"fingers": articulation_cfg})

    observation_space = 0   # Dynamic dict observations (asymmetric actor-critic)
    state_space = 0

    # cik_info loaded here so it is available via cfg.cik_info for the environment
    cik_info = load_from_json(cik_json_path)

    # Dynamic DexCube scale and mass calculation (exact logic from multi_env_cfg.py)
    sphere_radius = cik_info["joint_info"]["sphere_frame"][6]
    cube_density = cube_mass / (cube_size ** 3)
    cube_scale = scale * sphere_radius / base_radius
    robot_cube_mass = cube_density * (cube_size ** 3) * (sphere_radius / base_radius)

    # Randomization - always create EventCfg (curriculum or fixed depending on training flag)
    events: EventCfg = EventCfg(
        robot="robot",
        training=training,
        n_iter_per_stage=n_iter_per_stage,
        randomize_scale=randomize_scale,
        randomize_mass=randomize_mass,
        randomize_efforts=randomize_effort,
        randomize_vel=randomize_vel,
        cube_scale=cube_scale,
    )

    # in-hand object (DexCube with dynamic scale and mass)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
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
                max_depenetration_velocity=3.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=robot_cube_mass),
            scale=(cube_scale, cube_scale, cube_scale),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=(not randomize_scale))

    # reset
    reset_position_noise = 0.1  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    

