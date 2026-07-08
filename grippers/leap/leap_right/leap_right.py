import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os 
import math

USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "leap_right", "leap_right.usd"))

LEAP_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            retain_accelerations=True, 
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0, 
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0), 
        joint_pos={
        "joint_0": 0.0, 
        "joint_1": 0.0, 
        "joint_2": 0.0, 
        "joint_3": 0.0, 
        "joint_4": 0.0, 
        "joint_5": 0.0, 
        "joint_6": 0.0, 
        "joint_7": 0.0, 
        "joint_8": 0.0, 
        "joint_9": 0.0, 
        "joint_10": 0.0, 
        "joint_11": 0.0, 
        "joint_12": 0.0, 
        "joint_13": 1.5, 
        "joint_14": 0.0, 
        "joint_15": 0.0 
    },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit_sim={
                "joint.*": 0.21,
            },
            stiffness={
                "joint.*": 1.25,
            },
            damping={
                "joint.*": 0.46,
            },
            velocity_limit_sim= {  
                "joint.*": 1.56,
            },
            friction = 0.02,
            armature = 0.00149376,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
