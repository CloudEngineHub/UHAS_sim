import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os 

USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),"wuji_right", "wuji_right.usd"))


WUJI_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0), 
        joint_pos={
            # Finger 1 (thumb or first finger depending on your model)
            "right_finger1_joint1": 0.048, 
            "right_finger1_joint2": 0.0, 
            "right_finger1_joint3": 0.0, 
            "right_finger1_joint4": 0.0, 
            # Finger 2
            "right_finger2_joint1": 0.0, 
            "right_finger2_joint2": 0.0, 
            "right_finger2_joint3": 0.0, 
            "right_finger2_joint4": 0.0, 
            # Finger 3
            "right_finger3_joint1": 0.0, 
            "right_finger3_joint2": 0.0, 
            "right_finger3_joint3": 0.0, 
            "right_finger3_joint4": 0.0, 
            # Finger 4
            "right_finger4_joint1": 0.0, 
            "right_finger4_joint2": 0.0, 
            "right_finger4_joint3": 0.0, 
            "right_finger4_joint4": 0.0,
            # Finger 5
            "right_finger5_joint1": 0.0, 
            "right_finger5_joint2": 0.0, 
            "right_finger5_joint3": 0.0, 
            "right_finger5_joint4": 0.0,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_finger[1-5]_joint[1-4]"],
            effort_limit_sim={
                "right_finger.*_joint.*": 0.4,
            },
            stiffness={
                "right_finger.*_joint.*": 3.0,
            },
            damping={
                "right_finger.*_joint.*": 0.2,
            },
            velocity_limit_sim={  
                "right_finger.*_joint.*": 5.0,
            },
            friction = 0.02,
            armature = 0.00149376,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)