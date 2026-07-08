import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os 

USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "allegro_right", "allegro_right.usd"))

ALLEGRO_HAND_CFG = ArticulationCfg(
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
        "joint_12": 1.2, 
        "joint_13": 0.0, 
        "joint_14": 0.0, 
        "joint_15": 0.0 
    },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit_sim={
                "joint.*": 0.4, 
            },
            stiffness={
                "joint.*": 3.0,
            },
            damping={
                "joint.*": 0.2,
            },
            velocity_limit_sim= {  
                ".*": 5.0,
            },
            friction = 0.02,
            armature = 0.00149376,
            dynamic_friction = 0.02
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
