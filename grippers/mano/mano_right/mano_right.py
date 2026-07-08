import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os 

USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),"mano_right", "mano_right.usd"))


MANO_HAND_CFG = ArticulationCfg(
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
        "j_index1y": 0.0, 
        "j_index1z": 0.0, 
        "j_index2z": 0.0, 
        "j_index3z": 0.0, 
        "j_middle1y": 0.0, 
        "j_middle1z": 0.0, 
        "j_middle2z": 0.0, 
        "j_middle3z": 0.0, 
        "j_pinky1y": 0.0, 
        "j_pinky1z": 0.0, 
        "j_pinky2z": 0.0, 
        "j_pinky3z": 0.0, 
        "j_ring1y": 0.0, 
        "j_ring1z": 0.0, 
        "j_ring2z": 0.0, 
        "j_ring3z": 0.0,
        "j_thumb1x": 0.0, 
        "j_thumb1y": 0.0, 
        "j_thumb2y": 0.0,
        "j_thumb2z": 0.0, 
        "j_thumb3z": 0.0
    },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["j.*"],
            effort_limit_sim={
                "j.*": 0.4,
            },
            stiffness={
                "j.*": 3.0,
            },
            damping={
                "j.*": 0.2,
            },
            velocity_limit_sim= {  
                "j.*": 5.0,
            },
            friction = 0.02,
            armature = 0.00149376,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
