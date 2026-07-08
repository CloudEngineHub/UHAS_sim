# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os

USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),"shadow_right", "shadow_right.usd"))

SHADOW_HAND_CFG = ArticulationCfg(
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
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "(index_finger_joint|middle_finger_joint|ring_finger_joint)(1|2|3|4)", 
                "(little_finger_joint)(2|4|3|5)",
                "(thumb_joint)(1|2|3|4|5)"],
            effort_limit_sim={
                ".*": 0.4,
            },
            stiffness={
                ".*": 3.0,
            },
            damping={
                ".*": 0.2,
            },
            velocity_limit_sim= {  
                ".*": 5.0,
            },
            friction = 0.02,
            armature = 0.00149376,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)