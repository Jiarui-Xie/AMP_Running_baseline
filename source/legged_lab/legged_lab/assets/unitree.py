"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_rl_lab
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets import unitree_actuators

# Motor armature constants from SONIC deployment
# Source: GR00T-WholeBodyControl/gear_sonic_deploy/.../policy_parameters.hpp
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# PID gains: second-order critically-damped model (ω = 10Hz × 2π, ζ = 2.0)
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  # 14.25
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2  # 40.18
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2  # 99.10
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2  # 16.78

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ  # 0.91
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # 2.56
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # 6.31
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ  # 1.07


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


UNITREE_GO2_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/go2/usd/go2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ],
    # fmt: on
)


UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # hip_pitch grouped with 7520_22 per SONIC hardware; stiffness/damping tuned
        # between original training values and SONIC deployment for sim-to-real balance
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_.*": 3.0,
                ".*_knee_.*": 5.0,
            },
            armature=ARMATURE_7520_22,
        ),
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_yaw_.*": 80.0,
                "waist_yaw_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_.*": 2.0,
                "waist_yaw_joint": 4.0,
            },
            armature=ARMATURE_7520_14,
        ),
        "N5020-16-double": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50,
            velocity_limit_sim=37,
            stiffness={
                ".*_ankle_.*": 35.0,
                "waist_.*_joint": 40.0,
            },
            damping={
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 4.0,
            },
            armature=2.0 * ARMATURE_5020,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll.*"],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=30.0,
            damping=1.0,
            armature=ARMATURE_5020,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=25.0,
            damping=1.0,
            armature=ARMATURE_4010,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)
