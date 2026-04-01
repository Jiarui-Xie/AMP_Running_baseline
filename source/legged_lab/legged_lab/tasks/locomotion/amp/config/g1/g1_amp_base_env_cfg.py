"""G1 AMP environment config with 2 LaFan1 clips (walk + run).

Designed for learning omnidirectional locomotion with just 2 reference clips.
Extends standard G1 AMP rewards with denser locomotion signals.
"""

import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR

from .g1_amp_env_cfg import G1AmpEnvCfg, G1AmpRewards


@configclass
class G1AmpBaseRewards(G1AmpRewards):
    """
    is_alive, stand_still_joint_deviation, feet_contact_without_cmd
    """

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    track_lin_vel_xy_low_speed = RewTerm(
        func=mdp.track_lin_vel_xy_low_speed,
        weight=0.2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "max_speed": 1.0, "min_speed": 0.2}
    )

    is_alive = RewTerm(func=mdp.is_alive, weight=0.25)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )

    stand_still_joint_deviation = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
        },
    )

    # Relax base vertical velocity penalty to allow natural hip oscillation during
    # running.  Landing impact protection is already handled by feet_landing_vel_z.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.02)

    feet_too_near = None  # replaced by feet_y_distance

    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-1.5,
        params={
            "min_dist": 0.15,
            "max_dist": 0.30,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    elbow_hip_distance = RewTerm(
        func=mdp.body_pair_distance,
        weight=-10.0,
        params={
            "min_dist": 0.15,
            "asset_cfg_a": SceneEntityCfg("robot", body_names=["left_elbow_link", "right_elbow_link"]),
            "asset_cfg_b": SceneEntityCfg("robot", body_names=["left_hip_pitch_link", "right_hip_pitch_link"]),
        },
    )

    hands_height = RewTerm(
        func=mdp.hands_height,
        weight=-10.0,
        params={
            "min_height": 0.70,
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]),
        },
    )

    feet_landing_vel_z = RewTerm(
        func=mdp.feet_landing_vel_z,
        weight=-0.5,
        params={
            "height_threshold": 0.10,
            "base_vel_limit": 0.1,
            "vel_scale_with_cmd": 0.15,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )


@configclass
class G1AmpBaseEnvCfg(G1AmpEnvCfg):
    """G1 AMP environment with 2 LaFan1 clips for walk/run."""

    rewards: G1AmpBaseRewards = G1AmpBaseRewards()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # communication delay domain randomization: 0–1 steps, 10% probability
        # ------------------------------------------------------
        self.max_action_delay_steps = 1
        self.action_delay_probability = 0.1

        # ------------------------------------------------------
        # reduce pure-stand command probability
        # ------------------------------------------------------
        self.commands.base_velocity.rel_standing_envs = 0.01

        # ------------------------------------------------------
        # motion data — only 2 clips (walk + run) from base dataset
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            # BASE dataset (~70% total weight)
            "base/walk1_subject1": 1.0,
            "base/run1_subject2": 1.0,
            # walk_and_run dataset (~30% total weight) — walk, run, transitions
            "walk_and_run/C1_-_stand_to_run_stageii": 0.1,
            "walk_and_run/C3_-_run_stageii": 0.1,
            "walk_and_run/C4_-_run_to_walk_a_stageii": 0.1,
            "walk_and_run/C5_-_walk_to_run_stageii": 0.1,
            "walk_and_run/Walk_B4_-_Stand_to_Walk_Back_stageii": 0.1,
            "walk_and_run/Walk_B10_-_Walk_turn_left_45_stageii": 0.1,
            "walk_and_run/Walk_B13_-_Walk_turn_right_45_stageii": 0.1,
            "walk_and_run/Walk_B22_-_Side_step_left_stageii": 0.1,
            "walk_and_run/Walk_B23_-_Side_step_right_stageii": 0.1,
        }            

@configclass
class G1AmpBaseEnvCfg_PLAY(G1AmpBaseEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
