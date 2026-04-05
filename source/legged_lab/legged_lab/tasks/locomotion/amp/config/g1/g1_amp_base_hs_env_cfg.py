"""G1 AMP High-Speed environment config.

Inherits from G1AmpBaseEnvCfg and adds:
- Energy/power consumption penalty
- Gravity direction randomization equivalent to ±8° slope
- Flat terrain only (same as BASE)
- Speed range (0.5, 2.5) m/s, biased toward 1.5–2.5 by narrowing the low end
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp import events as isaaclab_events
import legged_lab.tasks.locomotion.amp.mdp as mdp

from .g1_amp_base_env_cfg import G1AmpBaseEnvCfg, G1AmpBaseRewards

# ±8° slope equivalent gravity perturbation in x/y: 9.81 * sin(8°) ≈ 1.36 m/s²
_GRAVITY_PERTURB = 9.81 * math.sin(math.radians(8.0))


@configclass
class G1AmpBaseHSRewards(G1AmpBaseRewards):
    """Rewards for high-speed training: adds energy penalty on top of BASE rewards."""

    energy = RewTerm(
        func=mdp.energy,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Extra energy penalty on hip-pitch + ankle joints (prone to overheating)
    energy_leg_hot_joints = RewTerm(
        func=mdp.energy,
        weight=-2e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
            )
        },
    )


@configclass
class G1AmpBaseHighSpeedEnvCfg(G1AmpBaseEnvCfg):
    """G1 AMP high-speed environment.

    Differences from BASE:
    - lin_vel_x range: (0.5, 2.5) — narrowed low end so 50% of samples fall in 1.5–2.5
    - Energy penalty reward
    - Flat terrain (same as BASE)
    """

    rewards: G1AmpBaseHSRewards = G1AmpBaseHSRewards()

    def __post_init__(self):
        super().__post_init__()

        # Speed range: (0.5, 2.5) — 50% of uniform samples land in [1.5, 2.5]
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 2.5)

        # Gravity direction randomization: ±8° equivalent slope (scene-wide, resampled every 5-10s)
        self.events.randomize_gravity = EventTerm(
            func=isaaclab_events.randomize_physics_scene_gravity,
            mode="interval",
            is_global_time=True,
            interval_range_s=(5.0, 10.0),
            params={
                "gravity_distribution_params": (
                    [-_GRAVITY_PERTURB, -_GRAVITY_PERTURB, 0.0],
                    [_GRAVITY_PERTURB, _GRAVITY_PERTURB, 0.0],
                ),
                "operation": "add",
            },
        )


@configclass
class G1AmpBaseHighSpeedEnvCfg_PLAY(G1AmpBaseHighSpeedEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
