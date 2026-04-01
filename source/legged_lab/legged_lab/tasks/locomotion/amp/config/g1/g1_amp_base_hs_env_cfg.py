"""G1 AMP High-Speed environment config — post-training variant for high-speed deployment.

Inherits from G1AmpBaseEnvCfg and adds:
- Energy/power consumption penalty
- Slope terrain (up to 6 degrees, randomized direction)
- Increased max speed to 4.0 m/s (min 0.5, skipping start-from-zero)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen

import legged_lab.tasks.locomotion.amp.mdp as mdp

from .g1_amp_base_env_cfg import G1AmpBaseEnvCfg, G1AmpBaseRewards


# ---------------------------------------------------------------------------
# Terrain: flat + pyramid slopes (0–6°) + gentle wave undulations
# Each 8×8 m tile is generated with a random slope angle in the given range.
# With num_rows=10, num_cols=20 → 200 tiles covering 4096 environments.
# Pyramid slope tiles have a central platform (1.5 m wide) with slopes going
# in all 4 directions, so robots naturally encounter forward / lateral /
# diagonal inclines depending on spawn position and heading.
# ---------------------------------------------------------------------------
SLOPE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        # 20% flat — ensures the robot still trains on flat ground
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.0, 0.005),
            noise_step=0.005,
            border_width=0.25,
        ),
        # 30% gentle slope: 0–3.4°
        "slope_gentle": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3,
            slope_range=(0.0, 0.06),
            platform_width=1.5,
            border_width=0.25,
        ),
        # 25% moderate slope: 3.4–6°
        "slope_moderate": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.06, 0.105),
            platform_width=1.5,
            border_width=0.25,
        ),
        # 15% inverted pyramid (valley/bowl): 0–6°
        "slope_inverted": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.105),
            platform_width=1.5,
            border_width=0.25,
        ),
        # 10% gentle wave undulations
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.1,
            amplitude_range=(0.02, 0.08),
            num_waves=3,
        ),
    },
)


@configclass
class G1AmpBaseHSRewards(G1AmpBaseRewards):
    """Rewards for high-speed post-training: adds energy penalty on top of BASE rewards."""

    energy = RewTerm(
        func=mdp.energy,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class G1AmpBaseHighSpeedEnvCfg(G1AmpBaseEnvCfg):
    """G1 AMP high-speed post-training environment.

    Differences from BASE:
    - lin_vel_x range: (0.5, 4.0) — no need to learn start-from-zero
    - Energy penalty reward
    - Slope terrain up to 6 degrees
    """

    rewards: G1AmpBaseHSRewards = G1AmpBaseHSRewards()

    def __post_init__(self):
        super().__post_init__()

        # Speed range: minimum 0.5 m/s (skip start-from-zero), max 4.0 m/s
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 4.0)

        # Terrain: switch from flat plane to slope generator
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = SLOPE_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None


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
