from dataclasses import MISSING

from isaaclab.utils import configclass

from .manager_based_animation_env_cfg import ManagerBasedAnimationEnvCfg

@configclass
class ManagerBasedAmpEnvCfg(ManagerBasedAnimationEnvCfg):
    """Configuration for a AMP environment with the manager-based workflow."""

    terminal_obs_groups: tuple[str, ...] = ("disc",)
    """Observation groups to preview before reset and export through ``extras["terminal_obs"]``."""

    max_action_delay_steps: int = 0
    """Maximum per-joint communication delay in control steps.

    Each joint in each environment independently draws a delay in [0, max_action_delay_steps]
    at every reset. Set to 0 to disable. At 50 Hz control (decimation=4, sim_dt=5ms),
    each step is 20ms, so a value of 1 covers 0–20ms per-joint latency.
    """

    action_delay_probability: float = 1.0
    """Per-joint probability of having a non-zero delay at reset.

    Each joint independently has this probability of drawing from [1, max_action_delay_steps];
    otherwise it gets delay=0. Default 1.0 means all joints can have delay.
    """
