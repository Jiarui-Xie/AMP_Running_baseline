"""RSL-RL PPO+AMP config for G1 high-speed training.

Same hyperparameters as BASE (task_style_lerp=0.6). The speed range is only
moderately above BASE (2.5 vs 3.0 max), so the demo clips still provide
useful style signal — no need to weaken or reinit the discriminator.
"""

from isaaclab.utils import configclass

from .rsl_rl_ppo_base_cfg import G1RslRlOnPolicyRunnerAmpBaseCfg


@configclass
class G1RslRlOnPolicyRunnerAmpBaseHSCfg(G1RslRlOnPolicyRunnerAmpBaseCfg):
    experiment_name = "g1_amp_base_hs"
