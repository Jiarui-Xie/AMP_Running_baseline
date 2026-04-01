"""RSL-RL PPO+AMP config for G1 high-speed post-training.

Post-trains from the BASE 2-clip checkpoint with reduced style weighting
so the policy focuses more on high-speed task performance while retaining
human-like motion style (task_style_lerp=0.25 → 75% task / 25% style).
"""

from isaaclab.utils import configclass
from legged_lab.rsl_rl import RslRlAmpCfg

from .rsl_rl_ppo_base_cfg import G1RslRlOnPolicyRunnerAmpBaseCfg


@configclass
class G1RslRlOnPolicyRunnerAmpBaseHSCfg(G1RslRlOnPolicyRunnerAmpBaseCfg):
    experiment_name = "g1_amp_base_hs"
    resume = True
    load_run = "g1_amp_base"
    load_checkpoint = "model_6200.pt"

    def __post_init__(self):
        # Reduce style influence: 75% task / 25% style (BASE was 0.6)
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 0.25
