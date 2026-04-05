"""RSL-RL PPO+AMP config for G1 high-speed training.

Resumes from BASE checkpoint. The HS speed range (0.5–2.5) is a subset of
BASE (-0.5–3.0) so the discriminator remains stable — no need to reinit.
"""

from isaaclab.utils import configclass

from .rsl_rl_ppo_base_cfg import G1RslRlOnPolicyRunnerAmpBaseCfg


@configclass
class G1RslRlOnPolicyRunnerAmpBaseHSCfg(G1RslRlOnPolicyRunnerAmpBaseCfg):
    experiment_name = "g1_amp_base_hs"
    resume = True
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
