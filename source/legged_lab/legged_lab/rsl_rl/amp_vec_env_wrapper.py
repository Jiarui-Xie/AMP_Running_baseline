"""Wrapper that adapts isaaclab's RslRlVecEnvWrapper for the AMP RSL-RL fork.

The AMP fork expects get_observations() and step() to return TensorDict (the full
obs_dict with all observation groups), while isaaclab's wrapper returns
(policy_tensor, extras_dict). This wrapper bridges the two interfaces.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class AmpVecEnvWrapper(RslRlVecEnvWrapper):
    """VecEnv wrapper for AMP RSL-RL fork compatibility.

    Overrides get_observations(), reset(), and step() to return the full
    observation dict (as TensorDict) instead of just the policy tensor.
    """

    def _obs_dict_to_tensordict(self, obs_dict: dict) -> TensorDict:
        """Convert nested obs dict to TensorDict."""
        flat = {}
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                flat[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}/{sub_key}"] = sub_value
            else:
                flat[key] = value
        return TensorDict(flat, batch_size=[])

    def get_observations(self) -> TensorDict:
        """Returns all observation groups as a TensorDict."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._obs_dict_to_tensordict(obs_dict)

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, info = self.env.reset()
        return self._obs_dict_to_tensordict(obs_dict), info

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # keep full obs dict in extras for compatibility
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._obs_dict_to_tensordict(obs_dict), rew, dones, extras
