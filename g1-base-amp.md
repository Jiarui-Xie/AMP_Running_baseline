# AMP 3-Clip Training Changes Summary

2-clip AMP locomotion (walk + run) for G1 29DOF humanoid using legged_lab.

## New Files

### `scripts/tools/retarget/csv_to_gmr_pkl.py`
Converter: LaFan1 CSV (36 cols) -> GMR-format pickle (`{fps, root_pos, root_rot, dof_pos}`).

### `source/.../amp/config/g1/g1_amp_3clip_env_cfg.py`
Environment config for 2-clip AMP. Subclasses `G1AmpEnvCfg` with:
- Motion data: only `walk1_subject1` and `run1_subject2` (equal weight)
- `track_lin_vel_xy_exp` weight: 1.0 -> **2.0** (boost velocity tracking)
- `joint_deviation_arms` weight: -0.05 -> **-0.01** (let AMP control arm style)
- Added rewards: `is_alive` (+0.5), `stand_still_joint_deviation` (-0.2), `feet_contact_without_cmd` (+0.5)
- Play config with 48 envs, forward-only velocity commands

### `source/.../amp/config/g1/agents/rsl_rl_ppo_3clip_cfg.py`
Agent config tuned for disc stability:
- `disc_learning_rate`: 1e-5 -> **5e-6** (prevent disc saturation)
- `grad_penalty_scale`: 15 -> **25** (keep disc uncertain)
- `task_style_lerp`: 0.5 (50/50 task/style blend)
- `style_reward_scale`: 5.0
- Symmetry: data augmentation + mirror loss (0.1)

### `source/.../data/MotionData/g1_29dof/amp/3clip/*.pkl`
Retargeted motion clips: `walk1_subject1.pkl`, `run1_subject2.pkl`, `fallAndGetUp1_subject1.pkl` (fall clip unused in current config).

### `source/.../rsl_rl/amp_vec_env_wrapper.py`
Bridges isaaclab's tuple env interface with AMPRunner's TensorDict expectation.

## Modified Files

### `scripts/rsl_rl/train.py`
- Import fix: `RslRlBaseRunnerCfg` -> `RslRlOnPolicyRunnerCfg`
- Added AMP wrapper branch: uses `AmpVecEnvWrapper` when `class_name == "AMPRunner"`

### `scripts/rsl_rl/play.py`
- Same import fix and AMP wrapper branch as train.py
- Added AMPRunner support for loading checkpoints

### `scripts/rsl_rl/cli_args.py`
- Import fix: `RslRlBaseRunnerCfg` -> `RslRlOnPolicyRunnerCfg`

### `source/.../envs/manager_based_amp_env.py`
- Removed `record_post_physics_decimation_step()` call (method doesn't exist in this version)

### `source/.../amp/config/g1/__init__.py`
- Registered two new gym tasks:
  - `LeggedLab-Isaac-AMP-G1-3Clip-v0` (training)
  - `LeggedLab-Isaac-AMP-G1-3Clip-Play-v0` (evaluation)

### `source/.../amp/config/g1/agents/rsl_rl_ppo_cfg.py`
- Removed `actor_obs_normalization=False` and `critic_obs_normalization=False` (not supported by this rsl_rl version)

### `source/.../amp/mdp/rewards.py`
- Added `feet_too_near()`: penalizes feet closer than threshold (prevents crossed legs)
- Added `feet_contact_without_cmd()`: rewards both feet grounded when velocity command is near zero

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| disc_learning_rate | 5e-6 | Halved from default to prevent saturation |
| grad_penalty_scale | 25 | Increased from 15 for stronger disc regularization |
| task_style_lerp | 0.5 | Equal task/style weight |
| style_reward_scale | 5.0 | |
| track_lin_vel_xy weight | 2.0 | Doubled to break velocity tracking plateau |
| joint_deviation_arms weight | -0.01 | Reduced 5x to let disc control arm style |
| num_envs | 4096 | RTX 4090 24GB |
| loss_type | LSGAN | |

## Training Command

```bash
cd /root/gpufree-data/legged_lab && python scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-G1-3Clip-v0 \
    --headless --max_iterations 50000 --num_envs 4096
```

## Play Command

```bash
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-3Clip-Play-v0 --num_envs 48
```

## Deployment

AMP policy deploys identically to standard RL — just a feedforward ONNX model. Only difference: expects 4-frame observation history (96 dims x 4 = 384 input). No discriminator or motion clips needed at inference.
