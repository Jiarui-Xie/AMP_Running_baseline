# BASE AMP Training — Changes Summary (Xie Jiarui)

2-clip AMP locomotion (walk + run) for G1 29DOF humanoid, modified for the 2026 Beijing Yizhuang Half Marathon Robot Race.

## New / Modified Files

### `source/.../amp/config/g1/g1_amp_base_env_cfg.py`
Environment config for BASE AMP. Subclasses `G1AmpEnvCfg` with:
- Motion data: `base/walk1_subject1` and `base/run1_subject2` (equal weight), plus walk_and_run clips at 0.1 weight each
- SONIC-style actuator model
- Per-joint action delay domain randomization (`max_action_delay_steps=1`, 10% probability)
- Reduced standing command probability (`rel_standing_envs=0.01`)
- Added rewards: `is_alive` (+0.25), `stand_still_joint_deviation` (-0.2), `feet_contact_without_cmd` (+0.5)
- `feet_y_distance`: replaces `feet_too_near`, enforces 15–30cm foot separation
- `elbow_hip_distance`: prevents elbow-hip collision (-10.0)
- `hands_height`: penalizes hands below 70cm (-10.0)
- `feet_landing_vel_z`: penalizes high-velocity landings (-0.5)
- Relaxed `lin_vel_z_l2` (-0.02) to allow natural hip oscillation during running
- Play config: 48 envs, zero velocity commands

### `source/.../amp/config/g1/agents/rsl_rl_ppo_base_cfg.py`
Agent config:
- `experiment_name`: `g1_amp_base`
- `disc_learning_rate`: 5e-6 (prevent disc saturation)
- `grad_penalty_scale`: 25 (stronger disc regularization)
- `task_style_lerp`: 0.6
- `style_reward_scale`: 5.0
- Symmetry: data augmentation + mirror loss (0.1)

### `source/.../data/MotionData/g1_29dof/amp/base/`
Retargeted motion clips: `walk1_subject1.pkl`, `run1_subject2.pkl`.

### `source/.../amp/mdp/rewards.py`
Added reward terms:
- `feet_y_distance()`: penalizes feet outside [min_dist, max_dist] lateral range
- `feet_landing_vel_z()`: penalizes high downward velocity at foot contact
- `body_pair_distance()`: generic minimum-distance penalty between two body groups
- `hands_height()`: penalizes hand links below a height threshold
- `feet_contact_without_cmd()`: rewards grounded feet when velocity command is near zero
- `track_lin_vel_xy_low_speed()`: additional tracking reward at low speeds

### `checkpoints/model_6200.pt`
Baseline checkpoint from the 2026-04-01 training run (6200 iterations).

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| disc_learning_rate | 5e-6 | Halved from default to prevent saturation |
| grad_penalty_scale | 25 | Increased from 15 for stronger disc regularization |
| task_style_lerp | 0.6 | 60% task / 40% style |
| style_reward_scale | 5.0 | |
| max_action_delay_steps | 1 | Communication latency DR |
| loss_type | LSGAN | |

## Training Command

```bash
python scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-G1-BASE-v0 \
    --headless --max_iterations 50000
```

## Play Command

```bash
python scripts/rsl_rl/play.py \
    --task LeggedLab-Isaac-AMP-G1-BASE-Play-v0 \
    --num_envs 48 --video \
    --checkpoint checkpoints/model_6200.pt
```

## Deployment

AMP policy deploys identically to standard RL — feedforward ONNX model. Expects 4-frame observation history (96 dims × 4 = 384 input). No discriminator or motion clips needed at inference.
