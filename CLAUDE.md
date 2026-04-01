# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

All Python commands must run through the Isaac Lab conda environment. Use the wrapper script:

```bash
bash .codex/run-in-env.sh python <script>
bash .codex/run-in-env.sh pytest source/legged_lab/test
bash .codex/run-in-env.sh pre-commit run --all-files
```

Before using the wrapper, `.codex/env.local.sh` must exist (copy from `.codex/env.local.sh.example` and fill in `CODEX_CONDA_ENV` and `ISAACLAB_PATH`). Never guess machine-specific paths.

**Do not use `git worktree`** — the `pip install -e` editable install breaks when the checkout path changes.

Install packages:
```bash
python -m pip install -e source/legged_lab  # main package
python -m pip install -e rsl_rl             # forked rsl_rl (must be feature/amp branch)
```

## Training & Playback

```bash
# Train (BASE 2-clip variant)
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-BASE-v0 --headless --max_iterations 50000

# Train on specific GPU
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-BASE-v0 --headless --device cuda:1 agent.device=cuda:1

# Play / record video
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-BASE-Play-v0 --num_envs 48 --video --checkpoint logs/rsl_rl/.../model_xxx.pt
```

Registered task IDs are in `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/__init__.py`. Adding a new task requires a `gym.register()` entry there.

## Motion Data Pipeline

Raw GMR-format pickles (`{fps, root_pos, root_rot, dof_pos}`) → retargeted `.pkl` via:
```bash
python scripts/tools/retarget/dataset_retarget.py \
    --robot g1 --input_dir temp/gmr_data/ --output_dir temp/lab_data/ \
    --config_file scripts/tools/retarget/config/g1_29dof.yaml --loop clamp
```
Place output under `source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/`. Large assets are stored in Git LFS (`git lfs pull` after cloning).

## Architecture Overview

### Env Hierarchy
```
ManagerBasedRLEnv (IsaacLab)
  └── ManagerBasedAnimationEnv      # adds motion_data_manager + animation_manager
        └── ManagerBasedAmpEnv      # adds AMP-specific step loop, terminal disc obs preview,
                                    # and per-env action delay buffer (comm latency DR)
```
`ManagerBasedAmpEnv.step()` captures discriminator observations *before* reset (`_preview_terminal_obs`) and merges them back into `extras["terminal_obs"]` for terminated envs — this is critical for AMP training correctness.

### Manager Pipeline
Each step: `ActionManager` → physics decimation loop → `AnimationManager.update()` → `TerminationManager` → `RewardManager` → `_reset_idx` → `CommandManager` → `ObservationManager`.

Custom managers added by this repo:
- **`MotionDataManager`** (`managers/motion_data_manager.py`): loads `.pkl` clips recursively, concatenates all frames into flat tensors with `motion_start_indices` for O(1) GPU indexing, interpolates states with `quat_slerp` + `lerp`.
- **`AnimationManager`** (`managers/animation_manager.py`): maintains per-env reference motion time, advances it each step, exposes multi-step lookahead buffers (`num_steps_to_use`) used by discriminator demo observations.
- **`PreviewObservationManager`** (`managers/preview_observation_manager.py`): extends IsaacLab's `ObservationManager` with a non-mutating `preview_group()` call that snapshots history buffers before reset.

### Config Inheritance (G1 AMP)
```
LocomotionAmpEnvCfg (amp_env_cfg.py)
  └── G1AmpEnvCfg (g1_amp_env_cfg.py)   # sets robot, 30-clip walk_and_run dataset, base rewards
        └── G1AmpBaseEnvCfg             # overrides with 2-clip base dataset, tuned rewards
```
Reward classes follow the same pattern: `G1AmpRewards` → `G1AmpBaseRewards`.

Agent configs live alongside env configs in `config/g1/agents/`. The `obs_groups` dict in the runner config maps runner roles (`policy`, `critic`, `discriminator`, `discriminator_demonstration`) to observation group names defined in the env cfg.

### Observation Groups
- **`policy`**: noisy, `history_length=5`, flattened — fed to actor
- **`critic`**: clean (privileged), `history_length=5` — fed to critic only
- **`disc`**: clean, `history_length=AMP_NUM_STEPS=4`, `flatten_history_dim=False` — sequence fed to discriminator
- **`disc_demo`**: reference motion state for same timestep window, also shape `(N, steps, dim)`

### Domain Randomization
`EventCfg` in `amp_env_cfg.py` defines: startup physics material + mass randomization, reset external force, interval push perturbation. Communication delay is handled separately via `ManagerBasedAmpEnvCfg.max_action_delay_steps` — per-env integer delay (uniform in `[0, max]`) applied to actions in `step()`, resampled at every reset.

### Actuator Config (`assets/unitree.py`)
`UNITREE_G1_29DOF_CFG` uses `UnitreeArticulationCfg` which extends `ArticulationCfg` with `joint_sdk_names` — the ordered list of joint names as seen by the hardware SDK, used for deployment mapping. Actuator groups (`N7520-14.3`, `N7520-22.5`, `N5020-16`, `W4010-25`) correspond to physical motor models with per-joint stiffness/damping.

## Code Style

Line limit: 120 chars. Formatter: `black`. Import sorter: `isort` (Black profile, `legged_lab` first-party). Pre-commit also runs `flake8` and `pyupgrade`. All code in English; communicate with the user in their language.
