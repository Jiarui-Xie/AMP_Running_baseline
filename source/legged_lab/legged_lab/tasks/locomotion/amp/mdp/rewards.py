from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet being closer than a threshold distance (prevents crossed legs)."""
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_distance_l1(
    env: ManagerBasedRLEnv,
    min_dist: float = 0.18,
    max_dist: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear penalty when horizontal foot spacing is outside [min_dist, max_dist]."""
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # xy only
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    too_close = (min_dist - distance).clamp(min=0)
    too_far = (distance - max_dist).clamp(min=0)
    return too_close + too_far


def feet_y_distance(
    env: ManagerBasedRLEnv,
    min_dist: float = 0.15,
    max_dist: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear penalty when foot spacing along the robot's local Y-axis (hip-width) is outside [min_dist, max_dist].

    Projects the foot-to-foot vector onto the robot base's local Y-axis so the penalty
    is invariant to heading and only measures lateral (hip-width) separation.
    """
    asset = env.scene[asset_cfg.name]
    # feet world positions: (N, 2, 3) — expects exactly 2 body_ids
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    diff = feet_pos[:, 0] - feet_pos[:, 1]  # (N, 3)
    # robot base quaternion (w, x, y, z)
    base_quat = asset.data.root_quat_w  # (N, 4)
    # project diff into robot local frame
    local_diff = math_utils.quat_apply_inverse(base_quat, diff)  # (N, 3)
    y_dist = local_diff[:, 1].abs()  # lateral distance
    too_close = (min_dist - y_dist).clamp(min=0)
    too_far = (y_dist - max_dist).clamp(min=0)
    return too_close + too_far


def feet_landing_vel_z(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.10,
    base_vel_limit: float = 0.5,
    vel_scale_with_cmd: float = 0.3,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize downward foot Z-velocity near ground and at the moment of landing.

    Two complementary triggers (OR logic):
      1. Height-based: foot height < height_threshold AND descending
      2. Contact-based: foot just landed this step (was in air, now in contact)

    The allowed impact velocity scales with commanded speed:
        allowed_vel = base_vel_limit + vel_scale_with_cmd * ||cmd_xy||

    Args:
        height_threshold: foot height (m) below which the height trigger activates.
        base_vel_limit: baseline allowed downward velocity (m/s) at zero command speed.
        vel_scale_with_cmd: extra allowed velocity per unit of command speed (m/s per m/s).
        command_name: velocity command manager name.
        sensor_cfg: contact sensor with body_ids for feet.
        asset_cfg: robot asset with body_ids for feet (expects 2 feet).
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, 2)
    feet_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # (N, 2)

    # command speed
    cmd = env.command_manager.get_command(command_name)[:, :2]  # (N, 2)
    cmd_speed = torch.norm(cmd, dim=1, keepdim=True)  # (N, 1)
    allowed_vel = base_vel_limit + vel_scale_with_cmd * cmd_speed  # (N, 1)

    # trigger 1: height-based (foot low AND descending)
    near_ground = feet_pos_z < height_threshold  # (N, 2)
    descending = feet_vel_z < 0  # (N, 2)
    height_trigger = near_ground & descending

    # trigger 2: just landed (contact_time > 0 but very small, meaning first contact step)
    # contact_time resets to 0 when in air; on first contact frame it equals dt
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # (N, 2)
    landing_frames = 2
    just_landed = contact_time < landing_frames * env.step_dt  # within first 2 control steps
    in_contact = contact_time > 0
    landing_trigger = just_landed & in_contact

    # penalize if either trigger fires
    active = height_trigger | landing_trigger  # (N, 2)
    excess = (-feet_vel_z - allowed_vel).clamp(min=0)  # (N, 2)
    penalty = excess.square() * active  # (N, 2)  quadratic shaping
    return penalty.sum(dim=1)  # (N,)


def hands_height(
    env: ManagerBasedRLEnv,
    min_height: float = 0.70,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize hands dropping below a minimum world-frame height (quadratic).

    Returns sum of squared deficit over both hands for a stronger gradient as
    hands drop further below the threshold.
    """
    asset = env.scene[asset_cfg.name]
    hands_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, 2)
    deficit = (min_height - hands_z).clamp(min=0)  # (N, 2)
    return deficit.square().sum(dim=1)  # (N,)


def body_pair_distance(
    env: ManagerBasedRLEnv,
    min_dist: float = 0.15,
    asset_cfg_a: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg_b: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize paired bodies being closer than *min_dist* (quadratic).

    ``asset_cfg_a`` and ``asset_cfg_b`` must resolve to the same number of
    body IDs.  Distances are computed element-wise between matching pairs
    (e.g. left_elbow↔left_hip, right_elbow↔right_hip).
    """
    asset_a = env.scene[asset_cfg_a.name]
    asset_b = env.scene[asset_cfg_b.name]
    pos_a = asset_a.data.body_pos_w[:, asset_cfg_a.body_ids, :]  # (N, P, 3)
    pos_b = asset_b.data.body_pos_w[:, asset_cfg_b.body_ids, :]  # (N, P, 3)
    dist = torch.norm(pos_a - pos_b, dim=-1)  # (N, P)
    deficit = (min_dist - dist).clamp(min=0)  # (N, P)
    return deficit.square().sum(dim=1)  # (N,)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Reward both feet being in contact when velocity command is near zero (stand still)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def track_lin_vel_xy_low_speed(
    env: ManagerBasedRLEnv, std: float, command_name: str, max_speed: float = 1.0, min_speed: float = 0.0
) -> torch.Tensor:
    """Extra reward for tracking linear velocity at low speeds, decaying linearly as speed increases up to max_speed.

    Only active when command speed > min_speed to avoid rewarding standing still.
    """
    base_reward = mdp.track_lin_vel_xy_exp(env, std=std, command_name=command_name)
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.norm(command, dim=1)
    decay_weight = torch.clamp(1.0 - (command_norm / max_speed), min=0.0)
    active_mask = (command_norm > min_speed).float()
    return base_reward * decay_weight * active_mask
