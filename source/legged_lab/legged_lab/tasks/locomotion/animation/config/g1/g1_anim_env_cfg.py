import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.animation.mdp as mdp

from legged_lab.managers import AnimationTermCfg as AnimTerm
from legged_lab.managers import MotionDataTermCfg as MotionDataTerm

from legged_lab.tasks.locomotion.animation.animation_env_cfg import AnimationEnvCfg
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG


@configclass
class G1AnimEnvCfg(AnimationEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot_anim = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_anim")
        self.scene.robot_anim.spawn.rigid_props.disable_gravity = True # type: ignore
        self.scene.robot_anim.spawn.articulation_props.enabled_self_collisions = False # type: ignore
        self.scene.robot_anim.spawn.activate_contact_sensors = False # type: ignore
        self.scene.robot_anim.spawn.collision_props = sim_utils.CollisionPropertiesCfg( # type: ignore
            collision_enabled=False
        ) 
        
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "walk"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            "B15_-__Walk_turn_around_stageii": 1.0, 
            "B22_-__side_step_left_stageii": 1.0, 
            "B23_-__side_step_right_stageii": 1.0, 
            "B10_-__Walk_turn_left_45_stageii": 1.0,
        }
        # self.motion_data.motion_dataset.key_link_names = [
        #     "left_ankle_roll_link",
        #     "right_ankle_roll_link", 
        #     "left_wrist_yaw_link",
        #     "right_wrist_yaw_link",
        # ]
        
        self.animation.animation.random_initialize = True
        self.animation.animation.num_steps_to_use = 10








