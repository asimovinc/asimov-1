from asimov_rl import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .asimov.asimov_stand_config import AsimovStandCfg, AsimovStandCfgPPO
from .asimov.asimov_stand_env import AsimovStandEnv

from asimov_rl.utils.task_registry import task_registry

task_registry.register("asimov_stand", AsimovStandEnv, AsimovStandCfg(), AsimovStandCfgPPO())
