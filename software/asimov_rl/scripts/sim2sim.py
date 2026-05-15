# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

import math
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from asimov_rl import LEGGED_GYM_ROOT_DIR
from asimov_rl.envs import *
from asimov_rl.utils import  Logger
import torch
import pygame
from asimov_rl.utils.helpers import get_load_path
import os
import time

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
joystick_use = False
joystick_opened = False
joystick = None

if joystick_use:
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    # Asimov MJCF sensor names: imu_quat (w,x,y,z), imu_ang_vel, imu_lin_vel.
    # scipy expects (x,y,z,w), so reorder [1,2,3,0].
    quat = data.sensor('imu_quat').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('imu_ang_vel').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    foot_positions = []
    foot_forces = []
    base_pos = None
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name is None:
            continue
        if 'ankle_roll' in body_name:
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double))
        if 'pelvis' in body_name:
            base_pos = data.xpos[i][:3].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q ) * kp - dq * kd
    return torque_out


def run_mujoco(policy, cfg, env_cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    print("Load mujoco xml from:", cfg.sim_config.mujoco_model_path)
    # load model xml
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    # simulation timestep
    model.opt.timestep = cfg.sim_config.dt
    # model data
    data = mujoco.MjData(model)
    num_actuated_joints = env_cfg.env.num_actions  # This should match the number of actuated joints in your model
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos


    mujoco.mj_step(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)
    target_q = np.zeros((env_cfg.env.num_actions), dtype=np.double)
    action = np.zeros((env_cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(env_cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, env_cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)

    stop_state_log = 40000

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
        # Handle joystick input in main thread (required for macOS)
        global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        if joystick_opened and joystick is not None:
            pygame.event.get()
            x_vel_cmd = -joystick.get_axis(1) * 1
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(2) * 1

        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
        q = q[-env_cfg.env.num_actions:]
        dq = dq[-env_cfg.env.num_actions:]

        base_z = base_pos[2]
        foot_z = foot_positions
        foot_force_z = foot_forces
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            ####### for stand only #######
            if hasattr(env_cfg.commands,"sw_switch"):
                vel_norm = np.sqrt(x_vel_cmd**2 + y_vel_cmd**2 + yaw_vel_cmd**2)
                if env_cfg.commands.sw_switch and vel_norm <= env_cfg.commands.stand_com_threshold:
                    count_lowlevel = 0

            obs = np.zeros([1, env_cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            if env_cfg.env.num_commands == 5:
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / env_cfg.rewards.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / env_cfg.rewards.cycle_time)
                obs[0, 2] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            if env_cfg.env.num_commands == 3:
                obs[0, 0] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            obs[0, env_cfg.env.num_commands:env_cfg.env.num_commands+env_cfg.env.num_actions] = (q - cfg.robot_config.default_dof_pos) * env_cfg.normalization.obs_scales.dof_pos
            obs[0, env_cfg.env.num_commands+env_cfg.env.num_actions:env_cfg.env.num_commands+2*env_cfg.env.num_actions] = dq * env_cfg.normalization.obs_scales.dof_vel
            obs[0, env_cfg.env.num_commands+2*env_cfg.env.num_actions:env_cfg.env.num_commands+3*env_cfg.env.num_actions] = action
            obs[0, env_cfg.env.num_commands+3*env_cfg.env.num_actions:env_cfg.env.num_commands+3*env_cfg.env.num_actions+3] = omega
            obs[0, env_cfg.env.num_commands+3*env_cfg.env.num_actions+3:env_cfg.env.num_commands+3*env_cfg.env.num_actions+6] = eu_ang

            ####### for stand only #######
            if env_cfg.env.add_stand_bool:
                vel_norm = np.sqrt(x_vel_cmd**2 + y_vel_cmd**2 + yaw_vel_cmd**2)
                stand_command = (vel_norm <= env_cfg.commands.stand_com_threshold)
                obs[0, -1] = stand_command

            print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

            obs = np.clip(obs, -env_cfg.normalization.clip_observations, env_cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
            for i in range(env_cfg.env.frame_stack):
                policy_input[0, i * env_cfg.env.num_single_obs : (i + 1) * env_cfg.env.num_single_obs] = hist_obs[i][0, :]

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)
            target_q = action * env_cfg.control.action_scale

        target_dq = np.zeros((env_cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        data.ctrl = tau
        applied_tau = data.actuator_force

        mujoco.mj_step(model, data)
        viewer.sync()
        if not viewer.is_running():
            break

        count_lowlevel += 1
        idx = 5
        dof_pos_target = target_q + cfg.robot_config.default_dof_pos
        if _ < stop_state_log:
            dict = {
                    'base_height': base_z,
                    'foot_z_l': foot_z[0],
                    'foot_z_r': foot_z[1],
                    'foot_forcez_l': foot_force_z[0],
                    'foot_forcez_r': foot_force_z[1],
                    'base_vel_x': v[0],
                    'command_x': x_vel_cmd,
                    'base_vel_y': v[1],
                    'command_y': y_vel_cmd,
                    'base_vel_z': v[2],
                    'base_vel_yaw': omega[2],
                    'command_yaw': yaw_vel_cmd,
                    'dof_pos_target': dof_pos_target[idx],
                    'dof_pos': q[idx],
                    'dof_vel': dq[idx],
                    'dof_torque': applied_tau[idx],
                    'cmd_dof_torque': tau[idx],
                }

            # add dof_pos_target
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos_target[{i}]'] = dof_pos_target[i].item()

            # add dof_pos
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos[{i}]'] = q[i].item()

            # add dof_torque
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_torque[{i}]'] = applied_tau[i].item()

            # add dof_vel
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_vel[{i}]'] = dq[i].item()
            logger.log_states(dict=dict)

        elif _== stop_state_log:
            logger.plot_states()

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str,
                        help='Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.')
    parser.add_argument('--task', type=str, required=True,
                        help='task name.')
    args = parser.parse_args()
    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    class Sim2simCfg():

        class sim_config:
            mujoco_model_path = env_cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            sim_duration = 100.0
            dt = 0.001
            decimation = 10

        class robot_config:
            # get PD gain
            kps = np.array([env_cfg.control.stiffness[joint] for joint in env_cfg.control.stiffness.keys()]*2, dtype=np.double)
            kds = np.array([env_cfg.control.damping[joint] for joint in env_cfg.control.damping.keys()]*2, dtype=np.double)

            tau_limit = 500. * np.ones(env_cfg.env.num_actions, dtype=np.double)  # 定义关节力矩的限制

            default_dof_pos = np.array(list(env_cfg.init_state.default_joint_angles.values()))

    # load model
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, 'exported_policies')
    if args.load_model == None:
        jit_path = os.listdir(root_path)
        jit_path.sort()
        model_path = os.path.join(root_path, jit_path[-1])
    else:
        model_path = os.path.join(root_path, args.load_model)
    jit_name = os.listdir(model_path)
    model_path = os.path.join(model_path,jit_name[-1])
    policy = torch.jit.load(model_path)
    print("Load model from:", model_path)

    run_mujoco(policy, Sim2simCfg(), env_cfg)
