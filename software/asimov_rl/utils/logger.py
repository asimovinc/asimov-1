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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        return
        self.plot_process = Process(target=self._plot)
        self.plot_process1 = Process(target=self._plot_position)
        # self.plot_process2 = Process(target=self._plot_torque)
        # self.plot_process3 = Process(target=self._plot_vel)
        # self.plot_process4 = Process(target=self._plot_tn_rms)
        # self.plot_process5 = Process(target=self._plot_tn)
        # self.plot_process6 = Process(target=self._plot_torque_vel)
        self.plot_process7 = Process(target=self._plot_position1)
        # self.plot_process8 = Process(target=self._plot_torque1)
        # self.plot_process9 = Process(target=self._plot_vel1)
        # self.plot_process10 = Process(target=self._plot_tn_rms1)
        # self.plot_process11 = Process(target=self._plot_tn1)
        # self.plot_process12 = Process(target=self._plot_torque_vel1)
        self.plot_process.start()
        self.plot_process1.start()
        # self.plot_process2.start()
        # self.plot_process3.start()
        # self.plot_process4.start()
        # self.plot_process5.start()
        # self.plot_process6.start()
        self.plot_process7.start()
        # self.plot_process8.start()
        # self.plot_process9.start()
        # self.plot_process10.start()
        # self.plot_process11.start()
        # self.plot_process12.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        if log["command_sin"]: a.plot(time, log["command_sin"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        if log["command_cos"]: a.plot(time, log["command_cos"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # # plot base vel z
        # a = axs[1, 0]
        # if log["command_sin"]: a.plot(time, log["command_sin"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='command sin', title='Command Sin')
        # a.legend()
        # # plot contact forces
        # a = axs[1, 1]
        # if log["command_cos"]: a.plot(time, log["command_cos"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='command cos', title='Command Cos')
        # a.legend()
        plt.show()

    def _plot_position(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_pos[0]"]: a.plot(time, log["dof_pos[0]"], label='measured')
        if log["dof_pos_target[0]"]: a.plot(time, log["dof_pos_target[0]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[0]')
        a.legend()

        a = axs[0, 1]
        if log["dof_pos[1]"]: a.plot(time, log["dof_pos[1]"], label='measured')
        if log["dof_pos_target[1]"]: a.plot(time, log["dof_pos_target[1]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[1]')
        a.legend()

        a = axs[0, 2]
        if log["dof_pos[2]"]: a.plot(time, log["dof_pos[2]"], label='measured')
        if log["dof_pos_target[2]"]: a.plot(time, log["dof_pos_target[2]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[2]')
        a.legend()

        a = axs[1, 0]
        if log["dof_pos[3]"]: a.plot(time, log["dof_pos[3]"], label='measured')
        if log["dof_pos_target[3]"]: a.plot(time, log["dof_pos_target[3]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[3]')
        a.legend()

        a = axs[1, 1]
        if log["dof_pos[4]"]: a.plot(time, log["dof_pos[4]"], label='measured')
        if log["dof_pos_target[4]"]: a.plot(time, log["dof_pos_target[4]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[4]')
        a.legend()

        a = axs[1, 2]
        if log["dof_pos[5]"]: a.plot(time, log["dof_pos[5]"], label='measured')
        if log["dof_pos_target[5]"]: a.plot(time, log["dof_pos_target[5]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[5]')
        a.legend()
        plt.show()

    def _plot_position1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_pos[6]"]: a.plot(time, log["dof_pos[6]"], label='measured')
        if log["dof_pos_target[6]"]: a.plot(time, log["dof_pos_target[6]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[6]')
        a.legend()

        a = axs[0, 1]
        if log["dof_pos[7]"]: a.plot(time, log["dof_pos[7]"], label='measured')
        if log["dof_pos_target[7]"]: a.plot(time, log["dof_pos_target[7]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[7]')
        a.legend()

        a = axs[0, 2]
        if log["dof_pos[8]"]: a.plot(time, log["dof_pos[8]"], label='measured')
        if log["dof_pos_target[8]"]: a.plot(time, log["dof_pos_target[8]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[8]')
        a.legend()

        a = axs[1, 0]
        if log["dof_pos[9]"]: a.plot(time, log["dof_pos[9]"], label='measured')
        if log["dof_pos_target[9]"]: a.plot(time, log["dof_pos_target[9]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[9]')
        a.legend()

        a = axs[1, 1]
        if log["dof_pos[10]"]: a.plot(time, log["dof_pos[10]"], label='measured')
        if log["dof_pos_target[10]"]: a.plot(time, log["dof_pos_target[10]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[10]')
        a.legend()

        a = axs[1, 2]
        if log["dof_pos[11]"]: a.plot(time, log["dof_pos[11]"], label='measured')
        if log["dof_pos_target[11]"]: a.plot(time, log["dof_pos_target[11]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[11]')
        a.legend()
        plt.show()

    def _plot_torque(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque[0]"]!=[]: a.plot(time, log["dof_torque[0]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[0]')
        a.legend()
        a = axs[0,1]
        if log["dof_torque[1]"]!=[]: a.plot(time, log["dof_torque[1]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[2]"]!=[]: a.plot(time, log["dof_torque[2]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[3]"]!=[]: a.plot(time, log["dof_torque[3]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[4]"]!=[]: a.plot(time, log["dof_torque[4]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[5]"]!=[]: a.plot(time, log["dof_torque[5]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[5]')
        a.legend()
        plt.show()

    def _plot_torque1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque[6]"]!=[]: a.plot(time, log["dof_torque[6]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[6]')
        a.legend()
        a = axs[0,1]
        if log["dof_torque[7]"]!=[]: a.plot(time, log["dof_torque[7]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[8]"]!=[]: a.plot(time, log["dof_torque[8]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[9]"]!=[]: a.plot(time, log["dof_torque[9]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[10]"]!=[]: a.plot(time, log["dof_torque[10]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[11]"]!=[]: a.plot(time, log["dof_torque[11]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque[11]')
        a.legend()
        plt.show()

    def _plot_vel(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_vel[0]"]: a.plot(time, log["dof_vel[0]"], label='measured')
        if log["dof_vel_target[0]"]: a.plot(time, log["dof_vel_target[0]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[0]')
        a = axs[0,1]
        a.legend()
        if log["dof_vel[1]"]: a.plot(time, log["dof_vel[1]"], label='measured')
        if log["dof_vel_target[1]"]: a.plot(time, log["dof_vel_target[1]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_vel[2]"]: a.plot(time, log["dof_vel[2]"], label='measured')
        if log["dof_vel_target[2]"]: a.plot(time, log["dof_vel_target[2]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_vel[3]"]: a.plot(time, log["dof_vel[3]"], label='measured')
        if log["dof_vel_target[3]"]: a.plot(time, log["dof_vel_target[3]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_vel[4]"]: a.plot(time, log["dof_vel[4]"], label='measured')
        if log["dof_vel_target[4]"]: a.plot(time, log["dof_vel_target[4]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_vel[5]"]: a.plot(time, log["dof_vel[5]"], label='measured')
        if log["dof_vel_target[5]"]: a.plot(time, log["dof_vel_target[5]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[5]')
        a.legend()
        plt.show()

    def _plot_vel1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_vel[6]"]: a.plot(time, log["dof_vel[6]"], label='measured')
        if log["dof_vel_target[6]"]: a.plot(time, log["dof_vel_target[6]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[6]')
        a = axs[0,1]
        a.legend()
        if log["dof_vel[7]"]: a.plot(time, log["dof_vel[7]"], label='measured')
        if log["dof_vel_target[7]"]: a.plot(time, log["dof_vel_target[7]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_vel[8]"]: a.plot(time, log["dof_vel[8]"], label='measured')
        if log["dof_vel_target[8]"]: a.plot(time, log["dof_vel_target[8]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_vel[9]"]: a.plot(time, log["dof_vel[9]"], label='measured')
        if log["dof_vel_target[9]"]: a.plot(time, log["dof_vel_target[9]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_vel[10]"]: a.plot(time, log["dof_vel[10]"], label='measured')
        if log["dof_vel_target[4]"]: a.plot(time, log["dof_vel_target[4]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_vel[11]"]: a.plot(time, log["dof_vel[11]"], label='measured')
        if log["dof_vel_target[5]"]: a.plot(time, log["dof_vel_target[11]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[11]')
        a.legend()
        plt.show()

    def _plot_tn_rms(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        a = axs[0, 0]
        if log["dof_torque[0]"] != [] and log["dof_vel[0]"] != []:
            vel_array = np.array(log["dof_vel[0]"])
            torque_array = np.array(log["dof_torque[0]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[0]')
        a.legend()

        a = axs[0,1]
        if log["dof_torque[1]"] != [] and log["dof_vel[1]"] != []:
            vel_array = np.array(log["dof_vel[1]"])
            torque_array = np.array(log["dof_torque[1]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[1]"] != [] and log["dof_vel[1]"] != []:
            vel_array = np.array(log["dof_vel[1]"])
            torque_array = np.array(log["dof_torque[1]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[3]"] != [] and log["dof_vel[3]"] != []:
            vel_array = np.array(log["dof_vel[3]"])
            torque_array = np.array(log["dof_torque[3]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[4]"] != [] and log["dof_vel[4]"] != []:
            vel_array = np.array(log["dof_vel[4]"])
            torque_array = np.array(log["dof_torque[4]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[5]"] != [] and log["dof_vel[5]"] != []:
            vel_array = np.array(log["dof_vel[5]"])
            torque_array = np.array(log["dof_torque[5]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[5]')
        a.legend()
        plt.show()

    def _plot_tn_rms1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        a = axs[0, 0]
        if log["dof_torque[6]"] != [] and log["dof_vel[6]"] != []:
            vel_array = np.array(log["dof_vel[6]"])
            torque_array = np.array(log["dof_torque[6]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[6]')
        a.legend()
        a = axs[0,1]
        if log["dof_torque[7]"] != [] and log["dof_vel[7]"] != []:
            vel_array = np.array(log["dof_vel[7]"])
            torque_array = np.array(log["dof_torque[7]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[8]"] != [] and log["dof_vel[8]"] != []:
            vel_array = np.array(log["dof_vel[8]"])
            torque_array = np.array(log["dof_torque[8]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[9]"] != [] and log["dof_vel[9]"] != []:
            vel_array = np.array(log["dof_vel[9]"])
            torque_array = np.array(log["dof_torque[9]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[10]"] != [] and log["dof_vel[10]"] != []:
            vel_array = np.array(log["dof_vel[10]"])
            torque_array = np.array(log["dof_torque[10]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[11]"] != [] and log["dof_vel[11]"] != []:
            vel_array = np.array(log["dof_vel[11]"])
            torque_array = np.array(log["dof_torque[11]"])

            rms_vel = np.sqrt(np.mean(vel_array**2))
            rms_torque = np.sqrt(np.mean(torque_array**2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[11]')
        a.legend()
        plt.show()

    def _plot_tn(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        a = axs[0, 0]
        if log["dof_torque[0]"] != [] and log["dof_vel[0]"] != []:
            vel_array = np.array(log["dof_vel[0]"])
            torque_array = np.array(log["dof_torque[0]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[0]')
        a.legend()

        a = axs[0,1]
        if log["dof_torque[1]"] != [] and log["dof_vel[1]"] != []:
            vel_array = np.array(log["dof_vel[1]"])
            torque_array = np.array(log["dof_torque[1]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[1]"] != [] and log["dof_vel[1]"] != []:
            vel_array = np.array(log["dof_vel[1]"])
            torque_array = np.array(log["dof_torque[1]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[3]"] != [] and log["dof_vel[3]"] != []:
            vel_array = np.array(log["dof_vel[3]"])
            torque_array = np.array(log["dof_torque[3]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[4]"] != [] and log["dof_vel[4]"] != []:
            vel_array = np.array(log["dof_vel[4]"])
            torque_array = np.array(log["dof_torque[4]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[5]"] != [] and log["dof_vel[5]"] != []:
            vel_array = np.array(log["dof_vel[5]"])
            torque_array = np.array(log["dof_torque[5]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[5]')
        a.legend()
        plt.show()

    def _plot_tn1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        a = axs[0, 0]
        if log["dof_torque[6]"] != [] and log["dof_vel[6]"] != []:
            vel_array = np.array(log["dof_vel[6]"])
            torque_array = np.array(log["dof_torque[6]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[6]')
        a.legend()
        a = axs[0,1]
        if log["dof_torque[7]"] != [] and log["dof_vel[7]"] != []:
            vel_array = np.array(log["dof_vel[7]"])
            torque_array = np.array(log["dof_torque[7]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[8]"] != [] and log["dof_vel[8]"] != []:
            vel_array = np.array(log["dof_vel[8]"])
            torque_array = np.array(log["dof_torque[8]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[9]"] != [] and log["dof_vel[9]"] != []:
            vel_array = np.array(log["dof_vel[9]"])
            torque_array = np.array(log["dof_torque[9]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[10]"] != [] and log["dof_vel[10]"] != []:
            vel_array = np.array(log["dof_vel[10]"])
            torque_array = np.array(log["dof_torque[10]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[11]"] != [] and log["dof_vel[11]"] != []:
            vel_array = np.array(log["dof_vel[11]"])
            torque_array = np.array(log["dof_torque[11]"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[11]')
        a.legend()
        plt.show()

    def _plot_torque_vel(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque[0]"]!=[]: a.plot(time, log["dof_torque[0]"], label='measured_torque')
        if log["dof_vel[0]"]: a.plot(time, log["dof_vel[0]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[0]')
        a.legend()
        a = axs[0,1]
        if log["dof_torque[1]"]!=[]: a.plot(time, log["dof_torque[1]"], label='measured_torque')
        if log["dof_vel[1]"]: a.plot(time, log["dof_vel[1]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[2]"]!=[]: a.plot(time, log["dof_torque[2]"], label='measured_torque')
        if log["dof_vel[2]"]: a.plot(time, log["dof_vel[2]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[3]"]!=[]: a.plot(time, log["dof_torque[3]"], label='measured_torque')
        if log["dof_vel[3]"]: a.plot(time, log["dof_vel[3]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[4]"]!=[]: a.plot(time, log["dof_torque[4]"], label='measured_torque')
        if log["dof_vel[4]"]: a.plot(time, log["dof_vel[4]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[5]"]!=[]: a.plot(time, log["dof_torque[5]"], label='measured_torque')
        if log["dof_vel[5]"]: a.plot(time, log["dof_vel[5]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[5]')
        a.legend()
        plt.show()

    def _plot_torque_vel1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque[6]"]!=[]: a.plot(time, log["dof_torque[6]"], label='measured_torque')
        if log["dof_vel[6]"]: a.plot(time, log["dof_vel[6]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[6]')
        a.legend()
        a = axs[0,1]
        if log["dof_torque[7]"]!=[]: a.plot(time, log["dof_torque[7]"], label='measured_torque')
        if log["dof_vel[7]"]: a.plot(time, log["dof_vel[7]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[8]"]!=[]: a.plot(time, log["dof_torque[8]"], label='measured_torque')
        if log["dof_vel[8]"]: a.plot(time, log["dof_vel[8]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[9]"]!=[]: a.plot(time, log["dof_torque[9]"], label='measured_torque')
        if log["dof_vel[9]"]: a.plot(time, log["dof_vel[9]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[10]"]!=[]: a.plot(time, log["dof_torque[10]"], label='measured_torque')
        if log["dof_vel[10]"]: a.plot(time, log["dof_vel[10]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[11]"]!=[]: a.plot(time, log["dof_torque[11]"], label='measured_torque')
        if log["dof_vel[11]"]: a.plot(time, log["dof_vel[11]"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity Torque[11]')
        a.legend()
        plt.show()


    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()