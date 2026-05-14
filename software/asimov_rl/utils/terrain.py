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


import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from asimov_rl.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        # each sub terrain length
        self.env_length = cfg.terrain_length
        # each sub terrain width
        self.env_width = cfg.terrain_width
        # each terrain type proportion
        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # self.platform is size of platform for some terrain type, like pit, gap, slope
        self.platform = cfg.platform
        # max_difficulty is based on num_rows
        # terrain difficulty is from 0 to max
        self.max_difficulty = (cfg.num_rows-1)/cfg.num_rows

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # border_size is whole terrain border
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        # whole terrain cols
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        # whole terrain rows
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        self.idx = 0

        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            # i j select row col position in whole terrain
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    # choice select terrain type, difficulty select row, row increase difficulty increase
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        rought_flat_min_height = - self.cfg.rough_flat_range[0] - difficulty * (self.cfg.rough_flat_range[1] - self.cfg.rough_flat_range[0]) / self.max_difficulty
        rought_flat_max_height = self.cfg.rough_flat_range[0] + difficulty * (self.cfg.rough_flat_range[1] - self.cfg.rough_flat_range[0]) / self.max_difficulty
        slope = self.cfg.slope_range[0] + difficulty * (self.cfg.slope_range[1] - self.cfg.slope_range[0]) / self.max_difficulty
        rought_slope_min_height = - self.cfg.rough_slope_range[0] - difficulty * (self.cfg.rough_slope_range[1] - self.cfg.rough_slope_range[0]) / self.max_difficulty
        rought_slope_max_height = self.cfg.rough_slope_range[0] + difficulty * (self.cfg.rough_slope_range[1] - self.cfg.rough_slope_range[0]) / self.max_difficulty
        stair_width = self.cfg.stair_width_range[0] + difficulty * (self.cfg.stair_width_range[1] - self.cfg.stair_width_range[0]) / self.max_difficulty
        stair_height = self.cfg.stair_height_range[0] + difficulty * (self.cfg.stair_height_range[1] - self.cfg.stair_height_range[0]) / self.max_difficulty
        discrete_obstacles_height = self.cfg.discrete_height_range[0] + difficulty * (self.cfg.discrete_height_range[1] - self.cfg.discrete_height_range[0]) / self.max_difficulty

        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        amplitude = 0.2 + 0.333 * difficulty
        if choice < self.proportions[0]:
            idx = 1
            return terrain
        elif choice < self.proportions[1]:
            idx = 2
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=rought_flat_min_height,
                                                 max_height=rought_flat_max_height,
                                                 step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            idx = 4
            if choice < self.proportions[2]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain,
                                                 slope=slope,
                                                 platform_size=self.platform)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=rought_slope_min_height,
                                                 max_height=rought_slope_max_height,
                                                 step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[5]:
            idx = 6
            if choice < self.proportions[4]:
                idx = 5
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain,
                                                 slope=slope,
                                                 platform_size=self.platform)
        elif choice < self.proportions[7]:
            idx = 8
            if choice<self.proportions[6]:
                idx = 7
                stair_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain,
                                                 step_width=stair_width,
                                                 step_height=stair_height,
                                                 platform_size=self.platform)
        elif choice < self.proportions[8]:
            idx = 9
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain,
                                                     discrete_obstacles_height,
                                                     rectangle_min_size,
                                                     rectangle_max_size,
                                                     num_rectangles,
                                                     platform_size=self.platform)
        elif choice < self.proportions[9]:
            idx = 10
            terrain_utils.wave_terrain(terrain,
                                       num_waves=3,
                                       amplitude=amplitude)
        elif choice < self.proportions[10]:
            idx = 11
            gap_terrain(terrain,
                        gap_size=gap_size,
                        platform_size=self.platform)
        else:
            idx = 12
            pit_terrain(terrain,
                        depth=pit_depth,
                        platform_size=self.platform)
        self.idx = idx
        return terrain

    # row col select position in whole terrain
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = self.idx

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
