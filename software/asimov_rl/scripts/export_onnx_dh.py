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

from asimov_rl import LEGGED_GYM_ROOT_DIR
import os
from asimov_rl.envs import *
from asimov_rl.utils import  get_args, task_registry
from datetime import datetime
import torch

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    models = [file for file in os.listdir(load_run)]
    models.sort(key=lambda m: "{0:0>15}".format(m))
    model = models[-1]

    load_path = os.path.join(load_run, model)
    return load_path

def export_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # load jit
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_policies')
    model_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    print("Load model from:", model_path)
    jit_model = torch.jit.load(model_path)
    jit_model.eval()
    
    current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                            train_cfg.runner.experiment_name, 'exported_onnx',
                            current_date_time)
    os.makedirs(root_path, exist_ok=True)
    dir_name = args.task.split('_')[0] + "_policy.onnx"
    path = os.path.join(root_path, dir_name)
    example_input = torch.randn(1,env_cfg.env.num_observations)
    # export onnx model
    torch.onnx.export(jit_model,               # JIT model
                    example_input,             # model example input
                    path,                      # model output path
                    export_params=True,        # export model params
                    opset_version=11,          # ONNX opset version
                    do_constant_folding=True,  # optimize constant variable folding
                    input_names=['input'],     # model input name
                    output_names=['output'],   # model output name
                    )
    print("Export onnx model to: ", path)
if __name__ == '__main__':
    args = get_args()
    if args.load_run == None:
        args.load_run = -1
    export_onnx(args)