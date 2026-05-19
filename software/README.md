# Asimov v1 — Locomotion RL (`software/`)

IsaacGym training + MuJoCo inference for the Asimov v1 humanoid, ported from
[`agibot_x1_train`](https://github.com/AgibotTech/agibot_x1_train).
Phase 1 scope: **12-DOF lower body** (upper body frozen as fixed joints).

基于 `agibot_x1_train` 移植的 Asimov v1 人形机器人运动控制：IsaacGym 训练 +
MuJoCo 推理。当前阶段：**12 自由度下肢**（上半身在 URDF 中冻结为固定关节）。

---

## 1. What was changed / 修改概要

The upstream X1 framework was adapted to Asimov v1. Key changes:

| Area | Change | 说明 |
|---|---|---|
| Model | `mjcf2urdf.py` converts `sim-model/xmls/asimov.xml` → URDF | 从 MuJoCo MJCF 生成 IsaacGym 用的 URDF |
| Model | `make_legs_mjcf.py` strips 15 non-leg joints → `asimov_legs.xml` | 剥离上肢/腰/颈/趾关节，得到与训练 URDF 一致的 12-DOF MJCF |
| Model | Elbow `ref` baked into body quat; right ankle_pitch sign mirrored | 修正肘部静止角与右踝镜像符号 |
| Package | `humanoid` → `asimov_rl`, IsaacGym imports fixed | 重命名包，修复 isaacgym/torch 导入顺序 |
| Config | Default pose rebuilt as balanced squat (COM over feet) | 重建平衡蹲姿，质心落在脚上 |
| Config | PD gains raised well above X1 baseline (hip_pitch 30→100, knee 100→200) | 显著提高 PD 增益（X1 增益在 Asimov 上拉不动腿） |

Diagnostic tools live in `tools/` (see §6).

---

## 2. Layout / 目录结构

```
software/
├── asimov_rl/
│   ├── algo/ppo/            PPO, ActorCriticDH, state estimator, long-history CNN
│   ├── envs/
│   │   ├── base/            legged_robot, base_task (shared)
│   │   └── asimov/          asimov_stand_config.py, asimov_stand_env.py
│   ├── scripts/             train / play / sim2sim / export
│   └── utils/               task_registry, terrain, helpers, logger
├── resources/robots/asimov_v1/
│   ├── urdf/asimov_v1.urdf          full 27-joint model
│   ├── urdf/asimov_v1_legs.urdf     12 active DOF (training)
│   └── meshes/                      28 STL meshes
├── tools/                   conversion + diagnostic scripts
├── logs/<exp>/              training output (checkpoints, tensorboard, exports)
└── setup.py
```

Robot MJCF is in the repo root: `../sim-model/xmls/asimov_legs.xml`.

---

## 3. Setup / 环境

- Python 3.8, PyTorch 1.13.1+cu117, **IsaacGym Preview 4**, MuJoCo 3.x
- IsaacGym must be importable; install separately from NVIDIA.

```bash
cd software
pip install -e .
```

On the training machine (CUDA 11.7), prefix every IsaacGym command with:

训练机（CUDA 11.7）每条 IsaacGym 命令前加：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
```

---

## 4. Training / 训练

```bash
python -m asimov_rl.scripts.train \
    --task=asimov_stand \
    --num_envs=8192 \
    --max_iterations=15000 \
    --headless \
    --run_name=my_run
```

| Flag | Meaning / 说明 |
|---|---|
| `--task` | Always `asimov_stand` / 任务名固定 |
| `--num_envs` | Parallel envs / 并行环境数（4090: 8192, 16G T4: 2048） |
| `--max_iterations` | With `--resume` this is **incremental** / resume 时为增量 |
| `--headless` | No viewer / 无渲染 |
| `--run_name` | Log dir name / 日志目录名 |
| `--resume --load_run=<name>` | Continue from a run's latest checkpoint / 续训 |

Checkpoints save to `logs/asimov_stand/exported_data/<timestamp><run_name>/model_<iter>.pt`
per `runner.save_interval` in the config.

Monitor with TensorBoard:

```bash
tensorboard --logdir=logs/asimov_stand --port=6006
```

Key reward signals to watch: `rew_feet_air_time` (should rise — proves the
policy is lifting feet during walking commands), `rew_tracking_lin_vel`,
`Mean episode length`.

关注 `rew_feet_air_time`（上升说明在迈步）、`rew_tracking_lin_vel`、episode 长度。

---

## 5. Visualize a policy in IsaacGym / 在 IsaacGym 中可视化

```bash
python asimov_rl/scripts/play.py \
    --task=asimov_stand \
    --num_envs=4 \
    --headless \
    --load_run=my_run \
    --checkpoint=2000
```

- `--checkpoint=-1` (or omit) loads the latest / 不指定则用最新
- Constants at the bottom of `play.py`:
  - `RENDER=True` to record video (needs GUI/X11) / 录像（需 GUI）
  - `FIX_COMMAND=True` forces a constant forward command (default in this repo)
    / 固定前进命令；`False` 用手柄
- Prints env-0 base position every simulated second so you can verify the
  robot actually moves (`base[0]` should grow).

---

## 6. Export the policy / 导出策略

`export_policy_dh.py` traces the actor + state-estimator + long-history CNN
into a single TorchScript module for deployment.

```bash
python asimov_rl/scripts/export_policy_dh.py \
    --task=asimov_stand \
    --load_run=my_run \
    --checkpoint=2000
```

- Reads `logs/asimov_stand/exported_data/<run>/model_<checkpoint>.pt`
- Writes `logs/asimov_stand/exported_policies/<timestamp>/policy_dh.jit`
- `--load_run`/`--checkpoint` default to `-1` (latest run / latest checkpoint)

读取训练 checkpoint，导出为单个 TorchScript 文件
`logs/asimov_stand/exported_policies/<时间戳>/policy_dh.jit`。

---

## 7. MuJoCo inference (sim2sim) / MuJoCo 推理

Validates the exported policy in MuJoCo (CPU-only, **no IsaacGym needed** —
can run on macOS / laptop).

在 MuJoCo 中验证导出的策略，纯 CPU，**无需 IsaacGym**，可在 mac/笔记本运行。

```bash
# Use normal `python`, NOT `mjpython` (joystick + mujoco_viewer need the main thread)
python asimov_rl/scripts/sim2sim.py \
    --task=asimov_stand \
    --load_model=<timestamp>
```

- `--load_model` is the timestamp folder name under
  `logs/asimov_stand/exported_policies/` (the dir containing `policy_dh.jit`)
- Loads the legs MJCF (`../sim-model/xmls/asimov_legs.xml`) which matches the
  training URDF exactly (verified by `tools/verify_urdf_mjcf_match.py`)
- A USB gamepad controls velocity commands; without one it defaults to a
  fixed command (edit the `x_vel_cmd, y_vel_cmd, yaw_vel_cmd` line near the
  top of `sim2sim.py`). Gamepad button 0 (Xbox A / PS X) resets the robot.
- Prints `base_x`, `vx`, knee angle & torque once per simulated second.

注意：macOS 用普通 `python` 启动（不是 `mjpython`），否则 pygame 手柄与
viewer 抢主线程会崩溃。手柄按钮 0 = 复位。

---

## 8. Diagnostic tools / 诊断工具 (`tools/`)

| Script | Purpose / 用途 |
|---|---|
| `mjcf2urdf.py` | MJCF → URDF（`--fix-joints` 冻结指定关节） |
| `make_legs_mjcf.py` | Strip non-leg joints → `asimov_legs.xml` |
| `verify_urdf_mjcf_match.py` | Confirm URDF and MJCF describe the same robot |
| `audit_dof_config.py` | Full DOF audit: order, limits, PD, swing-delta signs |
| `test_default_pose.py` | Zero-torque rollout, report COM offset from feet |
| `test_lift_foot.py` | Open-loop single-foot lift test (`--headless`, `--slow`) |

Run from the repo root, e.g.:

```bash
cd /path/to/asimov-1
python software/tools/audit_dof_config.py
python software/tools/verify_urdf_mjcf_match.py
```

---

## 9. Typical workflow / 典型流程

```
1. Train          python -m asimov_rl.scripts.train --task=asimov_stand --run_name=R --headless
2. Watch curves   tensorboard --logdir=logs/asimov_stand
3. Inspect (GPU)  python asimov_rl/scripts/play.py --task=asimov_stand --load_run=R --checkpoint=N
4. Export         python asimov_rl/scripts/export_policy_dh.py --task=asimov_stand --load_run=R --checkpoint=N
5. Verify (CPU)   python asimov_rl/scripts/sim2sim.py --task=asimov_stand --load_model=<timestamp>
```
