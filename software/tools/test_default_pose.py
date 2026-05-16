#!/usr/bin/env python3
"""
test_default_pose.py — Diagnose initial standing pose stability.

Loads the legs MJCF with default joint angles, applies zero control
torque, and lets gravity decide. Reports center-of-mass position,
support-polygon, and how the robot settles over the first 5 seconds.
This isolates "is the default pose physically stable" from "is the
policy bad."
"""

import os
import sys
import numpy as np
import mujoco
import time

# Add software/ to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asimov_rl.envs.asimov.asimov_stand_config import AsimovStandCfg

cfg = AsimovStandCfg()
mjcf_path = cfg.asset.xml_file.format(
    LEGGED_GYM_ROOT_DIR=os.path.join(os.path.dirname(__file__), "..")
)
mjcf_path = os.path.normpath(mjcf_path)
print(f"Loading MJCF: {mjcf_path}")

model = mujoco.MjModel.from_xml_path(mjcf_path)
model.opt.timestep = 0.001
data = mujoco.MjData(model)

# Set default standing pose
default_pose = np.array(list(cfg.init_state.default_joint_angles.values()))
data.qpos[:] = 0
data.qpos[2] = cfg.init_state.pos[2]
data.qpos[3] = 1.0  # quat w=1
data.qpos[-12:] = default_pose
mujoco.mj_forward(model, data)

# Get pelvis ID and foot IDs for COM/support analysis
pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
left_ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
right_ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

def report():
    com = data.subtree_com[1].copy()  # COM of root body subtree (= whole robot)
    pelvis = data.xpos[pelvis_id].copy()
    lf = data.xpos[left_ankle_id].copy()
    rf = data.xpos[right_ankle_id].copy()
    foot_mid = (lf + rf) / 2
    com_offset_xy = com[:2] - foot_mid[:2]
    pelvis_xy = pelvis[:2] - foot_mid[:2]
    print(f"  pelvis xyz=({pelvis[0]:+.4f}, {pelvis[1]:+.4f}, {pelvis[2]:+.4f})")
    print(f"  COM    xyz=({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f})")
    print(f"  L foot xyz=({lf[0]:+.4f}, {lf[1]:+.4f}, {lf[2]:+.4f})")
    print(f"  R foot xyz=({rf[0]:+.4f}, {rf[1]:+.4f}, {rf[2]:+.4f})")
    print(f"  COM offset from foot midpoint (XY): ({com_offset_xy[0]:+.4f}, {com_offset_xy[1]:+.4f})")
    print(f"     [+x = forward, +y = left]")
    quat = data.qpos[3:7]
    print(f"  base quat (w,x,y,z) = ({quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f})")

print("\n=== t=0 (just after reset, before any sim step) ===")
report()

# Run with zero torque for 2 seconds, headless
print("\n=== Running 2 seconds with zero torque (pure gravity, headless) ===")
data.ctrl[:] = 0
for step in range(2000):
    mujoco.mj_step(model, data)
    if step in (10, 100, 500, 1000, 1999):
        t = step * 0.001
        print(f"\n--- t={t:.3f}s ---")
        report()
