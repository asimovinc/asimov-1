#!/usr/bin/env python3
"""
make_legs_mjcf.py — Strip non-leg joints from asimov.xml to match the legs URDF.

The training policy (12 active DOF, asimov_v1_legs.urdf) freezes the upper
body via fixed joints in URDF. To use the same policy in sim2sim, the MuJoCo
model must have the same kinematic structure: legs articulated, everything
else welded to its parent. This script does that mechanically by removing
specific <joint> elements from the source MJCF — body links remain (so mass
and inertia still contribute), but they become rigid extensions of their
parent body.

Run:
  python software/tools/make_legs_mjcf.py \
      --src sim-model/xmls/asimov.xml \
      --dst sim-model/xmls/asimov_legs.xml
"""

import argparse
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

# Joints to remove (must match exactly the names in the MJCF).
# These are the 15 joints that asimov_v1_legs.urdf converts to <fixed>.
FIX_JOINTS = {
    "left_toe_joint", "right_toe_joint",
    "waist_yaw_joint",
    "neck_yaw_joint", "neck_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
}


def axis_angle_to_quat(axis, angle):
    """Return (w, x, y, z) for a rotation of `angle` rad around `axis`."""
    a = np.array(axis, dtype=float)
    a /= np.linalg.norm(a)
    half = angle / 2.0
    s = math.sin(half)
    return (math.cos(half), a[0] * s, a[1] * s, a[2] * s)


def quat_mul(q1, q2):
    """Hamilton product (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def parse_quat(s):
    if s is None:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(x) for x in s.split())


def parse_axis(s):
    return [float(x) for x in s.split()]


def fmt_quat(q):
    return " ".join(f"{x:.10g}" for x in q)


def strip_joints(root):
    """Remove <joint> elements whose name is in FIX_JOINTS.

    For joints with a non-zero `ref` attribute, bake the rotation into the
    parent body's quat so the resulting rigid pose matches qpos=ref (not 0).
    """
    removed = []
    baked_refs = []

    def process_body(body):
        # Find joints to remove in this body
        to_remove = []
        for j in body.findall("joint"):
            jname = j.get("name")
            if jname in FIX_JOINTS:
                ref_str = j.get("ref", "0")
                ref = float(ref_str)
                if abs(ref) > 1e-9:
                    # Bake ref rotation into the body's quat so qpos=0 (rigid)
                    # gives the same geometry as the original qpos=ref.
                    axis = parse_axis(j.get("axis", "0 0 1"))
                    body_quat = parse_quat(body.get("quat"))
                    ref_quat = axis_angle_to_quat(axis, ref)
                    new_quat = quat_mul(body_quat, ref_quat)
                    body.set("quat", fmt_quat(new_quat))
                    baked_refs.append((jname, ref, axis))
                to_remove.append(j)
                removed.append(jname)
        for j in to_remove:
            body.remove(j)
        # Recurse into nested bodies
        for sub in body.findall("body"):
            process_body(sub)

    # Find the top-level worldbody and walk all its body children
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")
    for top_body in worldbody.findall("body"):
        process_body(top_body)

    return removed, baked_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    tree = ET.parse(args.src)
    root = tree.getroot()

    removed, baked_refs = strip_joints(root)
    print(f"Removed {len(removed)} joints:")
    for n in removed:
        marker = " (baked non-zero ref)" if n in {b[0] for b in baked_refs} else ""
        print(f"  - {n}{marker}")
    if baked_refs:
        print()
        print("Baked non-zero refs into parent body quat:")
        for jname, ref, axis in baked_refs:
            print(f"  - {jname}: ref={ref:+.4f} axis={axis}")
    missing = FIX_JOINTS - set(removed)
    if missing:
        print(f"\nWARN: expected to remove these but didn't find them: {missing}")

    if root.tag == "mujoco":
        old_name = root.get("model", "")
        if old_name and "legs" not in old_name:
            root.set("model", old_name + "_legs")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(args.dst), xml_declaration=True, encoding="utf-8")
    print(f"\nWrote: {args.dst}")


if __name__ == "__main__":
    main()
