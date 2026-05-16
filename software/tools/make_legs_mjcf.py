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
import re
from pathlib import Path
from xml.etree import ElementTree as ET

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


def strip_joints(root):
    """Remove <joint name="..."> elements whose name is in FIX_JOINTS.
    Walk the body tree to find them. Bodies are kept; only their joint child
    is dropped, which welds the body rigidly to its parent."""
    removed = []
    # Recursive walk
    def walk(elem):
        # Collect joints to remove (don't modify during iteration)
        to_remove = []
        for child in elem:
            if child.tag == "joint":
                jname = child.get("name")
                if jname in FIX_JOINTS:
                    to_remove.append(child)
                    removed.append(jname)
        for j in to_remove:
            elem.remove(j)
        # Recurse into all children (including remaining joints, harmless)
        for child in elem:
            walk(child)
    walk(root)
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    tree = ET.parse(args.src)
    root = tree.getroot()

    removed = strip_joints(root)
    print(f"Removed {len(removed)} joints:")
    for n in removed:
        print(f"  - {n}")
    missing = FIX_JOINTS - set(removed)
    if missing:
        print(f"WARN: expected to remove these but didn't find them: {missing}")

    # Update <mujoco model="..."> name to make it obvious which file this is
    if root.tag == "mujoco":
        old_name = root.get("model", "")
        if old_name and "legs" not in old_name:
            root.set("model", old_name + "_legs")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(args.dst), xml_declaration=True, encoding="utf-8")
    print(f"\nWrote: {args.dst}")


if __name__ == "__main__":
    main()
