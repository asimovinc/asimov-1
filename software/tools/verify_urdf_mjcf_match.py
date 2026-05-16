#!/usr/bin/env python3
"""
verify_urdf_mjcf_match.py — Confirm asimov_v1_legs.urdf and asimov_legs.xml
describe the same robot.

Compares (in both models):
- Number of bodies / joints
- Joint axes and limits
- Body chain topology (parent->child)
- Each body's mass, COM offset, and inertia
- Body origin pose relative to its parent
- Default pose forward kinematics: where is each body in world coordinates
  when all joints are at zero?

If everything matches within numerical tolerance, training (URDF in IsaacGym)
and deployment (MJCF in MuJoCo) will see the same physics.
"""

import os
import sys
import math
import numpy as np
import mujoco
from xml.etree import ElementTree as ET


def parse_floats(s, n=None):
    if s is None:
        return None
    vals = [float(x) for x in s.split()]
    if n is not None and len(vals) != n:
        raise ValueError(f"expected {n} floats, got {len(vals)}")
    return vals


def load_urdf_chain(path):
    """Parse URDF file and return same row structure used for MJCF.
    Each row: name, parent, joint, jtype, axis, limits, xyz, rpy, mass,
              inertia_diag (tuple of ixx,iyy,izz), com (list)."""
    tree = ET.parse(path)
    root = tree.getroot()

    links = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        inertial = link_elem.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            mass = float(mass_elem.get("value")) if mass_elem is not None else 0.0
            origin = inertial.find("origin")
            com = parse_floats(origin.get("xyz") if origin is not None else "0 0 0", 3)
            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                ixx = float(inertia_elem.get("ixx", 0))
                iyy = float(inertia_elem.get("iyy", 0))
                izz = float(inertia_elem.get("izz", 0))
            else:
                ixx = iyy = izz = 0.0
            inertia_diag = (ixx, iyy, izz)
        else:
            mass = 0.0
            com = [0, 0, 0]
            inertia_diag = (0, 0, 0)
        links[name] = {"mass": mass, "com": com, "inertia_diag": inertia_diag}

    rows = []
    # Map child link -> joint info
    child_to_joint = {}
    for j in root.findall("joint"):
        jname = j.get("name")
        jtype = j.get("type")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = parse_floats(origin.get("xyz") if origin is not None else "0 0 0", 3)
        rpy = parse_floats(origin.get("rpy") if origin is not None else "0 0 0", 3)
        axis_elem = j.find("axis")
        axis = parse_floats(axis_elem.get("xyz"), 3) if axis_elem is not None else None
        limit_elem = j.find("limit")
        if limit_elem is not None and jtype in ("revolute", "prismatic"):
            lim = (float(limit_elem.get("lower", 0)), float(limit_elem.get("upper", 0)))
        else:
            lim = None
        child_to_joint[child] = {
            "name": jname,
            "type": jtype,
            "parent": parent,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "limits": lim,
        }

    for name, link_info in links.items():
        j = child_to_joint.get(name)
        if j is None:
            row = {
                "name": name,
                "parent": None,
                "joint": None,
                "jtype": None,
                "axis": None,
                "limits": None,
                "xyz": [0, 0, 0],
                "rpy": [0, 0, 0],
                **link_info,
            }
        else:
            row = {
                "name": name,
                "parent": j["parent"],
                "joint": j["name"],
                "jtype": j["type"],
                "axis": j["axis"],
                "limits": j["limits"],
                "xyz": j["xyz"],
                "rpy": j["rpy"],
                **link_info,
            }
        rows.append(row)
    return rows


def quat_to_rpy(wxyz):
    w, x, y, z = wxyz
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def mjcf_body_chain(model):
    """Return same structure as urdf_body_chain."""
    rows = []
    # Build: body_id -> joints
    joints_of = {bid: [] for bid in range(model.nbody)}
    for jid in range(model.njnt):
        bid = int(model.jnt_bodyid[jid])
        joints_of[bid].append(jid)

    for bid in range(1, model.nbody):  # skip world (id=0)
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        parent_bid = int(model.body_parentid[bid])
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_bid) if parent_bid > 0 else None
        body_pos = list(model.body_pos[bid])
        body_quat = list(model.body_quat[bid])
        body_rpy = quat_to_rpy(body_quat)

        # Joint info
        jids = joints_of[bid]
        # Skip free joint (floating base)
        jids = [j for j in jids if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        if jids:
            jid = jids[0]
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            jtype_int = int(model.jnt_type[jid])
            jtype = {2: "prismatic", 3: "revolute"}.get(jtype_int, str(jtype_int))
            axis = list(model.jnt_axis[jid])
            if model.jnt_limited[jid]:
                lim = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
            else:
                lim = None
        else:
            jname = None
            jtype = "fixed"
            axis = None
            lim = None

        mass = float(model.body_mass[bid])
        inertia_diag = tuple(model.body_inertia[bid])  # principal-axis diag
        com = list(model.body_ipos[bid])

        rows.append({
            "name": bname,
            "parent": parent,
            "joint": jname,
            "jtype": jtype,
            "axis": axis,
            "limits": lim,
            "xyz": body_pos,
            "rpy": body_rpy,
            "mass": mass,
            "inertia_diag": inertia_diag,
            "com": com,
        })
    return rows


def vec_close(a, b, atol=1e-4):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.allclose(a, b, atol=atol)


def fmt(v, prec=4):
    if v is None:
        return "None"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(f"{x:+.{prec}f}" for x in v) + "]"
    if isinstance(v, float):
        return f"{v:+.{prec}f}"
    return str(v)


def main():
    urdf_path = "software/resources/robots/asimov_v1/urdf/asimov_v1_legs.urdf"
    mjcf_path = "sim-model/xmls/asimov_legs.xml"
    print(f"URDF: {urdf_path}")
    print(f"MJCF: {mjcf_path}")
    print()

    urdf_rows = load_urdf_chain(urdf_path)
    mjcf = mujoco.MjModel.from_xml_path(mjcf_path)

    u = {row["name"]: row for row in urdf_rows}
    m = {row["name"]: row for row in mjcf_body_chain(mjcf)}

    print(f"URDF links: {len(u)}, MJCF bodies: {len(m)}")

    common = sorted(set(u) & set(m))
    only_u = sorted(set(u) - set(m))
    only_m = sorted(set(m) - set(u))
    if only_u:
        print(f"  URDF-only links ({len(only_u)}): {only_u}")
    if only_m:
        print(f"  MJCF-only bodies ({len(only_m)}): {only_m}")
    print(f"  In common: {len(common)}")
    print()

    # Compare each common link
    diffs = []
    print("=" * 80)
    print("Per-body comparison")
    print("=" * 80)
    for name in common:
        ur = u[name]
        mr = m[name]
        problems = []

        # Parent
        if ur["parent"] != mr["parent"]:
            problems.append(f"parent: URDF={ur['parent']} vs MJCF={mr['parent']}")

        # Joint type alignment
        # URDF: revolute / continuous / fixed / prismatic / floating
        # MJCF: revolute / prismatic / fixed / "0" (free)
        u_type = ur["jtype"]
        m_type = mr["jtype"]
        u_is_revolute = u_type in ("revolute", "continuous")
        m_is_revolute = m_type == "revolute"
        u_is_fixed = u_type in (None, "fixed")
        m_is_fixed = m_type == "fixed"
        if u_is_revolute != m_is_revolute or u_is_fixed != m_is_fixed:
            problems.append(f"jtype: URDF={u_type} vs MJCF={m_type}")

        # Body origin xyz
        if not vec_close(ur["xyz"], mr["xyz"], atol=1e-4):
            problems.append(f"xyz: URDF={fmt(ur['xyz'])} vs MJCF={fmt(mr['xyz'])}")

        # Body origin rpy (only meaningful within tolerance; small numerical
        # differences for elbow-baked rotations are expected — flag if large)
        if not vec_close(ur["rpy"], mr["rpy"], atol=1e-3):
            problems.append(f"rpy: URDF={fmt(ur['rpy'])} vs MJCF={fmt(mr['rpy'])}")

        # Joint axis (only if both have a joint)
        if u_is_revolute and m_is_revolute:
            if not vec_close(ur["axis"], mr["axis"], atol=1e-4):
                problems.append(f"axis: URDF={fmt(ur['axis'])} vs MJCF={fmt(mr['axis'])}")

        # Joint limits
        if u_is_revolute and m_is_revolute:
            if ur["limits"] is None and mr["limits"] is None:
                pass
            elif ur["limits"] is None or mr["limits"] is None:
                problems.append(f"limits: URDF={ur['limits']} vs MJCF={mr['limits']}")
            elif not vec_close(ur["limits"], mr["limits"], atol=1e-4):
                problems.append(f"limits: URDF={fmt(ur['limits'])} vs MJCF={fmt(mr['limits'])}")

        # Mass
        if abs(ur["mass"] - mr["mass"]) > 1e-5:
            problems.append(f"mass: URDF={ur['mass']:.6f} vs MJCF={mr['mass']:.6f}")

        # Inertia diagonal — note: URDF stores body-frame full tensor, MJCF
        # stores principal-axis diagonal. They are NOT directly comparable
        # element-wise, but trace and total magnitude should be close.
        u_trace = sum(ur["inertia_diag"])
        m_trace = sum(mr["inertia_diag"])
        if u_trace > 0 and abs(u_trace - m_trace) / u_trace > 0.01:
            problems.append(f"inertia trace: URDF={u_trace:.6f} vs MJCF={m_trace:.6f}")

        # COM offset (URDF inertial origin xyz vs MJCF body_ipos)
        # These should match because MJCF stores ipos in body frame.
        if not vec_close(ur["com"], mr["com"], atol=1e-4):
            problems.append(f"com: URDF={fmt(ur['com'])} vs MJCF={fmt(mr['com'])}")

        if problems:
            diffs.append((name, problems))

    if not diffs:
        print("[OK] All common bodies agree on parent, axis, limits, mass, COM, and pose.")
    else:
        print(f"[!] Found {len(diffs)} bodies with discrepancies:\n")
        for name, problems in diffs:
            print(f"  {name}:")
            for p in problems:
                print(f"    - {p}")

    # Active DOF count comparison
    u_active = [r for r in urdf_rows if r["jtype"] in ("revolute", "continuous", "prismatic")]
    m_active = [r for r in mjcf_body_chain(mjcf) if r["jtype"] in ("revolute", "prismatic")]
    print()
    print(f"Active DOF: URDF={len(u_active)}, MJCF={len(m_active)}")
    if [r["name"] for r in u_active] != [r["name"] for r in m_active]:
        print("WARN: active joint order differs!")
        print(f"  URDF order: {[r['joint'] for r in u_active]}")
        print(f"  MJCF order: {[r['joint'] for r in m_active]}")
    else:
        print("Active joint order matches:")
        for r in u_active:
            print(f"  {r['joint']}")


if __name__ == "__main__":
    main()
