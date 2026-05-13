#!/usr/bin/env python3
"""
mjcf2urdf.py — Convert Asimov v1 MJCF (sim-model/xmls/asimov.xml) to URDF.

Strategy: use MuJoCo's own model loader to resolve all `default class` inheritance
and quaternion math, then walk the body tree and emit URDF.

URDF/MJCF correspondence used here:
- MJCF <body> with pos/quat (in parent body frame)
    -> URDF <link>, and a <joint> from parent_link to this link whose
       <origin xyz="pos" rpy="euler(quat)"/> carries the pose offset.
- MJCF <inertial pos quat mass fullinertia/diaginertia> (in body frame)
    -> URDF <inertial>; <origin xyz=com_pos rpy=euler(com_quat)/>;
       <inertia ixx ixy ixz iyy iyz izz>.
- MJCF <joint pos axis range type> (in body frame, hinge/slide)
    -> URDF <joint type=revolute|continuous|prismatic>; <axis xyz/>;
       <limit lower upper effort velocity/>.  URDF joint origin == body pose
       offset above.  Joint pos offset (if non-zero) requires a fixed dummy
       link; this MJCF has all joint pos == 0, so we skip that complication.
- MJCF <geom type=mesh mesh=name> -> URDF <visual>/<collision> with mesh ref.
- MJCF capsule collision geoms (fromto, size=radius) -> URDF <collision> with
  <geometry><cylinder> approximation (capsules unsupported in URDF) plus
  spheres at the endpoints, OR simply use the mesh for collision and let
  IsaacGym/Bullet/etc. decimate. We choose the latter (mesh collision) for
  simplicity; physics engines handle it.

Free joint at the root is not emitted in URDF (the floating base is added by
the simulator's RL env wrapper, not the URDF itself).

Run:
  python software/tools/mjcf2urdf.py \
      --mjcf sim-model/xmls/asimov.xml \
      --out  software/resources/robots/asimov_v1/urdf/asimov_v1.urdf \
      --mesh-rel ../meshes
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco package not available — run with the lerobot conda env's python")


JOINT_TYPE_MAP = {
    mujoco.mjtJoint.mjJNT_HINGE: "revolute",
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
}


def quat_to_rpy(quat_wxyz):
    """MuJoCo quaternion (w, x, y, z) -> URDF (roll, pitch, yaw) in radians."""
    w, x, y, z = quat_wxyz
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def fmt_vec(v, prec=8):
    return " ".join(f"{x:.{prec}g}" for x in v)


def get_inertia_ixx_etc(model, body_id):
    """MuJoCo stores diagonal inertia in body frame; off-diagonals are in the
    inertial frame defined by body_iquat. We need the full inertia tensor in
    the same frame as URDF <inertial><origin>, which is the inertial frame
    (com_pos, com_quat). In that frame the inertia tensor is diagonal."""
    diag = model.body_inertia[body_id]  # ixx, iyy, izz in inertial principal frame
    return float(diag[0]), 0.0, 0.0, float(diag[1]), 0.0, float(diag[2])


def build_urdf(mjcf_path: Path, mesh_rel: str, robot_name: str) -> ET.Element:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))

    robot = ET.Element("robot", name=robot_name)

    # 1) Walk all bodies (skip world at index 0)
    body_names = []
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        body_names.append(bname)

    # collect mesh names + filenames
    mesh_files = {}
    for mid in range(model.nmesh):
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        # MuJoCo stores the file in model.mesh_pathadr (path inside paths)
        adr = model.mesh_pathadr[mid]
        fname_bytes = bytes(model.paths[adr:]).split(b"\x00", 1)[0]
        fname = fname_bytes.decode("utf-8", errors="replace")
        # In this MJCF, mesh name often == basename of file; we use the file
        mesh_files[mname] = fname or mname

    # group geoms by body
    geoms_by_body = {bid: [] for bid in range(model.nbody)}
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        geoms_by_body[bid].append(gid)

    # group joints by body (only one expected per body for this robot)
    joints_by_body = {bid: [] for bid in range(model.nbody)}
    for jid in range(model.njnt):
        bid = int(model.jnt_bodyid[jid])
        joints_by_body[bid].append(jid)

    # 2) For each non-world body, emit a <link>
    for bid in range(1, model.nbody):
        bname = body_names[bid]
        link = ET.SubElement(robot, "link", name=bname)

        mass = float(model.body_mass[bid])
        if mass > 0:
            inertial = ET.SubElement(link, "inertial")
            ipos = model.body_ipos[bid]
            iquat = model.body_iquat[bid]
            rpy = quat_to_rpy(iquat)
            ET.SubElement(
                inertial,
                "origin",
                xyz=fmt_vec(ipos),
                rpy=fmt_vec(rpy),
            )
            ET.SubElement(inertial, "mass", value=f"{mass:.10g}")
            ixx, ixy, ixz, iyy, iyz, izz = get_inertia_ixx_etc(model, bid)
            ET.SubElement(
                inertial,
                "inertia",
                ixx=f"{ixx:.10g}",
                ixy=f"{ixy:.10g}",
                ixz=f"{ixz:.10g}",
                iyy=f"{iyy:.10g}",
                iyz=f"{iyz:.10g}",
                izz=f"{izz:.10g}",
            )

        # geoms: split into visual (mesh, contype=0 typically) and collision
        for gid in geoms_by_body[bid]:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            gtype = model.geom_type[gid]
            grp = int(model.geom_group[gid])
            # gpos/gquat are in body frame
            gpos = model.geom_pos[gid]
            gquat = model.geom_quat[gid]
            rpy = quat_to_rpy(gquat)

            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                meshid = int(model.geom_dataid[gid])
                mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, meshid)
                fname = mesh_files.get(mname, mname + ".STL")
                # decide visual vs collision: MJCF default `class="visual"` has contype=0
                contype = int(model.geom_contype[gid])
                conaffinity = int(model.geom_conaffinity[gid])
                is_visual = (contype == 0 and conaffinity == 0) or grp == 2
                tag = "visual" if is_visual else "collision"
                node = ET.SubElement(link, tag, name=gname or f"{bname}_{tag}")
                ET.SubElement(node, "origin", xyz=fmt_vec(gpos), rpy=fmt_vec(rpy))
                geom = ET.SubElement(node, "geometry")
                ET.SubElement(geom, "mesh", filename=f"{mesh_rel}/{fname}")
            else:
                # primitive collision (capsule/sphere/box/cylinder/plane).
                # URDF supports box / cylinder / sphere. Map capsule -> cylinder
                # (lossy: misses hemispherical caps but adequate for self-collision).
                if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue  # floor is the world, not a body geom
                contype = int(model.geom_contype[gid])
                tag = "collision" if (contype != 0 or grp == 3) else "visual"
                size = model.geom_size[gid]

                if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    # MuJoCo capsule defined either by size=(r, halflen) with
                    # geom origin at body frame, OR by fromto. After compile,
                    # geom_pos is the midpoint and geom_size = (r, halflen, 0).
                    r = float(size[0])
                    halflen = float(size[1])
                    node = ET.SubElement(link, tag, name=gname or f"{bname}_{tag}")
                    ET.SubElement(node, "origin", xyz=fmt_vec(gpos), rpy=fmt_vec(rpy))
                    geom = ET.SubElement(node, "geometry")
                    ET.SubElement(
                        geom,
                        "cylinder",
                        radius=f"{r:.6g}",
                        length=f"{2*halflen:.6g}",
                    )
                elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                    node = ET.SubElement(link, tag, name=gname or f"{bname}_{tag}")
                    ET.SubElement(node, "origin", xyz=fmt_vec(gpos), rpy=fmt_vec(rpy))
                    geom = ET.SubElement(node, "geometry")
                    ET.SubElement(geom, "sphere", radius=f"{float(size[0]):.6g}")
                elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    node = ET.SubElement(link, tag, name=gname or f"{bname}_{tag}")
                    ET.SubElement(node, "origin", xyz=fmt_vec(gpos), rpy=fmt_vec(rpy))
                    geom = ET.SubElement(node, "geometry")
                    ET.SubElement(geom, "box", size=fmt_vec([2*size[0], 2*size[1], 2*size[2]]))
                # other primitives: skip with a warning
                else:
                    print(f"  WARN: unsupported geom type {gtype} on {bname}, skipped")

    # 3) Emit joints: each non-world body has one joint (parent->this)
    for bid in range(1, model.nbody):
        bname = body_names[bid]
        parent_bid = int(model.body_parentid[bid])
        parent_name = body_names[parent_bid] if parent_bid > 0 else None

        body_pos = model.body_pos[bid]
        body_quat = model.body_quat[bid]
        rpy = quat_to_rpy(body_quat)

        jids = joints_by_body[bid]

        if parent_bid == 0:
            # Root body. In URDF, root link has no parent joint.
            # If MJCF root has a freejoint, we drop it (sim wrapper provides floating base).
            continue

        if len(jids) == 0:
            # Fixed link
            joint = ET.SubElement(
                robot, "joint", name=f"{bname}_fixed", type="fixed"
            )
            ET.SubElement(joint, "origin", xyz=fmt_vec(body_pos), rpy=fmt_vec(rpy))
            ET.SubElement(joint, "parent", link=parent_name)
            ET.SubElement(joint, "child", link=bname)
            continue

        if len(jids) > 1:
            print(f"  WARN: {bname} has {len(jids)} joints; URDF supports only 1 per pair")

        jid = jids[0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        jtype_mj = int(model.jnt_type[jid])
        if jtype_mj == mujoco.mjtJoint.mjJNT_FREE:
            # floating base — emit nothing; sim wrapper will add it
            continue
        if jtype_mj not in JOINT_TYPE_MAP:
            print(f"  WARN: joint {jname} type {jtype_mj} unsupported, emitting fixed")
            joint = ET.SubElement(robot, "joint", name=jname, type="fixed")
            ET.SubElement(joint, "origin", xyz=fmt_vec(body_pos), rpy=fmt_vec(rpy))
            ET.SubElement(joint, "parent", link=parent_name)
            ET.SubElement(joint, "child", link=bname)
            continue

        urdf_type = JOINT_TYPE_MAP[jtype_mj]
        limited = bool(model.jnt_limited[jid])
        # MJCF axis is in body frame (= URDF joint frame here, since joint pos == 0)
        axis = model.jnt_axis[jid].copy()
        jpos = model.jnt_pos[jid]
        if np.linalg.norm(jpos) > 1e-9:
            print(f"  WARN: joint {jname} has non-zero pos {jpos} in body frame; "
                  "URDF cannot represent this directly. Effective behavior differs.")

        if urdf_type == "revolute" and not limited:
            urdf_type = "continuous"

        joint = ET.SubElement(robot, "joint", name=jname, type=urdf_type)
        ET.SubElement(joint, "origin", xyz=fmt_vec(body_pos), rpy=fmt_vec(rpy))
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=bname)
        ET.SubElement(joint, "axis", xyz=fmt_vec(axis))

        # Effort/velocity: URDF requires them for revolute/prismatic. MJCF
        # doesn't carry these as joint attrs (they're in actuators). Use
        # sane defaults; downstream env configs can override.
        if urdf_type in ("revolute", "prismatic"):
            lower, upper = (
                float(model.jnt_range[jid][0]),
                float(model.jnt_range[jid][1]),
            )
            if not limited:
                lower, upper = -math.pi, math.pi
            ET.SubElement(
                joint,
                "limit",
                lower=f"{lower:.6g}",
                upper=f"{upper:.6g}",
                effort="200",
                velocity="20",
            )
        elif urdf_type == "continuous":
            ET.SubElement(joint, "limit", effort="200", velocity="20")

        # Carry armature as <dynamics damping="..."> hint if present
        armature = float(model.dof_armature[model.jnt_dofadr[jid]])
        damping = float(model.dof_damping[model.jnt_dofadr[jid]])
        if armature > 0 or damping > 0:
            dyn = ET.SubElement(joint, "dynamics")
            if damping > 0:
                dyn.set("damping", f"{damping:.6g}")
            if armature > 0:
                dyn.set("friction", "0")  # placeholder; armature has no URDF analog

    return robot


def prettify(elem: ET.Element) -> str:
    from xml.dom import minidom
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def copy_meshes(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_dir.iterdir():
        if f.suffix.lower() in (".stl", ".obj", ".dae"):
            shutil.copy2(f, dst_dir / f.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mesh-src", type=Path, default=None,
                    help="Directory of source STL files (default: <mjcf parent>/../assets/meshes)")
    ap.add_argument("--mesh-rel", default="../meshes",
                    help="Relative path from URDF to mesh dir (URDF mesh filename prefix)")
    ap.add_argument("--name", default="asimov_v1")
    args = ap.parse_args()

    mjcf_path = args.mjcf.resolve()
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MJCF: {mjcf_path}")
    robot = build_urdf(mjcf_path, args.mesh_rel, args.name)

    out_path.write_text(prettify(robot), encoding="utf-8")
    print(f"Wrote URDF: {out_path}")

    # Copy meshes
    mesh_src = args.mesh_src or (mjcf_path.parent.parent / "assets" / "meshes")
    mesh_dst = (out_path.parent / args.mesh_rel).resolve()
    if mesh_src.is_dir():
        n = copy_meshes(mesh_src, mesh_dst)
        print(f"Copied {n} mesh files: {mesh_src} -> {mesh_dst}")
    else:
        print(f"WARN: mesh source dir not found: {mesh_src}")


if __name__ == "__main__":
    main()
