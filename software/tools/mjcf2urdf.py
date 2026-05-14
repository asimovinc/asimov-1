#!/usr/bin/env python3
"""
mjcf2urdf.py — Convert Asimov v1 MJCF (sim-model/xmls/asimov.xml) to URDF.

Strategy: parse the source MJCF XML *as text* (not through MuJoCo's compiled
model), so all positions/orientations are preserved literally. We only use
MuJoCo's compiler to resolve `default class` inheritance for attributes we
don't otherwise see (e.g. geom contype/conaffinity from a class).

Key fixes over the previous compiled-model approach:
- Mesh geoms keep their MJCF-literal pos/quat. MuJoCo's compiler shifts geom
  positions to the mesh's center of mass for internal inertia accounting,
  which silently broke our previous output.
- Inertials use the body-frame components directly from MJCF's
  fullinertia="ixx iyy izz ixy ixz iyz" with origin RPY=0. URDF technically
  allows a rotated inertial frame, but many physics engines (IsaacGym
  included) ignore the RPY and treat ixx/iyy/izz as body-frame diagonals.

URDF/MJCF correspondence:
- MJCF <body pos quat>  -> URDF parent->child <joint><origin xyz rpy/>
- MJCF <inertial pos fullinertia mass>  -> URDF <inertial>
- MJCF <joint type=hinge pos axis range>  -> URDF <joint type=revolute>
- MJCF <geom type=mesh mesh=name pos quat>  -> URDF <visual>/<collision>

Run:
  python software/tools/mjcf2urdf.py \
      --mjcf sim-model/xmls/asimov.xml \
      --out  software/resources/robots/asimov_v1/urdf/asimov_v1.urdf \
      --mesh-rel ../meshes
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def parse_floats(s, n=None):
    if s is None:
        return None
    vals = [float(x) for x in s.split()]
    if n is not None and len(vals) != n:
        raise ValueError(f"expected {n} floats, got {len(vals)}: {s}")
    return vals


def quat_wxyz_to_rpy(wxyz):
    """MuJoCo quaternion (w, x, y, z) -> URDF (roll, pitch, yaw)."""
    w, x, y, z = wxyz
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def get_attr_quat(elem):
    """Read quat attribute as (w,x,y,z); default (1,0,0,0)."""
    q = elem.get("quat")
    if q is None:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(parse_floats(q, 4))


def get_attr_pos(elem):
    p = elem.get("pos")
    if p is None:
        return (0.0, 0.0, 0.0)
    return tuple(parse_floats(p, 3))


def fmt_vec(v, prec=8):
    return " ".join(f"{x:.{prec}g}" for x in v)


# ---------------------------------------------------------------------------
# Default-class resolution
# ---------------------------------------------------------------------------

class Defaults:
    """Tracks the active <default> class stack as we walk the MJCF tree."""

    def __init__(self, mjcf_root):
        # Map class_name -> { 'geom': {attr: val}, 'joint': {attr: val}, ... }
        self.classes = {}
        self._collect(mjcf_root.find("default"), parent_classes={})

    def _collect(self, node, parent_classes):
        if node is None:
            return
        # The top-level <default> may have a name="main" implicit class; gather
        # its direct geom/joint children as the "main" class defaults.
        for child in node:
            if child.tag == "default":
                cname = child.get("class")
                if cname is None:
                    continue
                inherited = {k: dict(v) for k, v in parent_classes.items()}
                # Recurse: collect inner defaults first
                child_defaults = {}
                for sub in child:
                    if sub.tag == "default":
                        continue
                    # sub.tag is e.g. 'geom', 'joint'
                    child_defaults[sub.tag] = dict(sub.attrib)
                # Merge inherited + child's own
                merged = {k: dict(v) for k, v in inherited.items()}
                for tag, attrs in child_defaults.items():
                    merged.setdefault(tag, {}).update(attrs)
                self.classes[cname] = merged
                # Recurse into nested defaults under this class
                self._collect(child, merged)
            # else: top-level geom/joint defaults (no class) — collected as 'main'
        # Top-level geom/joint defaults form the implicit "main" class
        if "main" not in self.classes:
            main = {}
            for child in node:
                if child.tag == "default":
                    continue
                main[child.tag] = dict(child.attrib)
            if main:
                self.classes["main"] = main

    def get(self, class_name, tag):
        """Get the merged attribute dict for (class_name, tag), with fallback to 'main'."""
        out = {}
        if "main" in self.classes:
            out.update(self.classes["main"].get(tag, {}))
        if class_name and class_name in self.classes:
            out.update(self.classes[class_name].get(tag, {}))
        return out


def resolved_attrs(elem, defaults: Defaults):
    """Return elem's attributes merged with its class defaults."""
    cls = elem.get("class")
    base = defaults.get(cls, elem.tag)
    out = dict(base)
    out.update(elem.attrib)
    return out


# ---------------------------------------------------------------------------
# URDF emission
# ---------------------------------------------------------------------------

def emit_inertial(parent_urdf, mjcf_inertial):
    """MJCF <inertial pos quat? mass fullinertia|diaginertia> -> URDF <inertial>.

    URDF inertial origin RPY is forced to 0; we transform the principal-axis
    diaginertia to body frame when MJCF specifies it via diaginertia+quat.
    """
    pos = get_attr_pos(mjcf_inertial)
    mass = float(mjcf_inertial.get("mass"))

    fullinertia = mjcf_inertial.get("fullinertia")
    diaginertia = mjcf_inertial.get("diaginertia")
    iquat = get_attr_quat(mjcf_inertial)

    if fullinertia is not None:
        # MJCF order: ixx iyy izz ixy ixz iyz (already in body frame)
        ixx, iyy, izz, ixy, ixz, iyz = parse_floats(fullinertia, 6)
    elif diaginertia is not None:
        I1, I2, I3 = parse_floats(diaginertia, 3)
        # If inertial has a quat, rotate diag(I1,I2,I3) into body frame
        if iquat != (1.0, 0.0, 0.0, 0.0):
            w, x, y, z = iquat
            # Build rotation matrix R from quaternion
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
            ])
            I_principal = np.diag([I1, I2, I3])
            I_body = R @ I_principal @ R.T
            ixx, iyy, izz = I_body[0, 0], I_body[1, 1], I_body[2, 2]
            ixy, ixz, iyz = I_body[0, 1], I_body[0, 2], I_body[1, 2]
        else:
            ixx, iyy, izz = I1, I2, I3
            ixy = ixz = iyz = 0.0
    else:
        raise ValueError("inertial missing both fullinertia and diaginertia")

    inertial = ET.SubElement(parent_urdf, "inertial")
    ET.SubElement(inertial, "origin", xyz=fmt_vec(pos), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.10g}")
    ET.SubElement(
        inertial, "inertia",
        ixx=f"{ixx:.10g}", ixy=f"{ixy:.10g}", ixz=f"{ixz:.10g}",
        iyy=f"{iyy:.10g}", iyz=f"{iyz:.10g}", izz=f"{izz:.10g}",
    )


def parse_mesh_table(mjcf_root):
    """Return dict: mesh_name -> file path (as written in MJCF asset)."""
    table = {}
    asset = mjcf_root.find("asset")
    if asset is None:
        return table
    for m in asset.findall("mesh"):
        mname = m.get("name")
        mfile = m.get("file") or m.get("name")
        table[mname] = mfile
    return table


def emit_geoms(link_urdf, mjcf_body, defaults, mesh_table, mesh_rel):
    """Emit URDF <visual>/<collision> for each <geom> child of a body."""
    for geom in mjcf_body.findall("geom"):
        attrs = resolved_attrs(geom, defaults)
        gtype = attrs.get("type", "sphere")
        gname = attrs.get("name", "")

        contype = int(attrs.get("contype", 1))
        conaffinity = int(attrs.get("conaffinity", 1))
        group = int(attrs.get("group", 0))
        is_visual = (contype == 0 and conaffinity == 0) or group == 2
        is_collision = not is_visual
        # Some geoms are both visual+collision in MJCF; in URDF emit only one
        # (collision if it has contact, otherwise visual).
        tag = "collision" if is_collision else "visual"

        # Common origin (use literal MJCF values, not compiled)
        pos = parse_floats(attrs.get("pos", "0 0 0"), 3)
        quat = parse_floats(attrs.get("quat", "1 0 0 0"), 4)
        rpy = quat_wxyz_to_rpy(quat)

        if gtype == "mesh":
            mesh_name = attrs.get("mesh")
            if mesh_name is None:
                continue
            mesh_file = mesh_table.get(mesh_name, mesh_name)
            node = ET.SubElement(link_urdf, tag, name=gname or f"{tag}")
            ET.SubElement(node, "origin", xyz=fmt_vec(pos), rpy=fmt_vec(rpy))
            geom_node = ET.SubElement(node, "geometry")
            ET.SubElement(geom_node, "mesh", filename=f"{mesh_rel}/{mesh_file}")

        elif gtype == "capsule":
            # MJCF capsule: either size="r halflen" + pos/quat, or fromto="x1 y1 z1 x2 y2 z2" size="r"
            fromto = attrs.get("fromto")
            if fromto is not None:
                ft = parse_floats(fromto, 6)
                p1 = np.array(ft[:3]); p2 = np.array(ft[3:])
                mid = (p1 + p2) / 2
                axis_vec = p2 - p1
                length = float(np.linalg.norm(axis_vec))
                if length < 1e-9:
                    continue
                z_hat = axis_vec / length
                # Find rotation that takes +Z to z_hat
                z0 = np.array([0., 0., 1.])
                v = np.cross(z0, z_hat)
                c = float(np.dot(z0, z_hat))
                if np.linalg.norm(v) < 1e-9:
                    # Aligned or anti-aligned
                    if c > 0:
                        R = np.eye(3)
                    else:
                        R = np.diag([1, -1, -1])
                else:
                    s = np.linalg.norm(v)
                    vx = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0],
                    ])
                    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
                # Extract RPY from R
                cap_rpy = matrix_to_rpy(R)
                radius = float(attrs.get("size", "0.01").split()[0])
                node = ET.SubElement(link_urdf, tag, name=gname or f"{tag}")
                ET.SubElement(node, "origin", xyz=fmt_vec(mid), rpy=fmt_vec(cap_rpy))
                geom_node = ET.SubElement(node, "geometry")
                # URDF has no capsule — use cylinder approximation
                ET.SubElement(geom_node, "cylinder",
                              radius=f"{radius:.6g}", length=f"{length:.6g}")
            else:
                size = parse_floats(attrs.get("size"), 2)
                radius, halflen = size
                node = ET.SubElement(link_urdf, tag, name=gname or f"{tag}")
                ET.SubElement(node, "origin", xyz=fmt_vec(pos), rpy=fmt_vec(rpy))
                geom_node = ET.SubElement(node, "geometry")
                ET.SubElement(geom_node, "cylinder",
                              radius=f"{radius:.6g}", length=f"{2*halflen:.6g}")

        elif gtype == "sphere":
            r = float(attrs.get("size", "0.01").split()[0])
            node = ET.SubElement(link_urdf, tag, name=gname or f"{tag}")
            ET.SubElement(node, "origin", xyz=fmt_vec(pos), rpy=fmt_vec(rpy))
            geom_node = ET.SubElement(node, "geometry")
            ET.SubElement(geom_node, "sphere", radius=f"{r:.6g}")

        elif gtype == "box":
            sz = parse_floats(attrs.get("size"), 3)
            node = ET.SubElement(link_urdf, tag, name=gname or f"{tag}")
            ET.SubElement(node, "origin", xyz=fmt_vec(pos), rpy=fmt_vec(rpy))
            geom_node = ET.SubElement(node, "geometry")
            ET.SubElement(geom_node, "box",
                          size=fmt_vec([2*sz[0], 2*sz[1], 2*sz[2]]))

        elif gtype == "plane":
            continue  # world geom
        else:
            print(f"  WARN: unsupported geom type '{gtype}' on {mjcf_body.get('name')}, skipped")


def matrix_to_rpy(R):
    """Extract URDF RPY (roll,pitch,yaw) from 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


JOINT_TYPE_MAP = {
    "hinge": "revolute",
    "slide": "prismatic",
}


def walk_body(mjcf_body, parent_link_name, robot_urdf, defaults, mesh_table,
              mesh_rel, fix_joints):
    """Recursively emit URDF for a MJCF <body> and its descendants."""
    bname = mjcf_body.get("name")
    if bname is None:
        return

    # 1) Emit <link>
    link = ET.SubElement(robot_urdf, "link", name=bname)
    inertial_elem = mjcf_body.find("inertial")
    if inertial_elem is not None:
        emit_inertial(link, inertial_elem)
    emit_geoms(link, mjcf_body, defaults, mesh_table, mesh_rel)

    # 2) Emit <joint> from parent_link_name -> bname (if not root)
    if parent_link_name is not None:
        body_pos = get_attr_pos(mjcf_body)
        body_quat = get_attr_quat(mjcf_body)
        body_rpy = quat_wxyz_to_rpy(body_quat)

        mjcf_joints = [j for j in mjcf_body.findall("joint")]
        mjcf_freejoint = mjcf_body.find("freejoint")

        if mjcf_freejoint is not None:
            # Floating base — not emitted in URDF; the simulator provides it.
            pass
        elif not mjcf_joints:
            j = ET.SubElement(robot_urdf, "joint",
                              name=f"{bname}_fixed", type="fixed")
            ET.SubElement(j, "origin", xyz=fmt_vec(body_pos), rpy=fmt_vec(body_rpy))
            ET.SubElement(j, "parent", link=parent_link_name)
            ET.SubElement(j, "child", link=bname)
        else:
            if len(mjcf_joints) > 1:
                print(f"  WARN: {bname} has {len(mjcf_joints)} joints; URDF supports 1 per pair")
            mj = mjcf_joints[0]
            jname = mj.get("name")
            jtype_mj = mj.get("type", "hinge")
            jattrs = resolved_attrs(mj, defaults)

            if jname in fix_joints:
                j = ET.SubElement(robot_urdf, "joint", name=jname, type="fixed")
                ET.SubElement(j, "origin", xyz=fmt_vec(body_pos), rpy=fmt_vec(body_rpy))
                ET.SubElement(j, "parent", link=parent_link_name)
                ET.SubElement(j, "child", link=bname)
            elif jtype_mj not in JOINT_TYPE_MAP:
                print(f"  WARN: joint {jname} type {jtype_mj} unsupported -> fixed")
                j = ET.SubElement(robot_urdf, "joint", name=jname, type="fixed")
                ET.SubElement(j, "origin", xyz=fmt_vec(body_pos), rpy=fmt_vec(body_rpy))
                ET.SubElement(j, "parent", link=parent_link_name)
                ET.SubElement(j, "child", link=bname)
            else:
                urdf_type = JOINT_TYPE_MAP[jtype_mj]
                axis = parse_floats(jattrs.get("axis", "0 0 1"), 3)
                jpos = parse_floats(jattrs.get("pos", "0 0 0"), 3)
                if any(abs(p) > 1e-9 for p in jpos):
                    print(f"  WARN: joint {jname} has non-zero pos {jpos} (URDF cannot represent directly)")
                rng = jattrs.get("range")
                limited = jattrs.get("limited")
                # MuJoCo: if range is set, treat as limited unless explicitly false
                has_range = rng is not None and (limited != "false")
                if urdf_type == "revolute" and not has_range:
                    urdf_type = "continuous"

                j = ET.SubElement(robot_urdf, "joint", name=jname, type=urdf_type)
                ET.SubElement(j, "origin", xyz=fmt_vec(body_pos), rpy=fmt_vec(body_rpy))
                ET.SubElement(j, "parent", link=parent_link_name)
                ET.SubElement(j, "child", link=bname)
                ET.SubElement(j, "axis", xyz=fmt_vec(axis))

                if urdf_type in ("revolute", "prismatic"):
                    lo, hi = parse_floats(rng, 2)
                    ET.SubElement(j, "limit",
                                  lower=f"{lo:.6g}", upper=f"{hi:.6g}",
                                  effort="200", velocity="20")
                else:
                    ET.SubElement(j, "limit", effort="200", velocity="20")

                damping = jattrs.get("damping")
                if damping and float(damping) > 0:
                    ET.SubElement(j, "dynamics", damping=f"{float(damping):.6g}")

    # 3) Recurse into child bodies
    for child_body in mjcf_body.findall("body"):
        walk_body(child_body, bname, robot_urdf, defaults, mesh_table,
                  mesh_rel, fix_joints)


def build_urdf(mjcf_path: Path, mesh_rel: str, robot_name: str, fix_joints=None):
    fix_joints = set(fix_joints or [])
    tree = ET.parse(str(mjcf_path))
    mjcf_root = tree.getroot()

    defaults = Defaults(mjcf_root)
    mesh_table = parse_mesh_table(mjcf_root)

    worldbody = mjcf_root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")

    robot = ET.Element("robot", name=robot_name)

    # Find the (single) root body under worldbody
    root_bodies = worldbody.findall("body")
    if len(root_bodies) != 1:
        print(f"  WARN: found {len(root_bodies)} root bodies; using the first")
    walk_body(root_bodies[0], None, robot, defaults, mesh_table, mesh_rel,
              fix_joints)

    return robot


def prettify(elem):
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
    ap.add_argument("--mesh-src", type=Path, default=None)
    ap.add_argument("--mesh-rel", default="../meshes")
    ap.add_argument("--name", default="asimov_v1")
    ap.add_argument("--fix-joints", default="")
    args = ap.parse_args()

    mjcf_path = args.mjcf.resolve()
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MJCF: {mjcf_path}")
    fix_joints = [s.strip() for s in args.fix_joints.split(",") if s.strip()]
    if fix_joints:
        print(f"  fixing joints: {fix_joints}")
    robot = build_urdf(mjcf_path, args.mesh_rel, args.name, fix_joints=fix_joints)

    out_path.write_text(prettify(robot), encoding="utf-8")
    print(f"Wrote URDF: {out_path}")

    mesh_src = args.mesh_src or (mjcf_path.parent.parent / "assets" / "meshes")
    mesh_dst = (out_path.parent / args.mesh_rel).resolve()
    if mesh_src.is_dir():
        n = copy_meshes(mesh_src, mesh_dst)
        print(f"Copied {n} mesh files: {mesh_src} -> {mesh_dst}")
    else:
        print(f"WARN: mesh source dir not found: {mesh_src}")


if __name__ == "__main__":
    main()
