#!/usr/bin/env python3
"""Validate repo-local Asimov v1 hardware and simulation artifacts."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import generate_fabrication_manifest as fabrication_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
WIRING_PATH = REPO_ROOT / "electrical" / "wiring.yaml"
MJCF_PATH = REPO_ROOT / "sim-model" / "xmls" / "asimov.xml"
EXPECTED_SUBASSEMBLIES = ["100", "200", "300", "400", "500", "600", "700"]
INLINE_LIST_RE = re.compile(r"^\[(.*)\]$")


class ContractError(RuntimeError):
    pass


def repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_inline_list(value: str) -> list[str | int]:
    match = INLINE_LIST_RE.match(value.strip())
    if not match:
        raise ContractError(f"expected inline list, got {value!r}")
    result: list[str | int] = []
    for item in match.group(1).split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            result.append(int(item))
        else:
            result.append(strip_quotes(item))
    return result


def parse_wiring_yaml(path: Path) -> dict[str, object]:
    connectors: dict[str, dict[str, object]] = {}
    cables: dict[str, dict[str, object]] = {}
    connections: list[list[tuple[str, list[str | int]]]] = []
    section = None
    current_name = None
    current_connection: list[tuple[str, list[str | int]]] | None = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith(" ") and stripped.endswith(":"):
            section = stripped[:-1]
            current_name = None
            current_connection = None
            continue

        if section in {"connectors", "cables"}:
            if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
                current_name = strip_quotes(stripped[:-1])
                target = connectors if section == "connectors" else cables
                target[current_name] = {}
                continue
            if current_name is None or not line.startswith("    "):
                raise ContractError(f"{repo_path(path)}:{line_number}: invalid {section} entry")
            key, raw_value = stripped.split(":", 1)
            raw_value = raw_value.strip()
            target = connectors if section == "connectors" else cables
            if raw_value.startswith("["):
                target[current_name][key] = parse_inline_list(raw_value)
            elif key == "wirecount":
                target[current_name][key] = int(raw_value)
            else:
                target[current_name][key] = strip_quotes(raw_value)
            continue

        if section == "connections":
            if line.startswith("  -"):
                current_connection = []
                connections.append(current_connection)
                continue
            if current_connection is None or not line.startswith("    - "):
                raise ContractError(f"{repo_path(path)}:{line_number}: invalid connection entry")
            entry = stripped[2:].strip()
            key, raw_pins = entry.split(":", 1)
            current_connection.append((strip_quotes(key), parse_inline_list(raw_pins.strip())))
            continue

        raise ContractError(f"{repo_path(path)}:{line_number}: unknown section {section!r}")

    return {
        "connectors": connectors,
        "cables": cables,
        "connections": connections,
    }


def validate_fabrication_manifest() -> str:
    entries = fabrication_manifest.build_entries()
    errors = fabrication_manifest.check_manifest(entries)
    subassemblies = sorted({str(entry["subassembly"]) for entry in entries})
    part_keys = [
        (str(entry["subassembly"]), str(entry["part_id"]))
        for entry in entries
    ]
    duplicate_parts = sorted(
        "/".join(key) for key, count in Counter(part_keys).items() if count > 1
    )
    if subassemblies != EXPECTED_SUBASSEMBLIES:
        errors.append(
            "expected subassemblies "
            f"{', '.join(EXPECTED_SUBASSEMBLIES)}, got {', '.join(subassemblies)}"
        )
    if duplicate_parts:
        errors.append(f"duplicate fabrication part ids: {', '.join(duplicate_parts)}")
    if errors:
        raise ContractError("; ".join(errors))
    classes = Counter(str(entry["fabrication_class"]) for entry in entries)
    class_summary = ", ".join(f"{key}={classes[key]}" for key in sorted(classes))
    return f"fabrication manifest: {len(entries)} entries ({class_summary})"


def validate_mjcf() -> str:
    model = ET.parse(MJCF_PATH).getroot()
    compiler = model.find("compiler")
    if compiler is None:
        raise ContractError(f"{repo_path(MJCF_PATH)} is missing <compiler>")
    mesh_dir = (MJCF_PATH.parent / compiler.attrib.get("meshdir", "")).resolve()

    meshes = model.findall(".//mesh")
    missing_meshes: list[str] = []
    for mesh in meshes:
        file_name = mesh.attrib.get("file")
        if file_name and not (mesh_dir / file_name).exists():
            missing_meshes.append(file_name)
    if missing_meshes:
        raise ContractError(f"missing MuJoCo mesh files: {', '.join(missing_meshes)}")

    joints = {joint.attrib["name"] for joint in model.findall(".//joint") if "name" in joint.attrib}
    missing_joint_refs = [
        motor.attrib.get("joint", "")
        for motor in model.findall(".//motor")
        if motor.attrib.get("joint") not in joints
    ]
    if missing_joint_refs:
        raise ContractError(f"missing MuJoCo motor joint refs: {', '.join(missing_joint_refs)}")

    sensor_root = model.find("sensor")
    sensors = [
        sensor
        for sensor in (sensor_root if sensor_root is not None else [])
        if isinstance(sensor.tag, str)
    ]
    return (
        f"MuJoCo model: {len(meshes)} mesh refs, {len(joints)} joints, "
        f"{len(model.findall('.//motor'))} motors, {len(sensors)} sensors"
    )


def validate_wiring() -> str:
    parsed = parse_wiring_yaml(WIRING_PATH)
    connectors = parsed["connectors"]
    cables = parsed["cables"]
    connections = parsed["connections"]
    assert isinstance(connectors, dict)
    assert isinstance(cables, dict)
    assert isinstance(connections, list)

    connector_refs: set[str] = set()
    cable_refs: set[str] = set()
    errors: list[str] = []

    for index, chain in enumerate(connections, 1):
        for name, pins in chain:
            if name in connectors:
                connector_refs.add(name)
                pinlabels = set(connectors[name].get("pinlabels", []))
                for pin in pins:
                    if pin not in pinlabels:
                        errors.append(f"connection {index}: {name}.{pin}")
            elif name in cables:
                cable_refs.add(name)
                wirecount = int(cables[name].get("wirecount", 0))
                for pin in pins:
                    if not isinstance(pin, int) or pin < 1 or pin > wirecount:
                        errors.append(f"connection {index}: {name}.{pin}")
            else:
                errors.append(f"connection {index}: unknown endpoint {name}")

    unused_connectors = sorted(set(connectors) - connector_refs)
    unused_cables = sorted(set(cables) - cable_refs)
    if unused_connectors:
        errors.append(f"unused connectors: {', '.join(unused_connectors)}")
    if unused_cables:
        errors.append(f"unused cables: {', '.join(unused_cables)}")
    if errors:
        raise ContractError("; ".join(errors))

    return (
        f"WireViz harness: {len(connectors)} connectors, {len(cables)} cables, "
        f"{len(connections)} connections"
    )


def main() -> int:
    checks = [
        validate_fabrication_manifest,
        validate_mjcf,
        validate_wiring,
    ]
    for check in checks:
        try:
            print(check())
        except ContractError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
