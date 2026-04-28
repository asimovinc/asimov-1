#!/usr/bin/env python3
"""Generate the Asimov v1 fabrication manifest from the CAD tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CAD_ROOT = REPO_ROOT / "mechanical" / "ASV1"
CSV_PATH = REPO_ROOT / "mechanical" / "FABRICATION_MANIFEST.csv"
JSON_PATH = REPO_ROOT / "mechanical" / "FABRICATION_MANIFEST.json"
FABRICATION_ORDER = {
    "ALU_7075": 0,
    "SML_316L": 1,
    "MJF_PA12": 2,
    "OFF_THE_SHELF": 3,
}
CSV_FIELDS = [
    "subassembly",
    "fabrication_class",
    "part_id",
    "file_name",
    "path",
    "size_bytes",
    "sha256",
]


def repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def part_id_from_file(path: Path) -> str:
    name = path.name
    while name.lower().endswith(".step"):
        name = name[:-5]
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_entries(cad_root: Path = CAD_ROOT) -> list[dict[str, str | int]]:
    entries: list[dict[str, str | int]] = []
    for file_path in cad_root.glob("*/FABRICATION/*/*"):
        if not file_path.is_file() or not file_path.name.lower().endswith(".step"):
            continue
        fabrication_dir = file_path.parent
        subassembly_dir = fabrication_dir.parent.parent
        if subassembly_dir.parent != cad_root:
            continue
        entries.append(
            {
                "subassembly": subassembly_dir.name,
                "fabrication_class": fabrication_dir.name,
                "part_id": part_id_from_file(file_path),
                "file_name": file_path.name,
                "path": repo_path(file_path),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    entries.sort(
        key=lambda entry: (
            entry["subassembly"],
            FABRICATION_ORDER.get(str(entry["fabrication_class"]), 99),
            entry["fabrication_class"],
            entry["part_id"],
            entry["file_name"],
        )
    )
    return entries


def manifest_payload(entries: list[dict[str, str | int]]) -> dict[str, object]:
    subassemblies = sorted({str(entry["subassembly"]) for entry in entries})
    fabrication_classes = sorted(
        {str(entry["fabrication_class"]) for entry in entries},
        key=lambda value: (FABRICATION_ORDER.get(value, 99), value),
    )
    return {
        "schema_version": 1,
        "source_root": "mechanical/ASV1",
        "entry_count": len(entries),
        "subassemblies": subassemblies,
        "fabrication_classes": fabrication_classes,
        "entries": entries,
    }


def csv_text(entries: list[dict[str, str | int]]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(entries)
    return output.getvalue()


def json_text(entries: list[dict[str, str | int]]) -> str:
    return json.dumps(manifest_payload(entries), indent=2) + "\n"


def write_manifest(entries: list[dict[str, str | int]]) -> None:
    CSV_PATH.write_text(csv_text(entries), encoding="utf-8")
    JSON_PATH.write_text(json_text(entries), encoding="utf-8")


def check_manifest(entries: list[dict[str, str | int]]) -> list[str]:
    errors: list[str] = []
    expected_csv = csv_text(entries)
    expected_json = json_text(entries)

    if not CSV_PATH.exists():
        errors.append(f"missing {repo_path(CSV_PATH)}")
    elif CSV_PATH.read_text(encoding="utf-8") != expected_csv:
        errors.append(f"{repo_path(CSV_PATH)} is out of date")

    if not JSON_PATH.exists():
        errors.append(f"missing {repo_path(JSON_PATH)}")
    elif JSON_PATH.read_text(encoding="utf-8") != expected_json:
        errors.append(f"{repo_path(JSON_PATH)} is out of date")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or check the Asimov v1 fabrication manifest."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that committed manifest files match the CAD tree",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = build_entries()
    if args.check:
        errors = check_manifest(entries)
        if errors:
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            print(
                "run `python3 scripts/generate_fabrication_manifest.py` "
                "and commit the updated manifest",
                file=sys.stderr,
            )
            return 1
        print(f"fabrication manifest is current ({len(entries)} entries)")
        return 0

    write_manifest(entries)
    print(f"wrote {repo_path(CSV_PATH)}")
    print(f"wrote {repo_path(JSON_PATH)}")
    print(f"entries: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
