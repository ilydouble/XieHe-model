#!/usr/bin/env python3
"""Move previously imported samples absent from assignment_all into quarantine."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from import_e_drive_training_data import assignment_matches, read_assignment_patient_ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="隔离不在assignment_all白名单的既有导入文件")
    parser.add_argument("--import-manifest", required=True, type=Path)
    parser.add_argument("--assignment-xlsx", required=True, type=Path)
    parser.add_argument("--quarantine-dir", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="实际移动；默认仅生成预演清单")
    return parser.parse_args()


def build_plan(import_manifest: Path, assignment_xlsx: Path) -> dict:
    manifest = json.loads(import_manifest.read_text(encoding="utf-8"))
    patient_ids = read_assignment_patient_ids(assignment_xlsx)
    records = []
    for task, actions in manifest["actions"].items():
        for action in actions:
            if action["status"] != "imported" or assignment_matches(
                action["source_image"], patient_ids
            ):
                continue
            paths = [Path(action["destination_image"]), Path(action["destination_label"])]
            records.append(
                {
                    "task": task,
                    "source_image": action["source_image"],
                    "files": [
                        {
                            "path": str(path.resolve()),
                            "exists": path.is_file(),
                            "sha256": sha256_file(path) if path.is_file() else None,
                        }
                        for path in paths
                    ],
                }
            )
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_import_manifest": str(import_manifest.resolve()),
        "assignment_xlsx": str(assignment_xlsx.resolve()),
        "assignment_patient_ids": len(patient_ids),
        "records": records,
    }


def apply_plan(plan: dict, quarantine_dir: Path) -> None:
    for record in plan["records"]:
        for file_record in record["files"]:
            source = Path(file_record["path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            relative = Path(source.parent.parent.name) / source.parent.name / source.name
            destination = quarantine_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(source, destination)
            if sha256_file(destination) != file_record["sha256"]:
                raise RuntimeError(f"隔离后SHA-256不一致：{source}")
            file_record["quarantine_path"] = str(destination.resolve())
            file_record["status"] = "quarantined"


def main() -> int:
    args = parse_args()
    plan = build_plan(args.import_manifest, args.assignment_xlsx)
    args.quarantine_dir.mkdir(parents=True, exist_ok=True)
    if args.apply:
        apply_plan(plan, args.quarantine_dir)
    plan["mode"] = "apply" if args.apply else "dry_run"
    (args.quarantine_dir / "manifest.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(plan["records"]), "mode": plan["mode"]}, ensure_ascii=False))
    return int(any(not file["exists"] for record in plan["records"] for file in record["files"]))


if __name__ == "__main__":
    raise SystemExit(main())
