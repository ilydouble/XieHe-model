#!/usr/bin/env python3
"""Move imported pose samples from train into val/test by patient group."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_counts(dataset: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        images = {
            path.stem
            for path in (dataset / "images" / split).iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
        labels = {path.stem for path in (dataset / "labels" / split).glob("*.txt")}
        if images != labels:
            raise ValueError(f"{split}图像与标签不配对")
        counts[split] = len(images)
    return counts


def target_counts(total: int) -> dict[str, int]:
    train = round(total * 0.8)
    val = round(total * 0.1)
    return {"train": train, "val": val, "test": total - train - val}


def choose_groups(
    groups: dict[str, list[dict[str, Any]]], target: int, seed: int
) -> set[str]:
    """Find a deterministic patient subset with exactly target samples."""
    patients = sorted(groups)
    random.Random(seed).shuffle(patients)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for patient in patients:
        size = len(groups[patient])
        for count, selected in sorted(reachable.items(), reverse=True):
            new_count = count + size
            if new_count <= target and new_count not in reachable:
                reachable[new_count] = (*selected, patient)
        if target in reachable:
            return set(reachable[target])
    raise ValueError(f"无法按患者整组选择恰好{target}份样本")


def build_plan(
    dataset: Path, import_manifest: Path, *, seed: int = 20260812
) -> dict[str, Any]:
    dataset = dataset.resolve()
    counts_before = split_counts(dataset)
    targets = target_counts(sum(counts_before.values()))
    needed = {
        split: targets[split] - counts_before[split] for split in ("val", "test")
    }
    if any(value < 0 for value in needed.values()):
        raise ValueError(f"现有val/test已经超过目标：{needed}")

    imported = json.loads(import_manifest.read_text(encoding="utf-8"))
    actions = imported.get("actions", {}).get("six_point", [])
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        if action.get("status") != "imported":
            continue
        patient = action.get("assignment_patient_id")
        if not patient:
            raise ValueError(f"导入记录缺少患者ID：{action.get('source_image')}")
        groups[patient].append(action)

    val_patients = choose_groups(dict(groups), needed["val"], seed)
    remaining = {key: value for key, value in groups.items() if key not in val_patients}
    test_patients = choose_groups(remaining, needed["test"], seed + 1)
    destinations = {**{key: "val" for key in val_patients}, **{key: "test" for key in test_patients}}

    moves: list[dict[str, Any]] = []
    for patient, destination_split in sorted(destinations.items()):
        for action in sorted(groups[patient], key=lambda item: item["source_image"]):
            image = Path(action["destination_image"])
            label = Path(action["destination_label"])
            if image.parent.name != "train" or label.parent.name != "train":
                raise ValueError(f"新增样本不在train：{image.name}")
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"待移动图像或标签不存在：{image.name}")
            image_target = dataset / "images" / destination_split / image.name
            label_target = dataset / "labels" / destination_split / label.name
            if image_target.exists() or label_target.exists():
                raise FileExistsError(f"目标文件已存在：{image_target.name}")
            moves.append(
                {
                    "patient_id": patient,
                    "source_split": "train",
                    "destination_split": destination_split,
                    "image_source": str(image),
                    "image_destination": str(image_target),
                    "image_sha256": sha256_file(image),
                    "label_source": str(label),
                    "label_destination": str(label_target),
                    "label_sha256": sha256_file(label),
                    "status": "planned",
                }
            )

    projected = dict(counts_before)
    for split, amount in needed.items():
        projected["train"] -= amount
        projected[split] += amount
    if projected != targets:
        raise AssertionError(f"预计分区不等于目标：{projected} != {targets}")
    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "seed": seed,
        "policy": "保留现有val/test；仅将新增assignment样本按患者整组从train移入val/test",
        "counts_before": counts_before,
        "target_counts": targets,
        "projected_counts": projected,
        "selected_patients": {
            "val": sorted(val_patients),
            "test": sorted(test_patients),
        },
        "moves": moves,
        "mode": "dry_run",
    }


def apply_plan(plan: dict[str, Any]) -> None:
    for move in plan["moves"]:
        image_source = Path(move["image_source"])
        label_source = Path(move["label_source"])
        image_destination = Path(move["image_destination"])
        label_destination = Path(move["label_destination"])
        image_source.replace(image_destination)
        label_source.replace(label_destination)
        if sha256_file(image_destination) != move["image_sha256"]:
            raise RuntimeError(f"图像移动后SHA不一致：{image_destination.name}")
        if sha256_file(label_destination) != move["label_sha256"]:
            raise RuntimeError(f"标签移动后SHA不一致：{label_destination.name}")
        move["status"] = "moved"
    plan["mode"] = "apply"
    plan["applied_at"] = datetime.now().astimezone().isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="按患者重分六点Pose数据集")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--import-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.dataset, args.import_manifest, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.apply:
        apply_plan(plan)
    (args.output / "split_manifest.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "mode": plan["mode"],
        "counts_before": plan["counts_before"],
        "counts_after": split_counts(args.dataset),
        "target_counts": plan["target_counts"],
        "moved_samples": len(plan["moves"]),
        "val_patients": len(plan["selected_patients"]["val"]),
        "test_patients": len(plan["selected_patients"]["test"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
