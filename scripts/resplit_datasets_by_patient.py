#!/usr/bin/env python3
"""Resplit pose datasets by patient while preserving exact requested ratios."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from import_e_drive_training_data import assignment_matches, read_assignment_patient_ids


SPLITS = ("train", "val", "test")
BUSINESS_RE = re.compile(r"(?:^|_)((?:WFLX|FZSY|KWXF|WZSY|WFLQ)\d{4}P\d+)(?:_|\.)")
CR_RE = re.compile(r"^(\d+)__CR_TSPINE_")
UID_RE = re.compile(r"(1\.2\.156\.147522\.44\.410947\.\d+(?:\.\d+){2,})")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_key(name: str, assignment_ids: set[str]) -> tuple[str, bool]:
    matches = assignment_matches(name, assignment_ids)
    if len(matches) > 1:
        raise ValueError(f"文件名匹配多个assignment患者：{name} -> {matches}")
    if matches:
        return f"assignment:{matches[0]}", True
    for prefix, pattern in (("business", BUSINESS_RE), ("server", CR_RE)):
        match = pattern.search(name)
        if match:
            return f"{prefix}:{match.group(1)}", True
    if UID_RE.search(name):
        return f"unparsed_uid:{name}", False
    return f"unparsed:{name}", False


def dataset_items(root: Path, assignment_ids: set[str]) -> list[dict[str, Any]]:
    items = []
    for split in SPLITS:
        images = {path.stem: path for path in (root / "images" / split).glob("*") if path.is_file() and not path.name.startswith(".")}
        labels = {path.stem: path for path in (root / "labels" / split).glob("*.txt") if path.is_file() and not path.name.startswith(".")}
        if images.keys() != labels.keys():
            raise ValueError(f"{root}/{split}图像标签不配对")
        for stem in sorted(images):
            key, parsed = patient_key(images[stem].name, assignment_ids)
            items.append({"split": split, "image": images[stem], "label": labels[stem], "patient_key": key, "parsed": parsed})
    return items


def choose_exact(groups: list[tuple[str, int]], target: int, preferred: set[str]) -> set[str]:
    """Choose whole groups summing to target, maximizing preferred sample count."""
    if target == 0:
        return set()
    ordered = sorted(groups, key=lambda item: (hashlib.sha256(item[0].encode()).hexdigest(), item[0]))
    dp: dict[int, tuple[int, tuple[str, ...]]] = {0: (0, ())}
    for key, size in ordered:
        for total, (score, selected) in list(dp.items())[::-1]:
            new_total = total + size
            if new_total > target:
                continue
            candidate = (score + (size if key in preferred else 0), selected + (key,))
            current = dp.get(new_total)
            if current is None or candidate[0] > current[0] or (candidate[0] == current[0] and candidate[1] < current[1]):
                dp[new_total] = candidate
    if target not in dp:
        sizes = Counter(size for _, size in groups)
        raise ValueError(f"患者组无法精确补足{target}份；可用组大小={dict(sorted(sizes.items()))}")
    return set(dp[target][1])


def plan_dataset(
    root: Path,
    assignment_ids: set[str],
    targets: dict[str, int],
    preferred_targets: dict[str, str] | None = None,
) -> dict[str, Any]:
    items = dataset_items(root, assignment_ids)
    if sum(targets.values()) != len(items):
        raise ValueError(f"目标总量{sum(targets.values())}不等于数据量{len(items)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["patient_key"]].append(item)

    assignments: dict[str, str] = {}
    fixed_counts = Counter()
    train_candidates = []
    for key, members in groups.items():
        original = {member["split"] for member in members}
        parsed = members[0]["parsed"]
        if not parsed:
            split = members[0]["split"]
            assignments[key] = split
            fixed_counts[split] += len(members)
        elif "test" in original:
            assignments[key] = "test"
            fixed_counts["test"] += len(members)
        elif "val" in original:
            assignments[key] = "val"
            fixed_counts["val"] += len(members)
        else:
            train_candidates.append((key, len(members)))

    for split in ("test", "val"):
        needed = targets[split] - fixed_counts[split]
        if needed < 0:
            raise ValueError(f"优先保留现有{split}患者后已有{fixed_counts[split]}份，超过目标{targets[split]}")
        preferred = {key for key, desired in (preferred_targets or {}).items() if desired == split}
        chosen = choose_exact(train_candidates, needed, preferred)
        for key in chosen:
            assignments[key] = split
        train_candidates = [(key, size) for key, size in train_candidates if key not in chosen]
    for key, _ in train_candidates:
        assignments[key] = "train"

    projected = Counter()
    moves = []
    for item in items:
        destination_split = assignments[item["patient_key"]]
        projected[destination_split] += 1
        if destination_split == item["split"]:
            continue
        destination_image = root / "images" / destination_split / item["image"].name
        destination_label = root / "labels" / destination_split / item["label"].name
        if destination_image.exists() or destination_label.exists():
            raise FileExistsError(f"迁移目标已存在：{destination_image}")
        moves.append({
            "patient_key": item["patient_key"], "source_split": item["split"], "destination_split": destination_split,
            "image_source": str(item["image"].resolve()), "image_destination": str(destination_image.resolve()),
            "image_sha256": sha256_file(item["image"]), "label_source": str(item["label"].resolve()),
            "label_destination": str(destination_label.resolve()), "label_sha256": sha256_file(item["label"]),
        })
    if dict(projected) != targets:
        raise RuntimeError(f"计划分区不等于目标：{dict(projected)} != {targets}")
    leaks = []
    for key, members in groups.items():
        if members[0]["parsed"] and key not in assignments:
            leaks.append(key)
    return {
        "root": str(root.resolve()), "total": len(items),
        "counts_before": dict(Counter(item["split"] for item in items)), "target_counts": targets,
        "projected_counts": dict(projected), "parsed_groups": sum(members[0]["parsed"] for members in groups.values()),
        "unparsed_samples_fixed": sum(len(members) for members in groups.values() if not members[0]["parsed"]),
        "patient_assignments": assignments, "projected_patient_leaks": leaks, "moves": moves,
    }


def apply_moves(plans: dict[str, dict[str, Any]]) -> None:
    completed = []
    try:
        for plan in plans.values():
            for move in plan["moves"]:
                for source_key, destination_key, hash_key in (
                    ("image_source", "image_destination", "image_sha256"),
                    ("label_source", "label_destination", "label_sha256"),
                ):
                    source = Path(move[source_key]); destination = Path(move[destination_key])
                    destination.parent.mkdir(parents=True, exist_ok=True); shutil.move(source, destination)
                    completed.append((source, destination))
                    if sha256_file(destination) != move[hash_key]:
                        raise RuntimeError(f"移动后哈希不一致：{destination}")
                move["status"] = "moved"
    except Exception:
        for source, destination in reversed(completed):
            source.parent.mkdir(parents=True, exist_ok=True); shutil.move(destination, source)
        raise


def patient_leaks(root: Path, assignment_ids: set[str]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item in dataset_items(root, assignment_ids):
        if item["parsed"]:
            by_key[item["patient_key"]][item["split"]].append(item["image"].name)
    return [
        {"patient_key": key, "splits": dict(splits)}
        for key, splits in sorted(by_key.items()) if len(splits) > 1
    ]


def parse_targets(value: str) -> dict[str, int]:
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("目标格式必须为train,val,test")
    return dict(zip(SPLITS, parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-xlsx", required=True, type=Path); parser.add_argument("--pose-root", required=True, type=Path)
    parser.add_argument("--corner-root", required=True, type=Path); parser.add_argument("--pose-targets", type=parse_targets, default=parse_targets("1404,176,175"))
    parser.add_argument("--corner-targets", type=parse_targets, default=parse_targets("1999,250,250")); parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--apply", action="store_true"); args = parser.parse_args()
    ids = read_assignment_patient_ids(args.assignment_xlsx)
    pose = plan_dataset(args.pose_root.resolve(), ids, args.pose_targets)
    preferred = {key: split for key, split in pose["patient_assignments"].items() if key.startswith("assignment:")}
    corner = plan_dataset(args.corner_root.resolve(), ids, args.corner_targets, preferred)
    plans = {"six_point": pose, "spine_pose": corner}
    if args.apply:
        apply_moves(plans)
        for task, plan in plans.items():
            root = Path(plan["root"]); total, counts = len(dataset_items(root, ids)), Counter(x["split"] for x in dataset_items(root, ids))
            plan["counts_after"] = dict(counts); plan["patient_leaks_after"] = patient_leaks(root, ids)
            if total != plan["total"] or dict(counts) != plan["target_counts"] or plan["patient_leaks_after"]:
                raise RuntimeError(f"{task}重分后验证失败")
    manifest = {
        "schema_version": 1, "generated_at": datetime.now().astimezone().isoformat(),
        "policy": "patient_group_test_priority_then_val_exact_80_10_10_unparsed_fixed",
        "mode": "apply_verified" if args.apply else "dry_run", "datasets": plans,
    }
    args.record_dir.mkdir(parents=True, exist_ok=True)
    (args.record_dir / "resplit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({task: {"before": p["counts_before"], "projected": p["projected_counts"], "moves": len(p["moves"]), "unparsed_fixed": p["unparsed_samples_fixed"], "leaks_after": len(p.get("patient_leaks_after", []))} for task,p in plans.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
