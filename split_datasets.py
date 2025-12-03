#!/usr/bin/env python3
"""Utility to split YOLO pose and segmentation datasets into train/val/test folders."""

from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
from pathlib import Path
import re

VALID_IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split YOLO pose/seg datasets into train/val/test subsets."
    )
    default_base = Path(__file__).resolve().parent
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help="Base directory that contains seg-data and pose-data folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["seg", "pose"],
        default=["seg", "pose"],
        help="Datasets to process. Defaults to both seg and pose.",
    )
    parser.add_argument(
        "--seg-dir",
        type=Path,
        help="Optional explicit path to the segmentation dataset root.",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        help="Optional explicit path to the pose dataset root.",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--test",
        type=float,
        help="Test split ratio. Defaults to 1 - train - val when omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting (default: 42).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move files into split folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild splits even if existing split folders already contain files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the intended split counts without moving any files.",
    )
    return parser.parse_args()


def resolve_ratios(train_ratio: float, val_ratio: float, test_ratio: float | None) -> tuple[float, float, float]:
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Split ratios must be non-negative and sum to 1.0.")
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0 (got {total:.6f}).")
    return train_ratio, val_ratio, test_ratio


def compute_split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total == 0:
        return 0, 0, 0
    raw_counts = [total * r for r in ratios]
    floors = [int(math.floor(c)) for c in raw_counts]
    remainder = total - sum(floors)
    fractional = [c - math.floor(c) for c in raw_counts]
    order = sorted(range(len(ratios)), key=lambda idx: fractional[idx], reverse=True)
    for idx in order[:remainder]:
        floors[idx] += 1
    return tuple(floors)


def collect_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    image_files = {}
    for entry in images_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name.startswith('.'):  # ignore hidden artefacts like .DS_Store
            continue
        if entry.suffix.lower() not in VALID_IMAGE_EXTS:
            continue
        image_files.setdefault(entry.stem, entry)
    label_files = {}
    for entry in labels_dir.glob("*.txt"):
        if entry.name.startswith('.'):
            continue
        label_files.setdefault(entry.stem, entry)
    shared = sorted(set(image_files) & set(label_files))
    missing_images = sorted(set(label_files) - set(image_files))
    missing_labels = sorted(set(image_files) - set(label_files))
    pairs = [(image_files[name], label_files[name]) for name in shared]
    return pairs, missing_images, missing_labels


def count_split_files(images_dir: Path, labels_dir: Path) -> int:
    total = 0
    for root in (images_dir, labels_dir):
        for split in SPLIT_NAMES:
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            total += sum(1 for _ in split_dir.iterdir())
    return total


def flatten_existing_splits(images_dir: Path, labels_dir: Path) -> None:
    for root in (images_dir, labels_dir):
        for split in SPLIT_NAMES:
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            for item in split_dir.iterdir():
                if item.is_file():
                    target = root / item.name
                    if target.exists():
                        raise RuntimeError(f"Target {target} already exists while flattening splits.")
                    shutil.move(str(item), str(target))
            shutil.rmtree(split_dir)


def prepare_destination_dirs(images_dir: Path, labels_dir: Path) -> None:
    for root in (images_dir, labels_dir):
        for split in SPLIT_NAMES:
            target = root / split
            target.mkdir(parents=True, exist_ok=True)


def move_pairs(
    split_map: dict[str, list[tuple[Path, Path]]],
    images_dir: Path,
    labels_dir: Path,
    copy_files: bool,
) -> None:
    for split_name, items in split_map.items():
        image_target_dir = images_dir / split_name
        label_target_dir = labels_dir / split_name
        for image_path, label_path in items:
            dst_image = image_target_dir / image_path.name
            dst_label = label_target_dir / label_path.name
            if copy_files:
                shutil.copy2(image_path, dst_image)
                shutil.copy2(label_path, dst_label)
            else:
                shutil.move(str(image_path), str(dst_image))
                shutil.move(str(label_path), str(dst_label))


def replace_yaml_line(lines: list[str], key: str, value: str) -> list[str]:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]*)(.*)$")
    for idx, line in enumerate(lines):
        content = line.rstrip("\n")
        match = pattern.match(content)
        if match:
            prefix, _, suffix = match.groups()
            newline = "\n" if line.endswith("\n") else ""
            lines[idx] = f"{prefix}{value}{suffix}{newline}"
            return lines
    newline = "\n" if lines and not lines[-1].endswith("\n") else ""
    lines.append(f"{newline}{key}: {value}\n")
    return lines


def update_dataset_yaml(dataset_yaml: Path, dataset_root: Path) -> None:
    if not dataset_yaml.is_file():
        print(f"[WARN] dataset.yaml not found at {dataset_yaml}, skipped path update.")
        return
    try:
        lines = dataset_yaml.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        print(f"[WARN] Could not read {dataset_yaml} due to encoding issues.")
        return
    abs_path = str(dataset_root.resolve())
    lines = replace_yaml_line(lines, "path", abs_path)
    lines = replace_yaml_line(lines, "train", "images/train")
    lines = replace_yaml_line(lines, "val", "images/val")
    lines = replace_yaml_line(lines, "test", "images/test")
    dataset_yaml.write_text("".join(lines), encoding="utf-8")


def split_dataset(
    dataset_name: str,
    dataset_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    copy_files: bool,
    force: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"{dataset_root} must contain 'images' and 'labels' directories.")

    existing_split_items = count_split_files(images_dir, labels_dir)
    pairs, missing_images, missing_labels = collect_pairs(images_dir, labels_dir)
    dataset_yaml = dataset_root / "dataset.yaml"

    if missing_images:
        print(f"[WARN] {dataset_name}: {len(missing_images)} label files without images.", file=sys.stderr)
    if missing_labels:
        print(f"[WARN] {dataset_name}: {len(missing_labels)} images without labels.", file=sys.stderr)

    if existing_split_items and not pairs and not force:
        if not dry_run:
            update_dataset_yaml(dataset_yaml, dataset_root)
        print(f"[INFO] {dataset_name}: split directories already populated, skipping (use --force to rebuild).")
        return 0, 0, 0

    if existing_split_items and force:
        if dry_run:
            raise RuntimeError(f"{dataset_name}: cannot flatten existing splits during dry-run.")
        print(f"[INFO] {dataset_name}: flattening existing split directories before rebuilding.")
        flatten_existing_splits(images_dir, labels_dir)
        pairs, missing_images, missing_labels = collect_pairs(images_dir, labels_dir)

    if not pairs:
        print(f"[INFO] {dataset_name}: no paired image/label files found, nothing to split.")
        return 0, 0, 0

    rng = random.Random(seed)
    rng.shuffle(pairs)
    train_count, val_count, test_count = compute_split_counts(len(pairs), (train_ratio, val_ratio, test_ratio))
    split_map: dict[str, list[tuple[Path, Path]]] = {
        "train": pairs[:train_count],
        "val": pairs[train_count:train_count + val_count],
        "test": pairs[train_count + val_count:],
    }

    if dry_run:
        print(
            f"[DRY-RUN] {dataset_name}: total={len(pairs)} train={train_count} val={val_count} test={test_count}"
        )
        return train_count, val_count, test_count

    prepare_destination_dirs(images_dir, labels_dir)
    move_pairs(split_map, images_dir, labels_dir, copy_files)

    update_dataset_yaml(dataset_yaml, dataset_root)

    print(
        f"[INFO] {dataset_name}: split {len(pairs)} pairs -> train {train_count}, val {val_count}, test {test_count}."
    )
    return train_count, val_count, test_count


def main() -> None:
    args = parse_args()
    try:
        train_ratio, val_ratio, test_ratio = resolve_ratios(args.train, args.val, args.test)
    except ValueError as err:
        print(f"[ERROR] {err}")
        sys.exit(1)

    base_dir = args.base_dir.resolve()
    dataset_paths: dict[str, Path] = {}
    if "seg" in args.datasets:
        dataset_paths["seg"] = (args.seg_dir or base_dir / "seg-data").resolve()
    if "pose" in args.datasets:
        dataset_paths["pose"] = (args.pose_dir or base_dir / "pose-data").resolve()

    if not dataset_paths:
        print("[ERROR] No datasets selected for splitting.")
        sys.exit(1)

    success = True
    for name, path in dataset_paths.items():
        try:
            split_dataset(
                dataset_name=name,
                dataset_root=path,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=args.seed,
                copy_files=args.copy,
                force=args.force,
                dry_run=args.dry_run,
            )
        except Exception as err:  # noqa: BLE001
            success = False
            print(f"[ERROR] Failed to split '{name}' dataset: {err}", file=sys.stderr)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
