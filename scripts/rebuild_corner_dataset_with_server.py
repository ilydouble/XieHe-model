#!/usr/bin/env python3
"""Replace the inaccurate legacy corner set with the deduplicated server set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="移出旧743份角点，保留eap人工871份并融合服务器去重后的1381份。"
    )
    parser.add_argument("--server-root", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--quarantine-dir", required=True, type=Path)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--expected-legacy", type=int, default=743)
    parser.add_argument("--expected-retained", type=int, default=871)
    parser.add_argument("--expected-server", type=int, default=1381)
    parser.add_argument("--apply", action="store_true", help="实际重建；默认仅预演")
    parser.add_argument("--verify-existing-manifest", type=Path, help="只验证已完成重建及隔离哈希")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_files(root: Path) -> dict[Path, tuple[Path, Path]]:
    images = {
        Path(split) / path.name: path
        for split in SPLITS
        for path in sorted((root / "images" / split).glob("*"))
        if path.is_file() and not path.name.startswith("._")
    }
    labels = {
        Path(split) / path.with_suffix(".png").name: path
        for split in SPLITS
        for path in sorted((root / "labels" / split).glob("*.txt"))
        if path.is_file() and not path.name.startswith("._")
    }
    if images.keys() != labels.keys():
        missing_labels = sorted(str(path) for path in images.keys() - labels.keys())
        missing_images = sorted(str(path) for path in labels.keys() - images.keys())
        raise ValueError(
            f"图像标签不配对：missing_labels={missing_labels[:5]} missing_images={missing_images[:5]}"
        )
    return {relative: (images[relative], labels[relative]) for relative in sorted(images)}


def sanitize_label(path: Path) -> tuple[str, dict[str, object]]:
    kept: list[str] = []
    removed: list[int] = []
    classes: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 17:
            raise ValueError(f"标签列数不是17：{path}:{line_number}")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        coordinate_indices = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14)
        if any(not 0.0 <= values[index] <= 1.0 for index in coordinate_indices):
            raise ValueError(f"坐标越界：{path}:{line_number}")
        if any(values[index] not in (0.0, 1.0, 2.0) for index in (6, 9, 12, 15)):
            raise ValueError(f"visibility非法：{path}:{line_number}")
        tl, tr, br, bl = (
            (values[4], values[5]),
            (values[7], values[8]),
            (values[10], values[11]),
            (values[13], values[14]),
        )
        if not (tl[0] < tr[0] and bl[0] < br[0] and (tl[1] + tr[1]) < (bl[1] + br[1])):
            raise ValueError(f"TL/TR/BR/BL几何顺序异常：{path}:{line_number}")
        if 0 <= class_id <= 17:
            kept.append(" ".join(parts))
            classes.append(class_id)
        else:
            removed.append(class_id)
    if len(classes) != len(set(classes)):
        raise ValueError(f"存在重复椎体类别：{path}")
    missing = sorted(set(range(18)) - set(classes))
    if missing not in ([], [12]):
        raise ValueError(f"缺失类别超出已知例外：{path} missing={missing}")
    return "\n".join(kept) + "\n", {
        "classes": classes,
        "missing_classes": missing,
        "removed_classes": removed,
    }


def split_counts(paths: list[Path]) -> dict[str, int]:
    counts = Counter(path.parts[0] for path in paths)
    return {split: counts[split] for split in SPLITS}


def build_plan(
    server_root: Path,
    target_root: Path,
    *,
    expected_legacy: int,
    expected_retained: int,
    expected_server: int,
) -> dict[str, object]:
    target = paired_files(target_root)
    server = paired_files(server_root)
    retained = {path: pair for path, pair in target.items() if path.name.startswith("eap_")}
    legacy = {path: pair for path, pair in target.items() if not path.name.startswith("eap_")}
    actual = (len(legacy), len(retained), len(server))
    expected = (expected_legacy, expected_retained, expected_server)
    if actual != expected:
        raise ValueError(f"数据规模与安全门槛不符：actual={actual} expected={expected}")

    retained_names = {path.name for path in retained}
    collisions = sorted(path.name for path in server if path.name in retained_names)
    if collisions:
        raise ValueError(f"服务器文件名与保留人工数据冲突：{collisions[:10]}")

    server_records = []
    removed_class_files = 0
    missing_class_files = 0
    for relative, (image, label) in server.items():
        sanitized, info = sanitize_label(label)
        removed_class_files += bool(info["removed_classes"])
        missing_class_files += bool(info["missing_classes"])
        server_records.append(
            {
                "relative": str(relative),
                "source_image": str(image.resolve()),
                "source_label": str(label.resolve()),
                "source_image_sha256": sha256_file(image),
                "source_label_sha256": sha256_file(label),
                "sanitized_label_sha256": hashlib.sha256(sanitized.encode()).hexdigest(),
                **info,
            }
        )

    legacy_records = [
        {
            "relative": str(relative),
            "image": str(image.resolve()),
            "label": str(label.resolve()),
            "image_sha256": sha256_file(image),
            "label_sha256": sha256_file(label),
        }
        for relative, (image, label) in legacy.items()
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "strategy": "replace_legacy_743_with_server_1381_keep_eap_871",
        "server_root": str(server_root.resolve()),
        "target_root": str(target_root.resolve()),
        "before": {
            "legacy": len(legacy),
            "retained_eap": len(retained),
            "total": len(target),
            "split_counts": split_counts(list(target)),
        },
        "server": {
            "total": len(server),
            "split_counts": split_counts(list(server)),
            "files_with_removed_class_18_19": removed_class_files,
            "files_missing_class_12": missing_class_files,
        },
        "expected_after": {
            "total": len(retained) + len(server),
            "split_counts": {
                split: split_counts(list(retained))[split] + split_counts(list(server))[split]
                for split in SPLITS
            },
        },
        "legacy_records": legacy_records,
        "server_records": server_records,
    }


def write_manifest(plan: dict[str, object], record_dir: Path, mode: str) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    plan["mode"] = mode
    (record_dir / "rebuild_manifest.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def audit_dataset(root: Path) -> dict[str, object]:
    pairs = paired_files(root)
    image_hashes: dict[str, list[str]] = {}
    issues: list[str] = []
    missing_class_12: list[str] = []
    from PIL import Image

    for relative, (image, label) in pairs.items():
        try:
            with Image.open(image) as source:
                source.load()
        except Exception as error:
            issues.append(f"图像解码失败：{relative}: {error}")
        digest = sha256_file(image)
        image_hashes.setdefault(digest, []).append(str(relative))
        try:
            _, info = sanitize_label(label)
            if info["removed_classes"]:
                issues.append(f"仍有超范围类别：{relative}: {info['removed_classes']}")
            if info["missing_classes"] == [12]:
                missing_class_12.append(str(relative))
        except Exception as error:
            issues.append(str(error))
    duplicate_groups = [paths for paths in image_hashes.values() if len(paths) > 1]
    cross_split = [paths for paths in duplicate_groups if len({Path(path).parts[0] for path in paths}) > 1]
    return {
        "total": len(pairs),
        "split_counts": split_counts(list(pairs)),
        "issues": issues,
        "missing_class_12": missing_class_12,
        "exact_duplicate_groups": duplicate_groups,
        "cross_split_exact_duplicate_groups": cross_split,
    }


def verify_quarantine(plan: dict[str, object], quarantine_dir: Path) -> dict[str, object]:
    issues = []
    checked = 0
    for record in plan["legacy_records"]:  # type: ignore[index]
        relative = Path(record["relative"])
        image = quarantine_dir / "images" / relative
        label = quarantine_dir / "labels" / relative.with_suffix(".txt")
        for path, expected in ((image, record["image_sha256"]), (label, record["label_sha256"])):
            checked += 1
            if not path.is_file():
                issues.append(f"隔离文件缺失：{path}")
            elif sha256_file(path) != expected:
                issues.append(f"隔离文件哈希不一致：{path}")
    return {"samples": len(plan["legacy_records"]), "files_checked": checked, "issues": issues}


def apply_plan(plan: dict[str, object], target_root: Path, quarantine_dir: Path) -> None:
    if quarantine_dir.exists() and any(quarantine_dir.iterdir()):
        raise FileExistsError(f"隔离目录非空：{quarantine_dir}")
    moved: list[tuple[Path, Path]] = []
    copied: list[Path] = []
    try:
        for record in plan["legacy_records"]:  # type: ignore[index]
            relative = Path(record["relative"])
            for kind, suffix in (("images", relative.suffix), ("labels", ".txt")):
                source = target_root / kind / relative.with_suffix(suffix)
                destination = quarantine_dir / kind / relative.with_suffix(suffix)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, destination)
                moved.append((source, destination))

        for record in plan["server_records"]:  # type: ignore[index]
            relative = Path(record["relative"])
            destination_image = target_root / "images" / relative
            destination_label = target_root / "labels" / relative.with_suffix(".txt")
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["source_image"], destination_image)
            copied.append(destination_image)
            sanitized, _ = sanitize_label(Path(record["source_label"]))
            destination_label.write_text(sanitized, encoding="utf-8")
            copied.append(destination_label)
            if sha256_file(destination_image) != record["source_image_sha256"]:
                raise RuntimeError(f"图像复制后哈希不一致：{destination_image}")
            if sha256_file(destination_label) != record["sanitized_label_sha256"]:
                raise RuntimeError(f"标签写入后哈希不一致：{destination_label}")
    except Exception:
        for path in reversed(copied):
            path.unlink(missing_ok=True)
        for source, destination in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(destination, source)
        raise


def main() -> int:
    args = parse_args()
    if args.verify_existing_manifest:
        plan = json.loads(args.verify_existing_manifest.read_text(encoding="utf-8"))
        plan["audit_after"] = audit_dataset(args.target_root.resolve())
        plan["quarantine_verification"] = verify_quarantine(plan, args.quarantine_dir.resolve())
        failed = bool(
            plan["audit_after"]["issues"]
            or plan["audit_after"]["exact_duplicate_groups"]
            or plan["quarantine_verification"]["issues"]
        )
        write_manifest(plan, args.record_dir.resolve(), "apply_verified" if not failed else "apply_verification_failed")
        print(json.dumps({"mode": plan["mode"], "audit_after": plan["audit_after"], "quarantine_verification": plan["quarantine_verification"]}, ensure_ascii=False))
        return int(failed)
    plan = build_plan(
        args.server_root.resolve(),
        args.target_root.resolve(),
        expected_legacy=args.expected_legacy,
        expected_retained=args.expected_retained,
        expected_server=args.expected_server,
    )
    if args.apply:
        apply_plan(plan, args.target_root.resolve(), args.quarantine_dir.resolve())
        plan["audit_after"] = audit_dataset(args.target_root.resolve())
        plan["quarantine_verification"] = verify_quarantine(plan, args.quarantine_dir.resolve())
        if plan["audit_after"]["issues"] or plan["audit_after"]["exact_duplicate_groups"] or plan["quarantine_verification"]["issues"]:
            raise RuntimeError("重建后验证失败，请检查隔离目录和活动数据；详情已保留在内存中")
        write_manifest(plan, args.record_dir.resolve(), "apply")
    else:
        write_manifest(plan, args.record_dir.resolve(), "dry_run")
    print(json.dumps({
        "mode": plan["mode"],
        "before": plan["before"],
        "server": plan["server"],
        "expected_after": plan["expected_after"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
