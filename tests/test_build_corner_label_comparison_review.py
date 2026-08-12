import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review", ROOT / "scripts/build_corner_label_comparison_review.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def label(offset: float) -> str:
    rows = []
    for class_id in range(1, 18):
        x, y = 0.4 + offset, 0.1 + class_id * 0.04
        points = [(x, y), (x + 0.1, y), (x + 0.1, y + 0.02), (x, y + 0.02)]
        tokens = [class_id, x + 0.05, y + 0.01, 0.1, 0.02]
        for px, py in points:
            tokens += [px, py, 2]
        rows.append(" ".join(map(str, tokens)))
    return "\n".join(rows)


class ReviewPackageTest(unittest.TestCase):
    def test_builds_ranked_offline_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); server = root / "server"; current = root / "current"; out = root / "out"
            entries = []
            for index, offset in enumerate((0.01, 0.05), 1):
                for base in (server, current):
                    (base / "images/train").mkdir(parents=True, exist_ok=True)
                    (base / "labels/train").mkdir(parents=True, exist_ok=True)
                sp = server / f"images/train/s{index}.png"; cp = current / f"images/train/c{index}.png"
                Image.new("L", (200, 400), 100).save(sp); Image.new("L", (200, 400), 100).save(cp)
                (server / f"labels/train/s{index}.txt").write_text(label(offset))
                (current / f"labels/train/c{index}.txt").write_text(label(0))
                entries.append(f"{sp}\t{cp}\t1.0\told743")
            candidates = root / "pairs.tsv"; candidates.write_text("\n".join(entries))
            rows = MODULE.build(candidates, server, current, out, 2)
            self.assertEqual([row["server_image"] for row in rows], ["s2.png", "s1.png"])
            self.assertEqual(len(list((out / "previews").glob("*.jpg"))), 2)
            self.assertIn("两版都不准", (out / "打开人工确认页面.html").read_text())
            self.assertIn("localStorage", (out / "打开人工确认页面.html").read_text())
            self.assertTrue((out / "样本索引与确认结果.csv").is_file())
            self.assertEqual(len(list((out / "labels/server").glob("*.txt"))), 2)
            self.assertEqual(len(list((out / "labels/current743").glob("*.txt"))), 2)


if __name__ == "__main__":
    unittest.main()
