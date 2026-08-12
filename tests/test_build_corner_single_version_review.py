import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("single_review", ROOT / "scripts/build_corner_single_version_review.py")
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader; sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def label() -> str:
    rows=[]
    for c in range(1,18):
        y=.05+c*.045; pts=[(.4,y),(.6,y),(.6,y+.025),(.4,y+.025)]; z=[c,.5,y+.0125,.2,.025]
        for x,q in pts:z += [x,q,2]
        rows.append(" ".join(map(str,z)))
    return "\n".join(rows)


class SingleReviewTest(unittest.TestCase):
    def test_excludes_matched_and_builds_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ds=root/'ds'; out=root/'out'
            for i in range(5):
                (ds/'images/train').mkdir(parents=True,exist_ok=True);(ds/'labels/train').mkdir(parents=True,exist_ok=True)
                Image.new('L',(240,480),120).save(ds/f'images/train/{i}.png');(ds/f'labels/train/{i}.txt').write_text(label())
            candidates=root/'matches.tsv';candidates.write_text(f"{root/'server.png'}\t{ds/'images/train/0.png'}\t1\told743")
            rows=MODULE.build(ds,candidates,out,3)
            self.assertEqual(len(rows),3);self.assertNotIn('0.png',{r['image'] for r in rows})
            self.assertEqual(len(list((out/'previews').glob('*.jpg'))),3)
            self.assertIn('wrong_numbering',(out/'打开人工确认页面.html').read_text())
            self.assertIn('remaining_count', (out/'manifest.json').read_text())


if __name__ == '__main__': unittest.main()
