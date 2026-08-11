#!/usr/bin/env python3
"""Build an offline review package for conflicting duplicate annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


LABEL_SUFFIX = "_label.json"
SIX_POINT_LABELS = frozenset(("CL", "CR", "IL", "IR", "SL", "SR"))
VERTEBRA_NAMES = ("C7",) + tuple(f"T{i}" for i in range(1, 13)) + tuple(
    f"L{i}" for i in range(1, 6)
)
SPINE_LABELS = frozenset(
    f"{vertebra}-{corner}"
    for vertebra in VERTEBRA_NAMES
    for corner in range(1, 5)
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成重复图像多版本标注的离线人工核对包。")
    parser.add_argument("export_dir", type=Path, help="当前规范化图像和JSON目录")
    parser.add_argument("--audit", required=True, type=Path, help="含exact_duplicate_groups的历史审计JSON")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出核对包目录")
    parser.add_argument(
        "--fallback-dir",
        action="append",
        default=[],
        type=Path,
        help="当前目录缺失成员的只读恢复备份目录，可重复传入",
    )
    parser.add_argument("--max-image-height", type=int, default=1600, help="页面底图最大像素高度")
    parser.add_argument("--jpeg-quality", type=int, default=88, help="页面底图JPEG质量")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file(name: str, roots: list[Path]) -> Path:
    for root in roots:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"在当前数据和恢复备份中均未找到：{name}")


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def annotation_points(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    points: dict[str, dict[str, Any]] = {}
    items = data.get("vertebrae")
    if not isinstance(items, list):
        return points
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str) or label not in SIX_POINT_LABELS | SPINE_LABELS:
            continue
        coordinate: object = item.get("point")
        if label in SPINE_LABELS:
            corners = item.get("corners")
            coordinate = corners[0] if isinstance(corners, list) and corners else None
        if not isinstance(coordinate, dict):
            continue
        x, y = coordinate.get("x"), coordinate.get("y")
        if not finite_number(x) or not finite_number(y):
            continue
        points[label] = {
            "x": float(x),
            "y": float(y),
            "source": str(item.get("source", "unknown")),
            "task": "six" if label in SIX_POINT_LABELS else "spine",
        }
    return points


def task_delta(candidates: list[dict[str, Any]], labels: frozenset[str]) -> dict[str, Any]:
    maximum = 0.0
    total = 0.0
    count = 0
    pairwise: list[dict[str, Any]] = []
    label_sets = [labels & candidate["points"].keys() for candidate in candidates]
    structure_conflict = any(label_set != label_sets[0] for label_set in label_sets[1:])
    source_conflict = False
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left = candidates[left_index]["points"]
            right = candidates[right_index]["points"]
            common = sorted(labels & left.keys() & right.keys())
            source_mismatches = [
                label for label in common if left[label]["source"] != right[label]["source"]
            ]
            source_conflict = source_conflict or bool(source_mismatches)
            distances = [
                math.hypot(left[label]["x"] - right[label]["x"], left[label]["y"] - right[label]["y"])
                for label in common
            ]
            pair_max = max(distances, default=0.0)
            pair_mean = sum(distances) / len(distances) if distances else 0.0
            maximum = max(maximum, pair_max)
            total += sum(distances)
            count += len(distances)
            pairwise.append(
                {
                    "left": left_index,
                    "right": right_index,
                    "common": len(common),
                    "left_count": len(label_sets[left_index]),
                    "right_count": len(label_sets[right_index]),
                    "only_left": sorted(label_sets[left_index] - label_sets[right_index]),
                    "only_right": sorted(label_sets[right_index] - label_sets[left_index]),
                    "source_mismatches": source_mismatches,
                    "max_delta": pair_max,
                    "mean_delta": pair_mean,
                }
            )
    return {
        "max_delta": maximum,
        "mean_delta": total / count if count else 0.0,
        "structure_conflict": structure_conflict,
        "source_conflict": source_conflict,
        "has_conflict": maximum > 0 or structure_conflict or source_conflict,
        "pairwise": pairwise,
    }


def write_thumbnail(source_path: Path, output_path: Path, max_height: int, quality: int) -> tuple[int, int]:
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
    if image.height > max_height:
        width = max(1, round(image.width * max_height / image.height))
        image = image.resize((width, max_height), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=quality, optimize=True)
    return image.size


def csv_value(value: object) -> str:
    return "" if value is None else str(value)


def build_package(
    export_dir: Path,
    audit_path: Path,
    output_dir: Path,
    *,
    fallback_dirs: list[Path] | None = None,
    max_image_height: int = 1600,
    jpeg_quality: int = 88,
) -> dict[str, Any]:
    export_dir = export_dir.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    roots = [export_dir, *(path.expanduser().resolve() for path in (fallback_dirs or []))]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    raw_groups = audit.get("exact_duplicate_groups")
    if not isinstance(raw_groups, list):
        raise ValueError("审计JSON缺少exact_duplicate_groups数组")
    if not 1 <= max_image_height <= 10000:
        raise ValueError("max_image_height必须位于1到10000")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality必须位于1到100")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    groups: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for index, raw_group in enumerate(sorted(raw_groups, key=lambda item: str(item.get("sha256", ""))), 1):
        files = raw_group.get("files")
        if not isinstance(files, list) or len(files) < 2 or not all(isinstance(name, str) for name in files):
            raise ValueError(f"重复组{index}的files无效")
        audit_sha = str(raw_group.get("sha256", ""))
        candidates: list[dict[str, Any]] = []
        image_paths: list[Path] = []
        for member_index, image_name in enumerate(files, 1):
            image_path = resolve_file(image_name, roots)
            annotation_name = f"{Path(image_name).stem}{LABEL_SUFFIX}"
            annotation_path = resolve_file(annotation_name, roots)
            actual_sha = sha256_file(image_path)
            if audit_sha and actual_sha != audit_sha:
                raise ValueError(f"{image_name}的SHA-256与历史审计不一致")
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            points = annotation_points(data)
            source_counter = Counter(point["source"] for point in points.values())
            source_counts.update(source_counter)
            candidates.append(
                {
                    "index": member_index,
                    "image": image_name,
                    "annotation": annotation_name,
                    "image_source": str(image_path.parent),
                    "annotation_source": str(annotation_path.parent),
                    "original_filename": str(data.get("originalFilename", "")),
                    "image_id": data.get("imageId"),
                    "six_count": len(SIX_POINT_LABELS & points.keys()),
                    "spine_count": len(SPINE_LABELS & points.keys()),
                    "source_counts": dict(sorted(source_counter.items())),
                    "points": points,
                }
            )
            image_paths.append(image_path)

        thumbnail_name = f"group_{index:04d}.jpg"
        thumbnail_size = write_thumbnail(
            image_paths[0], image_dir / thumbnail_name, max_image_height, jpeg_quality
        )
        six_delta = task_delta(candidates, SIX_POINT_LABELS)
        spine_delta = task_delta(candidates, SPINE_LABELS)
        groups.append(
            {
                "id": f"G{index:04d}",
                "number": index,
                "sha256": audit_sha,
                "thumbnail": f"images/{thumbnail_name}",
                "thumbnail_width": thumbnail_size[0],
                "thumbnail_height": thumbnail_size[1],
                "candidate_count": len(candidates),
                "candidates": candidates,
                "six_delta": six_delta,
                "spine_delta": spine_delta,
            }
        )

    package_id = hashlib.sha256(
        (str(audit_path) + "\n" + "\n".join(group["sha256"] for group in groups)).encode("utf-8")
    ).hexdigest()[:16]
    manifest = {
        "schema_version": 1,
        "package_id": package_id,
        "source_export_dir": str(export_dir),
        "source_audit": str(audit_path),
        "fallback_dirs": [str(path) for path in roots[1:]],
        "group_count": len(groups),
        "candidate_count": sum(group["candidate_count"] for group in groups),
        "group_size_counts": dict(sorted(Counter(str(group["candidate_count"]) for group in groups).items())),
        "source_counts": dict(sorted(source_counts.items())),
        "groups": groups,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    (output_dir / "review_data.js").write_text(
        "window.REVIEW_PACKAGE=" + compact + ";\n", encoding="utf-8"
    )
    (output_dir / "打开核对页面.html").write_text(REVIEW_HTML, encoding="utf-8")

    with (output_dir / "初始核对结果.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("组号", "SHA256", "选择", "选中标注", "备注", "候选标注")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    "组号": group["id"],
                    "SHA256": group["sha256"],
                    "选择": "",
                    "选中标注": "",
                    "备注": "",
                    "候选标注": " | ".join(candidate["annotation"] for candidate in group["candidates"]),
                }
            )

    readme = f"""# 284组重复标注人工核对包

- 重复组：{len(groups)}组
- 候选标注：{manifest['candidate_count']}份
- 组大小：{manifest['group_size_counts']}
- 数据快照：移除532之前的有标注审计

双击 `打开核对页面.html` 开始。每组可选择图1、图2（若存在则有图3）或“都不对”。

选择和备注会自动保存在浏览器本地存储中。请定期点击“导出JSON”和“导出CSV”保存文件；换浏览器或移动目录后，应使用“导入JSON”恢复进度。

页面只用于记录人工结论，不会修改原图、JSON或当前规范化数据集。
"""
    (output_dir / "复核包说明.md").write_text(readme, encoding="utf-8")
    return manifest


REVIEW_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>重复图像标注人工核对</title>
<style>
:root{color-scheme:dark;--bg:#101318;--panel:#1b2028;--line:#333c49;--text:#edf2f7;--muted:#aeb8c5;--accent:#50b7ff;--ok:#4ad68b;--bad:#ff6b78}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}
header{position:sticky;top:0;z-index:5;background:rgba(16,19,24,.96);border-bottom:1px solid var(--line);padding:12px 18px;backdrop-filter:blur(8px)}
.toolbar,.nav,.choices,.toggles{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.toolbar{justify-content:space-between}.nav{margin-top:9px}
button,select,input,textarea{font:inherit;color:var(--text);background:#252c36;border:1px solid #465161;border-radius:8px;padding:8px 11px}button{cursor:pointer}button:hover{border-color:var(--accent)}button.primary{background:#124e72;border-color:#2d9fe6}.selected{outline:3px solid var(--ok)!important}.danger.selected{outline-color:var(--bad)!important}
#progress{font-weight:700}.bar{height:6px;background:#2a3039;border-radius:6px;overflow:hidden;margin-top:8px}.bar>div{height:100%;background:var(--ok);width:0}
main{padding:16px}.summary{background:var(--panel);padding:12px 14px;border:1px solid var(--line);border-radius:10px;margin-bottom:14px}.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;min-width:0}.card h2{font-size:17px;margin:0 0 6px;word-break:break-all}.meta{font-size:13px;color:var(--muted);word-break:break-all;margin-bottom:8px}.canvas-wrap{position:relative;background:#050607;overflow:auto;border-radius:8px}.canvas-wrap canvas{display:block;width:100%;height:auto}
.choices{justify-content:center;margin:16px 0}.choices button{font-size:17px;padding:11px 18px}.note{width:100%;min-height:74px;resize:vertical}.help{font-size:13px;color:var(--muted);line-height:1.55}.pill{display:inline-block;padding:2px 7px;border-radius:999px;background:#303846;margin-right:5px;font-size:12px}
@media(max-width:700px){header{position:static}.cards{grid-template-columns:1fr}main{padding:9px}}
</style></head><body>
<header><div class="toolbar"><div><strong>重复图像标注人工核对</strong> <span id="progress"></span></div><div class="toolbar"><button id="exportJson">导出JSON</button><button id="exportCsv">导出CSV</button><button id="importBtn">导入JSON</button><input id="importFile" type="file" accept="application/json" hidden></div></div>
<div class="nav"><button id="prev">← 上一组</button><button id="next">下一组 →</button><label>跳转 <input id="jump" type="number" min="1" style="width:90px"></label><select id="filter"><option value="all">全部</option><option value="pending">仅未核对</option><option value="chosen">已选图</option><option value="neither">都不对</option></select><label><input id="autoNext" type="checkbox" checked> 选择后自动下一组</label><span class="toggles"><label><input id="showSix" type="checkbox" checked>六点</label><label><input id="showSpine" type="checkbox" checked>脊柱点</label><label><input id="showLabels" type="checkbox">点名</label></span></div><div class="bar"><div id="bar"></div></div></header>
<main><section class="summary"><div id="groupTitle"></div><div id="metrics" class="muted"></div></section><section id="cards" class="cards"></section><div id="choices" class="choices"></div><label>备注（自动保存）<textarea id="note" class="note" placeholder="例如：六点选图1，脊柱两份都需要修改；如两个任务应分别选择，请在这里写清楚。"></textarea></label><p class="help">快捷键：1/2/3选择对应图，0选择都不对，←/→切换。选择的是“保留这份标注”，不是选择像素图；同组底图内容相同。结果保存在当前浏览器localStorage，请定期导出JSON备份。</p></main>
<script src="review_data.js"></script><script>
(()=>{'use strict';const pkg=window.REVIEW_PACKAGE;if(!pkg){document.body.innerHTML='无法加载review_data.js';return}const key='duplicate-review-'+pkg.package_id;let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}let current=0,filtered=[];const $=id=>document.getElementById(id);const imageCache=new Map();
function save(){localStorage.setItem(key,JSON.stringify(state));updateProgress()}
function reviewed(g){return !!state[g.id]?.choice}function rebuildFilter(){const f=$('filter').value;filtered=pkg.groups.filter(g=>f==='all'||(f==='pending'&&!reviewed(g))||(f==='chosen'&&state[g.id]?.choice?.startsWith('candidate:'))||(f==='neither'&&state[g.id]?.choice==='neither'));if(!filtered.length)filtered=pkg.groups.slice();current=Math.min(current,filtered.length-1)}
function updateProgress(){const done=pkg.groups.filter(reviewed).length;$('progress').textContent=`${done}/${pkg.group_count} 已完成`;$('bar').style.width=(done/pkg.group_count*100).toFixed(1)+'%'}
function colorFor(label){const six={CL:'#31d7ff',CR:'#ff525f',IL:'#52ef89',IR:'#ffc43d',SL:'#bb7cff',SR:'#ff62db'};if(six[label])return six[label];const name=label.split('-')[0],i=['C7','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','L1','L2','L3','L4','L5'].indexOf(name);return `hsl(${(i*31)%360} 90% 62%)`}
function loadImage(src){if(!imageCache.has(src)){imageCache.set(src,new Promise((ok,bad)=>{const im=new Image();im.onload=()=>ok(im);im.onerror=bad;im.src=src}))}return imageCache.get(src)}
async function draw(canvas,g,c){const im=await loadImage(g.thumbnail);canvas.width=im.naturalWidth;canvas.height=im.naturalHeight;const x=canvas.getContext('2d');x.drawImage(im,0,0);const sx=canvas.width,sy=canvas.height,pts=c.points;x.lineCap='round';
if($('showSpine').checked){for(const v of ['C7','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','L1','L2','L3','L4','L5']){const ls=[1,2,4,3].map(n=>`${v}-${n}`).filter(k=>pts[k]);if(ls.length>1){x.beginPath();ls.forEach((k,i)=>{const p=pts[k];i?x.lineTo(p.x*sx,p.y*sy):x.moveTo(p.x*sx,p.y*sy)});x.closePath();x.strokeStyle=colorFor(v+'-1');x.lineWidth=Math.max(2,sx/700);x.stroke()}for(let n=1;n<=4;n++){const k=`${v}-${n}`,p=pts[k];if(!p)continue;dot(x,p.x*sx,p.y*sy,colorFor(k),Math.max(2.5,sx/380),$('showLabels').checked?k:'')}}}
if($('showSix').checked){for(const [a,b] of [['CL','CR'],['IL','IR'],['SL','SR']])if(pts[a]&&pts[b]){x.beginPath();x.moveTo(pts[a].x*sx,pts[a].y*sy);x.lineTo(pts[b].x*sx,pts[b].y*sy);x.strokeStyle='#fff';x.lineWidth=Math.max(2,sx/450);x.stroke()}for(const k of ['CL','CR','IL','IR','SL','SR']){const p=pts[k];if(p)dot(x,p.x*sx,p.y*sy,colorFor(k),Math.max(6,sx/120),k)}}}
function dot(x,px,py,color,r,label){x.beginPath();x.arc(px,py,r,0,Math.PI*2);x.fillStyle=color;x.fill();x.strokeStyle='#fff';x.lineWidth=1.5;x.stroke();if(label){x.font=`bold ${Math.max(12,Math.round(r*1.7))}px sans-serif`;x.fillStyle=color;x.strokeStyle='#000';x.lineWidth=3;x.strokeText(label,px+r+3,py-r);x.fillText(label,px+r+3,py-r)}}
function render(){rebuildFilter();const g=filtered[current];if(!g)return;$('jump').max=pkg.group_count;$('jump').value=g.number;$('groupTitle').innerHTML=`<strong>${g.id}</strong>　SHA ${g.sha256.slice(0,16)}…　<span class="pill">${g.candidate_count}个候选</span>`;
const sixStructure=g.six_delta.structure_conflict?'是':'否',spineStructure=g.spine_delta.structure_conflict?'是':'否';$('metrics').textContent=`六点最大坐标差 ${g.six_delta.max_delta.toFixed(6)}、结构差异 ${sixStructure}；脊柱点最大坐标差 ${g.spine_delta.max_delta.toFixed(6)}、结构差异 ${spineStructure}`;const cards=$('cards');cards.innerHTML='';g.candidates.forEach((c,i)=>{const a=document.createElement('article');a.className='card';a.innerHTML=`<h2>图${i+1}｜${esc(c.annotation)}</h2><div class="meta">imageId=${esc(c.image_id)}｜六点 ${c.six_count}/6｜脊柱 ${c.spine_count}/72｜来源 ${esc(JSON.stringify(c.source_counts))}</div><div class="canvas-wrap"><canvas></canvas></div>`;cards.appendChild(a);draw(a.querySelector('canvas'),g,c)});
const choices=$('choices');choices.innerHTML='';g.candidates.forEach((c,i)=>{const b=document.createElement('button');b.textContent=`选择图${i+1}`;b.className=state[g.id]?.choice===`candidate:${i}`?'selected':'';b.onclick=()=>choose(`candidate:${i}`);choices.appendChild(b)});const no=document.createElement('button');no.textContent='两个/全部都不对';no.className='danger '+(state[g.id]?.choice==='neither'?'selected':'');no.onclick=()=>choose('neither');choices.appendChild(no);$('note').value=state[g.id]?.note||'';updateProgress()}
function esc(v){return String(v??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
function choose(choice){const g=filtered[current];state[g.id]={...(state[g.id]||{}),choice,updated_at:new Date().toISOString()};save();render();if($('autoNext').checked)setTimeout(()=>move(1),180)}function move(n){current=Math.max(0,Math.min(filtered.length-1,current+n));render();scrollTo({top:0,behavior:'smooth'})}
function result(){return {schema_version:1,package_id:pkg.package_id,exported_at:new Date().toISOString(),group_count:pkg.group_count,completed:pkg.groups.filter(reviewed).length,results:pkg.groups.map(g=>{const s=state[g.id]||{},m=/^candidate:(\d+)$/.exec(s.choice||''),idx=m?Number(m[1]):-1;return {group_id:g.id,sha256:g.sha256,choice:s.choice||'',selected_index:idx>=0?idx+1:null,selected_annotation:idx>=0?g.candidates[idx]?.annotation||'':'',note:s.note||'',updated_at:s.updated_at||'',candidates:g.candidates.map(c=>c.annotation)}})}}
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function jumpToNumber(){const n=Number($('jump').value),i=filtered.findIndex(g=>g.number===n);if(i>=0){current=i;render()}else{const target=pkg.groups[n-1];if(target){$('filter').value='all';rebuildFilter();current=n-1;render()}}}$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('filter').onchange=()=>{current=0;render()};$('jump').onchange=jumpToNumber;$('jump').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();jumpToNumber()}};$('note').oninput=()=>{const g=filtered[current];state[g.id]={...(state[g.id]||{}),note:$('note').value,updated_at:new Date().toISOString()};save()};
for(const id of ['showSix','showSpine','showLabels'])$(id).onchange=render;$('exportJson').onclick=()=>download(`重复标注核对结果_${Date.now()}.json`,JSON.stringify(result(),null,2),'application/json');$('exportCsv').onclick=()=>{const r=result(),q=v=>'"'+String(v??'').replaceAll('"','""')+'"',rows=[['组号','SHA256','选择','选中标注','备注','候选标注'].map(q).join(',')];r.results.forEach(x=>rows.push([x.group_id,x.sha256,x.choice,x.selected_annotation,x.note,x.candidates.join(' | ')].map(q).join(',')));download(`重复标注核对结果_${Date.now()}.csv`,'\ufeff'+rows.join('\r\n'),'text/csv')};$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=async e=>{const d=JSON.parse(await e.target.files[0].text());if(d.package_id!==pkg.package_id)return alert('核对包ID不一致，拒绝导入');for(const r of d.results||[])state[r.group_id]={choice:r.choice||'',note:r.note||'',updated_at:r.updated_at||new Date().toISOString()};save();render();alert('导入完成')};
document.addEventListener('keydown',e=>{if(e.target.matches('input,textarea,select'))return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key==='0')choose('neither');else if(/^[1-3]$/.test(e.key)){const i=Number(e.key)-1,g=filtered[current];if(i<g.candidate_count)choose(`candidate:${i}`)}});rebuildFilter();render()})();
</script></body></html>'''


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_package(
        args.export_dir,
        args.audit,
        args.output_dir,
        fallback_dirs=args.fallback_dir,
        max_image_height=args.max_image_height,
        jpeg_quality=args.jpeg_quality,
    )
    print(
        json.dumps(
            {key: manifest[key] for key in ("package_id", "group_count", "candidate_count", "group_size_counts")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
