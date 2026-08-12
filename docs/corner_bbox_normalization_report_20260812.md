# 椎体角点 bbox 统一报告

日期：2026-08-12

## 处理结论

`datasets/pose_corner_data` 的 bbox 已统一为每个椎体 `TL/TR/BR/BL` 四点的水平最小外接框，不增加边距。标准 YOLO Pose 的 bbox 是轴对齐框；椎体倾斜方向继续由四角点表达。

| 指标 | 结果 |
|---|---:|
| 标签文件 | 2,499 |
| 椎体实例 | 44,981 |
| 实际修改标签 | 1,381 |
| 实际修改实例 | 21,253 |
| 人工 `eap_` 标签修改 | 0/1,118 |
| bbox 之外字段变化 | 0 |
| 统一后不符合规则的实例 | 0 |

按当前 split 统计，修改行为 train 16,015 行、val 2,422 行、test 2,816 行。图像和 split 没有改变，当前规模仍为 train 1,999、val 250、test 250。

## 修改范围

每行标签只允许改变以下四个字段：

```text
class cx cy w h TLx TLy v TRx TRy v BRx BRy v BLx BLy v
      └───────┘
```

计算规则为：

```text
xmin = min(TLx, TRx, BRx, BLx)
xmax = max(TLx, TRx, BRx, BLx)
ymin = min(TLy, TRy, BRy, BLy)
ymax = max(TLy, TRy, BRy, BLy)

cx = (xmin + xmax) / 2
cy = (ymin + ymax) / 2
w  = xmax - xmin
h  = ymax - ymin
```

class、四个角点、visibility、每份标签的行数及行序均保持原值。唯一缺 V12 的已知样本仍然保留，类别统计仍为 V12 2,498 个、其余类别各 2,499 个。

## 备份和哈希

修改前的全部 2,499 份标签位于：

```text
datasets/pose_corner_bbox_backup_20260812/labels/{train,val,test}/
```

`datasets/pose_corner_bbox_backup_20260812/manifest.json` 记录每份标签修改前后 SHA-256。整体哈希为：

- 修改前标签树：`1a39d541ca26172dfb2828e99c7245ce4403e995fffacd5d44ea7e34ce6ebb8f`
- 修改后标签树：`75b6e465a3a0a957506607fdd8727646255bc99f9aed4461c0e369f573039645`
- 图像树修改前后均为：`7559f65d47719ae1ab735c65c484ea10ea1436f271305f296790e9e0cddde73d`

逐文件复核确认，备份标签与 manifest 中修改前哈希全部一致，活动标签与修改后哈希全部一致。

## 验证结果

- 2,499 份图像和标签数量不变；
- 44,981 行全部为 17 字段 YOLO Pose 格式；
- class、关键点坐标、visibility 和行序变化为 0；
- 44,981 行 bbox 均与四点水平外接框一致；
- 图像树哈希不变，因此图像内容、精确重复情况和患者级 split 均未被本次操作改变；
- 重复运行预演结果为 `changed_files=0`、`changed_rows=0`；
- `tests/test_normalize_corner_bboxes.py` 共 3 项测试通过。

## 复现命令

预演：

```bash
python3 scripts/normalize_corner_bboxes.py \
  datasets/pose_corner_data \
  --expected-labels 2499
```

正式执行必须提供一个新的空备份目录：

```bash
python3 scripts/normalize_corner_bboxes.py \
  datasets/pose_corner_data \
  --backup datasets/pose_corner_bbox_backup_YYYYMMDD \
  --expected-labels 2499 \
  --apply
```
