# Training export 数据审计

`scripts/audit_training_export.py` 以只读方式检查后端导出的 PNG 与 `*_label.json`。它不会删除、重命名或修改原始数据。

## 使用方法

在仓库根目录运行：

```bash
python3 scripts/audit_training_export.py \
  /path/to/training_export \
  --output /tmp/training_export_audit.json
```

完整扫描会读取所有 PNG 并计算 SHA-256。只想快速检查标注时可以加 `--skip-hashes`。默认只报告问题并返回退出码 0；用于 CI 时可用 `--fail-on error` 或 `--fail-on warning`。

## 检查内容

- 图像与 JSON 是否配对，并按文件内容识别 PNG/JPEG、校验图像尺寸（不盲信 `.png` 扩展名）；
- JSON 是否可解析，顶层字段、`imageId`、尺寸和数组结构是否正确；
- 坐标是否为有限数值并位于归一化范围 `[0,1]`；
- `type`、`source`、点结构与四角结构是否符合导出格式；
- 标准 78 项标签是否缺失、重复或出现未知标签；
- 左右点和每节椎体四角的基本顺序是否可疑；
- 哪些 PNG 内容完全相同，以及重复图像的标注是相同、冲突、部分缺失还是全部缺失；
- 哪些标注使用相同的 `originalFilename`。

结构、尺寸和非法坐标归为 `error`；不完整标签、几何顺序可疑和重复图像标注冲突归为 `warning`。几何告警只用于筛选人工复核对象，不代表自动确认标错。

## 去重注意事项

不要按文件名直接删除，也不要看到相同 SHA-256 就立即删除。先查看 JSON 报告中 `exact_duplicate_groups[].annotation_status`：

- `identical`：图像与归一化后的标注内容一致，通常可选一个保留；
- `conflicting`：同一图像存在不同标注，需要人工比较或合并；
- `partial`：重复组中只有部分图片有标注；
- `unlabeled`：重复组全部没有标注。

对于 `conflicting`，继续查看 `conflict_kind`：`source_only` 表示坐标/结构相同但 `ai/manual` 来源不同；`coordinates_or_structure` 表示坐标或标签结构本身也不同。

报告可能包含原始文件名，只应保存在受控的本地位置，不建议提交到 Git。
