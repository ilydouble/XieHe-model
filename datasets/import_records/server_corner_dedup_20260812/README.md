# 服务器角点数据精确去重记录（2026-08-12）

处理对象：`/Users/liruirui/Downloads/images` 与 `/Users/liruirui/Downloads/labels`。

按 PNG 文件 SHA-256 发现 5 个完全重复组，每组保留 1 个图像/标签对，将另一个图像/标签对移动到：

`/Users/liruirui/Downloads/server_corner_dedup_quarantine_20260812/`

保留规则：

1. 若重复图与 `202605AP_PUMCH_Data` 的患者号、检查日期及影像内容能够对应，保留对应正确的文件名和标签。
2. 若重复组两份标签也完全相同，固定保留文件名字典序靠前的一份。
3. 不合并或平均两份不同标签；被移除版本连同图片一起隔离，便于恢复和人工复核。

原始规模为 1386 对；隔离 5 对后预计为 1381 对。具体文件、哈希和理由见 `manifest.csv`。
