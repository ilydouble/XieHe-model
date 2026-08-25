# XieHe-System 正位线上推理后处理审计

审计日期：2026-08-25

## 结论

不应把所有后处理一起删除，但当前 Pose Corner 的“全局 y 排序 + 连续重编号”必须移除。它会覆盖模型已经学习到的 native class ID；最新20类模型在250张独立 test、当前线上640口径下，只把这一逻辑改为保留native class，V0–V17平均误差即可由20.812 px降到14.566 px，降低6.246 px，约30.0%。

六点Pose的最高置信单实例选择、box conf 0.5和点跨度合法性检查可继续保留。当前最新Pose在175张独立test上175/175均通过这些安全门，未发生拒绝。需要停止的是“Pose失败后静默用椎体框固定倍数估算六点并继续生成自动测量”，或至少必须把它改成明确的estimated结果，不能伪装成模型实测点。

本轮只审计与量化，没有修改`XieHe-System`线上代码。

## 实施更新

审计后已按结论修改`XieHe-System`：提交`64e9f4ad`完成Corner native class与Pose失败语义修正；随后提交`70c7794c`把Pose显式输入尺寸由兼容旧线上行为的640改为与最新版训练/正式评测一致的800。最新Pose在同一175张test上，800口径平均误差29.850 px，优于640口径33.742 px；本机CPU推理均时约由93.8 ms增至138.4 ms，线上GPU仍应记录实际延迟。

## 当前正式调用链

1. HTTP从对象存储读取影像字节，只用OpenCV按原尺寸BGR解码。
2. 同一张完整原图分别送入Pose和Pose Corner；没有裁黑边、ROI、二阶段或服务层resize。
3. 两个`model.predict`都没有显式指定`imgsz`、`conf`和`device`。当前本地Ultralytics环境解析为默认imgsz=640、候选conf=0.25。
4. 最终公共阈值是`CONF_THRESHOLD=0.5`。
5. Pose为空而Corner非空时，服务层用Corner结果固定几何估算六点。

工作副本的`model/ap/weights`目录不存在，所以能确认代码逻辑，但无法确认正式服务器实际权重SHA-256。上线前必须核对服务器权重是否确为本报告使用的最新版。

## 当前额外逻辑清单与建议

| 模块 | 当前逻辑 | 决策 | 原因 |
|---|---|---|---|
| Pose候选选择 | box conf≥0.5后取最高置信单实例 | 保留 | 单人正位任务只需要一个六点实例；175张test全部通过 |
| Pose坐标回写 | `xyn × 原图宽高` | 保留 | 已验证坐标换算误差小于0.001 px，不是偏差来源 |
| Pose跨度检查 | 横向不足原宽10%或纵向不足原高20%则拒绝 | 暂时保留 | 当前test零触发，可作为模型崩塌安全门；应记录拒绝原因 |
| Pose失败兜底 | 按L5、L3/L4、T1/T2/C7固定倍数估算六点 | 停止静默自动测量 | 这是启发式估计，不是新Pose输出；confidence=0仍会被前端载入 |
| Corner阈值 | box conf≥0.5 | 首轮保留 | 当前正式评测和业务V0–V17均使用0.5，先不要同时改多项变量 |
| Corner全局IoU | 不分模型类别，以IoU>0.3再次去重 | 删除/改写 | Ultralytics已做NMS；相邻椎体不应跨类别互相抑制 |
| Corner y排序 | 按四角平均y排序 | 仅保留作诊断或显示排序 | y顺序可以检查异常，但不能决定类别 |
| Corner连续重编号 | 排序后强制编号C7、T1…L5 | 必须删除 | 漏一节或多一节会让后续整段错号；20类语义下还有结构性冲突 |
| Corner native class | 当前完全忽略`result.boxes.cls` | 必须启用 | 每个class取最高置信候选，缺类保持缺失，不能用相邻目标补位 |
| V18/V19 | 当前由y rank产生`V18/V19` | 隔离为诊断 | 业务目录和Cobb当前只支持标准18节；先native选择，再过滤额外类 |
| 前端框内角点排序 | 单个椎体四角几何标准化 | 保留 | 不改变椎体类别，只规范TL/TR/BL/BR显示顺序 |

## Corner量化结果

模型：`corner_20class_roi_mixed_v1-2/weights/best.pt`，SHA-256 `7cde68587b25ff86e3f77480c33081a20b4e8d4e796d1dc2c9f2012cea69ad03`。

数据：修正后的250张独立Corner test，只评业务V0–V17；原图直推，imgsz=640，raw conf=0.25，最终conf=0.5。

| 指标 | 保留native class | 当前线上y重编号 | 差异 |
|---|---:|---:|---:|
| 已检出点平均误差 | 14.566 px | 20.812 px | native降低6.246 px |
| 点召回 | 99.089% | 99.067% | native高0.022个百分点 |
| PCK@20（漏点计失败） | 81.283% | 80.156% | native高1.127个百分点 |
| 完整V0–V17图像 | 234/250 | 233/250 | native多1张 |

250张中有28张的目标/类别分配发生变化。222张近似相同，native明显更好15张，y排序更好13张；但是y排序在少数漏检或多检病例产生整段错号，使总体误差显著恶化。线上y排序最终得到18节223张、19节10张、17节8张，其余9张只有8–16节。

同一新模型在正式800口径、native class下的既有结果为14.135 px、点召回99.67%、239/250张完整。因此Corner应单独显式设置`imgsz=800`；不要继续依赖Ultralytics默认640。这个尺寸调整应与后处理改动分别记录，便于回滚。

## 20类模型的输出契约

最新模型的训练语义为：

- class 0：C7
- class 1–12：T1–T12
- class 13–17：L1–L5
- class 18：L6
- class 19：T13，解剖位置在T12和L1之间

因此class 19不可能通过“最下方rank=19”正确恢复。当前业务仍只使用前18类时，正确做法是：先按native class完成每类最高置信候选选择，再输出class 0–17；class 18/19写入诊断信息或日志，不参与标准18节补位、Cobb排序或Pose兜底。

## 推荐的低风险改法

Corner只做以下选择，不再重建类别：

1. 读取`boxes.cls`、`boxes.conf`和对应四角。
2. 丢弃conf<0.5候选。
3. 每个native class只保留最高box confidence候选。
4. class 0–17按固定语义映射成C7、T1–T12、L1–L5。
5. 某class缺失时保留空槽并记录，不移动后续类别。
6. y中心只用于检查顺序异常、输出日志或界面排序。
7. class 18/19暂不进入现有Cobb与前端标准18节目录。

Pose保持直接原图单阶段推理，并显式固定与训练尺寸一致的`imgsz=800`。同一175张test的平均误差由640口径33.742 px降至800口径29.850 px；需要接受相应延迟增加，但不加入ROI或二阶段。Pose拒绝后返回明确失败；如业务一定要显示估算点，结果必须携带`source=estimated_from_corner`，并禁止直接生成自动测量或要求人工确认。

## 上线前验证

1. 为Corner增加四类契约测试：漏中间一节不移位、同类重复只取高置信、V18/V19不改变V0–V17、T13位于T12/L1之间也不触发重编号。
2. 为Pose增加三类测试：正常6点、box/跨度拒绝、拒绝后不静默输出自动测量。
3. 本地完整复跑175张Pose和250张Corner；Pose 800对照29.850 px结果，Corner 800对照14.135 px结果。
4. 对28张native/y排序不一致样本逐张可视化核验。
5. 灰度阶段同时记录服务器权重SHA、输入尺寸、原生类别、最终类别、拒绝/兜底原因和推理耗时。
6. 第一轮只改Corner类别后处理并固定参数；ROI、黑边裁剪和两阶段链路另开A/B，避免无法定位收益来源。

## 代码证据

- Pose直推、阈值、坐标回写和跨度拒绝：`XieHe-System/model/ap/infrastructure/yolo_inference.py:34`
- Pose固定几何估算：`XieHe-System/model/ap/infrastructure/yolo_inference.py:90`
- Corner跨类IoU、y排序与连续rank：`XieHe-System/model/ap/infrastructure/yolo_inference.py:122`
- Corner推理覆盖native class：`XieHe-System/model/ap/infrastructure/yolo_inference.py:161`
- Pose失败静默兜底：`XieHe-System/model/ap/application/measurement_service.py:23`
- 统一conf=0.5且权重路径未带版本：`XieHe-System/model/ap/config.py:6`
- 后端Cobb只支持标准18节：`XieHe-System/model/ap/domain/cobb.py:10`
- 前端正位角点几何排序：`XieHe-System/frontend/app/imaging/features/image-viewer/features/ai-measurement/usecases/aiDetectionUseCase.ts:203`
