# judge_lab：专科判官工具包

用主人在审核页留下的结论（通过 / 不通过 + 勾选的原因项），训练一组很小的“专科判官头”：

- 冻结的预训练图像骨干（SigLIP2 / DINOv2 / CLIP-L）把裁剪图变成特征向量；
- 每个缺陷项一个逻辑回归小头，输出 0–1 的缺陷概率。

另外附带：骨干比拼（线性探针）、按目标召回率给出建议阈值、考卷验收、新图影子打分，以及配方 / LoRA 版本 A/B 统计。
这套工具不生成图片，也不针对任何具体内容，只做和内容无关的缺陷分类。

## 三条规矩

1. **人工结论是唯一真值。** 标签只来自主人的审核记录。
2. **考卷永不训练。** 考卷名单里的图片不参与任何训练或拟合（probe / train / calibrate），只在 `exam` 中用来最终评估。代码在拟合前会核对；如果某个判官头的训练集和考卷有交集，`exam` 会直接报错退出。每条命令都会打印考卷数量。
3. **模型只出分数，不做判定。** `score` 只写概率和“是否超过阈值”，不写通过 / 不通过。阈值只是建议值，最终的线由主人审核、批准。

## 安装

```bash
pip install -r judge_lab/requirements.txt             # numpy / scikit-learn / pillow（+ pytest）
pip install -r judge_lab/requirements-backbones.txt   # 真实骨干：torch + transformers
```

所有命令都在 `mark-v4/` 目录下运行：`python -m judge_lab <命令>`。

## 输入格式

**1. 审核日志 JSONL**（一行一张图）：

```json
{"id": "img_0001", "image": "out/img_0001.png", "decision": "fail",
 "reasons": ["structure_break", "lighting"], "scores": {"lighting": 3},
 "recipe_hash": "a1b2", "versions": {"lora": "mark_taste_v1"}, "scene": "indoor"}
```

- `image` 可以是绝对路径，也可以是相对日志文件所在目录的路径。
- `reasons` 里**没勾选**的项目一律当作“OK”。
- 同一个 id 出现多次时，以最后一行为准（相当于重审）。

字段名不一样时，用 `--field-map` 改名，支持 `a.b` 这样的嵌套路径，也可以改通过 / 不通过的取值（见 `configs/field_map.example.json`）：

```json
{"id": "image_id", "image": "file", "decision": "verdict", "reasons": "fail_items",
 "scores": "item_scores", "versions": "meta.versions", "scene": "meta.scene",
 "decision_values": {"pass": ["pass", "通过"], "fail": ["fail", "不通过"]}}
```

**2. 考卷名单**：文本文件，一行一个 id，`#` 后面是注释。

**3. 预裁剪（可选）**：`<crops_root>/<image_id>/<region>_<k>.png|jpg`，其中 region 为 face / hand / body / tile / full。
没有预裁剪时：`full` 用整张图，`tile` 用整张图切成 2×2 四块。face / hand 没有裁剪的图会被跳过（人脸、手部检测由主人现有流程负责）；`body` 没有裁剪时回退到 `full`。

**4. 判官头配置**：`configs/heads_v1.json`，共 21 个项目，每项写明 `region`、`kind`（hard 硬门槛 / score 评分项）和 `label_rule`。

- 标签规则：不合格且勾选了该项 → 缺陷（1）；通过，或者不合格但没勾该项 → OK（0）。
- 评分项可以设 `"max_score": 4`：勾选且分数 ≤ 4 才算缺陷；勾选了但分数更高的样本不计入训练。
- `trainable: false` 的项目不自动训练，但仍出现在报告里：年龄偏小 / 偏大由主人在审片页人工判断（暂不装年龄估计模型）；动作和分镜不符、物理不合理是语义判断，交给 VLM 审核。
- 有的项目判官头只是辅助：乱码字主要靠 OCR；不是同一个人、像不像本人主要靠人脸特征相似度。

## 操作步骤

```bash
L="--log data/review_log.jsonl --exam data/exam.txt --crops-root data/crops"

# 0) 提取特征（会缓存，重跑时只算新图）
python -m judge_lab embed $L --backbone siglip2,dinov2,clip_l --out cache/

# 1) 骨干比拼：按图分组的 5 折交叉验证，报告 AUC，以及达到目标召回率时的召回和误拒率
python -m judge_lab probe $L --cache cache/ --backbones siglip2,dinov2,clip_l --out reports/probe.md

# 2) 训练最终判官头（每个项目自动选 probe 里最好的骨干；也可以用 --backbone siglip2 统一指定）
python -m judge_lab train $L --cache cache/ --probe reports/probe.json --out heads/

# 3) 建议阈值（用 out-of-fold 预测；硬门槛目标召回 0.95，评分项 0.80）
python -m judge_lab calibrate $L --cache cache/ --heads heads/ --out heads/thresholds.json

# 4) 考试验收（只用考卷）
python -m judge_lab exam $L --cache cache/ --heads heads/ --out reports/exam.md

# 5) 影子打分（新图，只记分数）
python -m judge_lab score --images new_batch/ --heads heads/ --crops-root new_crops/ --out shadow.jsonl
```

**阈值审批**：打开 `heads/thresholds.json`，确认或修改某一项的 `threshold`，再把 `status` 改成 `"approved"`，`approved_by` 填上名字。之后重跑 `calibrate` 只会更新 `proposed_threshold`，不会覆盖已批准的线（除非加 `--overwrite-approved`）。

**A/B 统计**：

```bash
python -m judge_lab stats ab --log data/review_log.jsonl --group-by versions.lora --scene-field scene \
       --baseline mark_taste_v1 --tolerance 0.05 --out reports/ab.md
python -m judge_lab stats n --p1 0.3 --p2 0.5 --power 0.8 --alpha 0.05   # 每组需要多少张
```

`stats ab` 输出：各组通过率及 Wilson 95% 置信区间、与基线差值的 Newcombe 置信区间、两比例 z 检验 p 值、分场景明细，以及回归检查（任何场景新版本比基线低超过容忍度就报警；加 `--fail-on-regression` 时退出码为 1）。

其他常用参数：`--items a,b` 只处理部分项目；`--model-id siglip2=<id 或本地路径>`；`--model-dir D:\models`（离线加载）；`--device cuda:1`；`--batch-size 32`；`--min-pos/--min-neg 15`；`--recall-hard/--recall-score`。

## 硬件

- RTX 4060 Ti 16GB 或 RTX 5080 16GB 都够用：骨干只做推理（fp16），三个骨干各自都在约 2 GB 显存以内（SigLIP2-so400m 连同文本塔一起加载，是最大的一个），batch 32 没问题；判官头是逻辑回归，CPU 上几秒就训完。
- 特征提取是唯一耗时的一步，结果会缓存在 `cache/<骨干>/<区域>.npz`，之后 probe / train / calibrate / exam 重跑基本不花时间。
- RTX 5080（Blackwell）需要 CUDA 12.8 以上的 torch 版本（`--index-url https://download.pytorch.org/whl/cu128`）。
- 首次使用真实骨干会从 Hugging Face 下载模型；离线时先放到本地，再用 `--model-dir` 指定。

## 产出物

| 文件 | 内容 |
|---|---|
| `reports/probe.md` / `probe.json` | 骨干比拼表；每个项目、每个区域的最佳骨干 |
| `heads/<项目>_v<N>.npz` | 判官头：coef、intercept、骨干与模型 id、区域、样本数、训练 id 列表及其 sha256（data_hash）、创建时间。版本号自动递增 |
| `heads/manifest.json` | 每个项目当前启用的版本和历史版本 |
| `heads/thresholds.json` | 建议阈值、OOF 召回、预计误拒率、审批状态 |
| `reports/exam.md` / `exam.json` | 考卷上的召回、误拒率、一致率、AUC、是否达标 |
| `shadow.jsonl` | 影子打分：每张图每个项目的概率和是否超过阈值（不含任何判定） |

## 设计取舍与限制

- **多实例简化**：一张图有多个裁剪（四块 tile、多张脸或手）时，训练时每个裁剪都继承整张图的标签，正样本也一样；预测时整张图的分数取各裁剪的最大值。缺陷通常只出现在一块里，所以正样本里会混进一些“其实没问题”的裁剪。这是为了简单而有意接受的近似；以后如果主人标出缺陷位置，可以换成更精确的做法。
- 判官头 = 标准化 + 类别平衡的逻辑回归（默认 C=0.1）。标准化已经并进 coef/intercept，所以使用时就是 `sigmoid(x·coef + b)`。
- 阈值来自 out-of-fold 预测，最终的头则用全部训练数据重新拟合，两者的分数分布会略有差异，所以要看 `exam` 结果再批准阈值。
- 召回、误拒率、一致率都按图计算：误拒率 = 主人判为 OK 的图里被标成超过阈值的比例。
- 样本少于 `--min-pos/--min-neg`（默认各 15）的项目不会训练，报告里会写明原因。
