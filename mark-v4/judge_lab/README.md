# judge_lab：专科判官工具包

用主人在审核页留下的结论（通过 / 不通过 + 勾选的原因项），训练一组很小的“专科判官头”：

- 冻结的预训练图像骨干（SigLIP2 / DINOv2 / CLIP-L）把裁剪图变成特征向量；
- 每个缺陷项一个逻辑回归小头，输出 0–1 的缺陷概率。

另外附带：骨干比拼（线性探针）、按目标召回率给出建议阈值、考卷验收、新图影子打分、链路自检、给看图大模型导出原因清单，以及配方 / LoRA 版本 A/B 统计。
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

## 直接读审片页的文件（推荐）

审片页写的两个文件可以原样喂给工具包，不用字段映射，也不用另外维护一份原因清单：

- `--log 审片记录.jsonl`：决定是「过 / 不过」，原因是对象（按显示名），时间取 `reviewed_at`，没有就取 `time`；过的那条 `scores` 是 `"default"`。记录里没有 `id` 时由图片路径生成 `<文件夹>__<文件名>`。
- `--config 缺陷清单.json`：就是审片页的原因清单。

**看哪块图只由“现在的清单”决定**：先看这一项自己写没写 `region`，没写就按它现在的分组（人员/光线/镜头/其他 → 整图；结构/文字 → 分块；手 → 手；身体 → 身体；身份/年龄/脸与肤质 → 脸）；「年龄」「动作与分镜」两组不自动训练。旧记录里每次点选存下来的 `group` 不看；记录里的点选靠名字、id 或 `merged_from`（曾用名）对上清单。所以把一个原因挪到别的分组后，旧点选照样算数，但已经训好的判官头还在看旧区域，要重新 `train`（`verify` 会提示）。可以给某一项加 `region` / `trainable` / `definition` 覆盖默认值。

审片页记录里工具包会读的字段：

| 字段 | 怎么用 |
|---|---|
| `decision`、`reasons[].item`、`reasons[].score`、`time` | 决定、原因、分数、审片时间 |
| `reasons[].carried` | `true` = 这个原因是从上一张自动带过来的，你没碰过；`false` = 你自己选的；没有这个字段 = 不清楚（旧记录）。见下面「标注口径」 |
| `reasons_complete` | `true` = 这张图的原因已经选全，没选的原因确实没有问题 |
| `batch_pass` | `true` = 整屏通过（旧记录靠备注 `整屏通过` 认出来） |
| `recommend` | `true` = 你点了「推荐进入」：这张图**一定进训练**，见下面 |
| `source` | 来源，例如「旧图补审」；可以用 `--source` / `--exclude-source` 筛 |
| `sha256`、`id` | 图片文件的 sha256；新记录的 `id` 是它的前 16 位。考卷名单可以按它登记 |
| `scene` | 场景；没有就从文件名取：`hallway_s1102.png` → `hallway` |
| `note` | 备注。**只在 `trace` 里显示**，不参加训练，也不进任何报告 |
| `seconds` | 审片用时。**只是工作量统计**（`verify` 报总分钟数和每张平均秒数），不用于训练 |

## 标注口径：一次点击（或没点）算什么

一条审片对某一个原因来说，是正例、反例，还是不计入，由下面的规则决定，按顺序看：

1. 这张图审的时候这个原因还没创建（或记录没有时间）→ **不计入**。
2. 判「过」→ **反例**。整屏通过的图分量打折（`batch_pass_weight`）；点了「推荐进入」的不打折。
3. 判「不过」、没选这个原因 → **不知道，不计入**。因为通常只点最明显的那个原因，没点不等于没有。
   记录带 `reasons_complete: true` 时才算反例。所以默认情况下，反例来自判过的图和“原因已选全”的不过图。
4. 判「不过」、选了这个原因 → **正例**。分量 = 带入系数 × 严重程度系数：
   - 带入系数：你自己选的（`carried: false`）= 1；自动带过来没碰过（`carried: true`）= `carried_weight`；不清楚（旧记录）= `carried_unknown_weight`。
   - 严重程度系数（只有计分项）：0 分（最严重）1.5，5 分 1.0，10 分 0.5；硬门项和没打分的是 1。
   - 计分项设了 `"max_score": 4` 时：分数高于 4 的点选算轻微，不计入。

四个旋钮，0 = 不计入，1 = 按满分量，中间的数 = 按这个分量参加训练：

| 旋钮 | 默认 | 含义 |
|---|---|---|
| `unselected_fail_weight` | 0 | 不过的图上没选的原因当反例的分量。0 = 不知道 |
| `batch_pass_weight` | 0.5 | 整屏通过的图当反例的分量 |
| `carried_weight` | 0 | 自动带过来、没碰过的原因当正例的分量。0 = 这张图对这个原因不计入 |
| `carried_unknown_weight` | 0.5 | 旧记录里不清楚是不是带过来的原因当正例的分量 |

在哪里改，后面的盖过前面的：内置默认值 → judge_lab 格式配置里的 `defaults.label_policy`（审片页的 缺陷清单.json 不认这一段）→ 某一项的 `label_rule` → 命令行 `--unselected-fail-weight` / `--batch-pass-weight` / `--carried-weight` / `--carried-unknown-weight`（所有读清单的命令都认）。

```json
{"defaults": {"label_policy": {"unselected_fail_weight": 0.3}},
 "items": {"lighting": {"name": "光线", "kind": "score", "region": "full",
                        "label_rule": {"positive_if": "selected", "batch_pass_weight": 1}}}}
```

现在用的是什么口径、记录里各有多少张，`verify` 的「标注口径」一行会写出来；每个判官头训练时用的口径存在它的 `label_policy` 里。

**默认口径的代价，要知道**：反例只剩判过的图时，一个头容易把“别的原因造成的不过”也打高分（它没见过“不过、但不是因为这一项”的例子）。`verify --signal` 的「只在不过的图里分原因」一列就是量这个的。想让每个头分清“为什么”，就在审片页把原因选全（记录带 `reasons_complete`）。

### 推荐进入

`recommend: true` 的图一定进训练：不会被 `--source` / `--exclude-source` 筛掉，整屏通过也按满分量算。
如果它同时在考卷名单里，这是硬冲突，工具不替你选：probe / train / calibrate 直接报错退出，`verify` 报「断」。把它从考卷名单里拿掉，或者取消推荐。

## 场景：别让判官头学成“认场景”

图是按「场景 × 种子」出的（例如 24 个场景各 4 张）。同一场景的图长得很像，如果交叉验证只保证“同一张图不拆开”，同场景的另外几张会落在训练那一边，头只要认出场景就能得高分，数字虚高。

- 场景取记录里的 `scene`，没有就从文件名取：`<场景>_s<数字>`。
- probe / calibrate / verify 的交叉验证默认 `--group-by auto`：过半的图有场景、且场景不少于 4 个时**按场景分组**（同一场景整组留出，测的是没见过的场景；没有场景的图各自成组），否则按图分组。`--group-by scene` 强制按场景（哪一项做不到就跳过那一项并写明原因），`--group-by image` 强制按图。
- 报告里写明每次用的是「按场景」还是「按图」。有场景信息却只能按图分组时会注明结果可能偏乐观。

`verify --signal` 的表：

| 列 | 含义 |
|---|---|
| 用你的标注 | 交叉验证 AUC |
| 同场景内打乱 | 把标注在**同一场景的图之间**打乱后再训练（每个场景的正例比例不变）。按图分组时是在全部图之间打乱 |
| 只看场景能猜多准 | 完全不看图，只凭同场景其它图的结果来猜这张（留一法）。场景帮不上忙时它会低于 0.5；场景不到两个时是「-」 |
| 只在不过的图里分原因 | 正例对比“不过、但不是这一项”的图（包括没选这一项、不计入训练的那些）。偏保守：那些图里有的其实也有这个问题，只是没点 |
| 结论 | 「学不出来」：AUC < 0.70，或只比打乱高不到 0.10；「和场景分不开」：只看场景就能猜到差不多一样准（≥ AUC − 0.03）；否则「学到了」 |

## 考卷名单（`--exam`）

下面几种写法都认，可以混着写：

```text
# 文本：一行一个，# 后面是注释
批次A__hallway_s1102                       ← id
D:\MarkV4\批次A\hallway_s1103.png          ← 图片路径
9f2c…（64 位十六进制）                      ← 图片文件的 sha256
9f2c4e1ab03d77e0                          ← 16 位 id（sha256 的前 16 位）
```

```json
["9f2c…", {"sha256": "9f2c…"}, {"id": "9f2c4e1ab03d77e0"}, {"image": "D:\\MarkV4\\批次A\\hallway_s1103.png"}]
```

也可以是 `{"items": [...]}`、`{"exam": [...]}`，或者一行一个对象的 JSONL。

一条审片只要 id 对上、由路径生成的 id 对上，或者 sha256 对上（整串，或前 16 位），就是考卷。sha256 先看记录里的字段；记录里没有、而名单里有按 sha256 登记的图时，才去读图片文件现算（同一次运行里按 路径+修改时间+大小 记住，不重复算）。图改了名、挪了位置，按 sha256 登记的考卷照样认得出来。名单里有一条看不懂就报错，不会悄悄跳过。

## 多个记录文件、私有记录、按来源筛选

```bash
python -m judge_lab verify --log 审片记录.jsonl 旧图补审.jsonl --private-log 私有审片.jsonl \
       --config 缺陷清单.json --exclude-source 测试
```

- `--log` 可以给多个文件。同一张图（图片路径相同，不分大小写和正反斜杠；或 id 相同）出现多次时，以最后读到的那条为准；文件按给的顺序读，私有记录最后读。
- `--private-log`：这些文件里的审片照常计数、照常参加训练，但屏幕和报告（md / json）里只出现数量：不出现它们的 id、路径、备注，也不出现对不上清单的原因文字。`trace` 遇到私有记录只说一句“这是私有记录”，加 `--show-private` 才显示。给看图大模型导出的例图永远不取私有记录。
  注意：特征缓存和判官头文件（`.npz`，本地二进制文件）里仍然存着这些图的 id，因为要靠它核对考卷、记住哪些图已经学过。
- `--source A,B` 只用这些来源的审片；`--exclude-source A,B` 去掉这些来源。「推荐进入」的图不受这两个参数影响。

## 私有原因

清单里某一项加上 `"private": true`，它的真名、id、曾用名、含义就不会出现在任何输出里（屏幕、报告、manifest.json、判官头文件、thresholds.json、影子打分、导出的提示词清单）：

- 它对外的 id 是 `p_<10 位哈希>`，名字是「私有项 <哈希后 4 位>」；真名和曾用名只用来对上你的点选，所以点选照样算数。
- 加 `--show-private`（每条命令都有，默认关）时，屏幕和报告里用真名；id 不变，所以判官头文件名和里面存的名字不随这个参数变（判官头、manifest、thresholds 里永远只存代号）。`score` 要显示真名还得给 `--config`。
- `export-prompts` 默认不导出私有原因，加 `--show-private` 才导出。
- 把一个已经训练过的原因改成私有后，旧判官头文件里还留着原名：`verify` 会提示有“多余的判官头”，删掉旧文件后重新 `train`。

## 链路自检：点击有没有到达判官头

```bash
# 每个环节通没通（不用显卡，几秒钟）
python -m judge_lab verify --log 审片记录.jsonl --config 缺陷清单.json --heads heads/ --crops-root crops/

# 再多做一步：用真实骨干验证“你的标注确实推动了权重”（要算特征）
python -m judge_lab verify --log 审片记录.jsonl --config 缺陷清单.json --signal siglip2 --cache cache/

# 看某一张图：每个点击变成了什么、分量多少、被哪个判官头学了
python -m judge_lab trace --log 审片记录.jsonl --config 缺陷清单.json --heads heads/ --image hallway_s1102
```

`verify` 逐项给出「通 / 注意 / 断」，有「断」时退出码为 1：

- 审片记录有没有全部读入；每次原因点选对不对得上清单（对不上 = 点了没用）；后来新增的原因有没有按时间正确计入；图片还在不在；看脸/手的项有没有切图。
- 「标注口径」：现在没选的原因、整屏通过、自动带过来的原因各按什么算，记录里各有多少。
- 「推荐进入」：有多少张；和考卷名单冲突就是「断」。
- 「含义」：有正例的原因里几个写了含义（`definition`），哪些还没写。
- 「审片用时」：总分钟数和每张平均秒数，只是工作量统计。
- 「判官头」：每一项攒了多少正例和反例、判官头学到了哪一张为止；原因换了分组或区域而旧头还在看旧区域时提示重新训练。
- 每一项的状态里同时写还差多少正例、多少反例，例如「还差 7 个正例、3 个反例」（`items list` 也一样）。
- 加 `--signal` 后见上面「场景」一节的表。

`trace` 对一张图显示：决定、时间、来源、是否整屏通过、是否推荐进入、备注、审片用时（仅统计，不用于训练），以及每个原因算正例 / 反例 / 不计入、为什么、训练分量、有没有学进判官头。

## 给看图大模型导出原因清单

```bash
python -m judge_lab export-prompts --log 审片记录.jsonl --config 缺陷清单.json --exam data/exam.txt \
       --out prompts.json --examples 3
```

写出 `prompts.json` 和同名的 `prompts.md`：每个原因的 id、名字、类型（硬门 / 计分）、分组、看哪里、含义，以及最多 N 张正例和 N 张反例的图片文件名（正例先取分量大的，也就是你自己选的、分数最严重的；反例只取判过的图）。例图不取考卷，也不取私有记录。没写含义的原因会标出来。「动作与分镜」这类不训练判官头的语义项也在里面，它们本来就是交给看图大模型的。

## 原因项怎么教给判官头

- 判官头不读文字，只从例子里学：哪些图算正例、反例，见上面「标注口径」。**每个原因各有一个头，所以点原因教的是“为什么不过”，不只是“过或不过”**（前提是有“不过、但不是这一项”的反例，见「默认口径的代价」）。
- **新增原因只统计它创建之后审过的图**（清单里的 `added`）。创建之前的图当时没有这个选项，不计入。
- **改名或合并**：在新项里写 `"merged_from": ["旧名字1", "旧名字2"]`，旧记录里的点选会算到新项上；新项的 `added` 要写最早那个旧项的创建时间。
- 含义（`definition`）是写给人和看图大模型看的，判官头不读。
- 正例和反例都攒到 `--min-pos` / `--min-neg`（默认各 15）以上，`probe` / `train` 才会训练这一项。`items list` 会说还差多少。
- 看脸/手的项需要切图（`--crops-root`）；没有切图时退而看整图分块，会粗，`verify` 会提示。

## 操作步骤

```bash
L="--log 审片记录.jsonl --config 缺陷清单.json --exam data/exam.txt --crops-root data/crops"

# 先看每个原因攒了多少例子
python -m judge_lab items list --log 审片记录.jsonl --config 缺陷清单.json --exam data/exam.txt

# 0) 提取特征（会缓存，重跑时只算新图）
python -m judge_lab embed $L --backbone siglip2,dinov2,clip_l --out cache/

# 1) 骨干比拼：分组 5 折交叉验证（有场景就按场景），报告 AUC，以及达到目标召回率时的召回和误拒率
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

`stats ab` 输出：各组通过率及 Wilson 95% 置信区间、与基线差值的 Newcombe 置信区间、两比例 z 检验 p 值、分场景明细，以及回归检查（任何场景新版本比基线低超过容忍度就报警；加 `--fail-on-regression` 时退出码为 1）。它也认多个 `--log`、`--private-log`（私有记录的场景合成一行「（私有记录）」）和来源筛选。

其他常用参数：`--items a,b` 只处理部分项目；`--model-id siglip2=<id 或本地路径>`；`--model-dir D:\models`（离线加载）；`--device cuda:1`；`--batch-size 32`；`--min-pos/--min-neg 15`；`--recall-hard/--recall-score`；`--group-by auto|scene|image`。

## 不用审片页时的输入格式

**1. 审核日志 JSONL**（一行一张图）：

```json
{"id": "img_0001", "image": "out/img_0001.png", "decision": "fail", "reviewed_at": "2026-11-13T10:05:00-08:00",
 "reasons": ["structure_break", "lighting"], "reasons_complete": true, "scores": {"lighting": 3},
 "recipe_hash": "a1b2", "versions": {"lora": "mark_taste_v1"}, "scene": "indoor"}
```

- `image` 可以是绝对路径，也可以是相对日志文件所在目录的路径。
- `reasons` 里没勾选的项目：这条记录带 `"reasons_complete": true` 才当作“OK”，否则是“不知道”（见「标注口径」）。原因写成字符串时没有 `carried` 信息，按“不清楚是不是带过来的”算（默认半个分量）；写成对象 `{"item": "lighting", "carried": false}` 才是满分量。
- 同一张图出现多次时，以最后一条为准（相当于重审）。

字段名不一样时，用 `--field-map` 改名，支持 `a.b` 这样的嵌套路径，也可以改通过 / 不通过的取值（见 `configs/field_map.example.json`）。`reasons_complete`、`batch_pass`、`recommend`、`source`、`sha256`、`note`、`seconds` 也可以改名：

```json
{"id": "image_id", "image": "file", "decision": "verdict", "reasons": "fail_items",
 "scores": "item_scores", "versions": "meta.versions", "scene": "meta.scene",
 "decision_values": {"pass": ["pass", "通过"], "fail": ["fail", "不通过"]}}
```

**2. 预裁剪（可选）**：`<crops_root>/<image_id>/<region>_<k>.png|jpg`，其中 region 为 face / hand / body / tile / full。
没有预裁剪时：`full` 用整张图，`tile` 用整张图切成 2×2 四块。face / hand 没有裁剪的图会被跳过（人脸、手部检测由主人现有流程负责）；`body` 没有裁剪时回退到 `full`。目录名里的 `image_id` 就是这条审片的 id（新记录是 sha256 的前 16 位）。

**3. 判官头配置**：`configs/heads_v1.json`，共 21 个项目，每项写明 `region`、`kind`（hard 硬门槛 / score 评分项）和 `label_rule`。用 `items add` 往自己的副本里加项目。

- 评分项可以设 `"max_score": 4`：勾选且分数 ≤ 4 才算缺陷；勾选了但分数更高的样本不计入训练。
- `trainable: false` 的项目不自动训练，但仍出现在报告里：年龄偏小 / 偏大由主人在审片页人工判断（暂不装年龄估计模型）；动作和分镜不符、物理不合理是语义判断，交给 VLM 审核。
- 有的项目判官头只是辅助：乱码字主要靠 OCR；不是同一个人、像不像本人主要靠人脸特征相似度。

## 硬件

- RTX 4060 Ti 16GB 或 RTX 5080 16GB 都够用：骨干只做推理（fp16），三个骨干各自都在约 2 GB 显存以内（SigLIP2-so400m 连同文本塔一起加载，是最大的一个），batch 32 没问题；判官头是逻辑回归，CPU 上几秒就训完。
- 特征提取是唯一耗时的一步，结果会缓存在 `cache/<骨干>/<区域>.npz`，之后 probe / train / calibrate / exam 重跑基本不花时间。
- RTX 5080（Blackwell）需要 CUDA 12.8 以上的 torch 版本（`--index-url https://download.pytorch.org/whl/cu128`）。
- 首次使用真实骨干会从 Hugging Face 下载模型；离线时先放到本地，再用 `--model-dir` 指定。

## 产出物

| 文件 | 内容 |
|---|---|
| `reports/probe.md` / `probe.json` | 骨干比拼表；每个项目、每个区域的最佳骨干；交叉验证是按场景还是按图 |
| `heads/<项目>_v<N>.npz` | 判官头：coef、intercept、骨干与模型 id、区域、样本数、训练时的标注口径、训练 id 列表及其 sha256（data_hash）、创建时间。版本号自动递增 |
| `heads/manifest.json` | 每个项目当前启用的版本和历史版本 |
| `heads/thresholds.json` | 建议阈值、OOF 召回、预计误拒率、分组方式、审批状态 |
| `reports/exam.md` / `exam.json` | 考卷上的召回、误拒率、一致率、AUC、是否达标 |
| `reports/verify.md` | 链路自检 |
| `prompts.json` / `prompts.md` | 每个原因的含义和例图文件名，给看图大模型写提示词用 |
| `shadow.jsonl` | 影子打分：每张图每个项目的概率和是否超过阈值（不含任何判定） |

## 设计取舍与限制

- **多实例简化**：一张图有多个裁剪（四块 tile、多张脸或手）时，训练时每个裁剪都继承整张图的标签，正样本也一样；预测时整张图的分数取各裁剪的最大值。缺陷通常只出现在一块里，所以正样本里会混进一些“其实没问题”的裁剪。这是为了简单而有意接受的近似；以后如果主人标出缺陷位置，可以换成更精确的做法。
- 判官头 = 标准化 + 类别平衡的逻辑回归（默认 C=0.1），每个样本带上「标注口径」给的分量。标准化已经并进 coef/intercept，所以使用时就是 `sigmoid(x·coef + b)`。
- 阈值来自 out-of-fold 预测，最终的头则用全部训练数据重新拟合，两者的分数分布会略有差异，所以要看 `exam` 结果再批准阈值。
- 召回、误拒率、一致率都按图计算，不带分量：误拒率 = 主人判为 OK 的图里被标成超过阈值的比例。“不知道”的图（不过、没选这一项、原因也没选全）不算对也不算错，考卷上也一样。
- 「只看场景能猜多准」用的是留一法，每个场景只有几张图时会偏低，所以它只在场景和标签绑得很紧时才报警；防止“认场景”主要靠按场景分组的交叉验证。场景只有十几二十个时，按场景分组的 AUC 本身上下浮动也不小（±0.1 左右），看趋势，别看小数点。
- 样本少于 `--min-pos/--min-neg`（默认各 15）的项目不会训练，报告里会写明原因。
