"""The chain the owner asked to be proven: a click on the review page -> a label -> a head's weights.

The fixture writes files in the review page's REAL formats (审片记录.jsonl and 缺陷清单.json as the page
writes them on the owner's machine: decisions 过 / 不过, reasons as objects spelled by display name and marked
`carried: false` (chosen by the owner, not carried over from the previous picture), no id, Windows-style paths,
scores "default" on a pass), with made-up pictures.

The records carry no `reasons_complete`, so under the default policy a failed picture says nothing about the
reasons that were NOT selected on it: the negatives of every reason are the passed pictures only."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from judge_lab.backbones import BackboneSpec
from judge_lab.cli import main
from judge_lab.data import (
    HeadItem, assess, canonicalize, explain_label, label_counts, load_config, load_log_ex, severity_weight, why_code,
)
from judge_lab.embeddings import ensure_embeddings
from judge_lab.heads import auc, build_item_data, file_stem, needed_regions, oof_scores

W, H, N = 48, 64, 240
STRUCT, LIGHT, LAYER, AGE = "结构断裂（直线断开、错位）", "光线不自然 / 像棚拍", "图层失调", "年龄不符合设定 · 偏小"
OLD_A, OLD_B = "图层割裂，看起来假", "图层错位"


def flags(i):
    return i % 3 == 0, i % 4 == 1, i % 5 == 2  # struct, light, layer


def picture(rng, struct, light, layer):
    a = rng.integers(0, 110, size=(H, W, 3)).astype(np.int32)
    if light:
        a[..., 0] += 140
    if layer:
        a[H // 2:, :, 1] += 140
    if struct:
        x, y = rng.integers(0, W - 10), rng.integers(0, H - 10)
        a[y:y + 10, x:x + 10] = 255
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def defects_list(merged=True):
    items = [
        {"name": "多余的人 / 肢体", "group": "人员", "kind": "hard"},
        {"name": STRUCT, "group": "结构", "kind": "hard"},
        {"name": "手指数量 / 形状不对", "group": "手", "kind": "hard"},
        {"name": AGE, "group": "年龄", "kind": "hard"},
        {"name": LIGHT, "group": "光线", "kind": "score"},
        {"name": "动作和分镜不符", "group": "动作与分镜", "kind": "score"},
    ]
    if merged:
        items.append({"name": LAYER, "group": "其他", "kind": "hard", "added": "2026-11-12 05:41",
                      "merged_from": [OLD_A, OLD_B], "definition": "人物和背景像两张图拼起来"})
    return {"version": 4, "time": "2026-11-13 09:00", "by": "Owner 在审片页新增", "items": items}


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    root = tmp_path_factory.mktemp("page")
    batch = root / "MarkV4" / "批次A"
    batch.mkdir(parents=True)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(N):
        st, li, la = flags(i)
        f = batch / f"hallway_s{1100 + i}.png"
        picture(rng, st, li, la).save(f)
        early = i < 20  # reviewed before the layer reason existed
        reasons = []
        if st:
            reasons.append({"item": STRUCT, "group": "结构", "kind": "hard", "score": None, "carried": False})
        if li:
            reasons.append({"item": LIGHT, "group": "光线", "kind": "score", "score": [1, 4, 8][i % 3], "carried": False})
        if la and not early:
            reasons.append({"item": OLD_A if i % 2 else OLD_B, "group": "其他", "kind": "hard", "score": None,
                            "carried": False})
        dec = "不过" if reasons else "过"
        rows.append({"time": "2026-11-12 03:00:00" if early else "2026-11-12 17:14:06",
                     "image": str(f).replace("/", "\\"), "recipe": "Moss·b68a4b9f", "seed": 1100 + i, "decision": dec,
                     "reasons": reasons,
                     "scores": "default" if dec == "过" else {r["item"]: r["score"] for r in reasons if r["kind"] == "score"},
                     "note": "", "seconds": None, "defects_version": 3, "by": "Owner"})
    log = root / "审片记录.jsonl"
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    cfg = root / "缺陷清单.json"
    cfg.write_text(json.dumps(defects_list(), ensure_ascii=False, indent=1), encoding="utf-8")
    old_cfg = root / "缺陷清单_未合并.json"
    old_cfg.write_text(json.dumps(defects_list(merged=False), ensure_ascii=False), encoding="utf-8")
    return {"root": root, "log": log, "cfg": cfg, "old_cfg": old_cfg}


def load(page, cfg="cfg"):
    reviews, st = load_log_ex(page["log"])
    items, _ = load_config(page[cfg])
    return reviews, st, items, canonicalize(reviews, items)


def by_name(items, name) -> HeadItem:
    return next(it for it in items.values() if it.name == name)


def test_real_page_records_load_without_any_mapping(page):
    reviews, st, items, orphans = load(page)
    assert st["lines"] == st["parsed"] == N and not st["skipped"] and st["derived_ids"] == N and st["no_time"] == 0
    r = next(x for x in reviews if x.id == "批次A__hallway_s1100")  # i=0: struct only
    assert r.decision == "fail" and r.reasons == [by_name(items, STRUCT).id] and r.image.is_file()
    assert r.recipe_hash == "Moss·b68a4b9f" and r.reviewed_at == "2026-11-12 03:00:00"
    assert sum(x.decision == "pass" for x in reviews) == sum(1 for i in range(N) if not any(flags(i)[:2]) and not (flags(i)[2] and i >= 20))
    assert not orphans
    light = by_name(items, LIGHT)
    assert {x.scores[light.id] for x in reviews if light.id in x.reasons} == {1.0, 4.0, 8.0}
    # regions come from the page's groups; age and storyboard items are never auto-trained
    assert by_name(items, STRUCT).region == "tile" and light.region == "full"
    assert by_name(items, "手指数量 / 形状不对").region == "hand" and by_name(items, "手指数量 / 形状不对").fallback_region == "tile"
    assert not by_name(items, AGE).trainable and not by_name(items, "动作和分镜不符").trainable


def test_merged_reason_keeps_the_clicks_made_under_both_old_names(page):
    reviews, _, items, orphans = load(page)
    layer = by_name(items, LAYER)
    assert not orphans and layer.aliases == (OLD_A, OLD_B)
    labels = {x.id: explain_label(x, layer)[0] for x in reviews}
    n_pos = sum(1 for i in range(N) if flags(i)[2] and i >= 20)
    assert sum(v == 1 for v in labels.values()) == n_pos > 30
    why = Counter(why_code(x, layer) for x in reviews)
    assert why["time"] == 20  # reviewed before the reason existed: not counted as OK
    # failed for another reason, this one not selected: unknown (the owner clicks the most obvious reason only)
    n_unknown = sum(1 for i in range(20, N) if (flags(i)[0] or flags(i)[1]) and not flags(i)[2])
    assert why["unknown"] == n_unknown > 50
    assert sum(v is None for v in labels.values()) == 20 + n_unknown
    # so the negatives are exactly the pictures passed after the reason existed
    assert sum(v == 0 for v in labels.values()) == sum(1 for i in range(20, N) if not any(flags(i))) > 60
    assert label_counts(reviews, layer) == {"pos": n_pos, "neg": why["pass"], "ambiguous": 20 + n_unknown}
    # without the merge entry the same clicks reach nothing, and verify says the chain is broken
    _, _, _, lost = load(page, "old_cfg")
    assert set(lost) == {OLD_A, OLD_B} and sum(lost.values()) == n_pos
    assert main(["verify", "--log", str(page["log"]), "--config", str(page["old_cfg"]),
                 "--out", str(page["root"] / "v_old.md")]) == 1
    assert "对不上清单" in (page["root"] / "v_old.md").read_text(encoding="utf-8")


def test_score_sets_the_training_weight(page):
    reviews, _, items, _ = load(page)
    light = by_name(items, LIGHT)
    w = {x.scores[light.id]: severity_weight(x, light) for x in reviews if light.id in x.reasons}
    assert w == {1.0: 1.4, 4.0: 1.1, 8.0: 0.7}
    assert severity_weight(next(x for x in reviews if x.decision == "pass"), light) == 1.0
    # the page marks every pick `carried: false` (the owner chose it), so the fitting weight is the severity alone
    assert {x.scores[light.id]: assess(x, light)[2] for x in reviews if light.id in x.reasons} == w
    assert all(x.carried == {k: False for k in x.reasons} for x in reviews)


def test_verify_and_trace_on_page_files(page, capsys):
    w = page["root"]
    args = ["--log", str(page["log"]), "--config", str(page["cfg"])]
    assert main(["verify", *args, "--signal", "fake", "--cache", str(w / "cache"), "--out", str(w / "verify.md")]) == 0
    md = (w / "verify.md").read_text(encoding="utf-8")
    assert "| 通 | 审片记录读入 |" in md and "| 通 | 原因 → 清单 |" in md and "| 通 | 图片文件 |" in md
    assert "还没有训练过任何判官头" in md and "缺切图" in md  # honest about what is not connected yet
    assert "| 原因 | 正例 | 反例 | 用你的标注 | 同场景内打乱 | 只看场景能猜多准 | 只在不过的图里分原因 | 结论 |" in md
    n_pass = sum(1 for i in range(N) if not any(flags(i)[:2]) and not (flags(i)[2] and i >= 20))
    n_layer = sum(1 for i in range(20, N) if flags(i)[2])
    late_pass = sum(1 for i in range(20, N) if not any(flags(i)))  # passed after the layer reason existed
    expect = {STRUCT: (N // 3, n_pass), LIGHT: (N // 4, n_pass), LAYER: (n_layer, late_pass)}
    for name in (STRUCT, LIGHT, LAYER):
        row = next(l for l in md.splitlines() if l.startswith(f"| {name} |") and "学到了" in l)
        cells = [c.strip() for c in row.strip("|").split("|")]
        # negatives are the passed pictures only: a failed picture is "unknown" for the reasons not selected on it
        assert (int(cells[1]), int(cells[2])) == expect[name], row
        real, shuffled, why = float(cells[3]), float(cells[4]), float(cells[6])
        assert real > 0.9 and shuffled < 0.65, row
        assert cells[5] == "-"  # every picture is the same scene (hallway): the scene cannot explain anything
        # Among failed pictures only. The colour cast and the layer split fill the picture and are told apart from
        # the other reasons. The small structure break is not: with passed pictures as the only negatives its head
        # also fires on the colour cast that some of its positives carry. verify says so instead of hiding it.
        assert (why < 0.6) if name == STRUCT else (why > 0.9), row
    assert f"学到了、但在不过的图里分不出是不是这一项的: {STRUCT}" in md
    # when the owner does mark every reason (here: the CLI knob that reads "not selected" as "absent"), every head
    # separates its reason from the others, and the failed pictures become negatives
    assert main(["verify", *args, "--signal", "fake", "--cache", str(w / "cache"), "--unselected-fail-weight", "1",
                 "--out", str(w / "verify_complete.md")]) == 0
    full = (w / "verify_complete.md").read_text(encoding="utf-8")
    for name in (STRUCT, LIGHT, LAYER):
        row = next(l for l in full.splitlines() if l.startswith(f"| {name} |") and "学到了" in l)
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert int(cells[2]) == (N - 20 if name == LAYER else N) - expect[name][0], row
        assert float(cells[3]) > 0.9 and float(cells[6]) > 0.9, row
    assert "不过但没选的原因 → 算反例" in full and "分不出是不是这一项" not in full
    # the per-item table counts the same way, and says how the clicks are being read
    row = next(l for l in md.splitlines() if l.startswith(f"| {STRUCT} | 硬门 |"))
    assert [c.strip() for c in row.strip("|").split("|")][3:6] == [str(N // 3), str(n_pass), str(N - N // 3 - n_pass)]
    assert "| 通 | 标注口径 |" in md and "不过但没选的原因 → 不计入（当作不知道）" in md
    assert "交叉验证分组: 按图 3 项" in md  # one scene only -> grouped by image, and not flagged as optimistic
    assert "偏乐观" not in md
    # train, then the same check shows the reviews as learned; a new review shows as not learned yet
    heads = w / "heads"
    assert main(["train", *args, "--exam", str(w / "none.txt"), "--cache", str(w / "cache"), "--backbone", "fake",
                 "--out", str(heads)]) == 2  # exam list missing -> refuses rather than guessing
    (w / "exam.txt").write_text("", encoding="utf-8")
    assert main(["train", *args, "--exam", str(w / "exam.txt"), "--cache", str(w / "cache"), "--backbone", "fake",
                 "--out", str(heads)]) == 0
    stems = {p.name for p in heads.glob("*.npz")}
    assert f"{file_stem(next(iter(load_config(page['cfg'])[0])))}_v1.npz" not in stems  # item 0 has no positives
    assert len(stems) == 3 and all(s.startswith("h_") for s in stems)  # Chinese ids -> safe file names
    capsys.readouterr()
    assert main(["trace", *args, "--heads", str(heads), "--image", "hallway_s1122"]) == 0  # i=22: layer only
    out = capsys.readouterr().out
    assert "图层失调" in out and "有问题（正例）" in out and "已学进判官头 v1" in out
    assert "| 1.00 |" in out  # owner-chosen hard-gate pick: full weight
    assert "其余 0 项" in out and "另有 6 项不知道有没有这个问题" in out  # failed: the other reasons are unknown
    assert main(["trace", *args, "--heads", str(heads), "--image", "hallway_s1102"]) == 0  # i=2: layer picture, reviewed early
    assert "当时还没有这些选项" in capsys.readouterr().out


def test_reasons_teach_why_not_just_pass_or_fail(page):
    """Among failed pictures only: a head trained on one reason separates 'failed for THIS reason' from
    'failed for another reason'; a head that only knows pass / fail cannot.

    This shows the mechanism when the owner DOES mark every reason, so the items are built with
    unselected_fail_weight = 1 (a failed picture is a full negative for the reasons not selected on it).
    Under the default policy (0) those pictures are unknown and never reach the head."""
    reviews, _, items, _ = load(page)
    spec = BackboneSpec.create("fake")
    its = [by_name(items, n) for n in (STRUCT, LIGHT, LAYER)]
    for it in its:
        it.label_rule = {**it.label_rule, "unselected_fail_weight": 1}
    caches = ensure_embeddings(spec, needed_regions(its) | {"full"}, [(r.id, r.image) for r in reviews], page["root"] / "cache", None)
    failed = {r.id for r in reviews if r.decision == "fail"}
    anyfail = HeadItem("any_fail", "不过", "hard", "tile", None, True, {"positive_if": "selected"}, 0.95)
    as_passfail = []
    for r in reviews:  # the coarse alternative: every failed picture carries the same single label
        c = type(r)(r.id, r.image, r.decision, ["any_fail"] if r.decision == "fail" else [], {}, reviewed_at=r.reviewed_at)
        as_passfail.append(c)
    ids, _, coarse = oof_scores(build_item_data(anyfail, as_passfail, caches), 5, 0.1)
    coarse = dict(zip(ids, coarse))
    out = {}
    for it in its:
        ids, y, s = oof_scores(build_item_data(it, reviews, caches), 5, 0.1)
        m = np.array([i in failed for i in ids])
        out[it.name] = (auc(y[m], s[m]), auc(y[m], np.array([coarse[i] for i in ids])[m]))
    print("\n只在不过的图里分原因 AUC（按原因训练的头, 只知道过/不过的头）:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in out.items()})
    assert all(a > 0.9 for a, _ in out.values()), out
    # the pass / fail head latches onto the most visible defect (the colour cast) and is near chance on the others
    coarse_scores = sorted(b for _, b in out.values())
    assert coarse_scores[0] < 0.6 and coarse_scores[1] < 0.6, out
    assert np.mean([a for a, _ in out.values()]) - np.mean(coarse_scores) > 0.3, out
