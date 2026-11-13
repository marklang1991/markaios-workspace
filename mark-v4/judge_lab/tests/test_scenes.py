"""Scene confound: pictures come as scenes x seeds, and a head must not be credited for recognising the scene.

The fixture: 24 scenes x 4 seeds (the shape of the owner's first real batch). Every scene has its own look.
  REAL  - a visible defect (green lower half) on two seeds of every scene: independent of the scene.
  CONF  - no visible mark at all; the label is decided by the scene alone (12 "bad" scenes)."""
import json

import numpy as np
import pytest
from PIL import Image

from judge_lab.backbones import BackboneSpec
from judge_lab.cli import main
from judge_lab.data import canonicalize, derive_scene, load_config, load_logs
from judge_lab.embeddings import ensure_embeddings
from judge_lab.heads import (
    auc, build_item_data, grouped_folds, oof_run, oof_scores, permuted, scene_folds, scene_only_auc,
)
from judge_lab.util import JudgeLabError, read_json

W, H, SCENES, SEEDS = 48, 64, 24, 4
REAL, CONF = "图层失调", "风格不对"


def has_real(k):
    return k % 2 == 0


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("scenes")
    batch = root / "批次S"
    batch.mkdir()
    rng = np.random.default_rng(11)
    bad = set(rng.permutation(SCENES)[:SCENES // 2].tolist())
    rows = []
    for sc in range(SCENES):
        look = rng.integers(0, 130, size=(4, 3, 3))  # the scene's own colour layout
        base = np.kron(look, np.ones((H // 4, W // 3, 1))).astype(np.int32)
        for k in range(SEEDS):
            a = base + rng.integers(0, 50, size=(H, W, 3))
            if has_real(k):
                a[H // 2:, :, 1] += 120
            f = batch / f"scene{sc:02d}_s{100 + k}.png"
            Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(f)
            reasons = ([{"item": CONF, "kind": "hard", "score": None, "carried": False}] if sc in bad else []) + \
                      ([{"item": REAL, "kind": "hard", "score": None, "carried": False}] if has_real(k) else [])
            rows.append({"time": "2026-11-12 17:14:06", "image": str(f), "decision": "不过" if reasons else "过",
                         "reasons": reasons, "reasons_complete": True})
    log = root / "审片记录.jsonl"
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    cfg = root / "缺陷清单.json"
    cfg.write_text(json.dumps({"version": 1, "items": [{"name": REAL, "group": "其他", "kind": "hard"},
                                                       {"name": CONF, "group": "其他", "kind": "hard"}]},
                              ensure_ascii=False), encoding="utf-8")
    reviews, _ = load_logs([log])
    items, _ = load_config(cfg)
    canonicalize(reviews, items)
    caches = ensure_embeddings(BackboneSpec.create("fake"), ["full"], [(r.id, r.image) for r in reviews],
                               root / "cache", None)
    return {"root": root, "log": log, "cfg": cfg, "reviews": reviews, "items": items, "caches": caches, "bad": bad}


def data(world, name, reviews=None):
    return build_item_data(world["items"][name], reviews or world["reviews"], world["caches"])


def test_scene_comes_from_the_record_or_the_file_stem(tmp_path, world):
    assert derive_scene("D:\\MarkV4\\批次A\\hallway_s1102.png") == "hallway"
    assert derive_scene("out/night_street_v2_s7.jpg") == "night_street_v2"
    assert derive_scene("out/img_0001.png") is None and derive_scene("out/_s12.png") is None and derive_scene(None) is None
    rows = [{"image": "a/hallway_s1102.png", "decision": "过"},
            {"image": "a/hallway_s1103.png", "decision": "过", "scene": "走廊·夜"},  # the record's own field wins
            {"image": "a/photo17.png", "decision": "过"}, {"id": "no_image", "decision": "过"}]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert [r.scene for r in load_logs([log])[0]] == ["hallway", "走廊·夜", None, None]
    assert {r.scene for r in world["reviews"]} == {f"scene{i:02d}" for i in range(SCENES)}


def test_scene_grouped_folds_never_split_a_scene(world):
    d = data(world, REAL)
    assert d.scenes is not None and len(d.scenes) == len(d.y) == SCENES * SEEDS
    for seed in range(3):
        folds, k = scene_folds(d, 5, seed)
        assert k == 5 and set(folds.tolist()) == set(range(5))
        for s in np.unique(d.scenes):
            assert len(set(folds[d.scenes == s].tolist())) == 1  # all seeds of a scene share one fold
        for f in range(5):
            assert not set(d.scenes[folds == f]) & set(d.scenes[folds != f])
            assert set(d.y[folds != f].tolist()) == {0, 1}  # every training split has both classes
    run = oof_run(d, 5, 0.1)
    assert (run.mode, run.k, run.n_scenes, run.note) == ("scene", 5, SCENES, "")
    assert oof_scores(d, 5, 0.1)[0] == run.ids and np.allclose(oof_scores(d, 5, 0.1)[2], run.s)  # same call, as a tuple
    # the image-grouped folds are still there, and do split scenes
    folds = grouped_folds(d.groups, d.y, 5, 0)
    assert any(len(set(folds[d.scenes == s].tolist())) > 1 for s in np.unique(d.scenes))


def test_when_scene_grouping_is_used(world):
    reviews = world["reviews"]
    # fewer than 4 scenes -> by image (as the two-scene synthetic set of conftest.py), unless forced
    three = [r for r in reviews if r.scene in ("scene00", "scene01", "scene02")]
    d = data(world, REAL, three)
    run = oof_run(d, 3, 0.1)
    assert run.mode == "image" and run.n_scenes == 3 and run.note
    assert oof_run(d, 3, 0.1, by="scene").mode == "scene"
    # fewer than half of the pictures have a scene -> by image; pictures without one are groups of their own
    d = data(world, REAL)
    half = d.scenes.copy()
    half[np.isin(d.scenes, [f"scene{i:02d}" for i in range(13, SCENES)])] = ""
    d.scenes = half
    assert oof_run(d, 5, 0.1).mode == "scene"  # 13 of 24 scenes kept: 52 of 96 pictures still have one
    folds, _ = scene_folds(d, 5, 0)
    assert all(len(set(folds[d.scenes == s].tolist())) == 1 for s in np.unique(d.scenes) if s)
    assert len(set(folds[d.scenes == ""].tolist())) > 1  # scene-less pictures are spread, one group each
    half[np.isin(d.scenes, ["scene11", "scene12"])] = ""
    assert oof_run(d, 5, 0.1).mode == "image"  # 44 of 96 now
    # no scenes at all
    d.scenes = np.array([""] * len(d.y), str)
    assert oof_run(d, 5, 0.1).mode == "image" and oof_run(d, 5, 0.1).n_scenes == 0
    with pytest.raises(JudgeLabError):
        oof_run(d, 5, 0.1, by="scene")
    with pytest.raises(JudgeLabError):
        oof_run(d, 5, 0.1, by="scenes")
    # every positive in ONE scene: no scene split can train on both classes -> auto falls back, forced refuses
    one = [r for r in reviews if r.scene == "scene00" or not has_real(int(r.image.stem[-1]))]
    d = data(world, REAL, one)
    assert d.n_pos == 2 and len(set(d.scenes[d.y == 1])) == 1
    run = oof_run(d, 5, 0.1)
    assert run.mode == "image" and run.k == 2 and "分不出" in run.note
    with pytest.raises(JudgeLabError):
        oof_run(d, 5, 0.1, by="scene")


def test_shuffle_within_scene_and_scene_only_baseline(world):
    real, conf = data(world, REAL), data(world, CONF)

    def per_scene(d):
        _, y, scenes = d.images()
        return {s: int(sum(v for v, x in zip(y, scenes) if x == s)) for s in set(scenes)}

    p = permuted(real, 3)
    assert per_scene(p) == per_scene(real) and p.w is None  # every scene keeps its share of positives
    assert (p.y != real.y).any()
    assert (permuted(conf, 3).y == conf.y).all()  # nothing varies inside a scene: nothing to shuffle
    g = permuted(conf, 3, within_scene=False)  # the old global shuffle
    assert per_scene(g) != per_scene(conf) and g.y.sum() == conf.y.sum()
    # the scene alone predicts CONF perfectly and REAL not at all
    assert scene_only_auc(conf) == 1.0
    assert scene_only_auc(real) < 0.5
    conf.scenes = np.array(["only"] * len(conf.y), str)
    assert scene_only_auc(conf) is None  # a single scene: nothing to compare


def test_scene_confounded_reason_is_not_credited(world):
    conf, real = data(world, CONF), data(world, REAL)
    by_image = oof_run(conf, 5, 0.1, by="image")
    by_scene = oof_run(conf, 5, 0.1, by="scene")
    a_image, a_scene = auc(by_image.y, by_image.s), auc(by_scene.y, by_scene.s)
    print(f"\n场景决定标签的原因: 按图 AUC {a_image:.3f}，按场景 AUC {a_scene:.3f}")
    assert a_image > 0.85  # other seeds of the same scene are in the training folds: the head recognises the scene
    assert abs(a_scene - 0.5) < 0.3 and a_image - a_scene > 0.2  # tested on scenes it has never seen: chance
    assert oof_run(conf, 5, 0.1).mode == "scene"  # and this is what "auto" uses here
    # a defect that is really in the picture survives scene grouping
    r = oof_run(real, 5, 0.1)
    assert r.mode == "scene" and auc(r.y, r.s) > 0.95


def _signal_rows(md):
    rows = {}
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 8 and cells[0] in (REAL, CONF):
            rows[cells[0]] = cells
    return rows


def test_verify_signal_reports_the_confound(world):
    w = world["root"]
    args = ["verify", "--log", str(world["log"]), "--config", str(world["cfg"]), "--signal", "fake",
            "--cache", str(w / "cache")]
    assert main([*args, "--out", str(w / "v_auto.md")]) == 0
    md = (w / "v_auto.md").read_text(encoding="utf-8")
    rows = _signal_rows(md)
    assert rows[REAL][7] == "学到了" and float(rows[REAL][3]) > 0.95
    assert rows[CONF][7] == "学不出来" and float(rows[CONF][3]) < 0.8  # scene-grouped: nothing left to learn
    assert rows[CONF][3] == rows[CONF][4]  # no label varies inside a scene, so shuffling within scenes changes nothing
    assert rows[CONF][5] == "1.000"  # 只看场景能猜多准
    assert "交叉验证分组: 按场景 2 项" in md and "偏乐观" not in md
    assert f"学不出来的: {CONF}（正反例几乎按场景分开" in md

    # forced image grouping: the head looks excellent, and verify says it cannot be told from knowing the scene
    assert main([*args, "--group-by", "image", "--out", str(w / "v_image.md")]) == 0
    md = (w / "v_image.md").read_text(encoding="utf-8")
    rows = _signal_rows(md)
    assert float(rows[CONF][3]) > 0.85 and float(rows[CONF][4]) < 0.7 and rows[CONF][7] == "和场景分不开"
    assert rows[REAL][7] == "学到了"
    assert "按图 2 项" in md and "结果可能偏乐观" in md and f"和场景分不开的: {CONF}" in md
    for text in ((w / "v_auto.md").read_text(encoding="utf-8"), md):
        assert f"| {CONF} | 48 | 48 |" in text
        assert not any(l.startswith(f"| {CONF} |") and "学到了" in l for l in text.splitlines())


def test_probe_and_calibrate_record_the_grouping(world, capsys):
    w = world["root"]
    (w / "exam.txt").write_text("", encoding="utf-8")
    args = ["--log", str(world["log"]), "--config", str(world["cfg"]), "--exam", str(w / "exam.txt"),
            "--cache", str(w / "cache")]
    assert main(["probe", *args, "--backbones", "fake", "--out", str(w / "probe.md")]) == 0
    js = read_json(w / "probe.json")
    res = {k: v["results"]["fake"] for k, v in js["items"].items()}
    assert {r["cv_mode"] for r in res.values()} == {"scene"} and js["cv"] == {
        "by_scene": 2, "by_image": 0, "by_image_although_scenes": 0}
    assert res[REAL]["auc"] > 0.95 and res[CONF]["auc"] < 0.8
    md = (w / "probe.md").read_text(encoding="utf-8")
    assert "| 交叉验证 |" in md and "按场景 2 次" in md and "| 按场景 |" in md
    assert main(["probe", *args, "--backbones", "fake", "--group-by", "image", "--out", str(w / "probe_img.md")]) == 0
    js = read_json(w / "probe_img.json")
    assert js["items"][CONF]["results"]["fake"]["auc"] > 0.85  # the optimistic number, and the report says so
    assert js["cv"]["by_image_although_scenes"] == 2
    assert "结果可能偏乐观" in (w / "probe_img.md").read_text(encoding="utf-8")
    assert "按图（可能偏乐观）" in (w / "probe_img.md").read_text(encoding="utf-8")
    assert main(["train", *args, "--backbone", "fake", "--out", str(w / "heads")]) == 0
    assert main(["calibrate", *args, "--heads", str(w / "heads")]) == 0
    th = read_json(w / "heads" / "thresholds.json")["items"]
    assert {e["cv_mode"] for e in th.values()} == {"scene"}
    assert main(["calibrate", *args, "--heads", str(w / "heads"), "--group-by", "image"]) == 0
    assert {e["cv_mode"] for e in read_json(w / "heads" / "thresholds.json")["items"].values()} == {"image"}
