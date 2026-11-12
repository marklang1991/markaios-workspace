"""End-to-end CLI round trip with the `fake` backbone: embed -> probe -> train -> calibrate -> exam -> score."""
import json

import numpy as np
import pytest

from judge_lab.backbones import REGISTRY, FakeBackbone, register_backbone
from judge_lab.cli import main
from judge_lab.data import load_config, load_exam
from judge_lab.heads import head_versions, load_active_heads, load_head, save_head
from judge_lab.util import read_json

from .conftest import common


@pytest.fixture(scope="module")
def run(ds, tmp_path_factory):
    register_backbone("fake_b", FakeBackbone, "fake-g8-d32-s1")
    w = tmp_path_factory.mktemp("work")
    args = common(ds, w / "cache")
    p = {"w": w, "args": args, "probe_md": w / "reports" / "probe.md", "heads": w / "heads"}
    no_cache = args[:-2]
    assert main(["embed", *no_cache, "--backbone", "fake,fake_b", "--out", str(w / "cache")]) == 0
    assert main(["probe", *args, "--backbones", "fake,fake_b", "--out", str(p["probe_md"])]) == 0
    assert main(["train", *args, "--probe", str(w / "reports" / "probe.json"), "--out", str(p["heads"])]) == 0
    assert main(["calibrate", *args, "--heads", str(p["heads"])]) == 0
    yield p
    REGISTRY.pop("fake_b", None)


def test_probe_report(run):
    md = run["probe_md"].read_text(encoding="utf-8")
    assert "| 项目 | 名称 | 类型 | 区域 | 骨干 | 正样本 | 负样本 |" in md
    assert "结构断裂" in md and "不自动训练 (trainable=false)" in md and "样本不足" in md
    js = read_json(run["probe_md"].with_suffix(".json"))
    assert js["exam"]["exam_in_log"] == 20 and js["exam"]["train_pool"] == 100
    sb = js["items"]["structure_break"]
    assert set(sb["results"]) == {"fake", "fake_b"}
    assert sb["results"]["fake"]["n_pos"] + sb["results"]["fake"]["n_neg"] == 100  # no exam images
    assert sb["best_backbone"] in ("fake", "fake_b") and sb["results"][sb["best_backbone"]]["auc"] > 0.8
    assert js["items"]["lighting"]["results"]["fake"]["auc"] > 0.9
    assert js["items"]["age_too_young"]["results"] == {}  # trainable=false: reported, never probed
    assert js["regions"]["tile"]["best_backbone"] and js["overall"]["best_backbone"]


def test_train_outputs_and_versioning(run, ds):
    heads_dir = run["heads"]
    manifest = read_json(heads_dir / "manifest.json")
    assert set(manifest["items"]) == {"structure_break", "lighting"}  # others lack samples / not trainable
    h = load_head(heads_dir / "structure_break_v1.npz")
    exam = load_exam(ds["exam"])
    assert h.meta["region"] == "tile" and h.meta["backbone"] in ("fake", "fake_b")
    assert len(h.train_ids) == 100 and not set(h.train_ids) & exam
    assert h.meta["n_pos"] == 34 and len(h.meta["data_hash"]) == 64 and h.meta["created_at"]
    assert h.coef.shape == (h.meta["dim"],)
    # retraining bumps the version and the manifest points at it
    assert main(["train", *run["args"], "--backbone", "fake", "--items", "lighting", "--out", str(heads_dir)]) == 0
    assert head_versions(heads_dir, "lighting") == [1, 2]
    manifest = read_json(heads_dir / "manifest.json")
    assert manifest["items"]["lighting"]["active"]["version"] == 2
    assert len(manifest["items"]["lighting"]["history"]) == 2


def test_non_trainable_never_trained(run, ds):
    out = run["w"] / "heads_low"
    assert main(["train", *run["args"], "--backbone", "fake", "--min-pos", "2", "--min-neg", "2",
                 "--out", str(out)]) == 0
    names = {p.name for p in out.glob("*.npz")}
    assert "composition_v1.npz" in names  # 3 positives, allowed by the lowered minimum
    items, _ = load_config()
    assert not any(n.startswith(k + "_v") for n in names for k, it in items.items() if not it.trainable)


def test_calibrate_thresholds(run):
    th = read_json(run["heads"] / "thresholds.json")
    assert "主人" in th["note"]
    for item_id, e in th["items"].items():
        assert e["status"] == "proposed" and e["oof_recall"] >= e["target_recall"]
        assert 0 <= e["expected_false_reject_rate"] <= 1
    assert th["items"]["structure_break"]["target_recall"] == 0.95
    assert th["items"]["lighting"]["target_recall"] == 0.8


def test_calibrate_keeps_owner_approved_threshold(run, tmp_path):
    th_path = tmp_path / "thresholds.json"
    th = read_json(run["heads"] / "thresholds.json")
    th["items"]["lighting"].update(status="approved", threshold=0.42, approved_by="owner")
    th_path.write_text(json.dumps(th), encoding="utf-8")
    assert main(["calibrate", *run["args"], "--heads", str(run["heads"]), "--out", str(th_path)]) == 0
    e = read_json(th_path)["items"]["lighting"]
    assert e["status"] == "approved" and e["threshold"] == 0.42 and e["approved_by"] == "owner"
    assert e["proposed_threshold"] != 0.42 and "approved_threshold_oof" in e


def test_exam_uses_only_exam_ids(run, ds):
    out = run["w"] / "reports" / "exam.md"
    assert main(["exam", *run["args"], "--heads", str(run["heads"]), "--out", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    assert "考试报告" in md and "trainable=false" in md
    res = read_json(out.with_suffix(".json"))
    sb = res["items"]["structure_break"]
    assert sb["n_pos"] + sb["n_neg"] == 20 and sb["n_pos"] == 6  # 102,105,108,111,114,117
    assert sb["threshold_source"] == "thresholds.json"
    for k in ("recall", "false_reject_rate", "agreement", "auc", "meets_target"):
        assert k in sb
    assert res["items"]["lighting"]["recall"] >= 0.8


def test_exam_refuses_contaminated_head(run, ds, tmp_path):
    items, _ = load_config()
    h = load_head(run["heads"] / "structure_break_v1.npz")
    save_head(tmp_path, items["structure_break"], h.coef, h.intercept, h.train_ids + ["img_100"], h.meta)
    assert main(["exam", *run["args"], "--heads", str(tmp_path), "--out", str(tmp_path / "e.md")]) == 2
    assert not (tmp_path / "e.md").exists()


def test_score_shadow_only(run, ds):
    out = run["w"] / "shadow.jsonl"
    listing = run["w"] / "new.txt"
    listing.write_text("\n".join(str(ds["images"] / f"img_{i:03d}.png") for i in (100, 105, 0)), encoding="utf-8")
    assert main(["score", "--images", str(listing), "--heads", str(run["heads"]), "--crops-root", str(ds["crops"]),
                 "--out", str(out)]) == 0
    assert main(["score", "--images", str(ds["images"]), "--heads", str(run["heads"]), "--out", str(out),
                 "--append"]) == 0
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 + 120
    text = out.read_text(encoding="utf-8").lower()
    assert '"decision"' not in text and '"pass"' not in text and '"fail"' not in text
    heads = load_active_heads(run["heads"])
    for rec in lines:
        assert rec["mode"] == "shadow" and set(rec["scores"]) == set(heads)
        for s in rec["scores"].values():
            assert 0.0 <= s["prob"] <= 1.0 and isinstance(s["over_threshold"], bool)
    by_id = {r["id"]: r for r in lines[:3]}
    assert by_id["img_105"]["scores"]["structure_break"]["n_crops"] == 4
    # shadow scores agree with the head applied to cached exam embeddings
    probs = np.array([r["scores"]["lighting"]["prob"] for r in lines[3:]])
    lit = np.array([i % 4 == 1 for i in range(120)])
    assert probs[lit].min() > probs[~lit].max()
