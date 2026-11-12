import json

import pytest

from judge_lab.data import FieldMap, HeadItem, Review, label_counts, label_for, load_config, load_exam, load_log, \
    split_exam
from judge_lab.util import JudgeLabError


def item(iid="lighting", kind="score", trainable=True, max_score=None):
    rule = {"positive_if": "selected"}
    if max_score is not None:
        rule["max_score"] = max_score
    return HeadItem(iid, iid, kind, "full", None, trainable, rule, 0.8)


def rev(decision, reasons=(), scores=None):
    return Review("x", None, decision, list(reasons), scores or {})


def test_label_rules_default():
    it = item()
    assert label_for(rev("pass"), it) == 0
    assert label_for(rev("pass", ["lighting"]), it) == 0  # pass wins
    assert label_for(rev("fail", ["structure_break"]), it) == 0  # not selected -> OK
    assert label_for(rev("fail", ["lighting"]), it) == 1
    assert label_for(rev("fail", ["lighting"], {"lighting": 9}), it) == 1  # no max_score -> selected = positive


def test_label_rules_score_threshold():
    it = item(max_score=4)
    assert label_for(rev("fail", ["lighting"], {"lighting": 3}), it) == 1
    assert label_for(rev("fail", ["lighting"], {"lighting": 4}), it) == 1
    assert label_for(rev("fail", ["lighting"], {"lighting": 7}), it) is None  # ambiguous, excluded
    assert label_for(rev("fail", ["lighting"]), it) == 1  # missing score -> positive by default
    it.label_rule["missing_score"] = "skip"
    assert label_for(rev("fail", ["lighting"]), it) is None
    hard = item("structure_break", kind="hard", max_score=4)  # max_score ignored for hard gates
    assert label_for(rev("fail", ["structure_break"]), hard) == 1


def test_default_config_has_21_items():
    items, defaults = load_config()
    assert len(items) == 21
    assert defaults["target_recall"] == {"hard": 0.95, "score": 0.8}
    assert sum(it.kind == "hard" for it in items.values()) == 9
    not_trainable = {k for k, it in items.items() if not it.trainable}
    assert not_trainable == {"age_too_young", "age_too_old", "action_vs_storyboard", "physics"}
    assert items["structure_break"].name == "结构断裂"
    assert items["limb_deform_or_clip"].fallback_region == "full"
    assert items["structure_break"].target_recall == 0.95 and items["lighting"].target_recall == 0.8


def test_log_and_exam(ds):
    reviews = load_log(ds["log"])
    assert len(reviews) == 120
    r0 = {r.id: r for r in reviews}["img_000"]
    assert r0.image.is_file() and r0.decision == "fail" and "structure_break" in r0.reasons
    assert r0.get("versions.lora") == "mark_taste_v1"
    exam = load_exam(ds["exam"])
    assert len(exam) == 20 and "img_100" in exam  # comment line ignored
    train, ex, summary = split_exam(reviews, exam)
    assert summary == {"exam_listed": 20, "exam_in_log": 20, "exam_not_in_log": 0, "train_pool": 100,
                       "total_reviews": 120}
    assert not {r.id for r in train} & exam and {r.id for r in ex} == exam


def test_label_counts_include_non_trainable(ds):
    items, _ = load_config()
    reviews = load_log(ds["log"])
    c = label_counts(reviews, items["age_too_young"])
    assert c["pos"] == 5 and c["pos"] + c["neg"] == 120  # i % 25 == 2 -> 2,27,52,77,102


def test_field_map(tmp_path):
    rec = {"image_id": "a1", "file": "pics/a1.png", "verdict": "不通过", "fail_items": "lighting,structure_break",
           "item_scores": {"lighting": "3"}, "meta": {"versions": {"lora": "v2"}, "scene": "夜景"}}
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps(rec, ensure_ascii=False) + "\n\n{bad json\n", encoding="utf-8")
    fm_path = tmp_path / "map.json"
    fm_path.write_text(json.dumps({
        "id": "image_id", "image": "file", "decision": "verdict", "reasons": "fail_items",
        "scores": "item_scores", "versions": "meta.versions", "scene": "meta.scene",
        "decision_values": {"pass": ["通过"], "fail": ["不通过"]}}, ensure_ascii=False), encoding="utf-8")
    (r,) = load_log(log, FieldMap.load(fm_path))
    assert r.id == "a1" and r.decision == "fail"
    assert r.reasons == ["lighting", "structure_break"] and r.scores == {"lighting": 3.0}
    assert r.image == tmp_path / "pics" / "a1.png"
    assert r.get("versions.lora") == "v2" and r.scene == "夜景"
    fm_path.write_text(json.dumps({"bogus": "x"}), encoding="utf-8")
    with pytest.raises(JudgeLabError):
        FieldMap.load(fm_path)
