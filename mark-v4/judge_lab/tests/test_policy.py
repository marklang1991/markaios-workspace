"""The labelling policy: what one review teaches one reason, and with what weight (data.assess).

Field feedback behind it: the owner usually clicks only the most obvious reason, a whole screen can be passed in
one go, and a reason can be carried over from the previous picture without the owner ever touching it."""
import json

import numpy as np
import pytest

from judge_lab.cli import main
from judge_lab.data import (
    POLICY_DEFAULTS, HeadItem, Review, apply_policy, assess, canonicalize, explain_label, label_for, load_config,
    load_logs, policy_of, severity_weight, why_code,
)
from judge_lab.embeddings import RegionCache
from judge_lab.heads import build_item_data
from judge_lab.util import JudgeLabError


def item(iid="lighting", kind="score", **rule):
    return HeadItem(iid, iid, kind, "full", None, True, {"positive_if": "selected", **rule}, 0.8)


def rev(decision, reasons=(), scores=None, **kw):
    return Review("x", None, decision, list(reasons), scores or {}, **kw)


def lw(review, it):
    """(label, weight) from assess."""
    label, _, weight = assess(review, it)
    return label, weight


def test_defaults():
    assert POLICY_DEFAULTS == {"unselected_fail_weight": 0.0, "batch_pass_weight": 0.5, "carried_weight": 0.0,
                               "carried_unknown_weight": 0.5, "severity_weight": True}
    # an item built by hand, without any knob in its label_rule, uses the defaults
    assert policy_of(item()) == POLICY_DEFAULTS


def test_pass_and_batch_pass():
    it = item("structure_break", "hard")
    assert lw(rev("pass"), it) == (0, 1.0)
    assert lw(rev("pass", batch_pass=True), it) == (0, 0.5)  # 整屏通过: a negative, at half weight
    assert why_code(rev("pass", batch_pass=True), it) == "batch"
    # 推荐进入 is never down-weighted for having been batch-passed
    assert lw(rev("pass", batch_pass=True, recommend=True), it) == (0, 1.0)
    off = item("structure_break", "hard", batch_pass_weight=0)
    assert lw(rev("pass", batch_pass=True), off) == (None, 0.0)  # 0 = excluded
    assert lw(rev("pass", batch_pass=True, recommend=True), off) == (0, 1.0)
    assert lw(rev("pass"), off) == (0, 1.0)  # an ordinary pass is untouched by the knob
    assert lw(rev("pass", batch_pass=True), item("structure_break", "hard", batch_pass_weight=1)) == (0, 1.0)


def test_unselected_reason_on_a_failed_picture_is_unknown_by_default():
    it = item("structure_break", "hard")
    other = rev("fail", ["lighting"])
    assert lw(other, it) == (None, 0.0) and why_code(other, it) == "unknown"
    assert "不知道" in explain_label(other, it)[1]
    # the record says the reasons are complete -> a full negative, whatever the knob says
    done = rev("fail", ["lighting"], reasons_complete=True)
    assert lw(done, it) == (0, 1.0) and why_code(done, it) == "complete"
    assert lw(done, item("structure_break", "hard", unselected_fail_weight=0.3)) == (0, 1.0)
    # the knob: in between = a negative with that fitting weight; 1 = a full negative
    assert lw(other, item("structure_break", "hard", unselected_fail_weight=0.3)) == (0, 0.3)
    assert lw(other, item("structure_break", "hard", unselected_fail_weight=1)) == (0, 1.0)
    # failed with no reason at all: unknown for every item
    assert lw(rev("fail"), it) == (None, 0.0)


def test_carried_reasons():
    hard = item("structure_break", "hard")
    chosen = rev("fail", ["structure_break"], carried={"structure_break": False})
    carried = rev("fail", ["structure_break"], carried={"structure_break": True})
    unknown = rev("fail", ["structure_break"])  # old record: no flag
    assert lw(chosen, hard) == (1, 1.0)  # the owner chose it
    assert lw(carried, hard) == (None, 0.0) and why_code(carried, hard) == "carried_off"  # never touched: excluded
    assert lw(unknown, hard) == (1, 0.5)  # still a positive, at half weight
    assert lw(carried, item("structure_break", "hard", carried_weight=0.4)) == (1, 0.4)
    assert lw(unknown, item("structure_break", "hard", carried_unknown_weight=1)) == (1, 1.0)
    assert lw(unknown, item("structure_break", "hard", carried_unknown_weight=0)) == (None, 0.0)
    assert lw(chosen, item("structure_break", "hard", carried_weight=0, carried_unknown_weight=0)) == (1, 1.0)


def test_positive_weight_is_carried_factor_times_severity():
    it = item()  # scoring item: 0 -> 1.5, 5 -> 1.0, 10 -> 0.5
    for score, sev in ((0, 1.5), (5, 1.0), (10, 0.5), (3, 1.2)):
        chosen = rev("fail", ["lighting"], {"lighting": score}, carried={"lighting": False})
        unknown = rev("fail", ["lighting"], {"lighting": score})
        assert severity_weight(chosen, it) == pytest.approx(sev)
        assert lw(chosen, it) == (1, pytest.approx(sev))
        assert lw(unknown, it) == (1, pytest.approx(0.5 * sev))
    half = item(carried_weight=0.5)
    assert lw(rev("fail", ["lighting"], {"lighting": 0}, carried={"lighting": True}), half) == (1, pytest.approx(0.75))
    flat = item(severity_weight=False)
    assert lw(rev("fail", ["lighting"], {"lighting": 0}, carried={"lighting": False}), flat) == (1, 1.0)
    # hard gates have no severity
    assert lw(rev("fail", ["structure_break"], {"structure_break": 0}, carried={"structure_break": False}),
              item("structure_break", "hard")) == (1, 1.0)


def test_order_of_the_checks():
    late = item("layer_seam", "hard")
    late.added_at = "2026-11-13T10:00:00+00:00"
    early = dict(reviewed_at="2026-11-12T10:00:00+00:00")
    # 1. reviewed before the reason existed: excluded, whatever else the record says
    for r in (rev("pass", **early), rev("pass", batch_pass=True, recommend=True, **early),
              rev("fail", ["lighting"], reasons_complete=True, **early),
              rev("fail", ["layer_seam"], carried={"layer_seam": False}, **early), rev("pass")):
        assert lw(r, late) == (None, 0.0) and why_code(r, late) == "time"
    # 2. a pass wins over reasons left on the record
    assert lw(rev("pass", ["lighting"], carried={"lighting": True}), item()) == (0, 1.0)
    # 4. the max_score rule comes before the carried factor
    strict = item(max_score=4)
    light = rev("fail", ["lighting"], {"lighting": 7}, carried={"lighting": False})
    assert lw(light, strict) == (None, 0.0) and why_code(light, strict) == "light"
    assert lw(rev("fail", ["lighting"], {"lighting": 3}, carried={"lighting": False}), strict) == (1, pytest.approx(1.2))
    assert lw(rev("fail", ["lighting"], {"lighting": 3}, carried={"lighting": True}), strict) == (None, 0.0)
    assert lw(rev("fail", ["lighting"], carried={"lighting": False}), strict) == (1, 1.0)  # no score: positive
    strict.label_rule["missing_score"] = "skip"
    assert lw(rev("fail", ["lighting"], carried={"lighting": False}), strict) == (None, 0.0)


def test_assess_is_the_single_source_of_truth():
    its = [item(), item("structure_break", "hard", unselected_fail_weight=0.3), item(max_score=4)]
    reviews = [rev("pass"), rev("pass", batch_pass=True), rev("fail", ["lighting"], {"lighting": 2}),
               rev("fail", ["structure_break"], carried={"structure_break": True}),
               rev("fail", ["lighting"], {"lighting": 9}, carried={"lighting": False}, reasons_complete=True)]
    for it in its:
        for r in reviews:
            label, why, weight = assess(r, it)
            assert explain_label(r, it) == (label, why) and label_for(r, it) == label
            assert (weight == 0.0) == (label is None) and why


def test_new_record_fields_are_read(tmp_path):
    rows = [
        {"id": "aaaaaaaaaaaaaaaa", "sha256": "A" * 64, "image": "x/hallway_s1.png", "decision": "不过",
         "reasons": [{"item": "光线", "kind": "score", "score": 3, "carried": True},
                     {"item": "结构断裂", "kind": "hard", "score": None, "carried": False},
                     {"item": "旧原因", "kind": "hard", "score": None}],
         "reasons_complete": True, "source": "旧图补审", "recommend": True, "note": "背景的门是歪的", "seconds": 12.5},
        {"image": "x/hallway_s2.png", "decision": "过", "note": "整屏通过", "seconds": None},  # old batch pass
        {"image": "x/hallway_s3.png", "decision": "过", "batch_pass": True, "note": ""},
        {"image": "x/hallway_s4.png", "decision": "过", "batch_pass": False, "note": "整屏通过", "seconds": "3"},
    ]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    a, b, c, d = load_logs([log])[0]
    assert a.carried == {"光线": True, "结构断裂": False} and a.reasons == ["光线", "结构断裂", "旧原因"]
    assert a.reasons_complete and a.recommend and a.source == "旧图补审" and a.note == "背景的门是歪的"
    assert a.seconds == 12.5 and a.sha256 == "a" * 64 and a.id == "aaaaaaaaaaaaaaaa" and not a.batch_pass
    assert b.batch_pass and not b.recommend and not b.reasons_complete and b.seconds is None and b.source is None
    assert b.sha256 is None and b.id == "x__hallway_s2"
    assert c.batch_pass and not d.batch_pass  # an explicit field beats the old note
    assert d.seconds is None  # not a number -> ignored
    # reasons spelled by name are rewritten to item ids, and so are their carried flags; merged names combine
    items = {"lighting": HeadItem("lighting", "光线", "score", "full", None, True, {}, 0.8),
             "structure_break": HeadItem("structure_break", "结构断裂", "hard", "tile", None, True, {}, 0.95,
                                         aliases=("旧原因",))}
    assert canonicalize([a], items) == {}
    assert a.reasons == ["lighting", "structure_break"]
    assert a.carried == {"lighting": True, "structure_break": False}  # chosen once = chosen
    assert a.scores == {"lighting": 3.0}
    assert lw(a, items["lighting"]) == (None, 0.0) and lw(a, items["structure_break"]) == (1, 1.0)


def test_policy_resolution_defaults_config_item_cli(tmp_path, capsys):
    cfg = tmp_path / "heads.json"
    cfg.write_text(json.dumps({
        "defaults": {"label_policy": {"unselected_fail_weight": 0.3, "batch_pass_weight": 1}},
        "items": {
            "a": {"name": "甲", "kind": "hard", "region": "full"},
            "b": {"name": "乙", "kind": "hard", "region": "full",
                  "label_rule": {"positive_if": "selected", "unselected_fail_weight": 1}},
        }}, ensure_ascii=False), encoding="utf-8")
    items, defaults = load_config(cfg)
    assert defaults["label_policy"] == {"unselected_fail_weight": 0.3, "batch_pass_weight": 1.0}
    pa, pb = policy_of(items["a"]), policy_of(items["b"])
    assert pa["unselected_fail_weight"] == 0.3 and pb["unselected_fail_weight"] == 1  # item beats the file default
    assert pa["batch_pass_weight"] == pb["batch_pass_weight"] == 1.0  # the file default beats the built-in 0.5
    assert pa["carried_unknown_weight"] == 0.5  # built-in default where nothing else speaks
    other = rev("fail", ["zzz"])
    assert lw(other, items["a"]) == (0, 0.3) and lw(other, items["b"]) == (0, 1.0)
    apply_policy(items, {"unselected_fail_weight": 0.0})  # the CLI beats everything
    assert lw(other, items["a"]) == (None, 0.0) and lw(other, items["b"]) == (None, 0.0)
    assert policy_of(items["a"])["batch_pass_weight"] == 1.0  # knobs not given on the CLI stay
    with pytest.raises(JudgeLabError):
        apply_policy(items, {"carried_weight": 1.5})
    # a typo in the file is an error, not a silently ignored knob
    cfg.write_text(json.dumps({"defaults": {"label_policy": {"unselected_weight": 1}}, "items": {}}), encoding="utf-8")
    with pytest.raises(JudgeLabError):
        load_config(cfg)
    # the review page's own list carries no file-level policy; an item's label_rule still works
    page = tmp_path / "缺陷清单.json"
    page.write_text(json.dumps({"defaults": {"label_policy": {"unselected_fail_weight": 1}}, "items": [
        {"name": "甲", "group": "结构", "kind": "hard"},
        {"name": "乙", "group": "结构", "kind": "hard", "label_rule": {"carried_weight": 1}}]}, ensure_ascii=False),
        encoding="utf-8")
    items, defaults = load_config(page)
    assert "label_policy" not in defaults
    assert policy_of(items["甲"]) == POLICY_DEFAULTS and policy_of(items["乙"])["carried_weight"] == 1.0


def test_cli_flag_overrides_the_policy(tmp_path, capsys):
    cfg = tmp_path / "heads.json"
    cfg.write_text(json.dumps({"items": {
        "a": {"name": "甲", "kind": "hard", "region": "full"},
        "b": {"name": "乙", "kind": "hard", "region": "full",
              "label_rule": {"positive_if": "selected", "unselected_fail_weight": 1}}}}, ensure_ascii=False),
        encoding="utf-8")
    rows = [{"id": "p1", "decision": "pass"}, {"id": "p2", "decision": "pass", "batch_pass": True},
            {"id": "f1", "decision": "fail", "reasons": [{"item": "a", "carried": False}]},
            {"id": "f2", "decision": "fail", "reasons": [{"item": "a", "carried": True}]},
            {"id": "f3", "decision": "fail", "reasons": ["b"]}]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def counts(*flags):
        capsys.readouterr()
        assert main(["items", "list", "--log", str(log), "--config", str(cfg), *flags]) == 0
        out = capsys.readouterr().out
        return {l.split("|")[1].strip(): [c.strip() for c in l.strip("|").split("|")][5:8]
                for l in out.splitlines() if l.startswith(("| a ", "| b "))}

    # default: a = f1 positive; f2 carried -> out; f3 unknown; passes negative.  b's own rule: f1, f2 are negatives
    assert counts() == {"a": ["1", "2", "2"], "b": ["1", "4", "0"]}
    assert counts("--unselected-fail-weight", "0") == {"a": ["1", "2", "2"], "b": ["1", "2", "2"]}  # beats b's rule
    assert counts("--unselected-fail-weight", "0.5", "--carried-weight", "1") == {"a": ["2", "3", "0"],
                                                                                 "b": ["1", "4", "0"]}
    assert counts("--batch-pass-weight", "0") == {"a": ["1", "1", "3"], "b": ["1", "3", "1"]}
    assert counts("--carried-unknown-weight", "0") == {"a": ["1", "2", "2"], "b": ["0", "4", "1"]}
    assert main(["items", "list", "--log", str(log), "--config", str(cfg), "--carried-weight", "2"]) == 2


def test_fitting_weights_reach_the_dataset():
    """build_item_data takes label AND weight from assess: negatives carry weights too."""
    rng = np.random.default_rng(0)
    reviews = [
        Review("pos_chosen", None, "fail", ["lighting"], {"lighting": 0}, carried={"lighting": False}),
        Review("pos_old", None, "fail", ["lighting"], {"lighting": 5}),
        Review("pos_carried", None, "fail", ["lighting"], {"lighting": 5}, carried={"lighting": True}),
        Review("neg_pass", None, "pass", [], {}),
        Review("neg_batch", None, "pass", [], {}, batch_pass=True),
        Review("neg_batch_recommended", None, "pass", [], {}, batch_pass=True, recommend=True),
        Review("fail_other", None, "fail", ["zzz"], {}),
        Review("fail_other_complete", None, "fail", ["zzz"], {}, reasons_complete=True),
    ]
    ids = np.array([r.id for r in reviews for _ in range(2)], str)  # two crops per picture
    cache = RegionCache(rng.standard_normal((len(ids), 8)).astype(np.float32), ids,
                        np.array(["c0", "c1"] * len(reviews), str), set(ids.tolist()), {})
    d = build_item_data(item(), reviews, {"full": cache})
    w = dict(zip(d.groups.tolist(), d.w.tolist()))
    assert w == {"pos_chosen": 1.5, "pos_old": 0.5, "neg_pass": 1.0, "neg_batch": 0.5,
                 "neg_batch_recommended": 1.0, "fail_other_complete": 1.0}
    assert (d.n_pos, d.n_neg, d.n_ambiguous) == (2, 4, 2) and len(d.y) == 12
    d = build_item_data(item(unselected_fail_weight=0.3, carried_weight=1), reviews, {"full": cache})
    w = dict(zip(d.groups.tolist(), d.w.tolist()))
    assert w["fail_other"] == 0.3 and w["pos_carried"] == 1.0 and (d.n_pos, d.n_neg, d.n_ambiguous) == (3, 5, 0)
