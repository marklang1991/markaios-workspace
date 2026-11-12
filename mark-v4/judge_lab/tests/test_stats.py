import json

import pytest

from judge_lab.cli import main
from judge_lab.data import Review
from judge_lab.stats import ab_report, newcombe_diff, sample_size, two_prop_ztest, wilson


def test_wilson_known_value():
    lo, hi = wilson(20, 47)
    assert lo == pytest.approx(0.295, abs=1e-3) and hi == pytest.approx(0.567, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == pytest.approx(1.0)


def test_newcombe_known_value():
    # Newcombe (1998) example: 56/70 vs 48/80 -> diff 0.2, CI [0.0524, 0.3339]
    d, lo, hi = newcombe_diff(56, 70, 48, 80)
    assert d == pytest.approx(0.2) and lo == pytest.approx(0.0524, abs=5e-4) and hi == pytest.approx(0.3339, abs=5e-4)


def test_ztest():
    z, p = two_prop_ztest(56, 70, 48, 80)
    assert z == pytest.approx(2.650, abs=1e-3) and p == pytest.approx(0.00805, abs=1e-4)  # matches scipy
    assert two_prop_ztest(5, 10, 5, 10)[1] == pytest.approx(1.0)


def test_sample_size():
    assert 91 <= sample_size(0.3, 0.5, 0.8, 0.05) <= 93
    assert sample_size(0.5, 0.3) == sample_size(0.3, 0.5)
    assert sample_size(0.3, 0.5, power=0.9) > sample_size(0.3, 0.5, power=0.8)


def _reviews(spec):
    out, n = [], 0
    for lora, scene, passes, total in spec:
        for i in range(total):
            out.append(Review(f"r{n}", None, "pass" if i < passes else "fail", [], {},
                              versions={"lora": lora}, scene=scene))
            n += 1
    return out


def test_ab_report_and_regression():
    reviews = _reviews([("v1", "indoor", 30, 50), ("v1", "night", 30, 50),
                        ("v2", "indoor", 40, 50), ("v2", "night", 20, 50)])
    res = ab_report(reviews, "versions.lora", "scene", tolerance=0.05)
    assert res["baseline"] == "v1"  # first seen
    assert res["groups"]["v1"]["rate"] == pytest.approx(0.6) and res["groups"]["v2"]["rate"] == pytest.approx(0.6)
    assert res["comparisons"]["v2"]["diff"] == pytest.approx(0.0)
    assert res["regression_flag"] and [r["scene"] for r in res["regressions"]] == ["night"]
    assert res["scenes"]["indoor"]["vs_baseline"]["v2"]["regression"] is False
    ok = ab_report(_reviews([("v1", "a", 10, 20), ("v2", "a", 12, 20)]), "versions.lora", "scene")
    assert not ok["regression_flag"]


def test_stats_cli(ds, tmp_path, capsys):
    out = tmp_path / "ab.md"
    assert main(["stats", "ab", "--log", str(ds["log"]), "--group-by", "versions.lora", "--scene-field", "scene",
                 "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(res["groups"]) == {"mark_taste_v1", "mark_taste_v2"} and res["groups"]["mark_taste_v1"]["n"] == 60
    assert "Wilson" in out.read_text(encoding="utf-8")
    assert main(["stats", "n", "--p1", "0.3", "--p2", "0.5", "--power", "0.8", "--alpha", "0.05"]) == 0
    assert "每组至少 93 张" in capsys.readouterr().out
