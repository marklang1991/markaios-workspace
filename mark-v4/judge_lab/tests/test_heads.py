import numpy as np
import pytest

from judge_lab.backbones import BackboneSpec
from judge_lab.data import load_config, load_exam, load_log, split_exam
from judge_lab.embeddings import ensure_embeddings
from judge_lab.heads import (
    build_item_data, evaluate_at, fit_linear, grouped_folds, image_max, oof_scores, predict_proba,
    threshold_for_recall,
)
from judge_lab.util import ExamLeakError


@pytest.fixture(scope="module")
def tile_data(ds, tmp_path_factory):
    items, _ = load_config()
    reviews = load_log(ds["log"])
    exam = load_exam(ds["exam"])
    train, _, _ = split_exam(reviews, exam)
    caches = ensure_embeddings(BackboneSpec.create("fake"), ["tile"], [(r.id, r.image) for r in reviews],
                               tmp_path_factory.mktemp("cache"), ds["crops"])
    return items["structure_break"], train, reviews, exam, caches


def test_grouped_folds_never_split_an_image(tile_data):
    it, train, _, exam, caches = tile_data
    d = build_item_data(it, train, caches, exclude_ids=exam)
    assert len(d.y) > d.n_pos + d.n_neg  # several crops per image
    for seed in range(3):
        folds = grouped_folds(d.groups, d.y, 5, seed)
        for g in np.unique(d.groups):
            assert len(set(folds[d.groups == g])) == 1
        for f in np.unique(folds):
            assert not set(d.groups[folds == f]) & set(d.groups[folds != f])
            img_labels = {g: y for g, y in zip(d.groups[folds == f], d.y[folds == f])}
            assert 0 < sum(img_labels.values()) < len(img_labels)  # stratified: both classes per fold


def test_exam_ids_cannot_reach_fitting(tile_data):
    it, _, reviews, exam, caches = tile_data
    with pytest.raises(ExamLeakError):
        build_item_data(it, reviews, caches, exclude_ids=exam)


def test_multi_instance_labels_and_max(tile_data):
    it, train, _, exam, caches = tile_data
    d = build_item_data(it, train, caches, exclude_ids=exam)
    # every crop of an image carries the image label
    for g in np.unique(d.groups):
        assert len(set(d.y[d.groups == g])) == 1
    agg = image_max(np.array([0.1, 0.9, 0.3]), np.array(["a", "a", "b"]), np.array(["t0", "t1", "t0"]))
    assert agg == {"a": (0.9, "t1"), "b": (0.3, "t0")}


def test_folded_scaler_matches_sklearn(tile_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    it, train, _, exam, caches = tile_data
    d = build_item_data(it, train, caches, exclude_ids=exam)
    coef, b = fit_linear(d.X, d.y, 0.1)
    ref = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    ref.fit(d.X, d.y)
    assert np.allclose(predict_proba(d.X, coef, b), ref.predict_proba(d.X)[:, 1], atol=1e-4)


def test_oof_threshold_hits_target_recall(tile_data):
    it, train, _, exam, caches = tile_data
    d = build_item_data(it, train, caches, exclude_ids=exam)
    ids, y, s = oof_scores(d, 5, 0.1)
    assert len(ids) == d.n_pos + d.n_neg
    for target in (0.8, 0.9, 0.95, 1.0):
        t = threshold_for_recall(y, s, target)
        assert t["recall"] >= target
        assert t["recall"] == evaluate_at(y, s, t["threshold"])["recall"]


def test_threshold_is_recall_feasible_and_min_false_reject():
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = rng.integers(0, 2, 60)
        y[:2] = [0, 1]
        s = rng.random(60) + 0.4 * y
        target = rng.choice([0.8, 0.95])
        t = threshold_for_recall(y, s, target)
        assert t["recall"] >= target
        feasible = [evaluate_at(y, s, c)["false_reject_rate"] for c in np.unique(s)
                    if evaluate_at(y, s, c)["recall"] >= target]
        assert t["false_reject_rate"] <= min(feasible) + 1e-12
    # perfectly separated: the line should catch every positive, not stop at the target
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])
    t = threshold_for_recall(y, s, 0.8)
    assert t["recall"] == 1.0 and t["false_reject_rate"] == 0.0 and 0.3 < t["threshold"] < 0.6
