"""Per-defect linear heads: datasets, grouped CV, fitting, thresholds, head files.

Multiple-instance simplification: an image with several crops (tiles, faces, hands)
contributes ALL of its crops with the image-level label, for negatives and positives
alike. At prediction time the image score is the max over its crops. Positive images
therefore add some noisy positive crops (a defect usually sits in one tile only); this
is a deliberate, documented trade-off for simplicity.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import HeadItem, Review, label_for
from .embeddings import RegionCache
from .util import JudgeLabError, assert_no_exam, now_iso, read_json, sha256_ids, write_json


# --------------------------------------------------------------------------- datasets
@dataclass
class ItemData:
    X: np.ndarray  # (n_crops, D)
    y: np.ndarray  # (n_crops,) image label repeated per crop
    groups: np.ndarray  # (n_crops,) image id per crop
    crop_names: np.ndarray
    n_pos: int  # images
    n_neg: int
    n_ambiguous: int
    n_no_crops: int

    @property
    def image_ids(self) -> list[str]:
        return sorted(set(self.groups.tolist()))


def region_rows(caches: dict[str, RegionCache], region: str, fallback: str | None,
                image_id: str) -> tuple[RegionCache, np.ndarray]:
    """Crops of `region` for this image, or of `fallback` when the region has none."""
    cache = caches[region]
    rows = cache.rows(image_id)
    if len(rows) == 0 and fallback:
        cache = caches[fallback]
        rows = cache.rows(image_id)
    return cache, rows


def needed_regions(items) -> set[str]:
    return {r for it in items for r in (it.region, it.fallback_region) if r}


def build_item_data(item: HeadItem, reviews: list[Review], caches: dict[str, RegionCache],
                    exclude_ids: set[str] | None = None) -> ItemData:
    if exclude_ids:
        assert_no_exam((r.id for r in reviews), exclude_ids, f"build_item_data({item.id})")
    Xs, ys, gs, ns = [], [], [], []
    pos = neg = amb = none = 0
    for r in reviews:
        lab = label_for(r, item)
        if lab is None:
            amb += 1
            continue
        cache, rows = region_rows(caches, item.region, item.fallback_region, r.id)
        if len(rows) == 0:
            none += 1
            continue
        pos += lab
        neg += 1 - lab
        Xs.append(cache.emb[rows])
        ys += [lab] * len(rows)
        gs += [r.id] * len(rows)
        ns += cache.crop_names[rows].tolist()
    X = np.concatenate(Xs) if Xs else np.zeros((0, 0), np.float32)
    return ItemData(X, np.array(ys, int), np.array(gs, str), np.array(ns, str), pos, neg, amb, none)


# --------------------------------------------------------------------------- model
def fit_linear(X: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, float]:
    """Standardise + balanced logistic regression; the scaler is folded into (coef, intercept)
    so a head is just p = sigmoid(x @ coef + intercept) on the L2-normalised embedding."""
    from sklearn.linear_model import LogisticRegression

    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-6)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit((X - mu) / sd, y)
    w = clf.coef_[0] / sd
    b = float(clf.intercept_[0] - np.dot(w, mu))
    return w.astype(np.float32), b


def predict_proba(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    z = np.clip(X.astype(np.float64) @ coef.astype(np.float64) + intercept, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def image_max(scores: np.ndarray, groups: np.ndarray, names: np.ndarray | None = None) -> dict[str, tuple[float, str]]:
    """Image-level score = max over its crops; also returns the arg-max crop name."""
    out: dict[str, tuple[float, str]] = {}
    for i, (s, g) in enumerate(zip(scores.tolist(), groups.tolist())):
        if g not in out or s > out[g][0]:
            out[g] = (s, "" if names is None else str(names[i]))
    return out


def grouped_folds(groups: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Stratified k-fold over IMAGES, expanded to crops: crops of one image never split."""
    from sklearn.model_selection import StratifiedKFold

    uniq, first = np.unique(groups, return_index=True)
    img_y = y[first]
    k = min(k, int(img_y.sum()), int((1 - img_y).sum()))
    if k < 2:
        raise JudgeLabError("交叉验证需要每类至少 2 张图")
    img_fold = np.empty(len(uniq), int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(uniq)), img_y)):
        img_fold[te] = f
    lookup = dict(zip(uniq.tolist(), img_fold.tolist()))
    return np.array([lookup[g] for g in groups.tolist()], int)


def oof_scores(d: ItemData, k: int, C: float, seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Out-of-fold image-level scores. Returns (image_ids, labels, scores)."""
    folds = grouped_folds(d.groups, d.y, k, seed)
    crop_scores = np.zeros(len(d.y))
    for f in np.unique(folds):
        te = folds == f
        coef, b = fit_linear(d.X[~te], d.y[~te], C)
        crop_scores[te] = predict_proba(d.X[te], coef, b)
    agg = image_max(crop_scores, d.groups)
    labels = dict(zip(d.groups.tolist(), d.y.tolist()))
    ids = sorted(agg)
    return ids, np.array([labels[i] for i in ids]), np.array([agg[i][0] for i in ids])


# --------------------------------------------------------------------------- metrics
def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def evaluate_at(labels: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    flag = scores >= thr
    pos, neg = labels == 1, labels == 0
    return {
        "threshold": float(thr),
        "recall": float(flag[pos].mean()) if pos.any() else None,
        "false_reject_rate": float(flag[neg].mean()) if neg.any() else None,
        "agreement": float((flag == pos).mean()) if len(labels) else None,
        "n_pos": int(pos.sum()),
        "n_neg": int(neg.sum()),
    }


def threshold_for_recall(labels: np.ndarray, scores: np.ndarray, target: float) -> dict:
    """Threshold with recall >= target and the lowest false-reject rate.

    t0 = the highest threshold reaching the target (minimises false rejects). Below t0 the
    line can still drop, at no extra false-reject cost, until it meets the next negative;
    we place it midway in that gap so it also catches as many extra positives as possible."""
    pos = np.sort(scores[labels == 1])[::-1]
    if len(pos) == 0:
        raise JudgeLabError("没有正样本，无法按召回率定阈值")
    need = max(1, int(np.ceil(target * len(pos) - 1e-9)))
    t0 = float(pos[need - 1])
    neg_below = scores[(labels == 0) & (scores < t0)]
    if len(neg_below):
        m = float(neg_below.max())
        t = (m + float(pos[pos > m].min())) / 2
    else:
        t = float(pos.min())
    res = evaluate_at(labels, scores, t)
    res["target_recall"] = target
    return res


# --------------------------------------------------------------------------- head files
@dataclass
class Head:
    item: str
    version: int
    coef: np.ndarray
    intercept: float
    meta: dict
    train_ids: list[str]
    path: Path

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_proba(X, self.coef, self.intercept)


def head_versions(heads_dir: Path, item_id: str) -> list[int]:
    pat = re.compile(rf"^{re.escape(item_id)}_v(\d+)\.npz$")
    d = Path(heads_dir)
    return sorted(int(m.group(1)) for p in d.glob("*.npz") if (m := pat.match(p.name))) if d.is_dir() else []


def save_head(heads_dir: Path, item: HeadItem, coef, intercept, train_ids: list[str], meta: dict) -> Head:
    import json

    heads_dir = Path(heads_dir)
    heads_dir.mkdir(parents=True, exist_ok=True)
    version = max(head_versions(heads_dir, item.id), default=0) + 1
    meta = {
        **meta, "item": item.id, "name": item.name, "kind": item.kind, "region": item.region,
        "fallback_region": item.fallback_region, "version": version,
        "data_hash": sha256_ids(train_ids), "created_at": now_iso(),
    }
    path = heads_dir / f"{item.id}_v{version}.npz"
    with open(path, "wb") as fh:
        np.savez(fh, coef=np.asarray(coef, np.float32), intercept=np.array(float(intercept)),
                 train_ids=np.array(sorted(train_ids), str), meta=np.array(json.dumps(meta, ensure_ascii=False)))
    manifest = load_manifest(heads_dir)
    entry = {"file": path.name, **{k: meta[k] for k in (
        "version", "backbone", "model_id", "region", "fallback_region", "kind", "name",
        "n_pos", "n_neg", "data_hash", "created_at")}}
    rec = manifest["items"].setdefault(item.id, {"active": None, "history": []})
    rec["active"] = entry
    rec["history"].append(entry)
    manifest["updated_at"] = now_iso()
    write_json(heads_dir / "manifest.json", manifest)
    return Head(item.id, version, np.asarray(coef, np.float32), float(intercept), meta, sorted(train_ids), path)


def load_head(path: Path) -> Head:
    import json

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return Head(meta["item"], int(meta["version"]), z["coef"].astype(np.float32), float(z["intercept"]),
                    meta, z["train_ids"].astype(str).tolist(), Path(path))


def load_manifest(heads_dir: Path) -> dict:
    p = Path(heads_dir) / "manifest.json"
    return read_json(p) if p.is_file() else {"items": {}, "updated_at": None}


def load_active_heads(heads_dir: Path, only: set[str] | None = None) -> dict[str, Head]:
    manifest = load_manifest(heads_dir)
    heads = {}
    for item_id, rec in manifest["items"].items():
        if rec.get("active") and (not only or item_id in only):
            heads[item_id] = load_head(Path(heads_dir) / rec["active"]["file"])
    if not heads:
        raise JudgeLabError(f"{heads_dir} 里没有可用的判官头（先运行 train）")
    return heads


def load_thresholds(path: Path) -> dict:
    return read_json(path) if Path(path).is_file() else {"items": {}}


def threshold_for_head(thresholds: dict, head: Head) -> float | None:
    """The active line for this head (None if not calibrated); warns on a head-version mismatch."""
    e = thresholds.get("items", {}).get(head.item)
    if not e or e.get("threshold") is None:
        return None
    if e.get("head_version") not in (None, head.version):
        print(f"[警告] {head.item}: 阈值是为 v{e['head_version']} 校准的，当前头是 v{head.version}，请重新 calibrate")
    return float(e["threshold"])
