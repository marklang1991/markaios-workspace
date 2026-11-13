"""Per-defect linear heads: datasets, grouped CV (by scene when there are scenes), fitting, thresholds, head files.

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

from .data import HeadItem, Review, assess
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
    w: np.ndarray | None = None  # (n_crops,) fitting weight from data.assess (positives and negatives)
    scenes: np.ndarray | None = None  # (n_crops,) scene of the crop's image, "" when it has none

    @property
    def image_ids(self) -> list[str]:
        return sorted(set(self.groups.tolist()))

    def images(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Per image (sorted by id): ids, labels, scenes ("" = none)."""
        uniq, first = np.unique(self.groups, return_index=True)
        scenes = [""] * len(uniq) if self.scenes is None else self.scenes[first].tolist()
        return uniq, self.y[first], scenes


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
                    exclude_ids: set[str] | None = None, labelled: bool = True) -> ItemData:
    """One item's crops with the label and fitting weight data.assess gives each review.
    labelled=False: every review that has crops, all labelled 0 (pictures to be scored, not fitted)."""
    if exclude_ids:
        assert_no_exam((r.id for r in reviews), exclude_ids, f"build_item_data({item.id})",
                       hide={r.id for r in reviews if r.private})
    Xs, ys, gs, ns, ws, ss = [], [], [], [], [], []
    pos = neg = amb = none = 0
    for r in reviews:
        lab, _, weight = assess(r, item) if labelled else (0, "", 1.0)
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
        ws += [weight] * len(rows)
        ss += [r.scene or ""] * len(rows)
        ns += cache.crop_names[rows].tolist()
    X = np.concatenate(Xs) if Xs else np.zeros((0, 0), np.float32)
    return ItemData(X, np.array(ys, int), np.array(gs, str), np.array(ns, str), pos, neg, amb, none,
                    np.array(ws, float), np.array(ss, str))


# --------------------------------------------------------------------------- model
def fit_linear(X: np.ndarray, y: np.ndarray, C: float, sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Standardise + balanced logistic regression; the scaler is folded into (coef, intercept)
    so a head is just p = sigmoid(x @ coef + intercept) on the L2-normalised embedding."""
    from sklearn.linear_model import LogisticRegression

    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-6)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit((X - mu) / sd, y, sample_weight=sample_weight)
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


def scene_folds(d: ItemData, k: int, seed: int = 0) -> tuple[np.ndarray, int] | None:
    """Stratified folds over SCENES, expanded to crops: every picture of a scene lands in the same fold, so a
    head is always tested on scenes it has not seen. A picture without a scene is a group of its own.
    k is lowered (down to 2) until every training split holds both classes; None if no k works."""
    from sklearn.model_selection import StratifiedGroupKFold

    uniq, img_y, scenes = d.images()
    grp = np.array([s if s else f"\x1f{g}" for s, g in zip(scenes, uniq.tolist())], str)
    for kk in range(min(k, len(set(grp.tolist()))), 1, -1):
        img_fold = np.full(len(uniq), -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            splits = list(StratifiedGroupKFold(n_splits=kk, shuffle=True, random_state=seed)
                          .split(np.zeros(len(uniq)), img_y, grp))
        for f, (tr, te) in enumerate(splits):
            if len(te) == 0 or len(set(img_y[tr].tolist())) < 2:
                break
            img_fold[te] = f
        else:
            lookup = dict(zip(uniq.tolist(), img_fold.tolist()))
            return np.array([lookup[g] for g in d.groups.tolist()], int), kk
    return None


@dataclass
class OofRun:
    ids: list[str]  # images, sorted
    y: np.ndarray  # image labels
    s: np.ndarray  # out-of-fold image scores
    mode: str  # "scene" = folds never split a scene | "image" = folds never split an image
    k: int  # folds actually used
    extra: dict[str, float] | None = None  # scores of the `extra` pictures, by image id
    n_scenes: int = 0  # distinct scenes among the images
    note: str = ""  # why scene grouping was not used although there are scenes


def oof_run(d: ItemData, k: int, C: float, seed: int = 0, by: str = "auto",
            extra: ItemData | None = None) -> OofRun:
    """Out-of-fold image-level scores.

    by: "scene" | "image" | "auto" = by scene when at least half of the images have one and there are >= 4
    distinct scenes (and such folds exist), else by image. Image grouping lets a head learn the scene instead of
    the defect whenever the pictures are scenes x seeds, so its numbers are optimistic then.
    extra: unlabelled pictures (not part of d) to score with the fold models: only the model that never saw the
    picture's scene when grouping by scene and that scene is in d, else the mean over all fold models."""
    if by not in ("auto", "scene", "image"):
        raise JudgeLabError(f"分组方式只能是 auto / scene / image，收到 {by!r}")
    uniq, _, scenes = d.images()
    n_with, distinct = sum(1 for x in scenes if x), len({x for x in scenes if x})
    folds, mode, note = None, "image", ""
    if by == "scene" and not distinct:
        raise JudgeLabError("按场景分组需要场景信息：记录里没有 scene，文件名也不是「场景_s数字」的样子")
    if by == "scene" or (by == "auto" and n_with * 2 >= len(uniq) and distinct >= 4):
        got = scene_folds(d, k, seed)
        if got is not None:
            (folds, k), mode = got, "scene"
        elif by == "scene":
            raise JudgeLabError("按场景分不出每一折都同时有正例和反例的分组（正例或反例只出现在一个场景里）")
        else:
            note = "按场景分不出每一折都同时有正例和反例的分组"
    elif by == "auto" and distinct:
        note = "场景不到 4 个，或多数图没有场景"
    if folds is None:
        folds = grouped_folds(d.groups, d.y, k, seed)
        k = len(np.unique(folds))
    crop_scores = np.zeros(len(d.y))
    held_out: dict[str, int] = {}  # scene (or image id) -> the fold that never trained on it
    if mode == "scene" and d.scenes is not None:
        held_out = {s: int(f) for s, f in zip(d.scenes.tolist(), folds.tolist()) if s}
    ex_fold = None
    if extra is not None and len(extra.y):
        ex_scenes = [""] * len(extra.y) if extra.scenes is None else extra.scenes.tolist()
        ex_fold = np.array([held_out.get(s, -1) for s in ex_scenes], int)  # -1: every fold model may score it
    ex_scores: dict[str, list[float]] = {}
    for f in np.unique(folds):
        te = folds == f
        coef, b = fit_linear(d.X[~te], d.y[~te], C, None if d.w is None else d.w[~te])
        crop_scores[te] = predict_proba(d.X[te], coef, b)
        if ex_fold is not None:
            m = (ex_fold == f) | (ex_fold == -1)
            for iid, (sc, _) in image_max(predict_proba(extra.X[m], coef, b), extra.groups[m]).items():
                ex_scores.setdefault(iid, []).append(sc)
    agg = image_max(crop_scores, d.groups)
    labels = dict(zip(d.groups.tolist(), d.y.tolist()))
    ids = sorted(agg)
    return OofRun(ids, np.array([labels[i] for i in ids]), np.array([agg[i][0] for i in ids]), mode, int(k),
                  None if extra is None else {i: float(np.mean(v)) for i, v in ex_scores.items()}, distinct, note)


def oof_scores(d: ItemData, k: int, C: float, seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Out-of-fold image-level scores, grouping chosen automatically. Returns (image_ids, labels, scores)."""
    run = oof_run(d, k, C, seed)
    return run.ids, run.y, run.s


def permuted(d: ItemData, seed: int, within_scene: bool = True) -> ItemData:
    """Same crops, image-level labels shuffled between images: the score a head reaches when labels mean nothing.
    within_scene (and the data has scenes): labels move only between pictures of the SAME scene, so every scene
    keeps its share of positives and only what tells pictures apart inside a scene is destroyed. Pictures with no
    scene, or alone in theirs, are shuffled among themselves."""
    from collections import Counter
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    uniq, img_y, scenes = d.images()
    img_y = img_y.copy()
    if within_scene and any(scenes):
        size = Counter(x for x in scenes if x)
        pool = np.array([x if x and size[x] > 1 else "" for x in scenes], str)
        for x in sorted(set(pool.tolist())):
            idx = np.flatnonzero(pool == x)
            img_y[idx] = rng.permutation(img_y[idx])
    else:
        rng.shuffle(img_y)
    lookup = dict(zip(uniq.tolist(), img_y.tolist()))
    return replace(d, y=np.array([lookup[g] for g in d.groups.tolist()], int), w=None)


def scene_only_auc(d: ItemData) -> float | None:
    """How well the scene alone predicts the label, without looking at any picture: each image is scored by the
    positive rate of the OTHER images of its scene (an image alone in its scene, or without one: the overall rate
    without it). Leave-one-out pulls this below 0.5 when the scene tells nothing, so a value near the head's own
    AUC is a real warning. None with a single class, or when fewer than two scenes exist (nothing to compare)."""
    _, y, scenes = d.images()
    y = y.astype(float)
    if len(set(y.tolist())) < 2 or len({x for x in scenes if x}) < 2:
        return None
    total: dict[str, list[float]] = {}
    for x, v in zip(scenes, y.tolist()):
        if x:
            t = total.setdefault(x, [0.0, 0.0])
            t[0] += v
            t[1] += 1
    pred = [(total[x][0] - v) / (total[x][1] - 1) if x and total[x][1] > 1 else (y.sum() - v) / (len(y) - 1)
            for x, v in zip(scenes, y.tolist())]
    return auc(y.astype(int), np.array(pred))


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


def file_stem(item_id: str) -> str:
    """File name for an item's head. Ids from the review page are Chinese names with '/', spaces, brackets:
    those get a short stable hash instead (the manifest and the head's meta keep the real id and name)."""
    if re.fullmatch(r"[A-Za-z0-9_\-]+", item_id):
        return item_id
    import hashlib
    return "h_" + hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:10]


def head_versions(heads_dir: Path, item_id: str) -> list[int]:
    pat = re.compile(rf"^{re.escape(file_stem(item_id))}_v(\d+)\.npz$")
    d = Path(heads_dir)
    return sorted(int(m.group(1)) for p in d.glob("*.npz") if (m := pat.match(p.name))) if d.is_dir() else []


def save_head(heads_dir: Path, item: HeadItem, coef, intercept, train_ids: list[str], meta: dict) -> Head:
    import json

    heads_dir = Path(heads_dir)
    heads_dir.mkdir(parents=True, exist_ok=True)
    version = max(head_versions(heads_dir, item.id), default=0) + 1
    meta = {  # public_name: a private reason's real name is never written into a head file
        **meta, "item": item.id, "name": item.public_name, "kind": item.kind, "region": item.region,
        "fallback_region": item.fallback_region, "version": version, "private": item.private,
        "data_hash": sha256_ids(train_ids), "created_at": now_iso(),
    }
    path = heads_dir / f"{file_stem(item.id)}_v{version}.npz"
    with open(path, "wb") as fh:
        np.savez(fh, coef=np.asarray(coef, np.float32), intercept=np.array(float(intercept)),
                 train_ids=np.array(sorted(train_ids), str), meta=np.array(json.dumps(meta, ensure_ascii=False)))
    manifest = load_manifest(heads_dir)
    entry = {"file": path.name, **{k: meta[k] for k in (
        "version", "backbone", "model_id", "region", "fallback_region", "kind", "name",
        "n_pos", "n_neg", "data_hash", "created_at")}}
    if item.private:
        entry["private"] = True
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
