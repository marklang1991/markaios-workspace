"""Review log loading (with field mapping), exam list, head config and label rules."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import JudgeLabError, read_json

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "heads_v1.json"

CANONICAL_FIELDS = ("id", "image", "decision", "reasons", "scores", "recipe_hash", "versions", "scene",
                    "reviewed_at")
REGIONS = ("face", "hand", "body", "tile", "full")
KINDS = ("hard", "score")


# --------------------------------------------------------------------------- field map
@dataclass
class FieldMap:
    fields: dict[str, str]  # canonical name -> source key (dotted path allowed)
    decision_values: dict[str, list[str]]  # "pass"/"fail" -> accepted raw values

    @classmethod
    def load(cls, path: Path | None) -> "FieldMap":
        fields = {k: k for k in CANONICAL_FIELDS}
        values = {"pass": ["pass"], "fail": ["fail"]}
        if path:
            data = dict(read_json(path))
            dv = data.pop("decision_values", None)
            for k, v in data.items():
                if k.startswith("_"):
                    continue  # comments
                if k not in CANONICAL_FIELDS:
                    raise JudgeLabError(f"field map: 未知字段 {k!r}，可用: {CANONICAL_FIELDS}")
                fields[k] = str(v)
            if dv:
                values = {d: [str(x) for x in dv.get(d, [d])] for d in ("pass", "fail")}
        return cls(fields, values)


def _get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


# --------------------------------------------------------------------------- reviews
@dataclass
class Review:
    id: str
    image: Path | None
    decision: str  # "pass" | "fail"
    reasons: list[str]
    scores: dict[str, float]
    recipe_hash: str | None = None
    versions: dict = field(default_factory=dict)
    scene: str | None = None
    raw: dict = field(default_factory=dict)
    reviewed_at: str | None = None  # ISO time of the owner's verdict

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted lookup; the first segment may be a canonical field (e.g. versions.lora)."""
        head, _, rest = path.partition(".")
        cur = getattr(self, head) if head in CANONICAL_FIELDS else self.raw.get(head)
        if rest:
            cur = _get_path(cur, rest)
        return default if cur is None else cur


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [s for s in re.split(r"[,，;；\s]+", v) if s]
    return [str(x) for x in v]


def _as_scores(v: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, x in (v or {}).items():
        try:
            out[str(k)] = float(x)
        except (TypeError, ValueError):
            pass
    return out


def load_log(path: Path, field_map: FieldMap | None = None) -> list[Review]:
    """Read a JSONL review log. Relative image paths resolve against the log's folder.
    Duplicate ids: the last line wins (a re-review overrides the earlier verdict)."""
    path = Path(path)
    fm = field_map or FieldMap.load(None)
    f = fm.fields
    to_decision = {str(v).lower(): d for d, vals in fm.decision_values.items() for v in vals}
    by_id: dict[str, Review] = {}
    bad = 0
    with path.open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rid = _get_path(rec, f["id"])
            decision = to_decision.get(str(_get_path(rec, f["decision"])).lower())
            if rid is None or decision is None:
                bad += 1
                continue
            img = _get_path(rec, f["image"])
            img_path = None
            if img:
                img_path = Path(str(img).replace("\\", "/"))
                if not img_path.is_absolute():
                    img_path = path.parent / img_path
            versions = _get_path(rec, f["versions"])
            scene = _get_path(rec, f["scene"])
            recipe = _get_path(rec, f["recipe_hash"])
            reviewed = _get_path(rec, f["reviewed_at"])
            by_id[str(rid)] = Review(
                id=str(rid),
                image=img_path,
                decision=decision,
                reasons=_as_list(_get_path(rec, f["reasons"])),
                scores=_as_scores(_get_path(rec, f["scores"])),
                recipe_hash=None if recipe is None else str(recipe),
                versions=versions if isinstance(versions, dict) else {},
                scene=None if scene is None else str(scene),
                raw=rec,
                reviewed_at=None if reviewed is None else str(reviewed),
            )
    if bad:
        print(f"[log] 警告: {bad} 行无法解析或缺少 id/decision，已跳过")
    return list(by_id.values())


def load_exam(path: Path | None) -> set[str]:
    """One image id per line; blank lines and '#' comments ignored."""
    if not path:
        return set()
    ids = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def split_exam(reviews: list[Review], exam_ids: set[str]) -> tuple[list[Review], list[Review], dict]:
    train = [r for r in reviews if r.id not in exam_ids]
    exam = [r for r in reviews if r.id in exam_ids]
    summary = {
        "exam_listed": len(exam_ids),
        "exam_in_log": len(exam),
        "exam_not_in_log": len(exam_ids) - len(exam),
        "train_pool": len(train),
        "total_reviews": len(reviews),
    }
    return train, exam, summary


# --------------------------------------------------------------------------- head config
@dataclass
class HeadItem:
    id: str
    name: str
    kind: str  # "hard" | "score"
    region: str
    fallback_region: str | None
    trainable: bool
    label_rule: dict
    target_recall: float
    note: str = ""
    added_at: str | None = None  # when the owner created this item; earlier reviews could not select it
    definition: str = ""  # the owner's one-line meaning, for humans and VLM prompts (heads learn from examples only)


def load_config(path: Path | None = None) -> tuple[dict[str, HeadItem], dict]:
    data = read_json(path or DEFAULT_CONFIG)
    defaults = data.get("defaults", {})
    recalls = defaults.get("target_recall", {"hard": 0.95, "score": 0.80})
    items: dict[str, HeadItem] = {}
    for iid, c in data["items"].items():
        kind, region = c.get("kind"), c.get("region")
        fb = c.get("fallback_region")
        if kind not in KINDS or region not in REGIONS or (fb and fb not in REGIONS):
            raise JudgeLabError(f"config: 项目 {iid} 的 kind/region 无效 ({kind}, {region}, {fb})")
        rule = c.get("label_rule") or {"positive_if": "selected"}
        if rule.get("positive_if", "selected") != "selected":
            raise JudgeLabError(f"config: 项目 {iid} 仅支持 positive_if=selected")
        items[iid] = HeadItem(
            id=iid,
            name=c.get("name", iid),
            kind=kind,
            region=region,
            fallback_region=fb,
            trainable=bool(c.get("trainable", True)),
            label_rule=rule,
            target_recall=float(c.get("target_recall", recalls[kind])),
            note=c.get("note", ""),
            added_at=c.get("added_at"),
            definition=c.get("definition", ""),
        )
    return items, defaults


def _parse_ts(v: str | None):
    from datetime import datetime, timezone
    if not v:
        return None
    try:
        t = datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def reviewed_after(reviewed_at: str | None, added_at: str) -> bool:
    """True only when the verdict is known to be no earlier than the item's creation."""
    r, a = _parse_ts(reviewed_at), _parse_ts(added_at)
    return r is not None and a is not None and r >= a


def label_for(review: Review, item: HeadItem) -> int | None:
    """1 = defect, 0 = OK, None = ambiguous (excluded).

    item has added_at, review older / untimed -> None (the owner could not have picked it yet)

    pass                         -> 0
    fail, item not selected      -> 0  (unselected items default to OK)
    fail, item selected          -> 1
      score item with max_score  -> 1 only if score <= max_score; selected but above -> None;
                                    selected without a score -> label_rule.missing_score
                                    ("positive" default, or "skip")
    """
    if item.added_at and not reviewed_after(review.reviewed_at, item.added_at):
        return None  # reviewed before the item existed (or no timestamp): the owner could not have selected it
    if review.decision == "pass" or item.id not in review.reasons:
        return 0
    max_score = item.label_rule.get("max_score")
    if item.kind == "score" and max_score is not None:
        s = review.scores.get(item.id)
        if s is None:
            return 1 if item.label_rule.get("missing_score", "positive") == "positive" else None
        return 1 if s <= float(max_score) else None
    return 1


def label_counts(reviews: list[Review], item: HeadItem) -> dict[str, int]:
    c = {"pos": 0, "neg": 0, "ambiguous": 0}
    for r in reviews:
        lab = label_for(r, item)
        c["ambiguous" if lab is None else ("pos" if lab else "neg")] += 1
    return c
