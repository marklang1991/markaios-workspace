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
PASS_WORDS = ("pass", "过", "通过")
FAIL_WORDS = ("fail", "不过", "不通过")
IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# The review page (检查台「审片」) keeps its own defect list, 缺陷清单.json, grouped in Chinese.
# Each group maps to the crop region a head looks at; two groups are never auto-trained.
GROUP_REGION = {"人员": "full", "结构": "tile", "文字": "tile", "手": "hand", "身体": "body", "身份": "face",
                "年龄": "face", "脸与肤质": "face", "光线": "full", "镜头": "full", "动作与分镜": "full", "其他": "full"}
GROUP_NO_TRAIN = {"年龄": "不自动训练：由主人在审片页人工判断", "动作与分镜": "语义判断，交给看图大模型"}


def derive_id(image: Any) -> str:
    """Stable, filesystem-safe id for a record that only carries an image path: '<folder>__<file stem>'."""
    parts = [x for x in str(image).replace("\\", "/").split("/") if x]
    if not parts:
        return ""
    stem = parts[-1].rsplit(".", 1)[0] if parts[-1].lower().endswith(IMG_SUFFIXES) else parts[-1]
    raw = f"{parts[-2]}__{stem}" if len(parts) > 1 else stem
    return re.sub(r'[<>:"/\\|?*\s]+', "_", raw)


def norm_key(v: Any) -> str:
    """Loose key for matching a reason to an item: case and whitespace do not matter."""
    return re.sub(r"\s+", "", str(v)).lower()


# --------------------------------------------------------------------------- field map
@dataclass
class FieldMap:
    fields: dict[str, str]  # canonical name -> source key (dotted path allowed)
    decision_values: dict[str, list[str]]  # "pass"/"fail" -> accepted raw values

    @classmethod
    def load(cls, path: Path | None) -> "FieldMap":
        fields = {k: k for k in CANONICAL_FIELDS}
        values = {"pass": list(PASS_WORDS), "fail": list(FAIL_WORDS)}
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


def _as_reasons(v: Any) -> tuple[list[str], dict[str, float]]:
    """Selected reasons and the scores given with them. Accepts 'a,b', ['a', 'b'] and the review page's
    [{'item': '皮肤塑料 / 磨皮', 'kind': 'score', 'score': 4}, ...]."""
    if not isinstance(v, (list, tuple)):
        return _as_list(v), {}
    names: list[str] = []
    scores: dict[str, float] = {}
    for x in v:
        if isinstance(x, dict):
            n = x.get("item") or x.get("id") or x.get("name")
            if n is None:
                continue
            names.append(str(n))
            try:
                if x.get("score") is not None:
                    scores[str(n)] = float(x["score"])
            except (TypeError, ValueError):
                pass
        else:
            names.append(str(x))
    return names, scores


def _as_scores(v: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(v, dict):  # the review page writes "default" for a pass
        return out
    for k, x in v.items():
        try:
            out[str(k)] = float(x)
        except (TypeError, ValueError):
            pass
    return out


def load_log_ex(path: Path, field_map: FieldMap | None = None) -> tuple[list[Review], dict]:
    """Read a JSONL review log and say what happened to every line.

    Works on the review page's own 审片记录.jsonl without a field map: decisions 过 / 不过, reasons as objects,
    no id (derived from the image path), time in `reviewed_at` or `time`.
    Relative image paths resolve against the log's folder. Duplicate ids: the last line wins (a re-review)."""
    path = Path(path)
    fm = field_map or FieldMap.load(None)
    f = fm.fields
    to_decision = {str(v).lower(): d for d, vals in fm.decision_values.items() for v in vals}
    by_id: dict[str, Review] = {}
    stats = {"lines": 0, "parsed": 0, "skipped": [], "derived_ids": 0, "no_time": 0, "re_reviews": 0}
    with path.open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped"].append((lineno, "不是合法 JSON"))
                continue
            raw_decision = _get_path(rec, f["decision"])
            decision = to_decision.get(str(raw_decision).lower())
            if decision is None:
                stats["skipped"].append((lineno, f"看不懂的决定: {raw_decision!r}"))
                continue
            img = _get_path(rec, f["image"])
            rid = _get_path(rec, f["id"])
            if rid is None and img:
                rid = derive_id(img)
                stats["derived_ids"] += 1
            if not rid:
                stats["skipped"].append((lineno, "既没有 id 也没有图片路径"))
                continue
            img_path = None
            if img:
                img_path = Path(str(img).replace("\\", "/"))
                if not img_path.is_absolute():
                    img_path = path.parent / img_path
            versions = _get_path(rec, f["versions"])
            scene = _get_path(rec, f["scene"])
            recipe = _get_path(rec, f["recipe_hash"])
            if recipe is None:
                recipe = rec.get("recipe")
            reviewed = _get_path(rec, f["reviewed_at"]) or rec.get("time")
            if not reviewed:
                stats["no_time"] += 1
            reasons, reason_scores = _as_reasons(_get_path(rec, f["reasons"]))
            scores = {**reason_scores, **_as_scores(_get_path(rec, f["scores"]))}
            if str(rid) in by_id:
                stats["re_reviews"] += 1
            by_id[str(rid)] = Review(
                id=str(rid),
                image=img_path,
                decision=decision,
                reasons=reasons,
                scores=scores,
                recipe_hash=None if recipe is None else str(recipe),
                versions=versions if isinstance(versions, dict) else {},
                scene=None if scene is None else str(scene),
                raw=rec,
                reviewed_at=None if not reviewed else str(reviewed),
            )
            stats["parsed"] += 1
    if stats["skipped"]:
        print(f"[log] 警告: {len(stats['skipped'])} 行被跳过，例如第 {stats['skipped'][0][0]} 行（{stats['skipped'][0][1]}）")
    return list(by_id.values()), stats


def load_log(path: Path, field_map: FieldMap | None = None) -> list[Review]:
    return load_log_ex(path, field_map)[0]


def load_exam(path: Path | None) -> set[str]:
    """One image id per line; blank lines and '#' comments ignored."""
    if not path:
        return set()
    ids = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
            if "/" in line or "\\" in line or line.lower().endswith(IMG_SUFFIXES):
                ids.add(derive_id(line))  # an image path: also match the id the log derives from it
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
    aliases: tuple = ()  # earlier names of this item (after a rename or a merge); old clicks still count


def _aliases(c: dict) -> tuple:
    out = []
    for k in ("aliases", "merged_from", "曾用名"):
        v = c.get(k) or []
        out += [v] if isinstance(v, str) else list(v)
    return tuple(str(x) for x in out)


def _items_from_defects(data: dict, recalls: dict) -> dict[str, HeadItem]:
    """The review page's 缺陷清单.json: {"version": n, "items": [{"name", "group", "kind", "added", ...}]}."""
    items: dict[str, HeadItem] = {}
    for c in data["items"]:
        name = str(c.get("name", "")).strip()
        kind, group = c.get("kind"), str(c.get("group") or "其他").strip()
        if not name or kind not in KINDS:
            raise JudgeLabError(f"缺陷清单: 这一项缺名称或类型无效: {c}")
        iid = str(c.get("id") or c.get("code") or c.get("slug") or c.get("key") or name)
        region = c.get("region") or GROUP_REGION.get(group, "full")
        if region not in REGIONS:
            raise JudgeLabError(f"缺陷清单: {name} 的 region 无效: {region}")
        note = c.get("note") or GROUP_NO_TRAIN.get(group, "")
        items[iid] = HeadItem(
            id=iid, name=name, kind=kind, region=region,
            # no face / hand / body crop for a picture -> fall back so the click is not lost (coarser; verify says so)
            fallback_region=c.get("fallback_region") or {"body": "full", "face": "tile", "hand": "tile"}.get(region),
            trainable=bool(c.get("trainable", group not in GROUP_NO_TRAIN)),
            label_rule=c.get("label_rule") or {"positive_if": "selected"},
            target_recall=float(c.get("target_recall", recalls[kind])),
            note=note, added_at=c.get("added_at") or c.get("added"),
            definition=str(c.get("definition") or c.get("定义") or ""), aliases=_aliases(c),
        )
    return items


def load_config(path: Path | None = None) -> tuple[dict[str, HeadItem], dict]:
    """Head config. Reads judge_lab's own format (items: {id: {...}}) and, directly, the review page's
    缺陷清单.json (items: [...]) so that there is ONE list of reasons: the one the owner edits on the page."""
    data = read_json(path or DEFAULT_CONFIG)
    defaults = data.get("defaults", {})
    recalls = defaults.get("target_recall", {"hard": 0.95, "score": 0.80})
    if isinstance(data.get("items"), list):
        items = _items_from_defects(data, recalls)
        build_resolver(items)
        return items, defaults
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
            aliases=_aliases(c),
        )
    build_resolver(items)
    return items, defaults


def build_resolver(items: dict[str, HeadItem]) -> dict[str, str]:
    """Any spelling the owner's click may carry (id, display name, an earlier name) -> item id."""
    out: dict[str, str] = {}
    for it in items.values():
        for k in (it.id, it.name, *it.aliases):
            key = norm_key(k)
            if out.get(key, it.id) != it.id:
                raise JudgeLabError(f"缺陷清单: 「{k}」同时指向 {out[key]} 和 {it.id}，名称或曾用名重复")
            out[key] = it.id
    return out


def canonicalize(reviews: list[Review], items: dict[str, HeadItem]) -> dict[str, int]:
    """Rewrite each review's reasons and score keys to item ids (in place, idempotent).
    Returns the reasons that match NO item, with how many reviews carry them: clicks no head will ever see."""
    resolver = build_resolver(items)
    orphans: dict[str, int] = {}
    for r in reviews:
        new_reasons = []
        for x in r.reasons:
            iid = resolver.get(norm_key(x))
            if iid is None:
                orphans[x] = orphans.get(x, 0) + 1
                iid = x
            if iid not in new_reasons:
                new_reasons.append(iid)
        r.reasons = new_reasons
        r.scores = {resolver.get(norm_key(k), k): v for k, v in r.scores.items()}
    return orphans


def _parse_ts(v: str | None):
    from datetime import datetime
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def reviewed_after(reviewed_at: str | None, added_at: str) -> bool:
    """True only when the verdict is known to be no earlier than the item's creation.
    Both stamps come from the same machine; if either lacks an offset they are compared as wall-clock time."""
    r, a = _parse_ts(reviewed_at), _parse_ts(added_at)
    if r is None or a is None:
        return False
    if r.tzinfo is None or a.tzinfo is None:
        r, a = r.replace(tzinfo=None), a.replace(tzinfo=None)
    return r >= a


def explain_label(review: Review, item: HeadItem) -> tuple[int | None, str]:
    """label_for plus the reason in plain words (for `trace` and `verify`)."""
    if item.added_at and not reviewed_after(review.reviewed_at, item.added_at):
        return None, ("这条审片没有时间，无法确认当时有没有这个选项" if not review.reviewed_at
                      else f"审片时间早于这一项的创建时间（{item.added_at}），当时还没有这个选项")
    if review.decision == "pass":
        return 0, "判过 → 这一项算正常"
    if item.id not in review.reasons:
        return 0, "判不过，但没选这一项 → 这一项算正常"
    max_score = item.label_rule.get("max_score")
    if item.kind == "score" and max_score is not None:
        sc = review.scores.get(item.id)
        if sc is None:
            ok = item.label_rule.get("missing_score", "positive") == "positive"
            return (1, "选了这一项（没打分）→ 有问题") if ok else (None, "选了但没打分，按配置不计入")
        if sc > float(max_score):
            return None, f"选了但分数 {sc:g} 高于 {max_score}，算轻微，不计入"
    return 1, "判不过并选了这一项 → 有问题"


def severity_weight(review: Review, item: HeadItem) -> float:
    """How much a positive counts when fitting: for a scoring item the owner's 0-10 score sets it
    (0 = worst -> 1.5, 5 -> 1.0, 10 -> 0.5). Hard gates, negatives and unscored picks count 1."""
    if item.kind != "score" or not item.label_rule.get("severity_weight", True):
        return 1.0
    sc = review.scores.get(item.id)
    if sc is None or review.decision == "pass" or item.id not in review.reasons:
        return 1.0
    return 0.5 + (10.0 - min(10.0, max(0.0, float(sc)))) / 10.0


def label_for(review: Review, item: HeadItem) -> int | None:
    """1 = defect, 0 = OK, None = excluded. See explain_label for the rules.
    Reasons must already be item ids (canonicalize) when the log spells them by display name."""
    return explain_label(review, item)[0]


def label_counts(reviews: list[Review], item: HeadItem) -> dict[str, int]:
    c = {"pos": 0, "neg": 0, "ambiguous": 0}
    for r in reviews:
        lab = label_for(r, item)
        c["ambiguous" if lab is None else ("pos" if lab else "neg")] += 1
    return c
