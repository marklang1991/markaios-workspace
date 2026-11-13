"""Review logs (one or several, with field mapping), exam registry, head config and the labelling policy."""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .util import JudgeLabError, read_json

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "heads_v1.json"

CANONICAL_FIELDS = ("id", "image", "decision", "reasons", "scores", "recipe_hash", "versions", "scene",
                    "reviewed_at")
# Newer fields of the review page; read under these names unless a field map renames them.
EXTRA_FIELDS = ("reasons_complete", "batch_pass", "recommend", "source", "sha256", "note", "seconds")
REGIONS = ("face", "hand", "body", "tile", "full")
KINDS = ("hard", "score")
PASS_WORDS = ("pass", "过", "通过")
FAIL_WORDS = ("fail", "不过", "不通过")
IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
BATCH_PASS_NOTE = "整屏通过"  # how old records marked a batch pass

# The review page (检查台「审片」) keeps its own defect list, 缺陷清单.json, grouped in Chinese.
# Each group maps to the crop region a head looks at; two groups are never auto-trained.
# The region always comes from the CURRENT list (group, or an explicit `region` on the item); the `group`
# stored inside old review records is ignored.
GROUP_REGION = {"人员": "full", "结构": "tile", "文字": "tile", "手": "hand", "身体": "body", "身份": "face",
                "年龄": "face", "脸与肤质": "face", "光线": "full", "镜头": "full", "动作与分镜": "full", "其他": "full"}
GROUP_NO_TRAIN = {"年龄": "不自动训练：由主人在审片页人工判断", "动作与分镜": "语义判断，交给看图大模型"}

# How a click (or the absence of one) becomes a training label. One knob per doubt; a weight of 0 means
# "unknown: leave this picture out for this reason", 1 means "counts fully".
# Resolved per item: these defaults < config `defaults.label_policy` < the item's `label_rule` < CLI flags.
POLICY_DEFAULTS = {
    "unselected_fail_weight": 0.0,  # failed picture, reason not selected (and the record is not `reasons_complete`)
    "batch_pass_weight": 0.5,       # negative from a 整屏通过 picture
    "carried_weight": 0.0,          # reason auto-carried from the previous picture, never touched by the owner
    "carried_unknown_weight": 0.5,  # old record: not known whether the reason was carried
    "severity_weight": True,        # scoring items: the owner's 0-10 score scales a positive (0 -> 1.5 ... 10 -> 0.5)
}
POLICY_WEIGHTS = tuple(k for k, v in POLICY_DEFAULTS.items() if not isinstance(v, bool))

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX16 = re.compile(r"[0-9a-f]{16}")
_SCENE = re.compile(r"^(.*)_s\d+$")


def derive_id(image: Any) -> str:
    """Stable, filesystem-safe id for a record that only carries an image path: '<folder>__<file stem>'."""
    parts = [x for x in str(image).replace("\\", "/").split("/") if x]
    if not parts:
        return ""
    stem = parts[-1].rsplit(".", 1)[0] if parts[-1].lower().endswith(IMG_SUFFIXES) else parts[-1]
    raw = f"{parts[-2]}__{stem}" if len(parts) > 1 else stem
    return re.sub(r'[<>:"/\\|?*\s]+', "_", raw)


def derive_scene(image: Any) -> str | None:
    """Pictures come as scenes x seeds: file stem 'hallway_s1102' -> scene 'hallway'."""
    if not image:
        return None
    name = str(image).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if name.lower().endswith(IMG_SUFFIXES) else name
    m = _SCENE.match(stem)
    return m.group(1) if m and m.group(1) else None


def image_key(image: Any, base: Path) -> str:
    """Identity of a picture across logs: its path with slashes unified, case ignored."""
    s = str(image).replace("\\", "/")
    if not (s.startswith("/") or re.match(r"^[A-Za-z]:/", s)):
        s = os.path.abspath(str(Path(base) / s)).replace("\\", "/")
    return posixpath.normpath(s).lower()


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
        known = CANONICAL_FIELDS + EXTRA_FIELDS
        fields = {k: k for k in known}
        values = {"pass": list(PASS_WORDS), "fail": list(FAIL_WORDS)}
        if path:
            data = dict(read_json(path))
            dv = data.pop("decision_values", None)
            for k, v in data.items():
                if k.startswith("_"):
                    continue  # comments
                if k not in known:
                    raise JudgeLabError(f"field map: 未知字段 {k!r}，可用: {known}")
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
    scene: str | None = None  # record field, else derived from the file stem ('hallway_s1102' -> 'hallway')
    raw: dict = field(default_factory=dict)
    reviewed_at: str | None = None  # ISO time of the owner's verdict
    carried: dict = field(default_factory=dict)  # reason -> True (auto-carried, untouched) / False (owner chose it);
    #                                              a reason missing here = unknown (old records)
    reasons_complete: bool = False  # the owner says every applicable reason is selected
    batch_pass: bool = False  # passed by 整屏通过, not looked at one by one
    recommend: bool = False  # 推荐进入: must enter training
    source: str | None = None
    sha256: str | None = None  # lower-case hex of the image file
    note: str = ""  # the owner's free text; shown by `trace` only
    seconds: float | None = None  # time spent on the picture: workload statistic only, never used for training
    private: bool = False  # came from a --private-log file: only ever counted, never shown

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted lookup; the first segment may be a canonical field (e.g. versions.lora)."""
        head, _, rest = path.partition(".")
        cur = getattr(self, head) if head in CANONICAL_FIELDS + EXTRA_FIELDS else self.raw.get(head)
        if rest:
            cur = _get_path(cur, rest)
        return default if cur is None else cur


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [s for s in re.split(r"[,，;；\s]+", v) if s]
    return [str(x) for x in v]


def _as_reasons(v: Any) -> tuple[list[str], dict[str, float], dict[str, bool]]:
    """Selected reasons, the scores given with them, and which were auto-carried.
    Accepts 'a,b', ['a', 'b'] and the review page's
    [{'item': '皮肤塑料 / 磨皮', 'kind': 'score', 'score': 4, 'carried': false}, ...]."""
    if not isinstance(v, (list, tuple)):
        return _as_list(v), {}, {}
    names: list[str] = []
    scores: dict[str, float] = {}
    flags: dict[str, bool | None] = {}
    for x in v:
        if isinstance(x, dict):
            n = x.get("item") or x.get("id") or x.get("name")
            if n is None:
                continue
            n, c = str(n), x.get("carried")
            names.append(n)
            try:
                if x.get("score") is not None:
                    scores[n] = float(x["score"])
            except (TypeError, ValueError):
                pass
        else:
            n, c = str(x), None
            names.append(n)
        c = c if isinstance(c, bool) else None
        flags[n] = _merge_carried(flags[n], c) if n in flags else c
    return names, scores, {k: c for k, c in flags.items() if c is not None}


def _merge_carried(a: bool | None, b: bool | None) -> bool | None:
    """Two clicks that are the same reason (after a merge): chosen by the owner > unknown > auto-carried."""
    if a is False or b is False:
        return False
    return None if a is None or b is None else True


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


def _as_number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def load_logs(paths: Iterable[Path], field_map: FieldMap | None = None,
              private_paths: Iterable[Path] = ()) -> tuple[list[Review], dict]:
    """Read one or several JSONL review logs and say what happened to every line.

    Works on the review page's own 审片记录.jsonl without a field map: decisions 过 / 不过, reasons as objects,
    no id (derived from the image path), time in `reviewed_at` or `time`.
    Relative image paths resolve against their log's folder.
    The same picture reviewed again (same id, or same image path with slashes and case ignored), in one file or
    across files: the last record wins; files are read in the order given, private ones last.
    Records from `private_paths` are flagged private: callers may count them but never show them."""
    fm = field_map or FieldMap.load(None)
    f = fm.fields
    to_decision = {str(v).lower(): d for d, vals in fm.decision_values.items() for v in vals}
    slots: list[Review | None] = []
    slot_of: dict[tuple[str, str], int] = {}  # ("id", id) / ("path", key) -> index in slots
    keys_of: dict[int, list[tuple[str, str]]] = {}
    stats = {"files": 0, "lines": 0, "parsed": 0, "skipped": [], "skipped_private": 0, "derived_ids": 0,
             "no_time": 0, "re_reviews": 0}
    sources = [(Path(p), False) for p in paths] + [(Path(p), True) for p in private_paths]
    for path, private in sources:
        stats["files"] += 1
        with path.open(encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1

                def skip(why: str) -> None:
                    if private:
                        stats["skipped_private"] += 1  # nothing of a private line is kept, not even its number
                    else:
                        stats["skipped"].append((lineno, why, path.name))

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skip("不是合法 JSON")
                    continue
                if not isinstance(rec, dict):
                    skip("不是一条记录（要 JSON 对象）")
                    continue
                raw_decision = _get_path(rec, f["decision"])
                decision = to_decision.get(str(raw_decision).lower())
                if decision is None:
                    skip(f"看不懂的决定: {raw_decision!r}")
                    continue
                img = _get_path(rec, f["image"])
                rid = _get_path(rec, f["id"])
                derived = rid is None and bool(img)
                if derived:
                    rid = derive_id(img)
                if not rid:
                    skip("既没有 id 也没有图片路径")
                    continue
                stats["derived_ids"] += derived
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
                reasons, reason_scores, carried = _as_reasons(_get_path(rec, f["reasons"]))
                scores = {**reason_scores, **_as_scores(_get_path(rec, f["scores"]))}
                note = _get_path(rec, f["note"])
                note = "" if note is None else str(note)
                batch = _get_path(rec, f["batch_pass"])
                sha = _get_path(rec, f["sha256"])
                sha = sha.strip().lower() if isinstance(sha, str) else ""
                source = _get_path(rec, f["source"])
                review = Review(
                    id=str(rid),
                    image=img_path,
                    decision=decision,
                    reasons=reasons,
                    scores=scores,
                    recipe_hash=None if recipe is None else str(recipe),
                    versions=versions if isinstance(versions, dict) else {},
                    scene=str(scene) if scene not in (None, "") else derive_scene(img),
                    raw=rec,
                    reviewed_at=None if not reviewed else str(reviewed),
                    carried=carried,
                    reasons_complete=_get_path(rec, f["reasons_complete"]) is True,
                    batch_pass=bool(batch) if batch is not None else note.strip() == BATCH_PASS_NOTE,
                    recommend=_get_path(rec, f["recommend"]) is True,
                    source=str(source) if source not in (None, "") else None,
                    sha256=sha if _HEX64.fullmatch(sha) else None,
                    note=note,
                    seconds=_as_number(_get_path(rec, f["seconds"])),
                    private=private,
                )
                keys = [("id", review.id)] + ([("path", image_key(img, path.parent))] if img else [])
                old = sorted({slot_of[k] for k in keys if k in slot_of})
                if old:
                    stats["re_reviews"] += 1
                    for i in old:  # every earlier record of this picture goes; the new one takes the first place
                        for k in keys_of.pop(i):
                            del slot_of[k]
                        slots[i] = None
                    at = old[0]
                else:
                    at = len(slots)
                    slots.append(None)
                slots[at] = review
                keys_of[at] = keys
                for k in keys:
                    slot_of[k] = at
                stats["parsed"] += 1
    if stats["skipped"]:
        n, why, name = stats["skipped"][0]
        print(f"[log] 警告: {len(stats['skipped'])} 行被跳过，例如 {name} 第 {n} 行（{why}）")
    if stats["skipped_private"]:
        print(f"[log] 警告: 私有记录里有 {stats['skipped_private']} 行被跳过（内容不显示）")
    reviews = [r for r in slots if r is not None]
    stats["reviews"], stats["private"] = len(reviews), sum(r.private for r in reviews)
    return reviews, stats


def load_log_ex(path: Path | Iterable[Path], field_map: FieldMap | None = None) -> tuple[list[Review], dict]:
    """One log (or a list of logs); see load_logs."""
    return load_logs([path] if isinstance(path, (str, os.PathLike)) else list(path), field_map)


def load_log(path: Path | Iterable[Path], field_map: FieldMap | None = None) -> list[Review]:
    return load_log_ex(path, field_map)[0]


def filter_sources(reviews: list[Review], keep: Iterable[str] | None = None,
                   drop: Iterable[str] | None = None) -> tuple[list[Review], dict]:
    """--source / --exclude-source. A 推荐进入 picture is never dropped: the owner said it must be trained on."""
    keep, drop = (set(keep) if keep else None), set(drop or ())
    out, info = [], {"dropped": 0, "kept_by_recommend": 0}
    for r in reviews:
        ok = (keep is None or r.source in keep) and r.source not in drop
        if not ok and r.recommend:
            info["kept_by_recommend"] += 1
            ok = True
        if ok:
            out.append(r)
        else:
            info["dropped"] += 1
    return out, info


# --------------------------------------------------------------------------- exam registry
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path | None) -> str | None:
    """sha256 of a file, remembered per (path, mtime, size) for the life of the process. None if unreadable."""
    if path is None:
        return None
    try:
        st = os.stat(path)
        key = (str(path), st.st_mtime_ns, st.st_size)
        if key not in _HASH_CACHE:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            _HASH_CACHE[key] = h.hexdigest()
        return _HASH_CACHE[key]
    except OSError:
        return None


class ExamRegistry(set):
    """Every token of the exam list: ids, path-derived ids, sha256 (64 hex, plus its first 16) and 16-hex ids.
    To old callers it is the plain set of exam ids; `entries` keeps one token set per listed picture."""

    def __init__(self, entries: Iterable[Iterable[str]] = ()):
        self.entries: list[frozenset[str]] = []
        seen: set[frozenset[str]] = set()
        for e in entries:
            e = frozenset(e)
            if e and e not in seen:
                seen.add(e)
                self.entries.append(e)
        super().__init__(t for e in self.entries for t in e)

    @property
    def hashes(self) -> set[str]:
        return {t for t in self if _HEX64.fullmatch(t) or _HEX16.fullmatch(t)}


def _exam_tokens(v: Any, where: str) -> set[str]:
    if isinstance(v, (str, int)) and not isinstance(v, bool):
        s = str(v).strip()
        low = s.lower()
        if not s:
            return set()
        if _HEX64.fullmatch(low):
            return {low, low[:16]}
        if _HEX16.fullmatch(low):
            return {s, low}
        toks = {s}
        if "/" in s or "\\" in s or low.endswith(IMG_SUFFIXES):
            toks.add(derive_id(s))  # an image path: also match the id the log derives from it
        return toks
    if isinstance(v, dict):
        toks: set[str] = set()
        sha = v.get("sha256")
        if sha:
            if not _HEX64.fullmatch(str(sha).strip().lower()):
                raise JudgeLabError(f"考卷名单{where}: sha256 要 64 位十六进制")
            toks |= _exam_tokens(str(sha), where)
        for k in ("id", "image", "path"):
            if v.get(k):
                toks |= _exam_tokens(str(v[k]), where)
        if not toks:
            raise JudgeLabError(f"考卷名单{where}: 看不懂这一项（要有 sha256、id、image 或 path 之一）")
        return toks
    raise JudgeLabError(f"考卷名单{where}: 看不懂这一项（要字符串，或带 sha256 / id / image 的对象）")


def load_exam(path: Path | None) -> ExamRegistry:
    """The exam list. Accepted forms:
      - text: one id, image path, sha256 (64 hex) or 16-hex id per line; blank lines and '#' comments ignored;
      - JSON: [...], {"items": [...]} or {"exam": [...]}, entries being such strings or objects with any of
        sha256 / id / image (or path);
      - JSONL: one such object per line.
    An entry that cannot be understood is an error, never skipped: a dropped exam entry would leak into training."""
    if not path:
        return ExamRegistry()
    text = Path(path).read_text(encoding="utf-8-sig")
    body = text.strip()
    if body[:1] in ("[", "{"):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            if body[:1] == "[":
                raise JudgeLabError(f"考卷名单 {Path(path).name} 不是合法 JSON") from None
            data = None  # several objects, one per line
        if data is not None:
            if isinstance(data, dict):
                inner = next((data[k] for k in ("items", "exam") if isinstance(data.get(k), list)), None)
                if inner is None and not any(data.get(k) for k in ("sha256", "id", "image", "path")):
                    raise JudgeLabError("考卷名单 JSON: 要有 items 或 exam 列表")
                data = [data] if inner is None else inner
            return ExamRegistry(_exam_tokens(v, f"第 {i} 项") for i, v in enumerate(data, 1))
    entries = []
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("{"):
            try:
                entries.append(_exam_tokens(json.loads(s), f"第 {lineno} 行"))
            except json.JSONDecodeError:
                raise JudgeLabError(f"考卷名单第 {lineno} 行不是合法 JSON") from None
        else:
            entries.append(_exam_tokens(line.split("#", 1)[0], f"第 {lineno} 行"))
    return ExamRegistry(entries)


def _review_tokens(r: Review, registry: ExamRegistry, want_hash: bool) -> set[str]:
    toks = {r.id, r.id.lower()} if _HEX16.fullmatch(r.id.lower()) else {r.id}
    if r.image is not None:
        toks.add(derive_id(r.image))
    if r.sha256 is None and want_hash and not toks & registry:
        r.sha256 = file_sha256(r.image)  # only read the file when nothing cheaper has matched
    if r.sha256:
        toks |= {r.sha256, r.sha256[:16]}
    return toks


def split_exam(reviews: list[Review], exam_ids: Iterable[str]) -> tuple[list[Review], list[Review], dict]:
    """A review is an exam picture if its id, its path-derived id or its sha256 (the record's field, else computed
    from the file when the registry lists hashes) is in the registry; full hash or its first 16 hex."""
    reg = exam_ids if isinstance(exam_ids, ExamRegistry) else ExamRegistry(_exam_tokens(x, "") for x in exam_ids)
    want_hash = bool(reg.hashes)
    train, exam, hit = [], [], set()
    for r in reviews:
        m = _review_tokens(r, reg, want_hash) & reg
        hit |= m
        (exam if m else train).append(r)
    in_log = sum(1 for e in reg.entries if e & hit)
    summary = {
        "exam_listed": len(reg.entries),
        "exam_in_log": len(exam),
        "exam_not_in_log": len(reg.entries) - in_log,
        "train_pool": len(train),
        "total_reviews": len(reviews),
    }
    return train, exam, summary


def exam_guard(registry: Iterable[str], exam: list[Review]) -> set[str]:
    """Ids that must never reach a fitting step: everything the registry lists plus the ids of the reviews it
    matched (a picture matched by its sha256 has an id the registry never mentions)."""
    return set(registry) | {r.id for r in exam}


# --------------------------------------------------------------------------- head config
def private_id(real: str) -> str:
    """Opaque id of a private reason: stable, and the same whether or not --show-private is given."""
    return "p_" + hashlib.sha1(str(real).encode("utf-8")).hexdigest()[:10]


def masked_name(opaque_id: str) -> str:
    return f"私有项 {opaque_id[-4:]}"


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
    group: str | None = None  # the review page's group (decides the region unless the item sets one)
    private: bool = False  # the real name / id / aliases / definition never leave the config file
    keys: tuple = field(default=(), repr=False)  # private item: real id, name, aliases - resolver keys only
    defined: bool | None = None  # private item: whether a definition exists (the text itself is dropped)

    @property
    def has_definition(self) -> bool:
        return bool(self.definition.strip()) if self.defined is None else self.defined

    @property
    def public_name(self) -> str:
        """The name that may be stored in head files and thresholds, whatever --show-private says."""
        return masked_name(self.id) if self.private else self.name


def _aliases(c: dict) -> tuple:
    out = []
    for k in ("aliases", "merged_from", "曾用名"):
        v = c.get(k) or []
        out += [v] if isinstance(v, str) else list(v)
    return tuple(str(x) for x in out)


def _privatise(it: HeadItem, show: bool, default_note: str = "") -> HeadItem:
    real_id = it.id
    it.id, it.private, it.defined = private_id(real_id), True, bool(it.definition.strip())
    it.keys = (real_id, it.name, *it.aliases)
    if not show:
        it.name, it.definition, it.aliases, it.note = masked_name(it.id), "", (), default_note
    return it


def check_policy(d: Any, where: str, strict: bool = False) -> dict:
    """The label-policy knobs found in `d`, validated. strict: any other key is an error (typo protection)."""
    if not d:
        return {}
    if not isinstance(d, dict):
        raise JudgeLabError(f"{where}: 标注口径要写成 {{名称: 值}}")
    out = {}
    for k, v in d.items():
        if k not in POLICY_DEFAULTS:
            if strict and not str(k).startswith("_"):
                raise JudgeLabError(f"{where}: 未知的标注口径 {k!r}，可用: {tuple(POLICY_DEFAULTS)}")
            continue
        if k in POLICY_WEIGHTS:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
                raise JudgeLabError(f"{where}: {k} 要是 0 到 1 之间的数，收到 {v!r}")
            out[k] = float(v)
        else:
            out[k] = bool(v)
    return out


def policy_of(item: HeadItem) -> dict:
    """The labelling policy in force for this item, every knob filled in."""
    return {k: item.label_rule.get(k, v) for k, v in POLICY_DEFAULTS.items()}


def apply_policy(items: dict[str, HeadItem], overrides: dict) -> None:
    """CLI flags: the last word, for every item."""
    overrides = check_policy(overrides, "命令行", strict=True)
    if overrides:
        for it in items.values():
            it.label_rule = {**it.label_rule, **overrides}


def _items_from_defects(data: dict, recalls: dict, show_private: bool) -> dict[str, HeadItem]:
    """The review page's 缺陷清单.json: {"version": n, "items": [{"name", "group", "kind", "added", ...}]}."""
    items: dict[str, HeadItem] = {}
    for n, c in enumerate(data["items"], 1):
        private = bool(c.get("private"))
        name = str(c.get("name", "")).strip()
        kind, group = c.get("kind"), str(c.get("group") or "其他").strip()
        shown = "一个私有项" if private else f"「{name}」"
        if not name or kind not in KINDS:
            raise JudgeLabError(f"缺陷清单: 第 {n} 项缺名称或类型无效" + ("" if private else f": {c}"))
        iid = str(c.get("id") or c.get("code") or c.get("slug") or c.get("key") or name)
        region = c.get("region") or GROUP_REGION.get(group, "full")
        if region not in REGIONS:
            raise JudgeLabError(f"缺陷清单: {shown} 的 region 无效: {region}")
        rule = c.get("label_rule") or {"positive_if": "selected"}
        check_policy(rule, f"缺陷清单 {shown}")
        it = HeadItem(
            id=iid, name=name, kind=kind, region=region,
            # no face / hand / body crop for a picture -> fall back so the click is not lost (coarser; verify says so)
            fallback_region=c.get("fallback_region") or {"body": "full", "face": "tile", "hand": "tile"}.get(region),
            trainable=bool(c.get("trainable", group not in GROUP_NO_TRAIN)),
            label_rule=rule,
            target_recall=float(c.get("target_recall", recalls[kind])),
            note=c.get("note") or GROUP_NO_TRAIN.get(group, ""), added_at=c.get("added_at") or c.get("added"),
            definition=str(c.get("definition") or c.get("定义") or ""), aliases=_aliases(c), group=group,
        )
        if private:
            _privatise(it, show_private, GROUP_NO_TRAIN.get(group, ""))
        items[it.id] = it
    return items


def load_config(path: Path | None = None, show_private: bool = False) -> tuple[dict[str, HeadItem], dict]:
    """Head config. Reads judge_lab's own format (items: {id: {...}}) and, directly, the review page's
    缺陷清单.json (items: [...]) so that there is ONE list of reasons: the one the owner edits on the page.

    An item with "private": true gets an opaque id (p_<hash>) and, unless show_private, a placeholder name and no
    definition; its real id / name / aliases survive only as resolver keys, so the owner's clicks still count."""
    data = read_json(path or DEFAULT_CONFIG)
    defaults = dict(data.get("defaults", {}))
    recalls = defaults.get("target_recall", {"hard": 0.95, "score": 0.80})
    if isinstance(data.get("items"), list):
        defaults.pop("label_policy", None)  # the page's own file carries no policy; use the item or the CLI
        items = _items_from_defects(data, recalls, show_private)
        build_resolver(items)
        return items, defaults
    base = check_policy(defaults.get("label_policy"), "config defaults.label_policy", strict=True)
    defaults["label_policy"] = base
    items: dict[str, HeadItem] = {}
    for iid, c in data["items"].items():
        private = bool(c.get("private"))
        shown = "一个私有项" if private else f"项目 {iid}"
        kind, region = c.get("kind"), c.get("region")
        fb = c.get("fallback_region")
        if kind not in KINDS or region not in REGIONS or (fb and fb not in REGIONS):
            raise JudgeLabError(f"config: {shown} 的 kind/region 无效 ({kind}, {region}, {fb})")
        rule = c.get("label_rule") or {"positive_if": "selected"}
        if rule.get("positive_if", "selected") != "selected":
            raise JudgeLabError(f"config: {shown} 仅支持 positive_if=selected")
        check_policy(rule, f"config {shown}")
        it = HeadItem(
            id=iid,
            name=c.get("name", iid),
            kind=kind,
            region=region,
            fallback_region=fb,
            trainable=bool(c.get("trainable", True)),
            label_rule={**base, **rule},
            target_recall=float(c.get("target_recall", recalls[kind])),
            note=c.get("note", ""),
            added_at=c.get("added_at"),
            definition=c.get("definition", ""),
            aliases=_aliases(c),
            group=c.get("group"),
        )
        if private:
            _privatise(it, show_private)
        items[it.id] = it
    build_resolver(items)
    return items, defaults


def build_resolver(items: dict[str, HeadItem]) -> dict[str, str]:
    """Any spelling the owner's click may carry (id, display name, an earlier name) -> item id."""
    out: dict[str, str] = {}
    for it in items.values():
        for k in (it.id, it.name, *it.aliases, *it.keys):
            key = norm_key(k)
            other = out.get(key, it.id)
            if other != it.id:
                if it.private or items[other].private:  # never print a private name, not even in an error
                    raise JudgeLabError(f"缺陷清单: 一个私有项的名称或曾用名和另一项重复（{other} 和 {it.id}）")
                raise JudgeLabError(f"缺陷清单: 「{k}」同时指向 {other} 和 {it.id}，名称或曾用名重复")
            out[key] = it.id
    return out


def canonicalize(reviews: list[Review], items: dict[str, HeadItem]) -> dict[str, int]:
    """Rewrite each review's reasons, score keys and carried flags to item ids (in place, idempotent).
    Returns the reasons that match NO item, with how many reviews carry them: clicks no head will ever see.
    Private records are left out of that dict (their text must not be shown): see private_orphans."""
    resolver = build_resolver(items)
    orphans: dict[str, int] = {}
    for r in reviews:
        new_reasons: list[str] = []
        carried: dict[str, bool | None] = {}
        for x in r.reasons:
            iid = resolver.get(norm_key(x))
            if iid is None:
                if not r.private:
                    orphans[x] = orphans.get(x, 0) + 1
                iid = x
            c = r.carried.get(x)
            carried[iid] = _merge_carried(carried[iid], c) if iid in carried else c
            if iid not in new_reasons:
                new_reasons.append(iid)
        r.reasons = new_reasons
        r.carried = {k: v for k, v in carried.items() if v is not None}
        r.scores = {resolver.get(norm_key(k), k): v for k, v in r.scores.items()}
    return orphans


def private_orphans(reviews: list[Review], items: dict[str, HeadItem]) -> int:
    """After canonicalize: how many clicks in private records match no item (a count is all that may be said)."""
    return sum(1 for r in reviews if r.private for x in r.reasons if x not in items)


# --------------------------------------------------------------------------- labels
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


def severity_weight(review: Review, item: HeadItem) -> float:
    """The severity factor of a positive: for a scoring item the owner's 0-10 score sets it
    (0 = worst -> 1.5, 5 -> 1.0, 10 -> 0.5). Hard gates, negatives and unscored picks count 1."""
    if item.kind != "score" or not item.label_rule.get("severity_weight", POLICY_DEFAULTS["severity_weight"]):
        return 1.0
    sc = review.scores.get(item.id)
    if sc is None or review.decision == "pass" or item.id not in review.reasons:
        return 1.0
    return 0.5 + (10.0 - min(10.0, max(0.0, float(sc)))) / 10.0


def _assess(review: Review, item: HeadItem) -> tuple[int | None, str, float, str]:
    """assess plus a short code naming the branch taken."""
    def knob(k: str) -> float:
        return float(item.label_rule.get(k, POLICY_DEFAULTS[k]))

    # 1. the reason did not exist yet when this picture was reviewed
    if item.added_at and not reviewed_after(review.reviewed_at, item.added_at):
        return None, ("这条审片没有时间，无法确认当时有没有这个选项" if not review.reviewed_at
                      else f"审片时间早于这一项的创建时间（{item.added_at}），当时还没有这个选项"), 0.0, "time"
    # 2. passed
    if review.decision == "pass":
        if review.batch_pass and review.recommend:
            return 0, "整屏通过，但你点了「推荐进入」→ 这一项算正常，按满分量", 1.0, "pass"
        if review.batch_pass:
            w = knob("batch_pass_weight")
            if w <= 0:
                return None, "整屏通过（没有逐张看）→ 按当前口径不计入", 0.0, "batch_off"
            if w < 1:
                return 0, f"整屏通过（没有逐张看）→ 这一项算正常，分量 {w:g}", w, "batch"
        return 0, "判过 → 这一项算正常", 1.0, "pass"
    # 3. failed, this reason not selected
    if item.id not in review.reasons:
        if review.reasons_complete:
            return 0, "判不过，原因已选全，没选这一项 → 这一项算正常", 1.0, "complete"
        w = knob("unselected_fail_weight")
        if w <= 0:
            return None, "判不过，但没选这一项 → 不知道有没有这个问题，不计入", 0.0, "unknown"
        return 0, f"判不过，没选这一项 → 按当前口径算正常，分量 {w:g}", w, "unselected"
    # 4. failed, this reason selected
    why = "判不过并选了这一项 → 有问题"
    max_score = item.label_rule.get("max_score")
    if item.kind == "score" and max_score is not None:
        sc = review.scores.get(item.id)
        if sc is None:
            if item.label_rule.get("missing_score", "positive") != "positive":
                return None, "选了但没打分，按配置不计入", 0.0, "no_score"
            why = "选了这一项（没打分）→ 有问题"
        elif sc > float(max_score):
            return None, f"选了但分数 {sc:g} 高于 {max_score}，算轻微，不计入", 0.0, "light"
    c = review.carried.get(item.id)
    if c is True:
        factor = knob("carried_weight")
        if factor <= 0:
            return None, "这个原因是从上一张自动带过来的，你没有碰过 → 不计入", 0.0, "carried_off"
        why += f"（从上一张自动带过来、没碰过，分量 ×{factor:g}）"
    elif c is False:
        factor = 1.0
    else:
        factor = knob("carried_unknown_weight")
        if factor <= 0:
            return None, "旧记录，不清楚这个原因是不是自动带过来的 → 按当前口径不计入", 0.0, "carried_off"
        if factor != 1:
            why += f"（旧记录，不清楚是不是自动带过来的，分量 ×{factor:g}）"
    return 1, why, factor * severity_weight(review, item), "selected"


def assess(review: Review, item: HeadItem) -> tuple[int | None, str, float]:
    """THE rule turning one review into one item's training example: (label, why in plain words, fitting weight).

    label 1 = defect, 0 = OK, None = left out (weight 0). Checked in this order:
      1. reviewed before the item existed (or no time on the review)        -> None
      2. passed                                                            -> 0, weight 1
           整屏通过 and not 推荐进入                                        -> 0 at batch_pass_weight (0 = None)
      3. failed, reason not selected: record says reasons_complete         -> 0, weight 1
           otherwise                                                       -> 0 at unselected_fail_weight (0 = None)
      4. failed, reason selected: the item's max_score rule may drop it    -> None
           else 1, weight = carried factor x severity factor, carried factor being
           1 (carried: false), carried_weight (true) or carried_unknown_weight (missing); 0 = None
    Reasons must already be item ids (canonicalize) when the log spells them by display name."""
    return _assess(review, item)[:3]


def why_code(review: Review, item: HeadItem) -> str:
    """Which branch of assess applied: time | pass | batch | batch_off | complete | unknown | unselected |
    selected | no_score | light | carried_off."""
    return _assess(review, item)[3]


def explain_label(review: Review, item: HeadItem) -> tuple[int | None, str]:
    """label_for plus the reason in plain words (for `trace` and `verify`)."""
    return assess(review, item)[:2]


def label_for(review: Review, item: HeadItem) -> int | None:
    """1 = defect, 0 = OK, None = excluded. See assess for the rules."""
    return assess(review, item)[0]


def label_counts(reviews: list[Review], item: HeadItem) -> dict[str, int]:
    c = {"pos": 0, "neg": 0, "ambiguous": 0}
    for r in reviews:
        lab = label_for(r, item)
        c["ambiguous" if lab is None else ("pos" if lab else "neg")] += 1
    return c
