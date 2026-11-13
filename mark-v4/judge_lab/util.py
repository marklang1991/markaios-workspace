"""Small shared helpers: errors, JSON/Markdown output, hashing, timestamps."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


class JudgeLabError(Exception):
    """User-facing error; the CLI prints it without a traceback."""


class ExamLeakError(JudgeLabError):
    """Raised whenever an exam id would reach a fitting step."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def assert_no_exam(ids: Iterable[str], exam_ids: set[str], where: str, hide: Iterable[str] = ()) -> None:
    """hide: ids of private records; they are counted in the error but never printed."""
    leaked = sorted(set(ids) & exam_ids)
    if leaked:
        shown = [i for i in leaked if i not in set(hide)][:5]
        eg = f" (例如 {shown})" if shown else "（私有记录，不显示）"
        raise ExamLeakError(f"{where}: {len(leaked)} 个考卷 id 混入了训练/拟合数据{eg}。考卷永不训练。")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(v: Any) -> str:
        return fmt(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def utf8_stdio() -> None:
    """Windows consoles default to a legacy code page; force UTF-8 output."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
