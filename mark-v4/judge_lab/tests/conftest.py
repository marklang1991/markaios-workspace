"""Synthetic dataset: noise images, some with a planted bright square in one quadrant
("structure_break", tile region) and some with a strong red cast ("lighting", full region).
Only the `fake` backbone is used, so nothing is ever downloaded."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

N_IMAGES = 120
EXAM_RANGE = range(100, 120)  # 20 exam ids
W, H = 48, 64


def is_struct(i: int) -> bool:
    return i % 3 == 0


def is_light(i: int) -> bool:
    return i % 4 == 1


def _image(rng, struct: bool, light: bool) -> Image.Image:
    a = rng.integers(0, 110, size=(H, W, 3)).astype(np.int32)
    if light:
        a[..., 0] += 140
    if struct:
        qx, qy = rng.integers(0, 2), rng.integers(0, 2)
        x = qx * W // 2 + rng.integers(0, W // 2 - 10)
        y = qy * H // 2 + rng.integers(0, H // 2 - 10)
        a[y : y + 10, x : x + 10] = 255
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make_dataset(root: Path, n: int = N_IMAGES, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        iid = f"img_{i:03d}"
        _image(rng, is_struct(i), is_light(i)).save(img_dir / f"{iid}.png")
        reasons, scores = [], {}
        if is_struct(i):
            reasons.append("structure_break")
        if is_light(i):
            reasons.append("lighting")
            scores["lighting"] = 3
        if i % 10 == 7 and not reasons:
            reasons.append("composition")
            scores["composition"] = 4
        if i % 25 == 2:
            reasons.append("age_too_young")
        lines.append({
            "id": iid,
            "image": f"images/{iid}.png",
            "decision": "fail" if reasons else "pass",
            "reasons": reasons,
            "reasons_complete": True,  # synthetic labels are complete: an unselected reason really is absent
            "scores": scores,
            "recipe_hash": f"r{i % 5}",
            "versions": {"lora": "mark_taste_v1" if i < 60 else "mark_taste_v2"},
            "scene": ["indoor", "outdoor"][(i // 2) % 2],
        })
    log = root / "review_log.jsonl"
    log.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    exam = root / "exam.txt"
    exam.write_text("# 考卷\n" + "\n".join(f"img_{i:03d}" for i in EXAM_RANGE) + "\n", encoding="utf-8")

    crops = root / "crops"
    for k in range(2):
        (crops / "img_000").mkdir(parents=True, exist_ok=True)
        _image(rng, False, False).resize((32, 32)).save(crops / "img_000" / f"face_{k}.png")
    (crops / "img_001").mkdir(parents=True, exist_ok=True)
    _image(rng, False, True).save(crops / "img_001" / "tile_0.png")  # overrides the auto 2x2 grid
    (crops / "img_003").mkdir(parents=True, exist_ok=True)
    _image(rng, True, False).save(crops / "img_003" / "body_0.jpg")
    return {"root": root, "log": log, "exam": exam, "crops": crops, "images": img_dir, "records": lines}


@pytest.fixture(scope="session")
def ds(tmp_path_factory) -> dict:
    return make_dataset(tmp_path_factory.mktemp("synth"))


def common(ds: dict, cache: Path) -> list[str]:
    return ["--log", str(ds["log"]), "--exam", str(ds["exam"]), "--crops-root", str(ds["crops"]),
            "--cache", str(cache)]
