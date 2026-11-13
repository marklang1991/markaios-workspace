"""Crops per region and the per-(backbone, region) embedding cache.

Crop sources, per image and region:
  1. pre-made crops  <crops_root>/<image_id>/<region>_<k>.(png|jpg|jpeg|webp)
                     (also looked up under the id derived from the image path, '<folder>__<file stem>':
                      the review page's newer records carry an id of their own, older crop folders do not know it)
  2. fallback        full -> the whole image; tile -> a 2x2 grid of the image
  3. otherwise none  (face/hand/body need the owner's pipeline crops; a head may set
                      fallback_region, e.g. body -> full)

Cache file: <cache>/<backbone>/<region>.npz with emb (N,D), image_ids (N,), crop_names (N,),
done_ids (ids already processed, including ones with zero crops) and a JSON meta string.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .backbones import BackboneSpec
from .data import derive_id
from .util import JudgeLabError, now_iso

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
PREP_SIZE = 512  # every crop is letterboxed to a square of this size before the backbone


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def letterbox(img: Image.Image, size: int = PREP_SIZE) -> Image.Image:
    """Pad to square (black) then resize, so processors that center-crop never cut content."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def grid_tiles(img: Image.Image, n: int = 2) -> list[tuple[str, Image.Image]]:
    w, h = img.size
    out = []
    for r in range(n):
        for c in range(n):
            box = (c * w // n, r * h // n, (c + 1) * w // n, (r + 1) * h // n)
            out.append((f"tile_r{r}c{c}", img.crop(box)))
    return out


def premade_crops(crops_root: Path | None, image_id: str, region: str, image_path: Path | None = None) -> list[Path]:
    """Pre-made crops of one image: the folder named by its id, else the folder named by its path-derived id."""
    if not crops_root:
        return []
    pat = re.compile(rf"^{re.escape(region)}_(\d+)\.(png|jpe?g|webp)$", re.IGNORECASE)
    for name in dict.fromkeys([image_id] + ([derive_id(image_path)] if image_path is not None else [])):
        d = Path(crops_root) / name
        if name and d.is_dir():
            found = [(int(m.group(1)), p) for p in d.iterdir() if (m := pat.match(p.name))]
            if found:
                return [p for _, p in sorted(found)]
    return []


def region_crops(image_id: str, image_path: Path | None, region: str, crops_root: Path | None):
    """Return [(crop_name, PIL image)] for one image and region (may be empty)."""
    pre = premade_crops(crops_root, image_id, region, image_path)
    if pre:
        return [(p.name, open_rgb(p)) for p in pre]
    if region not in ("full", "tile"):
        return []
    if image_path is None or not Path(image_path).is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    img = open_rgb(Path(image_path))
    return [("full", img)] if region == "full" else grid_tiles(img)


# --------------------------------------------------------------------------- cache
@dataclass
class RegionCache:
    emb: np.ndarray
    image_ids: np.ndarray
    crop_names: np.ndarray
    done_ids: set[str]
    meta: dict
    _index: dict | None = field(default=None, repr=False)

    @classmethod
    def empty(cls, meta=None) -> "RegionCache":
        return cls(np.zeros((0, 0), np.float32), np.array([], str), np.array([], str), set(), meta or {})

    def rows(self, image_id: str) -> np.ndarray:
        if self._index is None:
            idx: dict[str, list[int]] = {}
            for i, iid in enumerate(self.image_ids.tolist()):
                idx.setdefault(iid, []).append(i)
            self._index = {k: np.array(v) for k, v in idx.items()}
        return self._index.get(image_id, np.array([], dtype=int))

    def append(self, emb: np.ndarray, ids: list[str], names: list[str], done: set[str]) -> None:
        if len(ids):
            self.emb = emb if self.emb.size == 0 else np.concatenate([self.emb, emb])
            self.image_ids = np.concatenate([self.image_ids, np.array(ids, str)])
            self.crop_names = np.concatenate([self.crop_names, np.array(names, str)])
        self.done_ids |= done
        self._index = None


def cache_file(cache_dir: Path, backbone: str, region: str) -> Path:
    return Path(cache_dir) / backbone / f"{region}.npz"


def load_region_cache(path: Path) -> RegionCache:
    if not path.is_file():
        return RegionCache.empty()
    with np.load(path, allow_pickle=False) as z:
        return RegionCache(
            emb=z["emb"].astype(np.float32),
            image_ids=z["image_ids"].astype(str),
            crop_names=z["crop_names"].astype(str),
            done_ids=set(z["done_ids"].astype(str).tolist()),
            meta=json.loads(str(z["meta"])),
        )


def save_region_cache(path: Path, c: RegionCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            emb=c.emb.astype(np.float32),
            image_ids=c.image_ids.astype(str),
            crop_names=c.crop_names.astype(str),
            done_ids=np.array(sorted(c.done_ids), str),
            meta=np.array(json.dumps(c.meta, ensure_ascii=False)),
        )
    os.replace(tmp, path)


def compute_region(spec: BackboneSpec, region: str, items: list[tuple[str, Path | None]], crops_root):
    """Embed all crops of `items` for one region. Returns (emb, ids, names, done_ids, errors),
    errors being (image_id, message) pairs."""
    bs = max(1, spec.batch_size)
    embs, ids, names, done, errors = [], [], [], set(), []
    buf: list[tuple[str, str, Image.Image]] = []

    def flush():
        if buf:
            embs.append(spec.load().embed([letterbox(im) for _, _, im in buf]))
            ids.extend(b[0] for b in buf)
            names.extend(b[1] for b in buf)
            buf.clear()

    for iid, path in items:
        try:
            crops = region_crops(iid, path, region, crops_root)
        except (OSError, FileNotFoundError) as e:
            errors.append((iid, str(e)))
            continue
        buf.extend((iid, n, im) for n, im in crops)
        done.add(iid)
        if len(buf) >= bs:
            flush()
    flush()
    emb = np.concatenate(embs) if embs else np.zeros((0, 0), np.float32)
    return emb, ids, names, done, errors


def ensure_embeddings(
    spec: BackboneSpec,
    regions,
    items: list[tuple[str, Path | None]],
    cache_dir: Path,
    crops_root: Path | None = None,
    force: bool = False,
    hide=(),
) -> dict[str, RegionCache]:
    """Load the cache for each region and embed only the ids not processed yet.
    hide: ids of private records; a failure on one of them is counted, never printed."""
    hide = set(hide)
    out = {}
    for region in sorted(set(regions)):
        path = cache_file(cache_dir, spec.name, region)
        cache = RegionCache.empty() if force else load_region_cache(path)
        cached_mid = cache.meta.get("model_id")
        if cached_mid and cached_mid != spec.model_id:
            raise JudgeLabError(
                f"缓存 {path} 来自模型 {cached_mid}，当前是 {spec.model_id}。请换 --cache 目录或加 --force。"
            )
        missing = [(i, p) for i, p in items if i not in cache.done_ids]
        if missing:
            emb, ids, names, done, errors = compute_region(spec, region, missing, crops_root)
            cache.append(emb, ids, names, done)
            cache.meta = {"backbone": spec.name, "model_id": spec.model_id, "region": region,
                          "dim": int(cache.emb.shape[1]) if cache.emb.size else 0, "updated_at": now_iso()}
            save_region_cache(path, cache)
            shown = [(i, e) for i, e in errors if i not in hide]
            for i, e in shown[:5]:
                print(f"  [embed] 跳过 {i}: {e}")
            if len(shown) < len(errors):
                print(f"  [embed] 另有 {len(errors) - len(shown)} 张私有记录的图没处理成（不显示）")
            print(f"[embed] {spec.name}/{region}: 新处理 {len(done)} 张 → {len(ids)} 个裁剪；"
                  f"已有缓存 {len(items) - len(missing)} 张；失败 {len(errors)} 张")
        else:
            print(f"[embed] {spec.name}/{region}: 全部 {len(items)} 张已缓存")
        out[region] = cache
    return out
