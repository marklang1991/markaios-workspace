"""Pluggable frozen image backbones. Heavy deps (torch/transformers) are imported lazily.

Every backbone maps a list of RGB PIL images to an (N, D) float32 array of
L2-normalised embeddings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .util import JudgeLabError


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Backbone:
    def __init__(self, model_id: str, model_path: str, device: str, batch_size: int, fp16: bool):
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16

    def embed(self, images: list) -> np.ndarray:
        raise NotImplementedError


class FakeBackbone(Backbone):
    """Deterministic random projection of downsampled pixels. numpy only, for tests.
    model_id format: fake-g<grid>-d<dim>-s<seed> (default fake-g16-d64-s0)."""

    def __init__(self, model_id, model_path, device, batch_size, fp16):
        super().__init__(model_id, model_path, "cpu", batch_size, False)
        m = re.fullmatch(r"fake-g(\d+)-d(\d+)-s(\d+)", model_id)
        self.grid, dim, seed = (int(g) for g in m.groups()) if m else (16, 64, 0)
        rng = np.random.default_rng(seed)
        n_in = self.grid * self.grid * 3
        self.proj = (rng.standard_normal((n_in, dim)) / np.sqrt(n_in)).astype(np.float32)

    def embed(self, images):
        from PIL import Image

        px = [
            np.asarray(im.convert("RGB").resize((self.grid, self.grid), Image.BILINEAR), np.float32).ravel()
            for im in images
        ]
        x = np.stack(px) / 255.0 - 0.5
        return l2_normalize(x @ self.proj)


class _HFBackbone(Backbone):
    """Shared HF transformers plumbing: lazy import, fp16 on cuda, batched inference."""

    def __init__(self, model_id, model_path, device, batch_size, fp16):
        super().__init__(model_id, model_path, device, batch_size, fp16)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:  # pragma: no cover - depends on optional deps
            raise JudgeLabError(
                "真实骨干需要 torch + transformers: pip install -r judge_lab/requirements-backbones.txt"
            ) from e
        self.torch = torch
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(getattr(getattr(torch, "backends", None), "mps", None), "is_available", lambda: False)():
                device = "mps"  # Apple silicon (Mac Studio)
            else:
                device = "cpu"
        self.device = device
        self.dtype = torch.float16 if (fp16 and (device.startswith("cuda") or device == "mps")) else torch.float32
        # AutoImageProcessor = the image half of AutoProcessor (no tokenizer needed).
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device=self.device, dtype=self.dtype).eval()

    @staticmethod
    def _tensor(out):
        """get_image_features returns a tensor in transformers 4.x, an output object in newer versions."""
        if hasattr(out, "detach"):
            return out
        for attr in ("image_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        return out[0]

    def _features(self, inputs):
        return self._tensor(self.model.get_image_features(**inputs))

    def embed(self, images):
        torch = self.torch
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                inputs = self.processor(images=images[i : i + self.batch_size], return_tensors="pt")
                inputs = {
                    k: v.to(self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device)
                    for k, v in inputs.items()
                    if hasattr(v, "to")
                }
                chunks.append(self._features(inputs).float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))


class Siglip2Backbone(_HFBackbone):
    pass  # image features via get_image_features


class ClipLBackbone(_HFBackbone):
    pass  # projected image embeds via get_image_features


class Dinov2Backbone(_HFBackbone):
    def _features(self, inputs):
        return self.model(**inputs).pooler_output  # layer-normed CLS token


REGISTRY: dict[str, tuple[Callable[..., Backbone], str]] = {
    "siglip2": (Siglip2Backbone, "google/siglip2-so400m-patch14-384"),
    "dinov2": (Dinov2Backbone, "facebook/dinov2-large"),
    "clip_l": (ClipLBackbone, "openai/clip-vit-large-patch14"),
    "fake": (FakeBackbone, "fake-g16-d64-s0"),
}


def register_backbone(name: str, cls: Callable[..., Backbone], default_model_id: str) -> None:
    REGISTRY[name] = (cls, default_model_id)


def parse_model_overrides(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        name, sep, mid = p.partition("=")
        if not sep:
            raise JudgeLabError(f"--model-id 需要 NAME=ID 格式，收到 {p!r}")
        out[name.strip()] = mid.strip()
    return out


@dataclass
class BackboneSpec:
    """A backbone choice that is cheap to create; the model loads on first .load()."""

    name: str
    model_id: str
    model_dir: Path | None = None
    device: str = "auto"
    batch_size: int = 16
    fp16: bool = True
    _instance: Backbone | None = field(default=None, repr=False)

    @classmethod
    def create(cls, name, overrides=None, model_dir=None, device="auto", batch_size=16, fp16=True, model_id=None):
        if name not in REGISTRY:
            raise JudgeLabError(f"未知骨干 {name!r}，可用: {sorted(REGISTRY)}")
        mid = model_id or (overrides or {}).get(name) or REGISTRY[name][1]
        return cls(name, mid, Path(model_dir) if model_dir else None, device, batch_size, fp16)

    def model_path(self) -> str:
        """Local copy under --model-dir if present (by full id or by last path part), else the hub id."""
        if self.model_dir:
            for cand in (self.model_dir / self.model_id, self.model_dir / self.model_id.split("/")[-1]):
                if cand.is_dir():
                    return str(cand)
        return self.model_id

    def load(self) -> Backbone:
        if self._instance is None:
            cls = REGISTRY[self.name][0]
            print(f"[backbone] 加载 {self.name} ({self.model_path()}) ...")
            self._instance = cls(self.model_id, self.model_path(), self.device, self.batch_size, self.fp16)
        return self._instance
