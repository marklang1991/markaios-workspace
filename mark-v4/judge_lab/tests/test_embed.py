import numpy as np
import pytest
from PIL import Image

from judge_lab.backbones import REGISTRY, BackboneSpec, FakeBackbone, parse_model_overrides
from judge_lab.data import load_log
from judge_lab.embeddings import cache_file, ensure_embeddings, grid_tiles, letterbox, load_region_cache, \
    premade_crops, region_crops
from judge_lab.util import JudgeLabError


def test_registry_and_spec(tmp_path):
    assert {"siglip2", "dinov2", "clip_l", "fake"} <= set(REGISTRY)
    assert REGISTRY["siglip2"][1] == "google/siglip2-so400m-patch14-384"
    spec = BackboneSpec.create("siglip2", parse_model_overrides(["siglip2=my/siglip"]), model_dir=tmp_path)
    assert spec.model_id == "my/siglip" and spec.model_path() == "my/siglip"  # nothing loaded
    (tmp_path / "siglip").mkdir()
    assert spec.model_path() == str(tmp_path / "siglip")  # local copy found under --model-dir
    with pytest.raises(JudgeLabError):
        BackboneSpec.create("nope")


def test_hf_backbone_plumbing_with_stub_modules(monkeypatch):
    """Exercise the torch/transformers code path with stand-ins (nothing installed or downloaded)."""
    import contextlib
    import sys
    import types

    class T:  # minimal tensor
        def __init__(self, a):
            self.a = np.asarray(a, np.float32)

        def to(self, *a, **k):
            return self

        def is_floating_point(self):
            return True

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class Proc:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def __call__(self, images, return_tensors):
            return {"pixel_values": T([[i + 1.0, 1.0, 0.0] for i in range(len(images))])}

    class Model:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def to(self, **k):
            return self

        def eval(self):
            return self

        def get_image_features(self, pixel_values):  # newer transformers: an output object, not a tensor
            return types.SimpleNamespace(pooler_output=pixel_values)

        def __call__(self, pixel_values):
            return types.SimpleNamespace(pooler_output=pixel_values)

    fake_torch = types.SimpleNamespace(float16="fp16", float32="fp32", inference_mode=contextlib.nullcontext,
                                       cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoImageProcessor=Proc, AutoModel=Model))
    for name in ("siglip2", "dinov2", "clip_l"):
        bb = BackboneSpec.create(name, batch_size=2).load()
        assert bb.device == "cpu" and bb.dtype == "fp32"  # no cuda -> cpu, fp32
        e = bb.embed([Image.new("RGB", (8, 8))] * 5)
        assert e.shape == (5, 3) and np.allclose(np.linalg.norm(e, axis=1), 1.0)


def test_fake_backbone_is_deterministic_and_normalised():
    imgs = [Image.new("RGB", (40, 30), (i * 40, 10, 200)) for i in range(4)]
    a = FakeBackbone("fake-g16-d64-s0", "", "cpu", 8, False).embed(imgs)
    b = FakeBackbone("fake-g16-d64-s0", "", "cpu", 8, False).embed(imgs)
    assert a.shape == (4, 64) and np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_crops_premade_and_fallback(ds):
    img = ds["images"] / "img_005.png"
    assert [n for n, _ in region_crops("img_005", img, "tile", ds["crops"])] == \
        ["tile_r0c0", "tile_r0c1", "tile_r1c0", "tile_r1c1"]
    assert [n for n, _ in region_crops("img_005", img, "full", ds["crops"])] == ["full"]
    assert region_crops("img_005", img, "face", ds["crops"]) == []
    assert [p.name for p in premade_crops(ds["crops"], "img_000", "face")] == ["face_0.png", "face_1.png"]
    assert [n for n, _ in region_crops("img_001", None, "tile", ds["crops"])] == ["tile_0.png"]  # pre-made wins
    assert [n for n, _ in region_crops("img_003", None, "body", ds["crops"])] == ["body_0.jpg"]
    tiles = grid_tiles(Image.new("RGB", (48, 64)))
    assert all(t.size == (24, 32) for _, t in tiles)
    assert letterbox(Image.new("RGB", (48, 64)), 100).size == (100, 100)


def test_embed_cache_skips_cached(ds, tmp_path):
    reviews = load_log(ds["log"])
    items = [(r.id, r.image) for r in reviews]
    calls = []

    class Counting(FakeBackbone):
        def embed(self, images):
            calls.append(len(images))
            return super().embed(images)

    REGISTRY["fake_count"] = (Counting, "fake-g16-d64-s0")
    try:
        spec = BackboneSpec.create("fake_count", batch_size=32)
        caches = ensure_embeddings(spec, ["full", "tile", "face"], items[:50], tmp_path, ds["crops"])
        assert caches["full"].emb.shape == (50, 64)
        assert len(caches["tile"].image_ids) == 49 * 4 + 1  # img_001 has one pre-made tile
        assert len(caches["face"].image_ids) == 2 and set(caches["face"].image_ids) == {"img_000"}
        assert caches["face"].done_ids == {i for i, _ in items[:50]}  # zero-crop ids remembered
        first = sum(calls)

        caches = ensure_embeddings(spec, ["full", "tile", "face"], items[:50], tmp_path, ds["crops"])
        assert sum(calls) == first  # fully cached: backbone never called

        caches = ensure_embeddings(spec, ["full"], items[:55], tmp_path, ds["crops"])
        assert sum(calls) == first + 5 and caches["full"].emb.shape == (55, 64)
        on_disk = load_region_cache(cache_file(tmp_path, "fake_count", "full"))
        assert len(on_disk.image_ids) == 55 and on_disk.meta["model_id"] == "fake-g16-d64-s0"
        assert len(caches["full"].rows("img_054")) == 1

        other = BackboneSpec.create("fake_count", model_id="fake-g8-d32-s1")
        with pytest.raises(JudgeLabError):  # cache built by a different model id
            ensure_embeddings(other, ["full"], items[:55], tmp_path, ds["crops"])
    finally:
        REGISTRY.pop("fake_count")
