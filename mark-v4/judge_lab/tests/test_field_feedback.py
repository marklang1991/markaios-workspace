"""Field feedback from the first real batch: exam registry by sha256, 推荐进入, several logs, a private log,
source filters, private reasons, both shortfalls in `items list`, the prompt export, and verify / trace additions.

The fixture writes 16 scenes x 5 seeds of made-up pictures and records in the review page's current format
(id = first 16 hex of the file's sha256, sha256, source, batch_pass, recommend, note, seconds, reasons[].carried)."""
import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from judge_lab import data as data_mod
from judge_lab.backbones import BackboneSpec
from judge_lab.cli import main
from judge_lab.data import (
    ExamRegistry, canonicalize, derive_id, exam_guard, file_sha256, filter_sources, load_config, load_exam, load_logs,
    masked_name, private_id, private_orphans, split_exam,
)
from judge_lab.embeddings import ensure_embeddings
from judge_lab.heads import build_item_data, load_head, save_head
from judge_lab.util import ExamLeakError, JudgeLabError, read_json

W, H, SEEDS, N = 48, 64, 5, 80
STRUCT, LIGHT, STORY = "结构断裂", "光线不自然", "动作和分镜不符"
SECRET, SECRET_OLD, SECRET_DEF = "机密缺陷名", "机密旧名", "机密定义文字"
LOW = ["--min-pos", "8", "--min-neg", "8"]


def flags(i):
    return i % 3 == 0, i % 4 == 1, i % 5 == 2  # struct, light, secret


def defects(**struct_over):
    return {"version": 7, "items": [
        {"name": STRUCT, "group": "结构", "kind": "hard", "definition": "直线断开或错位", **struct_over},
        {"name": LIGHT, "group": "光线", "kind": "score"},
        {"name": SECRET, "group": "其他", "kind": "hard", "private": True, "definition": SECRET_DEF,
         "merged_from": [SECRET_OLD]},
        {"name": STORY, "group": "动作与分镜", "kind": "score", "definition": "动作和分镜脚本写的不一样"},
    ]}


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab")
    batch = root / "批次B"
    batch.mkdir()
    rng = np.random.default_rng(5)
    files, shas = [], []
    for i in range(N):
        st, li, se = flags(i)
        a = rng.integers(0, 110, size=(H, W, 3)).astype(np.int32)
        if li:
            a[..., 0] += 140
        if se:
            a[H // 2:, :, 1] += 140
        if st:
            x, y = rng.integers(0, W - 10), rng.integers(0, H - 10)
            a[y:y + 10, x:x + 10] = 255
        f = batch / f"scene{i // SEEDS:02d}_s{i % SEEDS}.png"
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(f)
        files.append(f)
        shas.append(hashlib.sha256(f.read_bytes()).hexdigest())
    cfg = root / "缺陷清单.json"
    cfg.write_text(json.dumps(defects(), ensure_ascii=False), encoding="utf-8")
    return {"root": root, "batch": batch, "files": files, "shas": shas, "cfg": cfg}


def record(lab, i, new=True, **over):
    """One review as the page writes it; new=False: an old record (no id, no sha256, no carried flags)."""
    st, li, se = flags(i)
    picks = ([(STRUCT, "hard", None)] if st else []) + ([(LIGHT, "score", [1, 4, 8][i % 3])] if li else []) + \
            ([(SECRET_OLD if i % 2 else SECRET, "hard", None)] if se else [])
    reasons = [{"item": n, "kind": k, "score": s, **({"carried": False} if new else {})} for n, k, s in picks]
    rec = {"time": "2026-11-12 17:14:06", "image": str(lab["files"][i]), "recipe": "Moss·b68a4b9f",
           "decision": "不过" if reasons else "过", "reasons": reasons, "note": "", "seconds": None}
    if new:
        rec.update(id=lab["shas"][i][:16], sha256=lab["shas"][i], source="新图", batch_pass=False, recommend=False,
                   seconds=4 + i % 5)
    rec.update(over)
    return rec


def write_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def text_outputs(folder):
    """Every text file the tools wrote under `folder`."""
    return {str(p): p.read_text(encoding="utf-8") for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix in (".md", ".json", ".jsonl", ".txt")}


def console(capsys):
    c = capsys.readouterr()
    return c.out + c.err


# --------------------------------------------------------------------------- exam registry
def test_exam_registry_forms(lab, tmp_path):
    files, shas = lab["files"], lab["shas"]
    old = write_log(tmp_path / "old.jsonl", [record(lab, i, new=False) for i in range(8)])  # hash computed from file
    new = write_log(tmp_path / "new.jsonl", [record(lab, i) for i in range(8)])  # hash is a record field
    entries = [shas[1], {"id": shas[2][:16]}, {"image": str(files[3])}, {"sha256": shas[4].upper(), "why": "holdout"},
               {"sha256": "f" * 64}]  # the last one is no picture of the log
    forms = {
        "list.txt": "\n".join(["# 考卷", shas[1], shas[2][:16].upper() + "   # 16 位 id", "",
                               str(files[3]).replace("/", "\\"), shas[4], "f" * 64]),
        "list.json": json.dumps(entries),
        "items.json": json.dumps({"version": 2, "id": "exam_v2", "items": entries}),
        "exam.json": json.dumps({"exam": entries}, indent=2),
        "list.jsonl": "\n".join(json.dumps(e if isinstance(e, dict) else {"sha256": e}) for e in entries),
    }
    for name, text in forms.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        for log in (old, new):
            reviews, _ = load_logs([log])
            reg = load_exam(path)
            assert len(reg.entries) == 5 and reg.hashes, name
            train, exam, summary = split_exam(reviews, reg)
            assert sorted(reviews.index(r) for r in exam) == [1, 2, 3, 4], (name, log.name)
            assert summary == {"exam_listed": 5, "exam_in_log": 4, "exam_not_in_log": 1, "train_pool": 4,
                               "total_reviews": 8}, (name, log.name)
            # the ids of the matched reviews are protected, though the registry never spells most of them
            guard = exam_guard(reg, exam)
            assert {r.id for r in exam} <= guard and not {r.id for r in train} & guard
    # a record's sha256 field matches even when the picture file is gone
    moved = write_log(tmp_path / "moved.jsonl", [record(lab, i, image=f"gone/scene_s{i}.png") for i in range(8)])
    _, exam, _ = split_exam(load_logs([moved])[0], load_exam(tmp_path / "list.jsonl"))
    assert [r.sha256 for r in exam] == [shas[1], shas[2], shas[4]]  # entry 3 is listed by path, and the path changed
    # the plain id list still works, and is still a set of ids to old callers
    (tmp_path / "ids.txt").write_text(f"{derive_id(files[0])}\n# 注释\n{derive_id(files[5])}\n", encoding="utf-8")
    reg = load_exam(tmp_path / "ids.txt")
    assert reg == {derive_id(files[0]), derive_id(files[5])} and not reg.hashes
    assert [r.id for r in split_exam(load_logs([old])[0], reg)[1]] == [derive_id(files[0]), derive_id(files[5])]
    assert load_exam(None) == set() and split_exam(load_logs([old])[0], set())[2]["exam_listed"] == 0
    # an entry that cannot be read is an error, never skipped
    for bad in ('[{"note": "no key"}]', '{"pictures": []}', '[broken', '{"sha256": "abc"}\n{broken', '[{"sha256": "xyz"}]',
                '[["nested"]]'):
        (tmp_path / "bad.json").write_text(bad, encoding="utf-8")
        with pytest.raises(JudgeLabError):
            load_exam(tmp_path / "bad.json")


def test_hashes_are_computed_lazily_and_cached(lab, tmp_path, monkeypatch):
    files, shas = lab["files"], lab["shas"]
    old = write_log(tmp_path / "old.jsonl", [record(lab, i, new=False) for i in range(6)])
    calls = []
    real = data_mod.file_sha256
    monkeypatch.setattr(data_mod, "file_sha256", lambda p: calls.append(p) or real(p))
    # no hash in the registry -> no picture file is read
    split_exam(load_logs([old])[0], ExamRegistry([{derive_id(files[0])}]))
    assert calls == []
    # hashes in the registry -> only the pictures that nothing cheaper matched are read
    reviews = load_logs([old])[0]
    _, exam, _ = split_exam(reviews, ExamRegistry([{derive_id(files[0])}, {shas[2]}]))
    assert [reviews.index(r) for r in exam] == [0, 2] and len(calls) == 5 and files[0] not in calls
    assert reviews[2].sha256 == shas[2] and reviews[0].sha256 is None
    # cached per (path, mtime, size): a changed file is hashed again
    f = tmp_path / "pic.bin"
    f.write_bytes(b"one")
    first = file_sha256(f)
    assert first == hashlib.sha256(b"one").hexdigest() and file_sha256(f) == first
    assert sum(1 for k in data_mod._HASH_CACHE if k[0] == str(f)) == 1
    f.write_bytes(b"other bytes")
    assert file_sha256(f) == hashlib.sha256(b"other bytes").hexdigest()
    assert file_sha256(tmp_path / "missing.png") is None and file_sha256(None) is None


def test_hash_matched_exam_picture_cannot_be_trained_on(lab, tmp_path, capsys):
    files, shas = lab["files"], lab["shas"]
    log = write_log(tmp_path / "log.jsonl", [record(lab, i, new=False) for i in range(N)])  # old records: no sha256
    exam_i = [0, 3, 5, 9, 12, 21, 22, 30, 41, 57]
    exam = tmp_path / "exam.json"
    exam.write_text(json.dumps({"items": [{"sha256": shas[i]} for i in exam_i[:5]] + [shas[i][:16] for i in exam_i[5:8]]
                                + [{"image": str(files[41])}, {"id": derive_id(files[57])}, {"sha256": "0" * 64}]}),
                    encoding="utf-8")
    exam_ids = {derive_id(files[i]) for i in exam_i}
    reviews, _ = load_logs([log])
    items, _ = load_config(lab["cfg"])
    canonicalize(reviews, items)
    reg = load_exam(exam)
    train, ex, summary = split_exam(reviews, reg)
    assert {r.id for r in ex} == exam_ids and summary["exam_listed"] == 11 and summary["exam_not_in_log"] == 1
    assert not exam_ids & {t for t in reg if t in {derive_id(f) for f in files[:40]}}  # matched by hash, not by id
    caches = ensure_embeddings(BackboneSpec.create("fake"), ["tile", "full"], [(r.id, r.image) for r in reviews],
                               tmp_path / "cache", None)
    struct = next(it for it in items.values() if it.name == STRUCT)
    with pytest.raises(ExamLeakError):  # the guard knows the ids of the hash-matched pictures
        build_item_data(struct, reviews, caches, exclude_ids=exam_guard(reg, ex))
    with pytest.raises(ExamLeakError):
        build_item_data(struct, train + [ex[0]], caches, exclude_ids=exam_guard(reg, ex))
    build_item_data(struct, train, caches, exclude_ids=exam_guard(reg, ex))

    base = ["--log", str(log), "--config", str(lab["cfg"]), "--exam", str(exam), "--cache", str(tmp_path / "cache")]
    heads = tmp_path / "heads"
    assert main(["train", *base, *LOW, "--backbone", "fake", "--out", str(heads)]) == 0
    assert "考卷名单 11 个，日志中命中 10 个（永不训练）；训练池 70 张" in console(capsys)
    trained = [load_head(p) for p in heads.glob("*.npz")]
    assert len(trained) == 3
    for h in trained:
        assert not set(h.train_ids) & exam_ids and len(h.train_ids) > 30
    assert main(["exam", *base, "--heads", str(heads), "--out", str(tmp_path / "exam_report.md")]) == 0
    res = read_json(tmp_path / "exam_report.json")
    assert res["exam"]["exam_in_log"] == 10 and res["items"][struct.id]["n_pos"] == sum(1 for i in exam_i if i % 3 == 0)
    # a head that did see a hash-matched exam picture is refused
    h = next(x for x in trained if x.item == struct.id)
    save_head(tmp_path / "dirty", struct, h.coef, h.intercept, h.train_ids + [derive_id(files[3])], h.meta)
    capsys.readouterr()
    assert main(["exam", *base, "--heads", str(tmp_path / "dirty"), "--out", str(tmp_path / "e2.md")]) == 2
    assert "考卷" in console(capsys) and not (tmp_path / "e2.md").exists()


def test_recommend_and_exam_conflict_is_never_settled_silently(lab, tmp_path, capsys):
    shas = lab["shas"]
    rows = [record(lab, i) for i in range(N)]
    rows[10]["recommend"] = True  # fine: not an exam picture
    rows[20]["recommend"] = True  # conflict: also on the exam list
    log = write_log(tmp_path / "log.jsonl", rows)
    exam = tmp_path / "exam.txt"
    exam.write_text("\n".join(shas[i] for i in (20, 25, 26)), encoding="utf-8")
    base = ["--log", str(log), "--config", str(lab["cfg"]), "--exam", str(exam)]
    cache = ["--cache", str(tmp_path / "cache")]
    for cmd in (["train", "--backbone", "fake", "--out", str(tmp_path / "heads")],
                ["probe", "--backbones", "fake", "--out", str(tmp_path / "probe.md")],
                ["calibrate", "--heads", str(tmp_path / "heads")]):
        capsys.readouterr()
        assert main([cmd[0], *base, *cache, *LOW, *cmd[1:]]) == 2
        err = capsys.readouterr().err
        assert "推荐进入" in err and "考卷" in err and shas[20][:16] in err
    assert not (tmp_path / "heads").exists() and not (tmp_path / "probe.md").exists()
    assert main(["verify", *base, "--out", str(tmp_path / "verify.md")]) == 1
    md = (tmp_path / "verify.md").read_text(encoding="utf-8")
    line = next(l for l in md.splitlines() if l.startswith("| 断 | 推荐进入 |"))
    assert "1 张图既标了「推荐进入」（必须进训练）又在考卷名单里（永不训练）" in line and shas[20][:16] in line
    # take it off the exam list (or un-recommend it) and everything runs again
    exam.write_text("\n".join(shas[i] for i in (25, 26)), encoding="utf-8")
    assert main(["verify", *base, "--out", str(tmp_path / "verify.md")]) == 0
    assert "| 通 | 推荐进入 | 2 张图标了「推荐进入」" in (tmp_path / "verify.md").read_text(encoding="utf-8")
    assert main(["train", *base, *cache, *LOW, "--backbone", "fake", "--out", str(tmp_path / "heads")]) == 0


# --------------------------------------------------------------------------- several logs, sources
def test_several_logs_the_later_record_wins(tmp_path, capsys):
    log1 = write_log(tmp_path / "a" / "审片记录.jsonl", [
        {"image": "D:\\MarkV4\\批次A\\hall_s1.png", "decision": "不过", "reasons": ["x"]},
        {"image": "D:\\MarkV4\\批次A\\hall_s2.png", "decision": "过"},
        {"id": "keep", "decision": "过"},
        {"id": "twice", "decision": "过"}, {"id": "twice", "decision": "不过"},  # re-review inside one file
    ])
    log2 = write_log(tmp_path / "b" / "补审.jsonl", [
        {"id": "abcdabcdabcdabcd", "image": "d:/markv4/批次A/HALL_S1.PNG", "decision": "过"},  # same picture, new id
        {"id": "keep", "decision": "不过"},  # same id
        {"image": "D:\\MarkV4\\批次A\\hall_s3.png", "decision": "过"},
    ])
    reviews, st = load_logs([log1, log2])
    got = {r.id: r.decision for r in reviews}
    assert got == {"abcdabcdabcdabcd": "pass", "批次A__hall_s2": "pass", "keep": "fail", "twice": "fail",
                   "批次A__hall_s3": "pass"}
    assert len(reviews) == len(got)  # ids are unique
    assert (st["files"], st["lines"], st["parsed"], st["re_reviews"], st["skipped"]) == (2, 8, 8, 3, [])
    # the other order: the first file's records are now the later ones
    got = {r.id: r.decision for r in load_logs([log2, log1])[0]}
    assert got["批次A__hall_s1"] == "fail" and "abcdabcdabcdabcd" not in got and got["keep"] == "pass"
    # one new record can replace two old ones (it shares its id with one and its path with the other)
    log3 = write_log(tmp_path / "c.jsonl", [{"id": "keep", "image": "D:/MarkV4/批次A/hall_s2.png", "decision": "不过"}])
    reviews, st = load_logs([log1, log3])
    assert sorted(r.id for r in reviews) == ["keep", "twice", "批次A__hall_s1"] and st["re_reviews"] == 2
    # every command takes several --log files
    cfg = tmp_path / "heads.json"
    cfg.write_text(json.dumps({"items": {"x": {"name": "甲", "kind": "hard", "region": "full"}}}), encoding="utf-8")
    assert main(["items", "list", "--log", str(log1), str(log2), "--config", str(cfg)]) == 0
    assert "训练池 5 张" in console(capsys)
    # (exit code 1: these made-up pictures are not on disk, which verify rightly calls a break)
    assert main(["verify", "--log", str(log1), str(log2), "--config", str(cfg), "--out", str(tmp_path / "v.md")]) == 1
    md = (tmp_path / "v.md").read_text(encoding="utf-8")
    assert "| 通 | 审片记录读入 | 2 个文件，8 行全部读入，共 5 张图（过 3 / 不过 2；重审 3 次，以最后一次为准） |" in md
    assert "| 断 | 图片文件 | 5 张图一张都找不到" in md


def test_source_filters_never_drop_a_recommended_picture(tmp_path, capsys):
    rows = [{"id": "n1", "decision": "过", "source": "新图"}, {"id": "n2", "decision": "不过", "source": "新图"},
            {"id": "o1", "decision": "过", "source": "旧图补审"},
            {"id": "o2", "decision": "过", "source": "旧图补审", "recommend": True},
            {"id": "t1", "decision": "过", "source": "测试"}, {"id": "u1", "decision": "过"}]
    log = write_log(tmp_path / "log.jsonl", rows)
    reviews, _ = load_logs([log])

    def ids(keep=None, drop=None):
        return [r.id for r in filter_sources(reviews, keep, drop)[0]]

    assert ids() == ["n1", "n2", "o1", "o2", "t1", "u1"]
    assert ids(keep=["新图"]) == ["n1", "n2", "o2"]  # o2: 推荐进入, kept although its source is not wanted
    assert ids(keep=["新图", "测试"]) == ["n1", "n2", "o2", "t1"]
    assert ids(drop=["旧图补审"]) == ["n1", "n2", "o2", "t1", "u1"]  # a record without a source is not "excluded"
    assert ids(drop=["旧图补审", "测试"]) == ["n1", "n2", "o2", "u1"]
    assert filter_sources(reviews, ["新图"], None)[1] == {"dropped": 3, "kept_by_recommend": 1}
    cfg = tmp_path / "heads.json"
    cfg.write_text(json.dumps({"items": {"x": {"name": "甲", "kind": "hard", "region": "full"}}}), encoding="utf-8")
    base = ["--log", str(log), "--config", str(cfg)]
    assert main(["items", "list", *base, "--source", "新图"]) == 0
    assert "训练池 3 张" in console(capsys)
    assert main(["items", "list", *base, "--exclude-source", "旧图补审,测试"]) == 0
    assert "训练池 4 张" in console(capsys)
    # (exit code 1 only because this made-up log has no pictures on disk)
    assert main(["verify", *base, "--exclude-source", "旧图补审", "--out", str(tmp_path / "v.md")]) == 1
    md = (tmp_path / "v.md").read_text(encoding="utf-8")
    assert "| 通 | 来源筛选 | 按 --source / --exclude-source 去掉 1 张；1 张「推荐进入」的图不受筛选影响，保留 |" in md
    assert "按来源筛选后用 5 张" in md


# --------------------------------------------------------------------------- private log
def test_private_log_is_counted_but_never_shown(lab, tmp_path, capsys):
    album = tmp_path / "in" / "私人相册XYZ"
    album.mkdir(parents=True)
    private_rows = []
    for k, i in enumerate((60, 61, 62, 63, 64, 65)):  # six pictures of the lab, under private names
        f = album / f"private_shot_s{k}.png"
        shutil.copy(lab["files"][i], f)
        private_rows.append(record(lab, i, new=False, image=str(f), id=f"PRIVID{k:04d}", note="秘密备注XYZ",
                                   source="私人"))
    private_rows[1]["reasons"] = private_rows[1]["reasons"] + [{"item": "私密原因ABC", "kind": "hard", "score": None}]
    private_rows.append({"id": "PRIVID9999", "image": str(album / "missing_shot_s9.png"), "decision": "过",
                         "note": "秘密备注XYZ"})  # its picture is gone
    private_rows.append({"id": "PRIVID8888", "image": str(album / "private_shot_s8.png"), "decision": "保密决定QQ"})
    public = write_log(tmp_path / "in" / "log.jsonl", [record(lab, i) for i in range(60)])
    private = write_log(tmp_path / "in" / "私有记录_SECRETLOG.jsonl", private_rows)
    secrets = ["私人相册XYZ", "private_shot", "missing_shot", "PRIVID", "秘密备注XYZ", "私密原因ABC", "保密决定QQ",
               "SECRETLOG"]
    out = tmp_path / "out"
    (tmp_path / "in" / "exam.txt").write_text(lab["shas"][0] + "\n", encoding="utf-8")
    base = ["--log", str(public), "--private-log", str(private), "--config", str(lab["cfg"]),
            "--exam", str(tmp_path / "in" / "exam.txt")]
    cache = ["--cache", str(tmp_path / "cache")]
    capsys.readouterr()
    # verify: the private orphan click and the unreadable private line break the chain, and are only counted
    assert main(["verify", *base, *cache, *LOW, "--signal", "fake", "--out", str(out / "verify.md")]) == 1
    assert main(["probe", *base, *cache, *LOW, "--backbones", "fake", "--out", str(out / "probe.md")]) == 0
    assert main(["train", *base, *cache, *LOW, "--backbone", "fake", "--out", str(out / "heads")]) == 0
    assert main(["calibrate", *base, *cache, *LOW, "--heads", str(out / "heads")]) == 0
    assert main(["exam", *base, *cache, "--heads", str(out / "heads"), "--out", str(out / "exam.md")]) == 0
    assert main(["embed", *base, "--backbone", "fake", "--out", str(tmp_path / "cache2")]) == 0
    assert main(["items", "list", *base]) == 0
    assert main(["trace", *base, "--heads", str(out / "heads"), "--last", "500"]) == 0
    assert main(["trace", *base, "--image", "PRIVID0001"]) == 0
    assert main(["export-prompts", *base, "--examples", "50", "--out", str(out / "prompts.json")]) == 0
    assert main(["export-prompts", *base, "--examples", "50", "--show-private", "--out", str(out / "prompts_all.json")]) == 0
    assert main(["stats", "ab", "--log", str(public), "--private-log", str(private), "--group-by", "recipe_hash",
                 "--out", str(out / "ab.md")]) == 0
    shown = console(capsys)
    files = text_outputs(out)
    assert len(files) >= 9 and "verify.md" in " ".join(files)
    for s in secrets:
        assert s not in shown, s
        for name, text in files.items():
            assert s not in text, (s, name)
    # ... but they are there: counted, trained on, and their problems reported as numbers
    md = files[str(out / "verify.md")]
    assert "（另有 1 个私有记录文件，其中 7 张图：只计数不显示）" in md
    assert "私有记录里 1 行（内容不显示）" in md and "私有记录里 1 次（内容不显示）" in md
    assert "1 / 67 张图找不到（私有记录，不显示）" in md
    assert "日志 67 张（其中私有记录 7 张，只计数不显示）" in shown
    assert "另有 1 张私有记录的图没处理成（不显示）" in shown
    assert shown.count("=== （私有记录：内容不显示；要看就加 --show-private）") == 7 + 1
    assert "（私有记录）" in files[str(out / "ab.md")]  # their scene is pooled, not named
    trained = {i for p in (out / "heads").glob("*.npz") for i in load_head(p).train_ids}
    assert {f"PRIVID{k:04d}" for k in range(6)} <= trained  # private pictures do teach the heads
    # example pictures for a prompt never come from private records, whatever the flags
    for name in ("prompts.json", "prompts_all.json"):
        for it in read_json(out / name)["items"]:
            assert all(x.startswith("scene") for x in it["positive_examples"] + it["negative_examples"])
    # asked for explicitly, the owner can look at a private record
    assert main(["trace", *base, "--image", "PRIVID0001", "--show-private"]) == 0
    shown = console(capsys)
    assert "=== PRIVID0001" in shown and "秘密备注XYZ" in shown and "私密原因ABC" in shown and "对不上清单" in shown
    reviews, st = load_logs([public], private_paths=[private])
    assert sum(r.private for r in reviews) == st["private"] == 7 and st["skipped_private"] == 1 and st["skipped"] == []
    items, _ = load_config(lab["cfg"])
    assert canonicalize(reviews, items) == {} and private_orphans(reviews, items) == 1


def test_exam_leak_error_never_prints_a_private_id():
    from judge_lab.util import assert_no_exam
    with pytest.raises(ExamLeakError) as e:
        assert_no_exam(["a", "PRIVID0001", "b"], {"a", "PRIVID0001"}, "train x", hide={"PRIVID0001"})
    assert "2 个考卷 id" in str(e.value) and "'a'" in str(e.value) and "PRIVID" not in str(e.value)
    with pytest.raises(ExamLeakError) as e:
        assert_no_exam(["PRIVID0001"], {"PRIVID0001"}, "train x", hide={"PRIVID0001"})
    assert "1 个考卷 id" in str(e.value) and "PRIVID" not in str(e.value)
    assert_no_exam(["a"], {"b"}, "train x")


# --------------------------------------------------------------------------- private reasons
def test_private_reason_is_masked_everywhere(lab, tmp_path, capsys):
    log = write_log(tmp_path / "in" / "log.jsonl", [record(lab, i) for i in range(N)])
    exam = tmp_path / "in" / "exam.txt"
    exam.write_text("\n".join(lab["shas"][70:]), encoding="utf-8")
    out, heads = tmp_path / "out", tmp_path / "out" / "heads"
    base = ["--log", str(log), "--config", str(lab["cfg"]), "--exam", str(exam)]
    cache = ["--cache", str(tmp_path / "cache")]
    pid = private_id(SECRET)
    alias = masked_name(pid)
    assert pid.startswith("p_") and len(pid) == 12 and alias == f"私有项 {pid[-4:]}"
    capsys.readouterr()
    assert main(["verify", *base, *cache, *LOW, "--signal", "fake", "--out", str(out / "verify.md")]) == 0
    assert main(["probe", *base, *cache, *LOW, "--backbones", "fake", "--out", str(out / "probe.md")]) == 0
    assert main(["train", *base, *cache, *LOW, "--backbone", "fake", "--out", str(heads)]) == 0
    assert main(["calibrate", *base, *cache, *LOW, "--heads", str(heads)]) == 0
    assert main(["exam", *base, *cache, "--heads", str(heads), "--out", str(out / "exam.md")]) == 0
    assert main(["verify", *base, "--heads", str(heads), "--out", str(out / "verify_heads.md")]) == 0
    assert main(["export-prompts", *base, "--out", str(out / "prompts.json")]) == 0
    assert main(["items", "list", *base]) == 0
    assert main(["trace", *base, "--heads", str(heads), "--image", "scene00_s2", "--all"]) == 0  # i=2: the private reason
    assert main(["score", "--images", str(lab["batch"]), "--heads", str(heads), "--out", str(out / "shadow.jsonl")]) == 0
    shown = console(capsys)
    files = text_outputs(out)
    assert {p.split("/")[-1] for p in files} >= {"verify.md", "probe.md", "probe.json", "manifest.json", "thresholds.json",
                                                "exam.md", "exam.json", "prompts.json", "prompts.md", "shadow.jsonl"}
    for s in (SECRET, SECRET_OLD, SECRET_DEF):
        assert s not in shown, s
        for name, text in files.items():
            assert s not in text, (s, name)
        for p in heads.glob("*.npz"):  # the head files themselves
            raw = p.read_bytes()
            assert not any(s.encode(enc) in raw for enc in ("utf-8", "utf-16-le", "utf-32-le")), (s, p.name)
            assert s not in json.dumps(load_head(p).meta, ensure_ascii=False)
    # the reason is there under its opaque id, and the clicks made under both of its names reached it
    manifest = read_json(heads / "manifest.json")
    n_pos = sum(1 for i in range(70) if flags(i)[2])
    assert manifest["items"][pid]["active"]["name"] == alias and manifest["items"][pid]["active"]["private"] is True
    assert manifest["items"][pid]["active"]["n_pos"] == n_pos == 14
    h = load_head(heads / f"{pid}_v1.npz")
    assert h.meta["name"] == alias and h.meta["private"] is True and h.item == pid
    assert read_json(heads / "thresholds.json")["items"][pid]["name"] == alias
    assert f"| {alias} | 硬门 | full | {n_pos} |" in files[str(out / "verify.md")]
    assert f"| {alias} | 14 |" in files[str(out / "verify.md")] and alias in shown and pid in shown
    assert read_json(out / "probe.json")["items"][pid]["name"] == alias
    shadow = [json.loads(l) for l in files[str(out / "shadow.jsonl")].splitlines()]
    assert all(r["scores"][pid]["name"] == alias for r in shadow)
    prompts = read_json(out / "prompts.json")
    assert pid not in {it["id"] for it in prompts["items"]} and prompts["private_items_not_exported"] == 1
    assert "含义" in files[str(out / "verify.md")] and f"还没写的: {LIGHT}" in files[str(out / "verify.md")]  # it has a definition

    # --show-private: the real name is used in what is shown ...
    assert main(["verify", *base, "--heads", str(heads), "--show-private", "--out", str(out / "verify_real.md")]) == 0
    shown = console(capsys)
    real = (out / "verify_real.md").read_text(encoding="utf-8")
    assert f"| {SECRET} | 硬门 | full | {n_pos} |" in real and alias not in real
    assert "判官头 v1（学了" in next(l for l in real.splitlines() if l.startswith(f"| {SECRET} |"))  # same head, same id
    assert main(["export-prompts", *base, "--show-private", "--out", str(out / "prompts_real.json")]) == 0
    entry = next(it for it in read_json(out / "prompts_real.json")["items"] if it["id"] == pid)
    assert entry["name"] == SECRET and entry["definition"] == SECRET_DEF and len(entry["positive_examples"]) == 3
    assert main(["items", "list", *base, "--show-private"]) == 0
    assert f"| {pid} | {SECRET} |" in console(capsys)
    # ... while ids, head files and what is stored in them do not depend on the flag
    heads2 = tmp_path / "heads_shown"
    assert main(["train", *base, *cache, *LOW, "--backbone", "fake", "--show-private", "--out", str(heads2)]) == 0
    assert SECRET in console(capsys)
    assert {p.name for p in heads2.glob("*.npz")} == {p.name for p in heads.glob("*.npz")}
    assert SECRET not in (heads2 / "manifest.json").read_text(encoding="utf-8")
    assert load_head(heads2 / f"{pid}_v1.npz").meta["name"] == alias
    assert main(["score", "--images", str(lab["files"][2]), "--heads", str(heads), "--show-private",
                 "--config", str(lab["cfg"]), "--out", str(tmp_path / "shadow_real.jsonl")]) == 0
    assert json.loads((tmp_path / "shadow_real.jsonl").read_text(encoding="utf-8"))["scores"][pid]["name"] == SECRET


def test_private_reason_in_config_and_errors(tmp_path, capsys):
    # judge_lab's own config format, with an explicit id: the id is hidden too
    cfg = tmp_path / "heads.json"
    cfg.write_text(json.dumps({"items": {
        "secret_code": {"name": SECRET, "kind": "hard", "region": "full", "private": True, "definition": SECRET_DEF,
                        "aliases": [SECRET_OLD], "note": "机密备注"},
        "plain": {"name": "普通", "kind": "hard", "region": "full"}}}, ensure_ascii=False), encoding="utf-8")
    items, _ = load_config(cfg)
    pid = private_id("secret_code")
    it = items[pid]
    assert set(items) == {pid, "plain"} and it.private and it.name == masked_name(pid) == it.public_name
    assert (it.definition, it.aliases, it.note) == ("", (), "") and it.has_definition
    assert "secret_code" not in repr(it) and SECRET not in repr(it) and SECRET_OLD not in repr(it)
    shown = load_config(cfg, show_private=True)[0][pid]
    assert (shown.name, shown.definition, shown.aliases) == (SECRET, SECRET_DEF, (SECRET_OLD,))
    assert shown.public_name == masked_name(pid) and shown.id == pid
    # clicks spelled by the real id, the real name or an earlier name all resolve, with or without the flag
    for show in (False, True):
        its, _ = load_config(cfg, show_private=show)
        log = tmp_path / "log.jsonl"
        write_log(log, [{"id": f"r{k}", "decision": "fail", "reasons": [n]}
                        for k, n in enumerate(("secret_code", SECRET, SECRET_OLD, "普通"))])
        reviews, _ = load_logs([log])
        assert canonicalize(reviews, its) == {} and [r.reasons for r in reviews] == [[pid], [pid], [pid], ["plain"]]
    # a clash with a private name is reported without printing it
    cfg.write_text(json.dumps({"items": [
        {"name": SECRET, "group": "其他", "kind": "hard", "private": True},
        {"name": "别的", "group": "其他", "kind": "hard", "merged_from": [SECRET]}]}, ensure_ascii=False), encoding="utf-8")
    for show in (False, True):
        with pytest.raises(JudgeLabError) as e:
            load_config(cfg, show_private=show)
        assert SECRET not in str(e.value) and "私有项" in str(e.value)
    capsys.readouterr()
    assert main(["items", "list", "--log", str(tmp_path / "log.jsonl"), "--config", str(cfg)]) == 2
    assert SECRET not in console(capsys)
    cfg.write_text(json.dumps({"items": [{"name": SECRET, "group": "其他", "kind": "oops", "private": True}]},
                              ensure_ascii=False), encoding="utf-8")
    with pytest.raises(JudgeLabError) as e:
        load_config(cfg)
    assert SECRET not in str(e.value)


# --------------------------------------------------------------------------- items list, export, verify, trace
def test_items_list_reports_both_shortfalls(tmp_path, capsys):
    cfg = tmp_path / "heads.json"
    cfg.write_text(json.dumps({"items": {k: {"name": k.upper(), "kind": "hard", "region": "full"} for k in "abcd"}}),
                   encoding="utf-8")
    rows = [{"id": f"p{k}", "decision": "pass"} for k in range(1)] + \
           [{"id": f"a{k}", "decision": "fail", "reasons": [{"item": "a", "carried": False}]} for k in range(2)] + \
           [{"id": f"b{k}", "decision": "fail", "reasons": [{"item": "b", "carried": False}]} for k in range(9)] + \
           [{"id": f"d{k}", "decision": "fail", "reasons": [{"item": "d", "carried": False}], "reasons_complete": True}
            for k in range(3)]
    log = write_log(tmp_path / "log.jsonl", rows)

    def states(*extra):
        capsys.readouterr()
        assert main(["items", "list", "--log", str(log), "--config", str(cfg), "--min-pos", "9", "--min-neg", "4",
                     *extra]) == 0
        out = capsys.readouterr().out
        return {c[0]: (c[5], c[6], c[7], c[8]) for c in ([x.strip() for x in l.strip("|").split("|")]
                                                         for l in out.splitlines() if l.startswith("| ")) if c[0] in "abcd"}

    # negatives: the pass and the three reasons_complete fails; the other fails are unknown for the reasons not picked
    assert states() == {"a": ("2", "4", "9", "还差 7 个正例"), "b": ("9", "4", "2", "可训练"),
                        "c": ("0", "4", "11", "还差 9 个正例"), "d": ("3", "1", "11", "还差 6 个正例、3 个反例")}
    assert states("--min-neg", "6")["b"] == ("9", "4", "2", "还差 2 个反例")
    assert states("--min-neg", "6")["a"] == ("2", "4", "9", "还差 7 个正例、2 个反例")
    assert states("--unselected-fail-weight", "1")["d"] == ("3", "12", "0", "还差 6 个正例")
    # verify words it the same way (exit code 1 only because this made-up log has no pictures on disk)
    assert main(["verify", "--log", str(log), "--config", str(cfg), "--min-pos", "9", "--min-neg", "4",
                 "--out", str(tmp_path / "v.md")]) == 1
    md = (tmp_path / "v.md").read_text(encoding="utf-8")
    assert "| D | 硬门 | full | 3 | 1 | 11 | - | 还差 6 个正例、3 个反例 |" in md
    assert "| A | 硬门 | full | 2 | 4 | 9 | - | 还差 7 个正例 |" in md
    assert "| B | 硬门 | full | 9 | 4 | 2 | - | 样本够了，还没训练 |" in md


def test_export_prompts(lab, tmp_path, capsys):
    rows = [record(lab, i) for i in range(N)]
    for i in (16, 17, 18, 19):  # passed pictures that were batch-passed count less, so they come last as examples
        if rows[i]["decision"] == "过":
            rows[i]["batch_pass"] = True
    log = write_log(tmp_path / "log.jsonl", rows)
    exam = tmp_path / "exam.txt"
    exam.write_text("\n".join(lab["shas"][:15]), encoding="utf-8")
    out = tmp_path / "prompts.json"
    assert main(["export-prompts", "--log", str(log), "--config", str(lab["cfg"]), "--exam", str(exam),
                 "--out", str(out), "--examples", "4"]) == 0
    assert "1 个还没有写含义" in console(capsys)
    js = read_json(out)
    by = {it["name"]: it for it in js["items"]}
    assert set(by) == {STRUCT, LIGHT, STORY} and js["private_items_not_exported"] == 1 and js["examples_per_side"] == 4
    assert js["items_without_definition"] == [LIGHT] and js["exam"]["exam_in_log"] == 15
    name_to_i = {f.name: i for i, f in enumerate(lab["files"])}
    for name, col in ((STRUCT, 0), (LIGHT, 1)):
        it = by[name]
        pos, neg = [name_to_i[x] for x in it["positive_examples"]], [name_to_i[x] for x in it["negative_examples"]]
        assert len(pos) == len(neg) == 4
        assert all(flags(i)[col] for i in pos) and all(not any(flags(i)) for i in neg)  # negatives: passed pictures
        assert all(i >= 15 for i in pos + neg)  # never an exam picture
        assert not any(rows[i]["batch_pass"] for i in neg)
        assert it["n_pos"] == sum(1 for i in range(15, N) if flags(i)[col])
    s = by[STRUCT]
    assert (s["id"], s["kind"], s["group"], s["region"], s["definition"], s["has_definition"], s["trainable"]) == \
        (STRUCT, "hard", "结构", "tile", "直线断开或错位", True, True)
    # the most severe scored positives come first (score 1 -> weight 1.4)
    light_pos = [name_to_i[x] for x in by[LIGHT]["positive_examples"]]
    assert all([1, 4, 8][i % 3] == 1 for i in light_pos)
    assert (by[LIGHT]["definition"], by[LIGHT]["has_definition"]) == ("", False)
    assert by[STORY]["trainable"] is False and by[STORY]["positive_examples"] == [] and by[STORY]["group"] == "动作与分镜"
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert f"## {STRUCT}" in md and "- 含义: 直线断开或错位" in md and "- 含义: **（还没有写含义）**" in md
    assert f"**还没有写含义的原因（1 个）**: {LIGHT}" in md and "另有 1 个私有原因没有导出" in md
    assert by[STRUCT]["positive_examples"][0] in md and SECRET not in md and SECRET_DEF not in md
    # --items narrows it; an .md path for --out still writes both files
    assert main(["export-prompts", "--log", str(log), "--config", str(lab["cfg"]), "--items", LIGHT,
                 "--out", str(tmp_path / "one.md"), "--examples", "1"]) == 0
    one = read_json(tmp_path / "one.json")
    assert [it["name"] for it in one["items"]] == [LIGHT] and len(one["items"][0]["positive_examples"]) == 1
    assert (tmp_path / "one.md").is_file()


def _rows_with_everything(lab):
    rows = [record(lab, i) for i in range(N)]
    rows[1]["note"] = "背景太亮了"
    for i in range(N):
        if rows[i]["decision"] == "过" and i >= 40:
            rows[i]["batch_pass"] = True
        if rows[i]["decision"] == "不过" and 2 <= i < 10:
            rows[i]["reasons_complete"] = True  # i = 2, 3, 5, 6, 7, 9
    rows[44]["recommend"] = True  # a batch-passed picture the owner recommends
    rows[9]["reasons"][1]["carried"] = True  # i=9: struct + light; the light pick was carried over, never touched
    del rows[21]["reasons"][0]["carried"]  # i=21: struct pick of unknown origin
    return rows


def test_verify_reports_policy_meaning_recommend_time_and_region(lab, tmp_path, capsys):
    rows = _rows_with_everything(lab)
    log = write_log(tmp_path / "log.jsonl", rows)
    (tmp_path / "exam.txt").write_text("", encoding="utf-8")
    base = ["--log", str(log), "--exam", str(tmp_path / "exam.txt")]
    assert main(["verify", *base, "--config", str(lab["cfg"]), "--out", str(tmp_path / "verify.md")]) == 0
    checks = {c[1]: (c[0], c[2]) for c in ([x.strip() for x in l.strip("|").split("|")]
                                           for l in (tmp_path / "verify.md").read_text(encoding="utf-8").splitlines()
                                           if l.startswith(("| 通 |", "| 注意 |", "| 断 |")))}
    n_fail = sum(1 for r in rows if r["decision"] == "不过")
    n_complete = sum(1 for r in rows if r.get("reasons_complete"))
    n_batch = sum(1 for r in rows if r["batch_pass"])
    n_picks = sum(len(r["reasons"]) for r in rows)
    assert (n_fail, n_complete, n_batch, n_picks) == (49, 6, 16, 63)
    level, text = checks["标注口径"]
    assert level == "通"
    assert "不过但没选的原因 → 不计入（当作不知道）" in text and "整屏通过的图 → 算反例，分量 0.5" in text
    assert "自动带过来、你没碰过的原因 → 不计入" in text and "不清楚是不是带过来的原因（旧记录）→ 算正例，分量 0.5" in text
    assert f"不过 {n_fail} 张，其中标了「原因已选全」的 {n_complete} 张；整屏通过 {n_batch} 张" in text
    assert f"自动带过来没碰过的 1 次、不清楚的 1 次、你自己选的 {n_picks - 2} 次" in text
    assert checks["推荐进入"] == ("通", "1 张图标了「推荐进入」：一定进训练，不受来源筛选影响，整屏通过也按满分量算；和考卷名单没有冲突")
    assert checks["审片用时"] == ("通", "80 张图有用时记录，共 8.0 分钟，平均每张 6.0 秒。这只是工作量统计，不用于训练")
    assert checks["含义"] == ("注意", f"有正例的 3 个原因里，2 个写了含义；还没写的: {LIGHT}。"
                                    "判官头不读含义，但人和看图大模型要靠它（export-prompts 会导出）")
    assert "判官头看的区域" not in checks
    # the CLI knobs show up in the same line
    assert main(["verify", *base, "--config", str(lab["cfg"]), "--batch-pass-weight", "0", "--carried-weight", "1",
                 "--out", str(tmp_path / "verify2.md")]) == 0
    md = (tmp_path / "verify2.md").read_text(encoding="utf-8")
    assert "整屏通过的图 → 不计入（当作不知道）" in md and "自动带过来、你没碰过的原因 → 算正例；" in md
    # counts follow the policy: light has 20 picks, one carried (out by default, in with --carried-weight 1)
    light_default = next(l for l in (tmp_path / "verify.md").read_text(encoding="utf-8").splitlines()
                         if l.startswith(f"| {LIGHT} | 计分 |"))
    light_carried = next(l for l in md.splitlines() if l.startswith(f"| {LIGHT} | 计分 |"))
    passes = N - n_fail
    assert f"| {LIGHT} | 计分 | full | 19 | {passes + 4} |" in light_default  # + reasons_complete fails without light
    assert f"| {LIGHT} | 计分 | full | 20 | {passes - 16 + 1 + 4} |" in light_carried  # batch passes out, 推荐进入 stays

    # a reason moved to another group after its head was trained: the head still looks at the old region
    heads = tmp_path / "heads"
    assert main(["train", *base, "--config", str(lab["cfg"]), *LOW, "--cache", str(tmp_path / "cache"),
                 "--backbone", "fake", "--out", str(heads)]) == 0
    assert load_head(next(heads.glob("*.npz"))).meta["label_policy"] == data_mod.POLICY_DEFAULTS
    # heads trained under one policy, verify run under another: said plainly
    assert main(["verify", *base, "--config", str(lab["cfg"]), "--heads", str(heads), "--carried-weight", "1",
                 "--out", str(tmp_path / "v_policy.md")]) == 0
    assert "| 注意 | 判官头的标注口径 | 3 个判官头训练时用的标注口径和现在的不一样" in (tmp_path / "v_policy.md").read_text(encoding="utf-8")
    moved = tmp_path / "缺陷清单_改组.json"
    moved.write_text(json.dumps(defects(group="脸与肤质"), ensure_ascii=False), encoding="utf-8")
    assert main(["verify", *base, "--config", str(moved), "--heads", str(heads), "--out", str(tmp_path / "v3.md")]) == 0
    md = (tmp_path / "v3.md").read_text(encoding="utf-8")
    line = next(l for l in md.splitlines() if l.startswith("| 注意 | 判官头看的区域 |"))
    assert f"{STRUCT}（判官头 v1 看 tile，清单现在是 face）" in line and "重新 train" in line and LIGHT not in line
    # records keep saying group 结构 for old clicks; only the CURRENT list decides the region
    assert load_config(moved)[0][STRUCT].region == "face"
    assert main(["verify", *base, "--config", str(lab["cfg"]), "--heads", str(heads), "--out", str(tmp_path / "v4.md")]) == 0
    assert "判官头看的区域" not in (tmp_path / "v4.md").read_text(encoding="utf-8")
    assert "判官头的标注口径" not in (tmp_path / "v4.md").read_text(encoding="utf-8")
    # a head left behind by a reason that is no longer in the list is counted, not named
    gone = tmp_path / "缺陷清单_少一项.json"
    gone.write_text(json.dumps({"items": defects()["items"][1:]}, ensure_ascii=False), encoding="utf-8")
    capsys.readouterr()
    assert main(["verify", *base, "--config", str(gone), "--heads", str(heads), "--out", str(tmp_path / "v5.md")]) == 1
    assert "| 注意 | 多余的判官头 | 判官头目录里有 1 个头对不上现在清单里的任何一项" in (tmp_path / "v5.md").read_text(encoding="utf-8")
    # no `seconds` anywhere -> no 审片用时 line
    quiet = write_log(tmp_path / "quiet.jsonl", [record(lab, i, seconds=None) for i in range(N)])
    assert main(["verify", "--log", str(quiet), "--config", str(lab["cfg"]), "--out", str(tmp_path / "v6.md")]) == 0
    assert "审片用时" not in (tmp_path / "v6.md").read_text(encoding="utf-8")


def test_trace_shows_note_source_flags_time_and_weight(lab, tmp_path, capsys):
    log = write_log(tmp_path / "log.jsonl", _rows_with_everything(lab))
    base = ["trace", "--log", str(log), "--config", str(lab["cfg"])]

    def trace(i, *extra):
        capsys.readouterr()
        assert main([*base, "--image", lab["files"][i].name, *extra]) == 0
        return capsys.readouterr().out

    out = trace(44)  # passed with the whole screen, then recommended
    assert "来源: 新图   整屏通过: 是   推荐进入: 是" in out and "审片用时: 8 秒（仅统计，不用于训练）" in out
    assert "场景: scene08" in out and "备注" not in out
    assert "其余 4 项: 这张图算“正常”的例子" in out and "分量" not in out  # full weight: recommended
    assert "整屏通过，但你点了「推荐进入」" in trace(44, "--all") and "| 1.00 |" in trace(44, "--all")
    out = trace(46)  # passed with the whole screen
    assert "整屏通过: 是   推荐进入: 否" in out and "其余 4 项: 这张图算“正常”的例子（分量 0.5）" in out
    assert "| 0.50 |" in trace(46, "--all")
    assert "另有 4 项不计入（整屏通过，按口径不算）" in trace(46, "--batch-pass-weight", "0")
    out = trace(1)  # light only, score 4, with a note; not marked complete
    assert "备注: 背景太亮了（只在这里显示，不用于训练）" in out and "原因已选全: 否" in out
    assert f"| {LIGHT} | 有问题（正例） | 判不过并选了这一项 → 有问题 | 4.000 | 1.10 |" in out
    assert "其余 0 项" in out and "另有 3 项不知道有没有这个问题（判不过但没选它们），不计入" in out
    out = trace(9)  # struct chosen, light carried over untouched; reasons marked complete
    assert "原因已选全: 是" in out
    assert f"| {STRUCT} | 有问题（正例） | 判不过并选了这一项 → 有问题 | - | 1.00 |" in out
    assert f"| {LIGHT} | 不计入 | 这个原因是从上一张自动带过来的，你没有碰过 → 不计入 | 1.000 | - | 不计入 |" in out
    assert "其余 2 项: 这张图算“正常”的例子" in out and "不知道" not in out
    assert f"| {LIGHT} | 有问题（正例） |" in trace(9, "--carried-weight", "0.5") and "| 0.70 |" in trace(9, "--carried-weight", "0.5")
    out = trace(21)  # struct pick with no carried flag (old record): half weight
    assert "不清楚是不是自动带过来的，分量 ×0.5" in out and "| 0.50 |" in out


def test_crops_are_found_under_the_path_derived_id_too(lab, tmp_path):
    """New records carry an id (sha256 prefix); crop folders made before that are named '<folder>__<file stem>'."""
    from judge_lab.embeddings import premade_crops, region_crops
    f = lab["files"][0]
    new_id, old_id = lab["shas"][0][:16], derive_id(f)
    assert old_id == "批次B__scene00_s0"
    (tmp_path / old_id).mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / old_id / "face_0.png")
    assert premade_crops(tmp_path, new_id, "face") == []
    assert [p.name for p in premade_crops(tmp_path, new_id, "face", f)] == ["face_0.png"]
    assert [n for n, _ in region_crops(new_id, f, "face", tmp_path)] == ["face_0.png"]
    (tmp_path / new_id).mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / new_id / "face_3.png")  # a folder named by the record's id wins
    assert [n for n, _ in region_crops(new_id, f, "face", tmp_path)] == ["face_3.png"]
    assert region_crops(new_id, f, "hand", tmp_path) == []
