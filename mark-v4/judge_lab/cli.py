"""Command line: python -m judge_lab <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .backbones import BackboneSpec, parse_model_overrides
from .data import (
    DEFAULT_CONFIG, FieldMap, HeadItem, Review, canonicalize, explain_label, label_counts, load_config, load_exam,
    load_log, load_log_ex, severity_weight, split_exam,
)
from .embeddings import IMG_EXTS, RegionCache, compute_region, ensure_embeddings, premade_crops
from .heads import (
    Head, auc, build_item_data, evaluate_at, fit_linear, image_max, load_active_heads, load_thresholds,
    load_head, load_manifest, needed_regions, oof_scores, permuted, region_rows, save_head, threshold_for_head,
    threshold_for_recall,
)
from .stats import ab_markdown, ab_report, sample_size
from .util import JudgeLabError, assert_no_exam, fmt, md_table, now_iso, read_json, sha256_ids, utf8_stdio, \
    write_json, write_text

DISCLAIMER = "模型只出分数，不做通过/不通过判定；阈值为建议值，最终线由主人审核批准。"


# --------------------------------------------------------------------------- shared context
@dataclass
class Ctx:
    reviews: list[Review]
    train: list[Review]
    exam: list[Review]
    exam_ids: set[str]
    summary: dict
    items: dict[str, HeadItem]


def load_ctx(args) -> Ctx:
    reviews = load_log(args.log, FieldMap.load(args.field_map))
    exam_ids = load_exam(getattr(args, "exam", None))
    train, exam, summary = split_exam(reviews, exam_ids)
    print(f"[数据] 日志 {summary['total_reviews']} 张；考卷名单 {summary['exam_listed']} 个，日志中命中 "
          f"{summary['exam_in_log']} 个（永不训练）；训练池 {summary['train_pool']} 张")
    if summary["exam_not_in_log"]:
        print(f"[数据] 注意: {summary['exam_not_in_log']} 个考卷 id 不在日志中")
    items, _ = load_config(args.config)
    orphans = canonicalize(reviews, items)
    if orphans:
        top = "、".join(f"「{k}」×{v}" for k, v in sorted(orphans.items(), key=lambda kv: -kv[1])[:5])
        print(f"[数据] 警告: {sum(orphans.values())} 次原因点选对不上清单里的任何一项，不会进任何判官头: {top}"
              "（运行 verify 查看全部）")
    if getattr(args, "items", None):
        wanted = set(args.items.split(","))
        if wanted - set(items):
            raise JudgeLabError(f"未知项目: {sorted(wanted - set(items))}")
        items = {k: v for k, v in items.items() if k in wanted}
    return Ctx(reviews, train, exam, exam_ids, summary, items)


def make_spec(args, name: str, model_id: str | None = None) -> BackboneSpec:
    return BackboneSpec.create(name, parse_model_overrides(args.model_id), args.model_dir, args.device,
                               args.batch_size, not args.no_fp16, model_id=model_id)


def target_recall(item: HeadItem, args) -> float:
    flag = args.recall_hard if item.kind == "hard" else args.recall_score
    return item.target_recall if flag is None else flag


def pairs(reviews: list[Review]):
    return [(r.id, r.image) for r in reviews]


def enough(d, args) -> str | None:
    if d.n_pos < args.min_pos or d.n_neg < args.min_neg:
        return f"样本不足 (正 {d.n_pos} / 负 {d.n_neg}，需要 ≥{args.min_pos}/{args.min_neg})"
    return None


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# --------------------------------------------------------------------------- embed
def cmd_embed(args) -> None:
    ctx = load_ctx(args)
    trainable = [it for it in ctx.items.values() if it.trainable]
    regions = _split(args.regions) if args.regions else sorted(needed_regions(trainable))
    for name in _split(args.backbone):
        # Exam images are embedded too (frozen backbone = no fitting); they are only read by `exam`.
        ensure_embeddings(make_spec(args, name), regions, pairs(ctx.reviews), args.out, args.crops_root, args.force)
    print(f"[embed] 完成。区域: {', '.join(regions)}；其中 {len(ctx.exam)} 张考卷图仅供 exam 评估使用")


# --------------------------------------------------------------------------- probe
def _mean_auc(results: dict, item_ids) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for iid in item_ids:
        for bb, r in results.get(iid, {}).items():
            if r.get("auc") is not None:
                acc.setdefault(bb, []).append(r["auc"])
    return {bb: float(np.mean(v)) for bb, v in acc.items()}


def _best(scores: dict[str, float]) -> str | None:
    return max(scores, key=scores.get) if scores else None


def cmd_probe(args) -> None:
    ctx = load_ctx(args)
    backbones = _split(args.backbones)
    trainable = [it for it in ctx.items.values() if it.trainable]
    results: dict[str, dict[str, dict]] = {it.id: {} for it in trainable}
    for name in backbones:
        caches = ensure_embeddings(make_spec(args, name), needed_regions(trainable), pairs(ctx.train),
                                   args.cache, args.crops_root)
        for it in trainable:
            d = build_item_data(it, ctx.train, caches, exclude_ids=ctx.exam_ids)
            r = {"n_pos": d.n_pos, "n_neg": d.n_neg, "n_crops": int(len(d.y)),
                 "n_no_crops": d.n_no_crops, "n_ambiguous": d.n_ambiguous}
            skip = enough(d, args)
            if skip:
                r["skipped"] = skip
            else:
                _, y, s = oof_scores(d, args.folds, args.C, args.seed)
                r.update(auc=auc(y, s), **threshold_for_recall(y, s, target_recall(it, args)))
            results[it.id][name] = r

    out = {"created_at": now_iso(), "toolkit_version": __version__, "exam": ctx.summary,
           "settings": {"folds": args.folds, "C": args.C, "seed": args.seed, "min_pos": args.min_pos,
                        "min_neg": args.min_neg, "backbones": backbones},
           "items": {}, "regions": {}, "overall": {}}
    rows = []
    for it in ctx.items.values():
        entry = {"name": it.name, "kind": it.kind, "region": it.region, "trainable": it.trainable,
                 "results": results.get(it.id, {}), "best_backbone": None}
        if not it.trainable:
            c = label_counts(ctx.train, it)
            rows.append([it.id, it.name, it.kind, it.region, "-", c["pos"], c["neg"], "-", "-", "-", "-", "-", "-",
                         "不自动训练 (trainable=false)" + (f"；{it.note}" if it.note else "")])
        else:
            for bb, r in entry["results"].items():
                rows.append([it.id, it.name, it.kind, it.region, bb, r["n_pos"], r["n_neg"], r["n_crops"],
                             r.get("auc"), r.get("target_recall", target_recall(it, args)), r.get("recall"),
                             r.get("false_reject_rate"), r.get("threshold"), r.get("skipped", "")])
            scored = {bb: r["auc"] for bb, r in entry["results"].items() if r.get("auc") is not None}
            entry["best_backbone"] = _best(scored)
        out["items"][it.id] = entry
    for region in sorted({it.region for it in trainable}):
        m = _mean_auc(results, [it.id for it in trainable if it.region == region])
        out["regions"][region] = {"best_backbone": _best(m), "mean_auc": m}
    m = _mean_auc(results, results)
    out["overall"] = {"best_backbone": _best(m), "mean_auc": m}

    s = ctx.summary
    md = [
        "# 骨干比拼（线性探针，按图分组的 5 折交叉验证）", "",
        f"- 生成时间: {out['created_at']}",
        f"- 考卷: 名单 {s['exam_listed']} 个，日志中 {s['exam_in_log']} 个，全部排除在外；训练池 {s['train_pool']} 张",
        f"- 设置: folds={args.folds}, C={args.C}, 最少正/负样本 {args.min_pos}/{args.min_neg}；"
        "召回/误拒率为图片级（取各裁剪最大分）", "",
        md_table(["项目", "名称", "类型", "区域", "骨干", "正样本", "负样本", "裁剪数", "AUC", "目标召回",
                  "召回", "误拒率", "阈值", "备注"], rows),
        "", "## 每个项目的最佳骨干（按 AUC）", "",
        md_table(["项目", "名称", "最佳骨干", "AUC"],
                 [[i, e["name"], e["best_backbone"] or "-",
                   e["results"][e["best_backbone"]]["auc"] if e["best_backbone"] else None]
                  for i, e in out["items"].items() if e["trainable"]]),
        "", "## 每个区域的最佳骨干（平均 AUC）", "",
        md_table(["区域", "最佳骨干"] + backbones,
                 [[rg, e["best_backbone"] or "-"] + [e["mean_auc"].get(bb) for bb in backbones]
                  for rg, e in out["regions"].items()]),
        "", f"整体最佳: **{out['overall']['best_backbone'] or '-'}**", "", DISCLAIMER, "",
    ]
    md_path = Path(args.out)
    write_text(md_path, "\n".join(md))
    write_json(md_path.with_suffix(".json"), out)
    print(f"[probe] 报告: {md_path}  JSON: {md_path.with_suffix('.json')}")


# --------------------------------------------------------------------------- train
def choose_backbone(item: HeadItem, probe: dict | None, flag: str | None) -> str | None:
    if flag:
        return flag
    if not probe:
        return None
    return (probe["items"].get(item.id, {}).get("best_backbone")
            or probe["regions"].get(item.region, {}).get("best_backbone")
            or probe["overall"].get("best_backbone"))


def cmd_train(args) -> None:
    if not args.probe and not args.backbone:
        raise JudgeLabError("请指定 --probe reports/probe.json 或 --backbone NAME")
    ctx = load_ctx(args)
    probe = read_json(args.probe) if args.probe else None
    plan: dict[str, list[HeadItem]] = {}
    for it in ctx.items.values():
        if not it.trainable:
            print(f"[train] 跳过 {it.id} ({it.name}): trainable=false")
            continue
        name = choose_backbone(it, probe, args.backbone)
        if name:
            plan.setdefault(name, []).append(it)
        else:
            print(f"[train] 跳过 {it.id}: probe 里没有可用骨干")
    trained = 0
    for name, its in plan.items():
        spec = make_spec(args, name)
        caches = ensure_embeddings(spec, needed_regions(its), pairs(ctx.train), args.cache, args.crops_root)
        for it in its:
            d = build_item_data(it, ctx.train, caches, exclude_ids=ctx.exam_ids)
            skip = enough(d, args)
            if skip:
                print(f"[train] 跳过 {it.id} ({it.name}): {skip}")
                continue
            assert_no_exam(d.image_ids, ctx.exam_ids, f"train {it.id}")
            coef, b = fit_linear(d.X, d.y, args.C, d.w)
            head = save_head(args.out, it, coef, b, d.image_ids, {
                "backbone": name, "model_id": spec.model_id, "dim": int(d.X.shape[1]),
                "n_pos": d.n_pos, "n_neg": d.n_neg, "n_crops": int(len(d.y)), "C": args.C,
                "label_rule": it.label_rule, "target_recall": target_recall(it, args),
                "exam_excluded": len(ctx.exam), "toolkit_version": __version__,
            })
            trained += 1
            print(f"[train] {head.path.name}: {it.name} 骨干={name} 正 {d.n_pos} / 负 {d.n_neg}")
    print(f"[train] 共训练 {trained} 个头 → {Path(args.out) / 'manifest.json'}")


# --------------------------------------------------------------------------- calibrate
def cmd_calibrate(args) -> None:
    ctx = load_ctx(args)
    heads = load_active_heads(args.heads, set(ctx.items))
    out_path = Path(args.out or Path(args.heads) / "thresholds.json")
    old = load_thresholds(out_path).get("items", {})
    out = {"note": DISCLAIMER, "updated_at": now_iso(), "exam": ctx.summary, "items": dict(old)}
    specs: dict[tuple, BackboneSpec] = {}
    for item_id, head in heads.items():
        it = ctx.items[item_id]
        key = (head.meta["backbone"], head.meta["model_id"])
        spec = specs.setdefault(key, make_spec(args, *key))
        caches = ensure_embeddings(spec, needed_regions([it]), pairs(ctx.train), args.cache, args.crops_root)
        d = build_item_data(it, ctx.train, caches, exclude_ids=ctx.exam_ids)
        skip = enough(d, args)
        if skip:
            print(f"[calibrate] 跳过 {item_id}: {skip}")
            continue
        if sha256_ids(d.image_ids) != head.meta["data_hash"]:
            print(f"[calibrate] 注意 {item_id}: 当前数据与 {head.path.name} 训练时不同，建议重新 train")
        _, y, s = oof_scores(d, args.folds, float(head.meta.get("C", args.C)), args.seed)
        t = threshold_for_recall(y, s, target_recall(it, args))
        entry = {
            "name": it.name, "kind": it.kind, "head_file": head.path.name, "head_version": head.version,
            "threshold": t["threshold"], "proposed_threshold": t["threshold"],
            "target_recall": t["target_recall"], "oof_recall": t["recall"],
            "expected_false_reject_rate": t["false_reject_rate"], "oof_auc": auc(y, s),
            "n_pos": t["n_pos"], "n_neg": t["n_neg"], "calibrated_at": now_iso(),
            "status": "proposed", "approved_by": None,
        }
        prev = old.get(item_id, {})
        if prev.get("status") == "approved" and not args.overwrite_approved:
            entry.update(status="approved", threshold=prev["threshold"], approved_by=prev.get("approved_by"),
                         approved_threshold_oof=evaluate_at(y, s, float(prev["threshold"])))
        out["items"][item_id] = entry
        print(f"[calibrate] {item_id} ({it.name}): 建议阈值 {t['threshold']:.4f} → OOF 召回 "
              f"{t['recall']:.3f} (目标 {t['target_recall']}), 预计误拒率 {fmt(t['false_reject_rate'])}"
              + ("  [已批准，保留主人设定的阈值]" if entry["status"] == "approved" else ""))
    write_json(out_path, out)
    print(f"[calibrate] → {out_path}。{DISCLAIMER}")


# --------------------------------------------------------------------------- exam
def cmd_exam(args) -> None:
    ctx = load_ctx(args)
    if not ctx.exam:
        raise JudgeLabError("日志中没有任何考卷图片，无法考试")
    heads = load_active_heads(args.heads, set(ctx.items))
    thresholds = load_thresholds(args.thresholds or Path(args.heads) / "thresholds.json")
    specs: dict[tuple, BackboneSpec] = {}
    res, rows = {}, []
    for it in ctx.items.values():
        head = heads.get(it.id)
        if head is None:
            why = "trainable=false" if not it.trainable else "没有判官头"
            rows.append([it.id, it.name, it.kind, "-", "-", "-", "-", "-", "-", "-", "-", why])
            continue
        assert_no_exam(head.train_ids, ctx.exam_ids, f"判官头 {head.path.name}")
        key = (head.meta["backbone"], head.meta["model_id"])
        spec = specs.setdefault(key, make_spec(args, *key))
        caches = ensure_embeddings(spec, needed_regions([it]), pairs(ctx.exam), args.cache, args.crops_root)
        d = build_item_data(it, ctx.exam, caches)
        if len(d.y) == 0:
            rows.append([it.id, it.name, it.kind, f"v{head.version}", 0, 0, "-", "-", "-", "-", "-", "考卷中无可用样本"])
            continue
        agg = image_max(head.predict(d.X), d.groups)
        labels = dict(zip(d.groups.tolist(), d.y.tolist()))
        ids = sorted(agg)
        y, s = np.array([labels[i] for i in ids]), np.array([agg[i][0] for i in ids])
        thr = threshold_for_head(thresholds, head)
        m = evaluate_at(y, s, 0.5 if thr is None else thr)
        target = target_recall(it, args)
        ok = None if m["recall"] is None else m["recall"] >= target
        res[it.id] = {**m, "auc": auc(y, s), "target_recall": target, "meets_target": ok,
                      "head_file": head.path.name, "threshold_source": "thresholds.json" if thr is not None else "默认 0.5"}
        rows.append([it.id, it.name, it.kind, f"v{head.version}", m["n_pos"], m["n_neg"], m["threshold"],
                     m["recall"], m["false_reject_rate"], m["agreement"], res[it.id]["auc"],
                     ("达标" if ok else "未达标") if ok is not None else "无正样本"])
    s = ctx.summary
    md = ["# 考试报告（只用考卷图片）", "", f"- 生成时间: {now_iso()}",
          f"- 考卷: 名单 {s['exam_listed']} 个，日志中 {s['exam_in_log']} 个；已核对所有判官头的训练集与考卷无交集", "",
          md_table(["项目", "名称", "类型", "头版本", "正样本", "负样本", "阈值", "召回", "误拒率", "一致率", "AUC",
                    "结果"], rows), "", DISCLAIMER, ""]
    md_path = Path(args.out)
    write_text(md_path, "\n".join(md))
    write_json(md_path.with_suffix(".json"), {"created_at": now_iso(), "exam": s, "items": res})
    print("\n".join(md))
    print(f"[exam] 报告: {md_path}")


# --------------------------------------------------------------------------- score (shadow)
def collect_images(src: Path) -> list[tuple[str, Path]]:
    src = Path(src)
    if not src.exists():
        raise JudgeLabError(f"--images 不存在: {src}")
    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    elif src.suffix.lower() in IMG_EXTS:
        paths = [src]
    else:  # text list, one path per line, relative to the list file
        paths = []
        for line in src.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                p = Path(line.replace("\\", "/"))
                paths.append(p if p.is_absolute() else src.parent / p)
    out, seen = [], set()
    for p in paths:
        iid, n = p.stem, 2
        while iid in seen:
            iid, n = f"{p.stem}__{n}", n + 1
        seen.add(iid)
        out.append((iid, p))
    return out


def cmd_score(args) -> None:
    heads = load_active_heads(args.heads, set(_split(args.items)) if args.items else None)
    thresholds = load_thresholds(args.thresholds or Path(args.heads) / "thresholds.json")
    images = collect_images(args.images)
    print(f"[score] {len(images)} 张图，{len(heads)} 个判官头（影子模式：只记分数，不做判定）")
    stamp = now_iso()
    recs = {iid: {"id": iid, "image": str(p), "mode": "shadow", "scored_at": stamp, "scores": {}, "no_crops": []}
            for iid, p in images}
    groups: dict[tuple, list[Head]] = {}
    for h in heads.values():
        groups.setdefault((h.meta["backbone"], h.meta["model_id"]), []).append(h)
    for key, hs in groups.items():
        spec = make_spec(args, *key)
        caches = {}
        for region in sorted({r for h in hs for r in (h.meta["region"], h.meta.get("fallback_region")) if r}):
            emb, ids, names, done, errors = compute_region(spec, region, images, args.crops_root)
            caches[region] = RegionCache(emb, np.array(ids, str), np.array(names, str), done, {})
            for e in errors:
                print(f"  [score] 跳过 {e}")
        for h in hs:
            thr = threshold_for_head(thresholds, h)
            for iid, _ in images:
                cache, rows = region_rows(caches, h.meta["region"], h.meta.get("fallback_region"), iid)
                if len(rows) == 0:
                    recs[iid]["no_crops"].append(h.item)
                    continue
                p = h.predict(cache.emb[rows])
                j = int(np.argmax(p))
                recs[iid]["scores"][h.item] = {
                    "name": h.meta.get("name", h.item), "prob": round(float(p[j]), 4),
                    "threshold": thr,
                    "over_threshold": None if thr is None else bool(p[j] >= thr),
                    "head_version": h.version, "max_crop": str(cache.crop_names[rows][j]), "n_crops": int(len(rows)),
                }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if args.append else "w", encoding="utf-8") as fh:
        for r in recs.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[score] → {out}")


# --------------------------------------------------------------------------- stats
def cmd_stats_ab(args) -> None:
    reviews = load_log(args.log, FieldMap.load(args.field_map))
    try:
        res = ab_report(reviews, args.group_by, args.scene_field or None, args.baseline, args.tolerance, args.alpha)
    except ValueError as e:
        raise JudgeLabError(str(e)) from e
    md = ab_markdown(res)
    print(md)
    if args.out:
        write_text(Path(args.out), md)
        write_json(Path(args.out).with_suffix(".json"), res)
        print(f"[stats] 报告: {args.out}")
    return 1 if (args.fail_on_regression and res["regression_flag"]) else 0


def cmd_stats_n(args) -> None:
    try:
        n = sample_size(args.p1, args.p2, args.power, args.alpha, not args.one_sided)
    except ValueError as e:
        raise JudgeLabError(str(e)) from e
    print(f"p1={args.p1} vs p2={args.p2}, power={args.power}, alpha={args.alpha}"
          f"{' (单侧)' if args.one_sided else ' (双侧)'}: 每组至少 {n} 张，共 {2 * n} 张")



# --------------------------------------------------------------------------- items (owner's defect list)
def cmd_items_list(args) -> None:
    """Per item: how many owner labels exist (exam excluded) and whether a head can be trained yet."""
    reviews = load_log(args.log, FieldMap.load(args.field_map))
    train, _, summary = split_exam(reviews, load_exam(args.exam))
    items, _ = load_config(args.config)
    canonicalize(reviews, items)
    rows = []
    for it in items.values():
        c = label_counts(train, it)
        ready = it.trainable and c["pos"] >= args.min_pos and c["neg"] >= args.min_neg
        state = "可训练" if ready else ("不自动训练" if not it.trainable else f"还差 {max(0, args.min_pos - c['pos'])} 个正例")
        rows.append([it.id, it.name, it.kind, it.region, it.added_at or "-", c["pos"], c["neg"], c["ambiguous"],
                     state, it.definition or "-"])
    print(f"[数据] 训练池 {summary['train_pool']} 张（考卷不计）")
    print(md_table(["id", "名称", "类型", "区域", "新增时间", "正例", "反例", "不计入", "状态", "定义"], rows))
    print("\n说明: 新增项目只统计它创建之后审过的图；创建之前的图当时没有这个选项，不算反例。")


def cmd_items_add(args) -> None:
    """Append one owner-defined item to a head config (never edits the packaged default)."""
    import re
    if Path(args.config).resolve() == DEFAULT_CONFIG.resolve():
        raise JudgeLabError("请用 --config 指定你自己的配置文件副本，不要改包内默认配置")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", args.id):
        raise JudgeLabError("id 只能用小写字母、数字、下划线，例如 layer_seam")
    data = read_json(args.config)
    if args.id in data.get("items", {}):
        raise JudgeLabError(f"项目 {args.id} 已存在")
    entry = {"name": args.name, "kind": args.kind, "region": args.region,
             "label_rule": {"positive_if": "selected"}, "added_at": args.added_at or now_iso(),
             "definition": args.definition or ""}
    if args.no_train:
        entry["trainable"] = False
    data.setdefault("items", {})[args.id] = entry
    write_json(args.config, data)
    load_config(args.config)  # validate
    print(f"已加入 {args.id}（{args.name}，{args.kind}，区域 {args.region}，从 {entry['added_at']} 起计数）")


# --------------------------------------------------------------------------- verify: is the chain connected?
def _active_heads(heads_dir) -> dict[str, Head]:
    if not heads_dir:
        return {}
    out = {}
    for item_id, rec in load_manifest(heads_dir)["items"].items():
        if rec.get("active"):
            f = Path(heads_dir) / rec["active"]["file"]
            if f.is_file():
                out[item_id] = load_head(f)
    return out


def cmd_verify(args) -> int:
    """Checks, on the real files, every link between a click on the review page and a head's weights."""
    reviews, st = load_log_ex(args.log, FieldMap.load(args.field_map))
    exam_ids = load_exam(args.exam)
    items, _ = load_config(args.config)
    orphans = canonicalize(reviews, items)
    train, exam, summary = split_exam(reviews, exam_ids)
    heads = _active_heads(args.heads)
    checks: list[tuple[str, str, str]] = []

    def add(level: str, title: str, detail: str) -> None:
        checks.append((level, title, detail))

    # 1. every line of the log is read
    n_pass = sum(r.decision == "pass" for r in reviews)
    if st["parsed"] == 0:
        add("断", "审片记录读入", f"{st['lines']} 行一条都没读进来" + (f"，例如: {st['skipped'][0][1]}" if st["skipped"] else ""))
    elif st["skipped"]:
        eg = "；".join(f"第 {n} 行 {why}" for n, why in st["skipped"][:3])
        add("断", "审片记录读入", f"{len(st['skipped'])} / {st['lines']} 行被跳过，这些审片不会被用到。{eg}")
    else:
        add("通", "审片记录读入", f"{st['lines']} 行全部读入，共 {len(reviews)} 张图（过 {n_pass} / 不过 {len(reviews) - n_pass}；"
            f"重审 {st['re_reviews']} 次，以最后一次为准）")

    # 2. every selected reason reaches an item
    fails = [r for r in reviews if r.decision == "fail"]
    clicks = sum(len(r.reasons) for r in fails)
    no_reason = sum(1 for r in fails if not r.reasons)
    if orphans:
        eg = "、".join(f"「{k}」×{v}" for k, v in sorted(orphans.items(), key=lambda kv: -kv[1]))
        add("断", "原因 → 清单", f"{sum(orphans.values())} / {clicks} 次点选对不上清单里的任何一项，不会进任何判官头: {eg}。"
            "把旧名字写进对应项的 merged_from（曾用名），或把这一项加回清单")
    else:
        add("通", "原因 → 清单", f"{clicks} 次原因点选全部对上清单（{len(items)} 项）")
    if no_reason:
        add("注意", "不过但没选原因", f"{no_reason} 张判不过却没有原因：它们只能教模型“这张不过”，教不了“为什么”")

    # 3. time stamps (needed by reasons added later)
    late = [it for it in items.values() if it.added_at]
    lost = {it.name: sum(1 for r in train if it.id in r.reasons and r.decision == "fail"
                         and explain_label(r, it)[0] is None and not it.label_rule.get("max_score"))
            for it in late}
    lost = {k: v for k, v in lost.items() if v}
    if lost:
        add("断", "新增原因的时间", "这些点选因为时间对不上而不计入: " + "、".join(f"{k}×{v}" for k, v in lost.items())
            + "。合并或改名后，新项的创建时间要写成最早那个旧项的创建时间")
    elif st["no_time"] and late:
        add("注意", "新增原因的时间", f"{st['no_time']} 条审片没有时间；对 {len(late)} 个后来新增的原因，这些图不计入（不会被当成反例）")
    else:
        add("通", "新增原因的时间", f"{len(late)} 个后来新增的原因，创建之后的审片都按时间正确计入")

    # 4. the pictures are still on disk
    gone = [r for r in reviews if r.image is None or not r.image.is_file()]
    if gone and len(gone) == len(reviews):
        add("断", "图片文件", f"{len(gone)} 张图一张都找不到（例如 {gone[0].image}）：没有图就提不了特征，训练不了")
    elif gone:
        add("注意", "图片文件", f"{len(gone)} / {len(reviews)} 张图找不到（例如 {gone[0].image}），这些审片进不了训练")
    else:
        add("通", "图片文件", f"{len(reviews)} 张图都在盘上")

    # 5. crops for face / hand items
    need = [it for it in items.values() if it.trainable and it.region in ("face", "hand")]
    if need:
        blocked, coarse = [], []
        for it in need:
            labelled = [r for r in train if label_for_cached(r, it) is not None]
            have = sum(1 for r in labelled if premade_crops(args.crops_root, r.id, it.region)) if args.crops_root else 0
            pos = sum(1 for r in labelled if label_for_cached(r, it) == 1)
            if labelled and have < len(labelled):
                (coarse if it.fallback_region else blocked).append(f"{it.name}（切图 {have}/{len(labelled)} 张，正例 {pos}）")
        if blocked:
            add("断", "脸 / 手切图", "这些项只看脸或手的切图，现在没有切图，点了也训不了: " + "；".join(blocked))
        if coarse:
            add("注意", "脸 / 手切图", f"{len(coarse)} 个看脸或手的项缺切图，暂时退而看整图分块，判得会粗；接上切图后要重新训练: "
                + "；".join(coarse[:4]) + (" 等" if len(coarse) > 4 else ""))
        if not blocked and not coarse:
            add("通", "脸 / 手切图", f"{len(need)} 个看脸或手的项，切图齐全")

    # 6. per item: how much the owner has taught it, and whether a head has learned it
    rows = []
    for it in items.values():
        c = label_counts(train, it)
        scored = sum(1 for r in train if label_for_cached(r, it) == 1 and it.id in r.scores)
        head = heads.get(it.id)
        if not it.trainable:
            state = it.note or "不自动训练"
        elif head:
            seen = set(head.train_ids)
            new = sum(1 for r in train if label_for_cached(r, it) is not None and r.id not in seen)
            state = f"判官头 v{head.version}（学了 {len(seen)} 张）；之后新增 {new} 张还没学进去"
        elif c["pos"] >= args.min_pos and c["neg"] >= args.min_neg:
            state = "样本够了，还没训练"
        else:
            state = f"还差 {max(0, args.min_pos - c['pos'])} 个正例"
        rows.append([it.name, "硬门" if it.kind == "hard" else "计分", it.region, c["pos"], c["neg"], c["ambiguous"],
                     scored if it.kind == "score" else "-", state])
    if not heads:
        add("注意", "判官头", "还没有训练过任何判官头：到目前为止，审片只进了记录和配方统计，没有改变任何模型")
    else:
        stale = sum(1 for it in items.values() if it.id in heads and any(
            label_for_cached(r, it) is not None and r.id not in set(heads[it.id].train_ids) for r in train))
        add("通" if not stale else "注意", "判官头", f"{len(heads)} 个判官头在用" + (f"，其中 {stale} 个有新审片还没学进去（重新 train）" if stale else "，都包含最新审片"))

    # 7. optional: do the labels move the weights, and does a head learn WHY rather than pass/fail?
    sig_rows = []
    if args.signal:
        spec = make_spec(args, args.signal)
        its = [it for it in items.values() if it.trainable]
        caches = ensure_embeddings(spec, needed_regions(its), pairs(train), args.cache, args.crops_root)
        failed = {r.id for r in train if r.decision == "fail"}
        weak = []
        for it in its:
            d = build_item_data(it, train, caches, exclude_ids=exam_ids)
            if enough(d, args):
                continue
            ids, y, sc = oof_scores(d, args.folds, args.C, args.seed)
            real = auc(y, sc)
            shuf = float(np.mean([auc(*oof_scores(permuted(d, args.seed + 1 + k), args.folds, args.C, args.seed)[1:])
                                  for k in range(3)]))
            m = np.array([i in failed for i in ids])
            why = auc(y[m], sc[m]) if m.sum() >= 6 and 0 < y[m].sum() < m.sum() else None
            ok = real is not None and real >= 0.70 and real - shuf >= 0.10
            if not ok:
                weak.append(it.name)
            sig_rows.append([it.name, d.n_pos, d.n_neg, real, shuf, why, "学到了" if ok else "学不出来"])
        if not sig_rows:
            add("注意", "标注 → 权重", "还没有哪一项的样本够做这个检查")
        elif weak:
            add("注意", "标注 → 权重", f"{len(sig_rows) - len(weak)} / {len(sig_rows)} 项学到了；学不出来的: {'、'.join(weak)}"
                "（样本少、前后标准不一致、区域不对，或这类问题骨干看不出来）")
        else:
            add("通", "标注 → 权重", f"{len(sig_rows)} 项都学到了：用你的标注训练明显好于把标注打乱")

    worst = "断" if any(c[0] == "断" for c in checks) else ("注意" if any(c[0] == "注意" for c in checks) else "通")
    head_line = {"断": "链路有断点：下面标「断」的地方，你的点击现在到不了模型", "注意": "链路是通的，有几处要留意",
                 "通": "链路全通"}[worst]
    md = ["# 链路自检：审片页的点击有没有到达判官头", "", f"- 时间: {now_iso()}", f"- 审片记录: {args.log}",
          f"- 原因清单: {args.config}", f"- 考卷: {summary['exam_listed']} 个（日志中 {summary['exam_in_log']} 个，永不训练）",
          "", f"**结论: {head_line}**", "", md_table(["状态", "环节", "说明"], checks), "",
          "## 每一项学到哪一步", "",
          md_table(["原因", "类型", "看哪里", "正例", "反例", "不计入", "带分数的正例", "状态"], rows), ""]
    if sig_rows:
        md += ["## 你的标注有没有改变权重（交叉验证，没见过的图上测）", "",
               md_table(["原因", "正例", "反例", "用你的标注", "把标注打乱", "只在不过的图里分原因", "结论"], sig_rows), "",
               "- 「用你的标注」明显高于「把标注打乱」（约 0.5）= 权重是被你的点选推动的，不是巧合。",
               "- 「只在不过的图里分原因」高 = 这个头分得清“因这一项不过”和“因别的原因不过”，学的是原因，不只是过或不过。", ""]
    out = Path(args.out)
    write_text(out, "\n".join(md))
    print(f"[verify] {head_line}")
    for lv, t, dsc in checks:
        print(f"  [{lv}] {t}: {dsc}")
    print(f"[verify] 报告: {out}")
    return 1 if worst == "断" else 0


def label_for_cached(review: Review, item: HeadItem) -> int | None:
    return explain_label(review, item)[0]


def cmd_trace(args) -> int:
    """For one reviewed picture: what each click became, and which head has learned it."""
    reviews, _ = load_log_ex(args.log, FieldMap.load(args.field_map))
    exam_ids = load_exam(args.exam)
    items, _ = load_config(args.config)
    orphans = canonicalize(reviews, items)
    heads = _active_heads(args.heads)
    if args.image:
        key = args.image.replace("\\", "/").lower()
        picked = [r for r in reviews if key in r.id.lower() or key in str(r.image or "").replace("\\", "/").lower()]
    else:
        picked = reviews[-args.last:]
    if not picked:
        raise JudgeLabError("审片记录里找不到这张图")
    for r in picked:
        print(f"\n=== {r.id}\n  图片: {r.image}\n  决定: {'过' if r.decision == 'pass' else '不过'}   审片时间: {r.reviewed_at or '没有记录'}")
        if r.id in exam_ids:
            print("  这是考卷图：永不参加训练，只用来验收判官头")
        rows, normal, off = [], 0, 0
        for it in items.values():
            lab, why = explain_label(r, it)
            picked_it = it.id in r.reasons and r.decision == "fail"
            if lab == 0 and not args.all:
                normal += 1
                continue
            if lab is None and not picked_it and not args.all:
                off += 1
                continue
            head = heads.get(it.id)
            if r.id in exam_ids:
                where = "考卷，不训练"
            elif not it.trainable:
                where = it.note or "不自动训练"
            elif lab is None:
                where = "不计入"
            elif head is None:
                c = label_counts([x for x in reviews if x.id not in exam_ids], it)
                where = f"还没有判官头（正例 {c['pos']} 个）"
            elif r.id in set(head.train_ids):
                where = f"已学进判官头 v{head.version}（{head.meta.get('created_at', '')[:16]}）"
            else:
                where = f"还没学进去：判官头 v{head.version} 训练时没有这张图，重新 train 才会进去"
            wgt = severity_weight(r, it) if lab == 1 else None
            rows.append([it.name, {1: "有问题（正例）", 0: "正常（反例）", None: "不计入"}[lab], why,
                         r.scores.get(it.id, "-") if it.kind == "score" else "-", "-" if wgt is None else f"{wgt:.2f}", where])
        for x in r.reasons:
            if x in orphans:
                rows.append([x, "对不上清单", "这个原因不在清单里，也不是任何一项的曾用名", "-", "-", "到不了任何判官头"])
        if rows:
            print(md_table(["原因", "这张图教给模型的", "为什么", "你打的分", "训练权重", "到了哪里"], rows))
        print(f"  其余 {normal} 项: 这张图算“正常”的例子" + (f"；另有 {off} 项不计入（当时还没有这些选项）" if off else ""))
    return 0

# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m judge_lab", description="专科判官工具包 judge_lab " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def data(sp, exam_required=True):
        sp.add_argument("--log", type=Path, required=True, help="审核日志 JSONL")
        sp.add_argument("--exam", type=Path, required=exam_required, help="考卷 id 列表（永不训练）")
        sp.add_argument("--field-map", type=Path, help="字段映射 JSON")
        sp.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="判官头配置 (默认 configs/heads_v1.json)")
        sp.add_argument("--crops-root", type=Path, help="预裁剪目录 <root>/<image_id>/<region>_<k>.png")
        sp.add_argument("--items", help="只处理这些项目（逗号分隔）")

    def backbone(sp):
        sp.add_argument("--model-id", action="append", metavar="NAME=ID", help="覆盖模型 id，可重复")
        sp.add_argument("--model-dir", type=Path, help="本地模型目录（离线）")
        sp.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu")
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--no-fp16", action="store_true", help="cuda 上也用 fp32")

    def cache(sp):
        sp.add_argument("--cache", type=Path, default=Path("cache"), help="特征缓存目录")

    def cv(sp, folds=True):
        sp.add_argument("--C", type=float, default=0.1, help="逻辑回归正则强度倒数")
        sp.add_argument("--min-pos", type=int, default=15)
        sp.add_argument("--min-neg", type=int, default=15)
        sp.add_argument("--recall-hard", type=float, help="硬门槛目标召回（默认看配置: 0.95）")
        sp.add_argument("--recall-score", type=float, help="评分项目标召回（默认看配置: 0.80）")
        if folds:
            sp.add_argument("--folds", type=int, default=5)
            sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("embed", help="提取并缓存特征")
    data(sp, exam_required=False)
    backbone(sp)
    sp.add_argument("--backbone", default="siglip2", help="逗号分隔，可多个")
    sp.add_argument("--regions", help="默认: 配置中可训练项目用到的区域")
    sp.add_argument("--out", type=Path, default=Path("cache"))
    sp.add_argument("--force", action="store_true", help="忽略已有缓存重新计算")
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("probe", help="骨干比拼（线性探针）")
    data(sp), backbone(sp), cache(sp), cv(sp)
    sp.add_argument("--backbones", default="siglip2,dinov2,clip_l")
    sp.add_argument("--out", type=Path, default=Path("reports/probe.md"))
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("train", help="训练最终判官头")
    data(sp), backbone(sp), cache(sp), cv(sp, folds=False)
    sp.add_argument("--probe", type=Path, help="probe 输出的 JSON（按项目选最佳骨干）")
    sp.add_argument("--backbone", help="所有项目统一用这个骨干（优先于 --probe）")
    sp.add_argument("--out", type=Path, default=Path("heads"))
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("calibrate", help="按目标召回率给出建议阈值")
    data(sp), backbone(sp), cache(sp), cv(sp)
    sp.add_argument("--heads", type=Path, default=Path("heads"))
    sp.add_argument("--out", type=Path, help="默认 <heads>/thresholds.json")
    sp.add_argument("--overwrite-approved", action="store_true", help="连主人已批准的阈值也覆盖")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("exam", help="只在考卷上评估（验收）")
    data(sp), backbone(sp), cache(sp)
    sp.add_argument("--recall-hard", type=float)
    sp.add_argument("--recall-score", type=float)
    sp.add_argument("--heads", type=Path, default=Path("heads"))
    sp.add_argument("--thresholds", type=Path, help="默认 <heads>/thresholds.json")
    sp.add_argument("--out", type=Path, default=Path("reports/exam.md"))
    sp.set_defaults(func=cmd_exam)

    sp = sub.add_parser("score", help="影子模式打分（只出分数）")
    backbone(sp)
    sp.add_argument("--images", type=Path, required=True, help="图片目录、单张图片或路径列表 txt")
    sp.add_argument("--crops-root", type=Path)
    sp.add_argument("--heads", type=Path, default=Path("heads"))
    sp.add_argument("--thresholds", type=Path)
    sp.add_argument("--items")
    sp.add_argument("--out", type=Path, default=Path("shadow.jsonl"))
    sp.add_argument("--append", action="store_true")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("stats", help="A/B 统计")
    ssub = sp.add_subparsers(dest="stats_cmd", required=True)
    ab = ssub.add_parser("ab", help="分组通过率、置信区间、显著性、分场景回归检查")
    ab.add_argument("--log", type=Path, required=True)
    ab.add_argument("--field-map", type=Path)
    ab.add_argument("--group-by", required=True, help="如 versions.lora 或 recipe_hash")
    ab.add_argument("--scene-field", default="scene", help="空字符串关闭分场景")
    ab.add_argument("--baseline", help="基线分组（默认: 日志里最先出现的）")
    ab.add_argument("--tolerance", type=float, default=0.05, help="回归容忍度（通过率绝对值）")
    ab.add_argument("--alpha", type=float, default=0.05)
    ab.add_argument("--out", type=Path)
    ab.add_argument("--fail-on-regression", action="store_true", help="有回归时退出码 1")
    ab.set_defaults(func=cmd_stats_ab)
    n = ssub.add_parser("n", help="每组所需样本量")
    n.add_argument("--p1", type=float, required=True)
    n.add_argument("--p2", type=float, required=True)
    n.add_argument("--power", type=float, default=0.8)
    n.add_argument("--alpha", type=float, default=0.05)
    n.add_argument("--one-sided", action="store_true")
    n.set_defaults(func=cmd_stats_n)

    sp = sub.add_parser("items", help="主人的缺陷清单：查看样本数 / 新增项目")
    isub = sp.add_subparsers(dest="items_cmd", required=True)
    il = isub.add_parser("list", help="每个项目有多少正例/反例，够不够训练")
    il.add_argument("--log", type=Path, required=True)
    il.add_argument("--exam", type=Path)
    il.add_argument("--field-map", type=Path)
    il.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    il.add_argument("--min-pos", type=int, default=15)
    il.add_argument("--min-neg", type=int, default=15)
    il.set_defaults(func=cmd_items_list)
    ia = isub.add_parser("add", help="把审片页新增的原因加进判官头配置")
    ia.add_argument("--config", type=Path, required=True, help="你自己的配置副本（不能是包内默认）")
    ia.add_argument("--id", required=True, help="英文 id，与审片页一致，例如 layer_seam")
    ia.add_argument("--name", required=True, help="中文名，例如 图层割裂")
    ia.add_argument("--kind", choices=["hard", "score"], required=True)
    ia.add_argument("--region", choices=["face", "hand", "body", "tile", "full"], default="full")
    ia.add_argument("--definition", help="一句话定义：什么样算这个问题")
    ia.add_argument("--added-at", help="创建时间 ISO 格式（默认现在）")
    ia.add_argument("--no-train", action="store_true", help="只记录，不训练判官头（例如语义类）")
    ia.set_defaults(func=cmd_items_add)

    sp = sub.add_parser("verify", help="链路自检：审片页的每个点击有没有到达判官头")
    sp.add_argument("--log", type=Path, required=True, help="审片记录 JSONL（审片页写的那个文件）")
    sp.add_argument("--config", type=Path, required=True, help="原因清单：审片页的 缺陷清单.json，或 judge_lab 配置")
    sp.add_argument("--exam", type=Path)
    sp.add_argument("--field-map", type=Path)
    sp.add_argument("--crops-root", type=Path)
    sp.add_argument("--heads", type=Path, help="判官头目录（有的话检查哪些审片已学进去）")
    sp.add_argument("--signal", metavar="BACKBONE", help="再做一步：用这个骨干验证标注确实推动了权重（要算特征）")
    backbone(sp), cache(sp), cv(sp)
    sp.add_argument("--out", type=Path, default=Path("reports/verify.md"))
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("trace", help="看一张图的每个点击变成了什么、被哪个判官头学了")
    sp.add_argument("--log", type=Path, required=True)
    sp.add_argument("--config", type=Path, required=True)
    sp.add_argument("--exam", type=Path)
    sp.add_argument("--field-map", type=Path)
    sp.add_argument("--heads", type=Path)
    sp.add_argument("--image", help="图片名或 id 的一部分；不给就看最近几张")
    sp.add_argument("--last", type=int, default=3)
    sp.add_argument("--all", action="store_true", help="每一项都列出来")
    sp.set_defaults(func=cmd_trace)
    return p


def main(argv=None) -> int:
    utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except JudgeLabError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename or e}", file=sys.stderr)
        return 2
