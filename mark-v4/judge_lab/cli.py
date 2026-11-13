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
from .data import DEFAULT_CONFIG, FieldMap, HeadItem, Review, label_counts, load_config, load_exam, load_log, split_exam
from .embeddings import IMG_EXTS, RegionCache, compute_region, ensure_embeddings
from .heads import (
    Head, auc, build_item_data, evaluate_at, fit_linear, image_max, load_active_heads, load_thresholds,
    needed_regions, oof_scores, region_rows, save_head, threshold_for_head, threshold_for_recall,
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
            coef, b = fit_linear(d.X, d.y, args.C)
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
