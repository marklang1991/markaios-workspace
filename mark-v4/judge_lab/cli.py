"""Command line: python -m judge_lab <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .backbones import BackboneSpec, parse_model_overrides
from .data import (
    DEFAULT_CONFIG, POLICY_DEFAULTS, POLICY_WEIGHTS, ExamRegistry, FieldMap, HeadItem, Review, apply_policy, assess,
    canonicalize, exam_guard, filter_sources, label_counts, label_for, load_config, load_exam, load_logs, policy_of,
    private_orphans, split_exam, why_code,
)
from .embeddings import IMG_EXTS, RegionCache, compute_region, ensure_embeddings, premade_crops
from .heads import (
    Head, auc, build_item_data, evaluate_at, fit_linear, image_max, load_active_heads, load_thresholds,
    load_head, load_manifest, needed_regions, oof_run, permuted, region_rows, save_head, scene_only_auc,
    threshold_for_head, threshold_for_recall,
)
from .stats import ab_markdown, ab_report, sample_size
from .util import JudgeLabError, assert_no_exam, fmt, md_table, now_iso, read_json, sha256_ids, utf8_stdio, \
    write_json, write_text

DISCLAIMER = "模型只出分数，不做通过/不通过判定；阈值为建议值，最终线由主人审核批准。"
CV_NAME = {"scene": "按场景", "image": "按图"}
OPTIMISTIC = "有场景信息却只能按图分组：同一场景的图会同时出现在训练和测试里，结果可能偏乐观"
WITHIN_FAIL_LOW = 0.60  # below this, a head does not tell its reason from the other reasons among failed pictures


# --------------------------------------------------------------------------- shared context
@dataclass
class Ctx:
    reviews: list[Review]  # every review read (after the source filter), exam pictures included
    train: list[Review]
    exam: list[Review]
    exam_ids: set[str]  # never to be fitted: the registry's tokens plus the ids of the reviews it matched
    summary: dict
    items: dict[str, HeadItem]
    log_stats: dict = field(default_factory=dict)
    orphans: dict[str, int] = field(default_factory=dict)  # clicks matching no item (public records: text -> n)
    private_orphans: int = 0  # the same in private records: a count only
    registry: ExamRegistry = field(default_factory=ExamRegistry)
    policy: dict = field(default_factory=dict)  # labelling policy in force (an item's own label_rule may differ)
    conflicts: list[Review] = field(default_factory=list)  # 推荐进入 AND on the exam list
    source_info: dict | None = None
    hide: set[str] = field(default_factory=set)  # ids of private records that must not be printed


def _split(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def cli_policy(args) -> dict:
    return {k: getattr(args, k) for k in POLICY_WEIGHTS if getattr(args, k, None) is not None}


def hidden(r: Review, args) -> bool:
    return r.private and not args.show_private


def policy_text(p: dict) -> str:
    def neg(w):
        return "不计入（当作不知道）" if w <= 0 else ("算反例" if w >= 1 else f"算反例，分量 {w:g}")

    def pos(w):
        return "不计入" if w <= 0 else ("算正例" if w >= 1 else f"算正例，分量 {w:g}")

    return (f"不过但没选的原因 → {neg(p['unselected_fail_weight'])}（标了「原因已选全」的图除外：按满分量算反例）；"
            f"整屏通过的图 → {neg(p['batch_pass_weight'])}；"
            f"自动带过来、你没碰过的原因 → {pos(p['carried_weight'])}；"
            f"不清楚是不是带过来的原因（旧记录）→ {pos(p['carried_unknown_weight'])}")


def conflict_text(conflicts: list[Review], args) -> str:
    shown = [r.id for r in conflicts if not hidden(r, args)][:5]
    eg = f"（例如 {'、'.join(shown)}）" if shown else "（私有记录，不显示）"
    return (f"{len(conflicts)} 张图既标了「推荐进入」（必须进训练）又在考卷名单里（永不训练）{eg}。"
            "工具不替你选：把它从考卷名单里拿掉，或者取消推荐")


def load_ctx(args, fit: bool = False, announce: bool = True) -> Ctx:
    """Logs -> source filter -> reason list (+ policy) -> exam split. fit: the command fits something, so a
    picture that is both 推荐进入 and exam is an error rather than a warning."""
    reviews, st = load_logs(args.log, FieldMap.load(args.field_map), getattr(args, "private_log", None) or ())
    src = None
    if args.source or args.exclude_source:
        reviews, src = filter_sources(reviews, _split(args.source) or None, _split(args.exclude_source))
    items, defaults = load_config(args.config, show_private=args.show_private)
    cli = cli_policy(args)
    apply_policy(items, cli)
    policy = {**POLICY_DEFAULTS, **defaults.get("label_policy", {}), **cli}
    orphans = canonicalize(reviews, items)
    n_private_orphans = private_orphans(reviews, items)
    registry = load_exam(getattr(args, "exam", None))
    train, exam, summary = split_exam(reviews, registry)
    conflicts = [r for r in exam if r.recommend]
    if announce:
        n_priv = sum(r.private for r in reviews)
        print(f"[数据] 日志 {summary['total_reviews']} 张" + (f"（其中私有记录 {n_priv} 张，只计数不显示）" if n_priv else "")
              + f"；考卷名单 {summary['exam_listed']} 个，日志中命中 {summary['exam_in_log']} 个（永不训练）；"
              f"训练池 {summary['train_pool']} 张")
        if src:
            print(f"[数据] 按来源筛选: 去掉 {src['dropped']} 张"
                  + (f"；{src['kept_by_recommend']} 张「推荐进入」的图不受筛选影响，保留" if src["kept_by_recommend"] else ""))
        if summary["exam_not_in_log"]:
            print(f"[数据] 注意: {summary['exam_not_in_log']} 个考卷条目不在日志中")
        if orphans:
            top = "、".join(f"「{k}」×{v}" for k, v in sorted(orphans.items(), key=lambda kv: -kv[1])[:5])
            print(f"[数据] 警告: {sum(orphans.values())} 次原因点选对不上清单里的任何一项，不会进任何判官头: {top}"
                  "（运行 verify 查看全部）")
        if n_private_orphans:
            print(f"[数据] 警告: 私有记录里有 {n_private_orphans} 次原因点选对不上清单（内容不显示）")
        print(f"[数据] 标注口径: {policy_text(policy)}")
    if conflicts:
        if fit:
            raise JudgeLabError(conflict_text(conflicts, args))
        if announce:
            print(f"[数据] 冲突: {conflict_text(conflicts, args)}")
    if getattr(args, "items", None):
        wanted = set(_split(args.items))
        if wanted - set(items):
            raise JudgeLabError(f"未知项目: {sorted(wanted - set(items))}")
        items = {k: v for k, v in items.items() if k in wanted}
    return Ctx(reviews, train, exam, exam_guard(registry, exam), summary, items, st, orphans, n_private_orphans,
               registry, policy, conflicts, src, {r.id for r in reviews if hidden(r, args)})


def make_spec(args, name: str, model_id: str | None = None) -> BackboneSpec:
    return BackboneSpec.create(name, parse_model_overrides(args.model_id), args.model_dir, args.device,
                               args.batch_size, not args.no_fp16, model_id=model_id)


def target_recall(item: HeadItem, args) -> float:
    flag = args.recall_hard if item.kind == "hard" else args.recall_score
    return item.target_recall if flag is None else flag


def pairs(reviews: list[Review]):
    return [(r.id, r.image) for r in reviews]


def embed(spec: BackboneSpec, regions, reviews: list[Review], ctx: Ctx, args, cache=None, force: bool = False):
    return ensure_embeddings(spec, regions, pairs(reviews), cache or args.cache, args.crops_root, force, hide=ctx.hide)


def enough(d, args) -> str | None:
    if d.n_pos < args.min_pos or d.n_neg < args.min_neg:
        return f"样本不足 (正 {d.n_pos} / 负 {d.n_neg}，需要 ≥{args.min_pos}/{args.min_neg})"
    return None


def shortfall(n_pos: int, n_neg: int, args) -> str:
    """「还差 7 个正例、3 个反例」; a side that already has enough is left out; '' when both do."""
    parts = ([f"{args.min_pos - n_pos} 个正例"] if n_pos < args.min_pos else []) + \
            ([f"{args.min_neg - n_neg} 个反例"] if n_neg < args.min_neg else [])
    return "还差 " + "、".join(parts) if parts else ""


def cv_label(run) -> str:
    return CV_NAME[run.mode] + ("（可能偏乐观）" if run.mode == "image" and run.n_scenes >= 2 else "")


# --------------------------------------------------------------------------- embed
def cmd_embed(args) -> None:
    ctx = load_ctx(args)
    trainable = [it for it in ctx.items.values() if it.trainable]
    regions = _split(args.regions) if args.regions else sorted(needed_regions(trainable))
    for name in _split(args.backbone):
        # Exam images are embedded too (frozen backbone = no fitting); they are only read by `exam`.
        embed(make_spec(args, name), regions, ctx.reviews, ctx, args, cache=args.out, force=args.force)
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
    ctx = load_ctx(args, fit=True)
    backbones = _split(args.backbones)
    trainable = [it for it in ctx.items.values() if it.trainable]
    results: dict[str, dict[str, dict]] = {it.id: {} for it in trainable}
    for name in backbones:
        caches = embed(make_spec(args, name), needed_regions(trainable), ctx.train, ctx, args)
        for it in trainable:
            d = build_item_data(it, ctx.train, caches, exclude_ids=ctx.exam_ids)
            r = {"n_pos": d.n_pos, "n_neg": d.n_neg, "n_crops": int(len(d.y)),
                 "n_no_crops": d.n_no_crops, "n_ambiguous": d.n_ambiguous}
            skip = enough(d, args)
            if not skip:
                try:
                    run = oof_run(d, args.folds, args.C, args.seed, by=args.group_by)
                except JudgeLabError as e:
                    skip = str(e)
            if skip:
                r["skipped"] = skip
            else:
                r.update(auc=auc(run.y, run.s), cv_mode=run.mode, cv_folds=run.k, cv_scenes=run.n_scenes,
                         cv_label=cv_label(run), **threshold_for_recall(run.y, run.s, target_recall(it, args)))
            results[it.id][name] = r

    out = {"created_at": now_iso(), "toolkit_version": __version__, "exam": ctx.summary,
           "settings": {"folds": args.folds, "C": args.C, "seed": args.seed, "min_pos": args.min_pos,
                        "min_neg": args.min_neg, "backbones": backbones, "group_by": args.group_by,
                        "label_policy": ctx.policy},
           "items": {}, "regions": {}, "overall": {}}
    rows = []
    for it in ctx.items.values():
        entry = {"name": it.name, "kind": it.kind, "region": it.region, "trainable": it.trainable,
                 "results": results.get(it.id, {}), "best_backbone": None}
        if not it.trainable:
            c = label_counts(ctx.train, it)
            rows.append([it.id, it.name, it.kind, it.region, "-", c["pos"], c["neg"], "-", "-", "-", "-", "-", "-", "-",
                         "不自动训练 (trainable=false)" + (f"；{it.note}" if it.note else "")])
        else:
            for bb, r in entry["results"].items():
                rows.append([it.id, it.name, it.kind, it.region, bb, r["n_pos"], r["n_neg"], r["n_crops"],
                             r.get("auc"), r.get("target_recall", target_recall(it, args)), r.get("recall"),
                             r.get("false_reject_rate"), r.get("threshold"), r.get("cv_label", "-"),
                             r.get("skipped", "")])
            scored = {bb: r["auc"] for bb, r in entry["results"].items() if r.get("auc") is not None}
            entry["best_backbone"] = _best(scored)
        out["items"][it.id] = entry
    for region in sorted({it.region for it in trainable}):
        m = _mean_auc(results, [it.id for it in trainable if it.region == region])
        out["regions"][region] = {"best_backbone": _best(m), "mean_auc": m}
    m = _mean_auc(results, results)
    out["overall"] = {"best_backbone": _best(m), "mean_auc": m}

    modes = [r for rs in results.values() for r in rs.values() if "cv_mode" in r]
    n_scene = sum(r["cv_mode"] == "scene" for r in modes)
    n_opt = sum(r["cv_mode"] == "image" and r["cv_scenes"] >= 2 for r in modes)
    out["cv"] = {"by_scene": n_scene, "by_image": len(modes) - n_scene, "by_image_although_scenes": n_opt}
    s = ctx.summary
    md = [
        "# 骨干比拼（线性探针，分组交叉验证）", "",
        f"- 生成时间: {out['created_at']}",
        f"- 考卷: 名单 {s['exam_listed']} 个，日志中 {s['exam_in_log']} 个，全部排除在外；训练池 {s['train_pool']} 张",
        f"- 设置: folds={args.folds}, C={args.C}, 最少正/负样本 {args.min_pos}/{args.min_neg}；"
        "召回/误拒率为图片级（取各裁剪最大分）",
        f"- 交叉验证分组（--group-by {args.group_by}）: 按场景 {n_scene} 次，按图 {len(modes) - n_scene} 次。"
        "「按场景」= 同一场景的图整组留出，测的是没见过的场景；「按图」= 只保证同一张图不拆开"
        + (f"。其中 {n_opt} 次{OPTIMISTIC}" if n_opt else ""),
        f"- 标注口径: {policy_text(ctx.policy)}", "",
        md_table(["项目", "名称", "类型", "区域", "骨干", "正样本", "负样本", "裁剪数", "AUC", "目标召回",
                  "召回", "误拒率", "阈值", "交叉验证", "备注"], rows),
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
    ctx = load_ctx(args, fit=True)
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
        caches = embed(spec, needed_regions(its), ctx.train, ctx, args)
        for it in its:
            d = build_item_data(it, ctx.train, caches, exclude_ids=ctx.exam_ids)
            skip = enough(d, args)
            if skip:
                print(f"[train] 跳过 {it.id} ({it.name}): {skip}")
                continue
            assert_no_exam(d.image_ids, ctx.exam_ids, f"train {it.id}", hide=ctx.hide)
            coef, b = fit_linear(d.X, d.y, args.C, d.w)
            head = save_head(args.out, it, coef, b, d.image_ids, {
                "backbone": name, "model_id": spec.model_id, "dim": int(d.X.shape[1]),
                "n_pos": d.n_pos, "n_neg": d.n_neg, "n_crops": int(len(d.y)), "C": args.C,
                "label_rule": it.label_rule, "label_policy": policy_of(it), "target_recall": target_recall(it, args),
                "exam_excluded": len(ctx.exam), "toolkit_version": __version__,
            })
            trained += 1
            print(f"[train] {head.path.name}: {it.name} 骨干={name} 正 {d.n_pos} / 负 {d.n_neg}")
    print(f"[train] 共训练 {trained} 个头 → {Path(args.out) / 'manifest.json'}")


# --------------------------------------------------------------------------- calibrate
def cmd_calibrate(args) -> None:
    ctx = load_ctx(args, fit=True)
    heads = load_active_heads(args.heads, set(ctx.items))
    out_path = Path(args.out or Path(args.heads) / "thresholds.json")
    old = load_thresholds(out_path).get("items", {})
    out = {"note": DISCLAIMER, "updated_at": now_iso(), "exam": ctx.summary, "items": dict(old)}
    specs: dict[tuple, BackboneSpec] = {}
    for item_id, head in heads.items():
        it = ctx.items[item_id]
        key = (head.meta["backbone"], head.meta["model_id"])
        spec = specs.setdefault(key, make_spec(args, *key))
        caches = embed(spec, needed_regions([it]), ctx.train, ctx, args)
        d = build_item_data(it, ctx.train, caches, exclude_ids=ctx.exam_ids)
        skip = enough(d, args)
        if skip:
            print(f"[calibrate] 跳过 {item_id}: {skip}")
            continue
        if sha256_ids(d.image_ids) != head.meta["data_hash"]:
            print(f"[calibrate] 注意 {item_id}: 当前数据与 {head.path.name} 训练时不同，建议重新 train")
        try:
            run = oof_run(d, args.folds, float(head.meta.get("C", args.C)), args.seed, by=args.group_by)
        except JudgeLabError as e:
            print(f"[calibrate] 跳过 {item_id}: {e}")
            continue
        y, s = run.y, run.s
        t = threshold_for_recall(y, s, target_recall(it, args))
        entry = {
            "name": it.public_name, "kind": it.kind, "head_file": head.path.name, "head_version": head.version,
            "threshold": t["threshold"], "proposed_threshold": t["threshold"],
            "target_recall": t["target_recall"], "oof_recall": t["recall"],
            "expected_false_reject_rate": t["false_reject_rate"], "oof_auc": auc(y, s),
            "n_pos": t["n_pos"], "n_neg": t["n_neg"], "cv_mode": run.mode, "cv_folds": run.k,
            "calibrated_at": now_iso(), "status": "proposed", "approved_by": None,
        }
        prev = old.get(item_id, {})
        if prev.get("status") == "approved" and not args.overwrite_approved:
            entry.update(status="approved", threshold=prev["threshold"], approved_by=prev.get("approved_by"),
                         approved_threshold_oof=evaluate_at(y, s, float(prev["threshold"])))
        out["items"][item_id] = entry
        print(f"[calibrate] {item_id} ({it.name}): 建议阈值 {t['threshold']:.4f} → OOF 召回 "
              f"{t['recall']:.3f} (目标 {t['target_recall']}), 预计误拒率 {fmt(t['false_reject_rate'])}，交叉验证{cv_label(run)}"
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
        assert_no_exam(head.train_ids, ctx.exam_ids, f"判官头 {head.path.name}", hide=ctx.hide)
        key = (head.meta["backbone"], head.meta["model_id"])
        spec = specs.setdefault(key, make_spec(args, *key))
        caches = embed(spec, needed_regions([it]), ctx.exam, ctx, args)
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
          f"- 考卷: 名单 {s['exam_listed']} 个，日志中 {s['exam_in_log']} 个；已核对所有判官头的训练集与考卷无交集",
          f"- 标注口径（考卷上同样适用，不知道的不算对也不算错）: {policy_text(ctx.policy)}", "",
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
    # head files only ever hold a private reason's placeholder name; the real one comes from the list, on request
    real_names = ({k: it.name for k, it in load_config(args.config, show_private=True)[0].items() if it.private}
                  if args.show_private and args.config else {})
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
            for iid, e in errors:
                print(f"  [score] 跳过 {iid}: {e}")
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
                    "name": real_names.get(h.item) or h.meta.get("name", h.item), "prob": round(float(p[j]), 4),
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
    reviews, _ = load_logs(args.log, FieldMap.load(args.field_map), args.private_log or ())
    if args.source or args.exclude_source:
        reviews, src = filter_sources(reviews, _split(args.source) or None, _split(args.exclude_source))
        print(f"[数据] 按来源筛选: 去掉 {src['dropped']} 张"
              + (f"；{src['kept_by_recommend']} 张「推荐进入」的图不受筛选影响，保留" if src["kept_by_recommend"] else ""))
    try:
        res = ab_report(reviews, args.group_by, args.scene_field or None, args.baseline, args.tolerance, args.alpha,
                        private_scene=None if args.show_private else "（私有记录）")
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
    ctx = load_ctx(args, announce=False)
    rows = []
    for it in ctx.items.values():
        c = label_counts(ctx.train, it)
        state = "不自动训练" if not it.trainable else (shortfall(c["pos"], c["neg"], args) or "可训练")
        rows.append([it.id, it.name, it.kind, it.region, it.added_at or "-", c["pos"], c["neg"], c["ambiguous"],
                     state, it.definition or ("（私有，不显示）" if it.private and it.has_definition else "-")])
    print(f"[数据] 训练池 {ctx.summary['train_pool']} 张（考卷不计）")
    print(f"[数据] 标注口径: {policy_text(ctx.policy)}")
    if ctx.conflicts:
        print(f"[数据] 冲突: {conflict_text(ctx.conflicts, args)}")
    print(md_table(["id", "名称", "类型", "区域", "新增时间", "正例", "反例", "不计入", "状态", "定义"], rows))
    print("\n说明: 「不计入」= 这张图对这一项既不算正例也不算反例：判不过但没选这一项的图（不知道有没有这个问题）、"
          "按口径不算的整屏通过和自动带过来的原因，以及这一项创建之前审的图（当时没有这个选项）。")


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


def _signal(it: HeadItem, train: list[Review], caches, ctx: Ctx, args) -> dict | None:
    """Does the owner's labelling move this item's weights, and is it the defect or the scene that is learned?"""
    d = build_item_data(it, train, caches, exclude_ids=ctx.exam_ids)
    if enough(d, args):
        return None
    # failed pictures where this reason was not selected: unknown for fitting, but still worth scoring
    unknown = [r for r in train if why_code(r, it) == "unknown"]
    extra = build_item_data(it, unknown, caches, labelled=False) if unknown else None
    try:
        run = oof_run(d, args.folds, args.C, args.seed, by=args.group_by, extra=extra)
    except JudgeLabError as e:
        return {"name": it.name, "n_pos": d.n_pos, "n_neg": d.n_neg, "error": str(e)}
    real = auc(run.y, run.s)
    shuffled = []
    for k in range(3):
        try:
            pr = oof_run(permuted(d, args.seed + 1 + k, within_scene=run.mode == "scene"), args.folds, args.C,
                         args.seed, by=run.mode)
        except JudgeLabError:
            continue
        a = auc(pr.y, pr.s)
        if a is not None:
            shuffled.append(a)
    shuf = float(np.mean(shuffled)) if shuffled else 0.5
    scene = scene_only_auc(d)
    failed = {r.id for r in train if r.decision == "fail"}
    pos = run.s[run.y == 1].tolist()
    others = [s for i, y, s in zip(run.ids, run.y.tolist(), run.s.tolist()) if y == 0 and i in failed]
    others += list((run.extra or {}).values())
    why = (auc(np.array([1] * len(pos) + [0] * len(others)), np.array(pos + others))
           if len(pos) >= 3 and len(others) >= 3 else None)
    if real is None or real < 0.70 or real - shuf < 0.10:
        verdict = "学不出来"
    elif scene is not None and scene >= real - 0.03:
        verdict = "和场景分不开"
    else:
        verdict = "学到了"
    return {"name": it.name, "n_pos": d.n_pos, "n_neg": d.n_neg, "real": real, "shuffled": shuf, "scene_only": scene,
            "within_fail": why, "verdict": verdict, "run": run}


def cmd_verify(args) -> int:
    """Checks, on the real files, every link between a click on the review page and a head's weights."""
    ctx = load_ctx(args, announce=False)
    reviews, train, items, st, summary = ctx.reviews, ctx.train, ctx.items, ctx.log_stats, ctx.summary
    heads = _active_heads(args.heads)
    checks: list[tuple[str, str, str]] = []

    def add(level: str, title: str, detail: str) -> None:
        checks.append((level, title, detail))

    # 1. every line of the log(s) is read
    n_pass = sum(r.decision == "pass" for r in reviews)
    n_skipped = len(st["skipped"]) + st["skipped_private"]
    eg = "；".join((f"{name} " if st["files"] > 1 else "") + f"第 {n} 行 {why}" for n, why, name in st["skipped"][:3])
    if st["skipped_private"]:
        eg += ("；" if eg else "") + f"私有记录里 {st['skipped_private']} 行（内容不显示）"
    if st["parsed"] == 0:
        add("断", "审片记录读入", f"{st['lines']} 行一条都没读进来" + (f"。{eg}" if eg else ""))
    elif n_skipped:
        add("断", "审片记录读入", f"{n_skipped} / {st['lines']} 行被跳过，这些审片不会被用到。{eg}")
    else:
        add("通", "审片记录读入", (f"{st['files']} 个文件，" if st["files"] > 1 else "")
            + f"{st['lines']} 行全部读入，共 {st['reviews']} 张图"
            + (f"，按来源筛选后用 {len(reviews)} 张" if ctx.source_info else "")
            + f"（过 {n_pass} / 不过 {len(reviews) - n_pass}；重审 {st['re_reviews']} 次，以最后一次为准"
            + (f"；其中私有记录 {st['private']} 张，只计数不显示" if st["private"] else "") + "）")
    if ctx.source_info:
        add("通", "来源筛选", f"按 --source / --exclude-source 去掉 {ctx.source_info['dropped']} 张"
            + (f"；{ctx.source_info['kept_by_recommend']} 张「推荐进入」的图不受筛选影响，保留"
               if ctx.source_info["kept_by_recommend"] else ""))

    # 2. every selected reason reaches an item
    fails = [r for r in reviews if r.decision == "fail"]
    clicks = sum(len(r.reasons) for r in fails)
    no_reason = sum(1 for r in fails if not r.reasons)
    n_orphans = sum(ctx.orphans.values()) + ctx.private_orphans
    if n_orphans:
        eg = "、".join(f"「{k}」×{v}" for k, v in sorted(ctx.orphans.items(), key=lambda kv: -kv[1]))
        if ctx.private_orphans:
            eg += ("；" if eg else "") + f"私有记录里 {ctx.private_orphans} 次（内容不显示）"
        add("断", "原因 → 清单", f"{n_orphans} / {clicks} 次点选对不上清单里的任何一项，不会进任何判官头: {eg}。"
            "把旧名字写进对应项的 merged_from（曾用名），或把这一项加回清单")
    else:
        add("通", "原因 → 清单", f"{clicks} 次原因点选全部对上清单（{len(items)} 项）")
    if no_reason:
        add("注意", "不过但没选原因", f"{no_reason} 张判不过却没有原因：它们只能教模型“这张不过”，教不了“为什么”")

    # 3. time stamps (needed by reasons added later)
    late = [it for it in items.values() if it.added_at]
    lost = {it.name: sum(1 for r in train if it.id in r.reasons and r.decision == "fail"
                         and why_code(r, it) == "time") for it in late}
    lost = {k: v for k, v in lost.items() if v}
    if lost:
        add("断", "新增原因的时间", "这些点选因为时间对不上而不计入: " + "、".join(f"{k}×{v}" for k, v in lost.items())
            + "。合并或改名后，新项的创建时间要写成最早那个旧项的创建时间")
    elif st["no_time"] and late:
        add("注意", "新增原因的时间", f"{st['no_time']} 条审片没有时间；对 {len(late)} 个后来新增的原因，这些图不计入（不会被当成反例）")
    else:
        add("通", "新增原因的时间", f"{len(late)} 个后来新增的原因，创建之后的审片都按时间正确计入")

    # 4. the pictures are still on disk (a private record's path is never shown)
    gone = [r for r in reviews if r.image is None or not r.image.is_file()]
    eg = next((f"（例如 {r.image}）" for r in gone if not hidden(r, args)), "（私有记录，不显示）")
    if gone and len(gone) == len(reviews):
        add("断", "图片文件", f"{len(gone)} 张图一张都找不到{eg}：没有图就提不了特征，训练不了")
    elif gone:
        add("注意", "图片文件", f"{len(gone)} / {len(reviews)} 张图找不到{eg}，这些审片进不了训练")
    else:
        add("通", "图片文件", f"{len(reviews)} 张图都在盘上")

    # 5. crops for face / hand items
    need = [it for it in items.values() if it.trainable and it.region in ("face", "hand")]
    if need:
        blocked, coarse = [], []
        for it in need:
            labelled = [r for r in train if label_for(r, it) is not None]
            have = (sum(1 for r in labelled if premade_crops(args.crops_root, r.id, it.region, r.image))
                    if args.crops_root else 0)
            pos = sum(1 for r in labelled if label_for(r, it) == 1)
            if labelled and have < len(labelled):
                (coarse if it.fallback_region else blocked).append(f"{it.name}（切图 {have}/{len(labelled)} 张，正例 {pos}）")
        if blocked:
            add("断", "脸 / 手切图", "这些项只看脸或手的切图，现在没有切图，点了也训不了: " + "；".join(blocked))
        if coarse:
            add("注意", "脸 / 手切图", f"{len(coarse)} 个看脸或手的项缺切图，暂时退而看整图分块，判得会粗；接上切图后要重新训练: "
                + "；".join(coarse[:4]) + (" 等" if len(coarse) > 4 else ""))
        if not blocked and not coarse:
            add("通", "脸 / 手切图", f"{len(need)} 个看脸或手的项，切图齐全")

    # 6. how the clicks are being read right now (the labelling policy), with what the records contain
    picks = [r.carried.get(x) for r in fails for x in r.reasons]
    own_rule = sum(1 for it in items.values() if any(policy_of(it)[k] != ctx.policy[k] for k in POLICY_WEIGHTS))
    add("通", "标注口径", policy_text(ctx.policy) + (f"（清单里有 {own_rule} 项另有单独设置）" if own_rule else "")
        + f"。现有记录：不过 {len(fails)} 张，其中标了「原因已选全」的 {sum(r.reasons_complete for r in fails)} 张；"
        f"整屏通过 {sum(r.batch_pass for r in reviews)} 张；原因点选里自动带过来没碰过的 {sum(c is True for c in picks)} 次、"
        f"不清楚的 {sum(c is None for c in picks)} 次、你自己选的 {sum(c is False for c in picks)} 次")

    # 7. 推荐进入: must be trained on, so it cannot also be an exam picture
    n_rec = sum(r.recommend for r in reviews)
    if ctx.conflicts:
        add("断", "推荐进入", conflict_text(ctx.conflicts, args) + "。解决之前 probe / train / calibrate 会直接报错")
    else:
        add("通", "推荐进入", f"{n_rec} 张图标了「推荐进入」：一定进训练，不受来源筛选影响，整屏通过也按满分量算；和考卷名单没有冲突"
            if n_rec else "没有图标「推荐进入」")
    unhashed = sum(1 for r in train if not r.sha256) if ctx.registry.hashes else 0
    if unhashed:
        add("注意", "考卷名单", f"名单里有按 sha256 登记的图，但 {unhashed} 张审片既没有 sha256 字段、图片文件也读不出来，"
            "没法核对它们是不是考卷")

    # 8. per item: how much the owner has taught it, and whether a head has learned it
    rows, counts = [], {it.id: label_counts(train, it) for it in items.values()}
    for it in items.values():
        c = counts[it.id]
        scored = sum(1 for r in train if it.id in r.scores and label_for(r, it) == 1)
        head = heads.get(it.id)
        if not it.trainable:
            state = it.note or "不自动训练"
        elif head:
            seen = set(head.train_ids)
            new = sum(1 for r in train if label_for(r, it) is not None and r.id not in seen)
            state = f"判官头 v{head.version}（学了 {len(seen)} 张）；之后新增 {new} 张还没学进去"
        else:
            state = shortfall(c["pos"], c["neg"], args) or "样本够了，还没训练"
        rows.append([it.name, "硬门" if it.kind == "hard" else "计分", it.region, c["pos"], c["neg"], c["ambiguous"],
                     scored if it.kind == "score" else "-", state])

    # 9. the meaning of each reason in use (for people and for a vision LLM prompt)
    used = [it for it in items.values() if counts[it.id]["pos"] > 0]
    undefined = [it.name for it in used if not it.has_definition]
    if not used:
        add("通", "含义", "还没有哪个原因有正例")
    else:
        add("注意" if undefined else "通", "含义", f"有正例的 {len(used)} 个原因里，{len(used) - len(undefined)} 个写了含义"
            + (f"；还没写的: {'、'.join(undefined[:5])}{' 等' if len(undefined) > 5 else ''}。"
               "判官头不读含义，但人和看图大模型要靠它（export-prompts 会导出）" if undefined else ""))

    # 10. time spent: reported, never used
    timed = [r.seconds for r in reviews if r.seconds is not None]
    if timed:
        add("通", "审片用时", f"{len(timed)} 张图有用时记录，共 {sum(timed) / 60:.1f} 分钟，平均每张 {sum(timed) / len(timed):.1f} 秒。"
            "这只是工作量统计，不用于训练")

    # 11. heads
    if not heads:
        add("注意", "判官头", "还没有训练过任何判官头：到目前为止，审片只进了记录和配方统计，没有改变任何模型")
    else:
        known = {k: h for k, h in heads.items() if k in items}
        stale = sum(1 for k, h in known.items() if any(
            label_for(r, items[k]) is not None and r.id not in set(h.train_ids) for r in train))
        add("通" if not stale else "注意", "判官头", f"{len(known)} 个判官头在用" + (f"，其中 {stale} 个有新审片还没学进去（重新 train）" if stale else "，都包含最新审片"))
        moved = [f"{items[k].name}（判官头 v{h.version} 看 {h.meta.get('region')}，清单现在是 {items[k].region}）"
                 for k, h in known.items() if h.meta.get("region") != items[k].region]
        if moved:
            add("注意", "判官头看的区域", "这些原因的分组或区域在判官头训练之后改过，旧头还在按旧区域打分，请重新 train: " + "；".join(moved))
        other_policy = [items[k].name for k, h in known.items()
                        if h.meta.get("label_policy") and h.meta["label_policy"] != policy_of(items[k])]
        if other_policy:
            add("注意", "判官头的标注口径", f"{len(other_policy)} 个判官头训练时用的标注口径和现在的不一样（上面的正例、反例数是按现在的口径算的）: "
                + "、".join(other_policy[:5]) + (" 等" if len(other_policy) > 5 else "") + "。要按现在的口径，就重新 train")
        if len(known) < len(heads):
            add("注意", "多余的判官头", f"判官头目录里有 {len(heads) - len(known)} 个头对不上现在清单里的任何一项（改过 id，或后来设成了私有），"
                "它们不会被用到；如果是后来设成私有的，旧文件里还留着原来的名字，删掉后重新 train")

    # 12. optional: do the labels move the weights, and is it the defect that is learned, not the scene?
    sig_rows, sig_notes = [], []
    if args.signal:
        spec = make_spec(args, args.signal)
        its = [it for it in items.values() if it.trainable]
        caches = embed(spec, needed_regions(its), train, ctx, args)
        res = [r for r in (_signal(it, train, caches, ctx, args) for it in its) if r]
        failed_cv = [r for r in res if "error" in r]
        res = [r for r in res if "error" not in r]
        sig_rows = [[r["name"], r["n_pos"], r["n_neg"], r["real"], r["shuffled"], r["scene_only"], r["within_fail"],
                     r["verdict"]] for r in res]
        weak = [r["name"] + ("（正反例几乎按场景分开，同一场景里没有可对比的图）" if (r["scene_only"] or 0) >= 0.9 else "")
                for r in res if r["verdict"] == "学不出来"]
        mixed = [r["name"] for r in res if r["verdict"] == "和场景分不开"]
        by_scene = [r["name"] for r in res if r["run"].mode == "scene"]
        by_image = [r["name"] for r in res if r["run"].mode == "image"]
        optimistic = [r["name"] for r in res if r["run"].mode == "image" and r["run"].n_scenes >= 2]
        sig_notes.append("- 交叉验证分组: " + "；".join(
            x for x in (f"按场景 {len(by_scene)} 项（同一场景的图整组留出，测的是没见过的场景）" if by_scene else "",
                        f"按图 {len(by_image)} 项" if by_image else "") if x)
            + (f"。{OPTIMISTIC}: {'、'.join(optimistic)}" if optimistic else ""))
        for r in failed_cv:
            sig_notes.append(f"- {r['name']}: 没做成（{r['error']}）")
        # learned, yet among failed pictures the head cannot tell this reason from the others
        blurred = [r["name"] for r in res if r["verdict"] == "学到了" and r["within_fail"] is not None
                   and r["within_fail"] < WITHIN_FAIL_LOW]
        if not sig_rows:
            add("注意", "标注 → 权重", "还没有哪一项的样本够做这个检查" + (f"；{len(failed_cv)} 项分不出组" if failed_cv else ""))
        elif weak or mixed or blurred:
            add("注意", "标注 → 权重", f"{len(res) - len(weak) - len(mixed)} / {len(res)} 项学到了"
                + (f"；学不出来的: {'、'.join(weak)}（样本少、前后标准不一致、区域不对，或这类问题骨干看不出来）" if weak else "")
                + (f"；和场景分不开的: {'、'.join(mixed)}（只凭“是哪个场景”就能猜得差不多一样准，分不清学到的是缺陷还是场景；"
                   "要在更多场景里各攒一些正例和反例）" if mixed else "")
                + (f"；学到了、但在不过的图里分不出是不是这一项的: {'、'.join(blurred)}（反例只有判过的图时，头容易把别的原因造成的不过"
                   "也打高分；也可能那些图其实也有这个问题只是没点。在审片页把原因选全，记录带 reasons_complete，就能分清）" if blurred else ""))
        else:
            add("通", "标注 → 权重", f"{len(res)} 项都学到了：用你的标注训练明显好于把标注打乱，也不是靠认场景")
        if optimistic:
            add("注意", "交叉验证分组", f"{len(optimistic)} 项{OPTIMISTIC}")

    worst = "断" if any(c[0] == "断" for c in checks) else ("注意" if any(c[0] == "注意" for c in checks) else "通")
    head_line = {"断": "链路有断点：下面标「断」的地方，你的点击现在到不了模型", "注意": "链路是通的，有几处要留意",
                 "通": "链路全通"}[worst]
    n_private_logs = len(getattr(args, "private_log", None) or ())
    md = ["# 链路自检：审片页的点击有没有到达判官头", "", f"- 时间: {now_iso()}",
          f"- 审片记录: {'、'.join(str(p) for p in args.log)}"
          + (f"（另有 {n_private_logs} 个私有记录文件，其中 {st['private']} 张图：只计数不显示）" if n_private_logs else ""),
          f"- 原因清单: {args.config}", f"- 考卷: {summary['exam_listed']} 个（日志中 {summary['exam_in_log']} 个，永不训练）",
          "", f"**结论: {head_line}**", "", md_table(["状态", "环节", "说明"], checks), "",
          "## 每一项学到哪一步", "",
          md_table(["原因", "类型", "看哪里", "正例", "反例", "不计入", "带分数的正例", "状态"], rows), "",
          "- 「不计入」= 对这一项既不算正例也不算反例：判不过但没选它的图（不知道有没有这个问题）、按口径不算的整屏通过和自动带过来的原因、"
          "这一项创建之前审的图。", ""]
    if sig_rows or sig_notes:
        md += ["## 你的标注有没有改变权重（交叉验证，没见过的图上测）", "",
               md_table(["原因", "正例", "反例", "用你的标注", "同场景内打乱", "只看场景能猜多准", "只在不过的图里分原因", "结论"],
                        sig_rows), "", *sig_notes,
               "- 「用你的标注」明显高于「同场景内打乱」= 权重是被你的点选推动的，不是巧合。按场景分组时，标注只在同一场景的图之间打乱"
               "（每个场景的正例比例不变）；按图分组时在全部图之间打乱（约 0.5）。",
               "- 「只看场景能猜多准」= 完全不看图，只凭同场景其它图的结果来猜这张。它接近「用你的标注」时结论写「和场景分不开」："
               "分不清头学到的是缺陷还是场景。低于 0.5 或「-」= 场景帮不上忙。",
               "- 「只在不过的图里分原因」高 = 这个头分得清“因这一项不过”和“因别的原因不过”，学的是原因，不只是过或不过。"
               "对照的那一边包括“判不过但没选这一项”的图；其中有些其实也有这个问题（只是你没点），所以这个数是保守的，只会偏低。"
               "接近 0.5 = 这个头把别的原因造成的不过也当成了这一项：反例只有判过的图时常见，原因选全（reasons_complete）后会改善。", ""]
    out = Path(args.out)
    write_text(out, "\n".join(md))
    print(f"[verify] {head_line}")
    for lv, t, dsc in checks:
        print(f"  [{lv}] {t}: {dsc}")
    print(f"[verify] 报告: {out}")
    return 1 if worst == "断" else 0


def cmd_trace(args) -> int:
    """For one reviewed picture: what each click became, and which head has learned it."""
    ctx = load_ctx(args, announce=False)
    reviews, items = ctx.reviews, ctx.items
    exam_ids = {r.id for r in ctx.exam}
    heads = _active_heads(args.heads)
    if args.image:
        key = args.image.replace("\\", "/").lower()
        picked = [r for r in reviews if key in r.id.lower() or key in str(r.image or "").replace("\\", "/").lower()]
    else:
        picked = reviews[-args.last:]
    if not picked:
        raise JudgeLabError("审片记录里找不到这张图")
    yes = {True: "是", False: "否"}
    for r in picked:
        if hidden(r, args):
            print("\n=== （私有记录：内容不显示；要看就加 --show-private）")
            continue
        print(f"\n=== {r.id}\n  图片: {r.image}\n  决定: {'过' if r.decision == 'pass' else '不过'}   审片时间: {r.reviewed_at or '没有记录'}")
        print(f"  来源: {r.source or '没有记录'}   整屏通过: {yes[r.batch_pass]}   推荐进入: {yes[r.recommend]}"
              + (f"   原因已选全: {yes[r.reasons_complete]}" if r.decision == "fail" else "")
              + (f"   场景: {r.scene}" if r.scene else ""))
        if r.note.strip():
            print(f"  备注: {r.note.strip()}（只在这里显示，不用于训练）")
        if r.seconds is not None:
            print(f"  审片用时: {r.seconds:g} 秒（仅统计，不用于训练）")
        if r.id in exam_ids:
            print("  这是考卷图：永不参加训练，只用来验收判官头")
            if r.recommend:
                print("  冲突: 这张图同时标了「推荐进入」。工具不替你选：从考卷名单里拿掉它，或者取消推荐")
        rows, normal, tally, light = [], 0, {}, set()
        for it in items.values():
            lab, why, wgt = assess(r, it)
            picked_it = it.id in r.reasons and r.decision == "fail"
            if lab == 0 and not args.all:
                normal += 1
                light.add(wgt)
                continue
            if lab is None and not picked_it and not args.all:
                code = why_code(r, it)
                tally[code] = tally.get(code, 0) + 1
                continue
            head = heads.get(it.id)
            if r.id in exam_ids:
                where = "考卷，不训练"
            elif not it.trainable:
                where = it.note or "不自动训练"
            elif lab is None:
                where = "不计入"
            elif head is None:
                c = label_counts(ctx.train, it)
                where = f"还没有判官头（正例 {c['pos']} 个）"
            elif r.id in set(head.train_ids):
                where = f"已学进判官头 v{head.version}（{head.meta.get('created_at', '')[:16]}）"
            else:
                where = f"还没学进去：判官头 v{head.version} 训练时没有这张图，重新 train 才会进去"
            rows.append([it.name, {1: "有问题（正例）", 0: "正常（反例）", None: "不计入"}[lab], why,
                         r.scores.get(it.id, "-") if it.kind == "score" else "-", "-" if lab is None else f"{wgt:.2f}", where])
        for x in r.reasons:
            if x not in items:
                rows.append([x, "对不上清单", "这个原因不在清单里，也不是任何一项的曾用名", "-", "-", "到不了任何判官头"])
        if rows:
            print(md_table(["原因", "这张图教给模型的", "为什么", "你打的分", "训练权重", "到了哪里"], rows))
        part = (f"（分量 {next(iter(light)):g}）" if len(light) == 1 and light != {1.0} else "")
        tail = [f"另有 {n} 项{txt}" for code, txt in (
            ("unknown", "不知道有没有这个问题（判不过但没选它们），不计入"), ("time", "不计入（当时还没有这些选项）"),
            ("batch_off", "不计入（整屏通过，按口径不算）")) if (n := tally.get(code))]
        print(f"  其余 {normal} 项: 这张图算“正常”的例子{part}" + "".join(f"；{t}" for t in tail))
    return 0


# --------------------------------------------------------------------------- export for a vision LLM prompt
def cmd_export_prompts(args) -> int:
    """Per reason: its meaning in the owner's words plus a few example file names, for writing a VLM prompt.
    Examples never come from exam pictures or private records; a private reason is exported only on request."""
    ctx = load_ctx(args)
    pool = [r for r in ctx.train if not r.private and r.image is not None]  # private records: never, whatever the flags
    out_items, n_hidden = [], 0
    for it in ctx.items.values():
        if it.private and not args.show_private:
            n_hidden += 1
            continue
        pos, neg = [], []
        for r in pool:
            lab, _, w = assess(r, it)
            if lab == 1:
                pos.append((w, r.image.name))
            elif lab == 0 and r.decision == "pass":
                neg.append((w, r.image.name))

        def top(xs: list) -> list[str]:  # heaviest first (owner-chosen, most severe); the log's order breaks ties
            names: list[str] = []
            for _, name in sorted(xs, key=lambda t: -t[0]):
                if name not in names:
                    names.append(name)
            return names[:max(0, args.examples)]

        c = label_counts(ctx.train, it)
        out_items.append({
            "id": it.id, "name": it.name, "kind": it.kind, "group": it.group, "region": it.region,
            "definition": it.definition, "has_definition": it.has_definition, "trainable": it.trainable,
            "n_pos": c["pos"], "n_neg": c["neg"], "positive_examples": top(pos), "negative_examples": top(neg),
        })
    undefined = [e["name"] for e in out_items if not e["has_definition"]]
    data = {"created_at": now_iso(), "toolkit_version": __version__, "exam": ctx.summary,
            "examples_per_side": args.examples, "private_items_not_exported": n_hidden,
            "items_without_definition": undefined, "items": out_items,
            "note": "例图只来自非考卷、非私有的审片；反例只取判过的图。含义是主人的原话，判官头不读它。"}
    json_path = Path(args.out) if Path(args.out).suffix.lower() == ".json" else Path(args.out).with_suffix(".json")
    write_json(json_path, data)
    kind = {"hard": "硬门（有就不过）", "score": "计分（0–10 分，越低越严重）"}
    md = ["# 原因清单：含义和例图（给看图大模型写提示词用）", "", f"- 生成时间: {data['created_at']}",
          f"- 共 {len(out_items)} 个原因" + (f"；另有 {n_hidden} 个私有原因没有导出" if n_hidden else ""),
          f"- 每个原因最多 {args.examples} 张正例、{args.examples} 张反例。例图只来自非考卷、非私有的审片；反例只取判过的图",
          "- " + (f"**还没有写含义的原因（{len(undefined)} 个）**: {'、'.join(undefined)}" if undefined else "每个原因都写了含义"), ""]
    for e in out_items:
        md += [f"## {e['name']}", "",
               f"- id: `{e['id']}`；{kind[e['kind']]}；分组: {e['group'] or '-'}；看: {e['region']}"
               + ("" if e["trainable"] else "；不训练判官头，由人或看图大模型判断"),
               f"- 含义: {e['definition']}" if e["has_definition"] else "- 含义: **（还没有写含义）**",
               f"- 正例（{e['n_pos']} 张里取）: {'、'.join(e['positive_examples']) or '还没有'}",
               f"- 反例（判过的图）: {'、'.join(e['negative_examples']) or '还没有'}", ""]
    write_text(json_path.with_suffix(".md"), "\n".join(md))
    print(f"[export-prompts] {len(out_items)} 个原因 → {json_path} 和 {json_path.with_suffix('.md')}"
          + (f"；{len(undefined)} 个还没有写含义" if undefined else "")
          + (f"；{n_hidden} 个私有原因没有导出（加 --show-private 才导出）" if n_hidden else ""))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m judge_lab", description="专科判官工具包 judge_lab " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def private(sp):
        sp.add_argument("--show-private", action="store_true",
                        help="显示私有原因的真名和私有记录的内容（默认只出现代号和数量）")

    def logs(sp):
        sp.add_argument("--log", type=Path, nargs="+", required=True, metavar="FILE",
                        help="审片记录 JSONL，可以给多个；同一张图以后面文件里的为准")
        sp.add_argument("--private-log", type=Path, nargs="+", default=[], metavar="FILE",
                        help="私有审片记录：照常计数和训练，但屏幕和报告里只出现数量")
        sp.add_argument("--source", metavar="A,B", help="只用这些来源（记录里的 source）的审片")
        sp.add_argument("--exclude-source", metavar="A,B", help="去掉这些来源的审片（「推荐进入」的图不会被去掉）")
        sp.add_argument("--field-map", type=Path, help="字段映射 JSON")
        private(sp)

    def data(sp, exam_required=False, config_required=False, crops=True, items=True):
        logs(sp)
        sp.add_argument("--exam", type=Path, required=exam_required,
                        help="考卷名单（永不训练）：id / 图片路径 / sha256，文本、JSON 或 JSONL")
        sp.add_argument("--config", type=Path, required=config_required, **({} if config_required else {"default": DEFAULT_CONFIG}),
                        help="原因清单：审片页的 缺陷清单.json，或 judge_lab 配置 (默认 configs/heads_v1.json)")
        if crops:
            sp.add_argument("--crops-root", type=Path, help="预裁剪目录 <root>/<image_id>/<region>_<k>.png")
        if items:
            sp.add_argument("--items", help="只处理这些项目（逗号分隔）")
        g = sp.add_argument_group("标注口径（0 = 不计入，1 = 按满分量；不写就用配置里的或默认值）")
        g.add_argument("--unselected-fail-weight", type=float, metavar="W", help="判不过但没选的原因当反例的分量（默认 0 = 不知道）")
        g.add_argument("--batch-pass-weight", type=float, metavar="W", help="整屏通过的图当反例的分量（默认 0.5）")
        g.add_argument("--carried-weight", type=float, metavar="W", help="自动带过来、没碰过的原因当正例的分量（默认 0）")
        g.add_argument("--carried-unknown-weight", type=float, metavar="W", help="旧记录里不清楚是否带过来的原因当正例的分量（默认 0.5）")

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
            sp.add_argument("--group-by", choices=["auto", "scene", "image"], default="auto",
                            help="交叉验证怎么分组：scene = 同一场景的图整组留出；image = 只保证同一张图不拆开；"
                                 "auto = 场景够多（≥4 个、过半的图有场景）就按场景")

    sp = sub.add_parser("embed", help="提取并缓存特征")
    data(sp)
    backbone(sp)
    sp.add_argument("--backbone", default="siglip2", help="逗号分隔，可多个")
    sp.add_argument("--regions", help="默认: 配置中可训练项目用到的区域")
    sp.add_argument("--out", type=Path, default=Path("cache"))
    sp.add_argument("--force", action="store_true", help="忽略已有缓存重新计算")
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("probe", help="骨干比拼（线性探针）")
    data(sp, exam_required=True), backbone(sp), cache(sp), cv(sp)
    sp.add_argument("--backbones", default="siglip2,dinov2,clip_l")
    sp.add_argument("--out", type=Path, default=Path("reports/probe.md"))
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("train", help="训练最终判官头")
    data(sp, exam_required=True), backbone(sp), cache(sp), cv(sp, folds=False)
    sp.add_argument("--probe", type=Path, help="probe 输出的 JSON（按项目选最佳骨干）")
    sp.add_argument("--backbone", help="所有项目统一用这个骨干（优先于 --probe）")
    sp.add_argument("--out", type=Path, default=Path("heads"))
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("calibrate", help="按目标召回率给出建议阈值")
    data(sp, exam_required=True), backbone(sp), cache(sp), cv(sp)
    sp.add_argument("--heads", type=Path, default=Path("heads"))
    sp.add_argument("--out", type=Path, help="默认 <heads>/thresholds.json")
    sp.add_argument("--overwrite-approved", action="store_true", help="连主人已批准的阈值也覆盖")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("exam", help="只在考卷上评估（验收）")
    data(sp, exam_required=True), backbone(sp), cache(sp)
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
    sp.add_argument("--config", type=Path, help="原因清单；只在 --show-private 时用来查私有原因的真名")
    private(sp)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("stats", help="A/B 统计")
    ssub = sp.add_subparsers(dest="stats_cmd", required=True)
    ab = ssub.add_parser("ab", help="分组通过率、置信区间、显著性、分场景回归检查")
    logs(ab)
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
    private(n)
    n.set_defaults(func=cmd_stats_n)

    sp = sub.add_parser("items", help="主人的缺陷清单：查看样本数 / 新增项目")
    isub = sp.add_subparsers(dest="items_cmd", required=True)
    il = isub.add_parser("list", help="每个项目有多少正例/反例，够不够训练")
    data(il, crops=False, items=False)
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
    private(ia)
    ia.set_defaults(func=cmd_items_add)

    sp = sub.add_parser("verify", help="链路自检：审片页的每个点击有没有到达判官头")
    data(sp, config_required=True, items=False)
    sp.add_argument("--heads", type=Path, help="判官头目录（有的话检查哪些审片已学进去）")
    sp.add_argument("--signal", metavar="BACKBONE", help="再做一步：用这个骨干验证标注确实推动了权重（要算特征）")
    backbone(sp), cache(sp), cv(sp)
    sp.add_argument("--out", type=Path, default=Path("reports/verify.md"))
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("trace", help="看一张图的每个点击变成了什么、被哪个判官头学了")
    data(sp, config_required=True, crops=False, items=False)
    sp.add_argument("--heads", type=Path)
    sp.add_argument("--image", help="图片名或 id 的一部分；不给就看最近几张")
    sp.add_argument("--last", type=int, default=3)
    sp.add_argument("--all", action="store_true", help="每一项都列出来")
    sp.set_defaults(func=cmd_trace)

    sp = sub.add_parser("export-prompts", help="导出每个原因的含义和例图文件名，给看图大模型写提示词用")
    data(sp, config_required=True, crops=False)
    sp.add_argument("--out", type=Path, default=Path("prompts.json"), help="JSON 文件；旁边会再写一份同名 .md")
    sp.add_argument("--examples", type=int, default=3, help="每个原因最多几张正例、几张反例")
    sp.set_defaults(func=cmd_export_prompts)
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
