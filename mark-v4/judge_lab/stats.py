"""Pass-rate statistics for recipe / LoRA A/B comparisons (stdlib only)."""
from __future__ import annotations

import math
from statistics import NormalDist

from .data import Review
from .util import fmt, md_table

_N = NormalDist()


def z_for(alpha: float, two_sided: bool = True) -> float:
    return _N.inv_cdf(1 - alpha / 2 if two_sided else 1 - alpha)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = 1.959964) -> tuple[float, float, float]:
    """CI for p1 - p2 by Newcombe's hybrid score method (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def two_prop_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test; returns (z, two-sided p)."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k1 / n1 - k2 / n2) / se
    return z, 2 * (1 - _N.cdf(abs(z)))


def sample_size(p1: float, p2: float, power: float = 0.8, alpha: float = 0.05, two_sided: bool = True) -> int:
    """Images needed per arm to detect p1 vs p2 (normal approximation, no continuity correction)."""
    if p1 == p2:
        raise ValueError("p1 和 p2 不能相等")
    za, zb = z_for(alpha, two_sided), _N.inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(num * num / (p1 - p2) ** 2)


def _tally(reviews, key_fn):
    out: dict = {}
    for r in reviews:
        key = key_fn(r)
        if key is None:
            continue
        k, n = out.get(key, (0, 0))
        out[key] = (k + (r.decision == "pass"), n + 1)
    return out


def ab_report(reviews: list[Review], group_by: str, scene_field: str | None = "scene",
              baseline: str | None = None, tolerance: float = 0.05, alpha: float = 0.05,
              private_scene: str | None = "（私有记录）") -> dict:
    """private_scene: records from a private log are pooled under this label in the per-scene table (a scene
    name comes from the file name, which must not be shown); None = show their real scene."""
    z = z_for(alpha)

    def scene_of(r):
        if private_scene and r.private:
            return private_scene
        return str(r.get(scene_field, "(无场景)"))

    def group(r):
        g = r.get(group_by)
        return None if g is None else str(g)

    groups = _tally(reviews, group)  # insertion order = first appearance in the log
    missing = sum(1 for r in reviews if group(r) is None)
    if not groups:
        raise ValueError(f"日志里没有字段 {group_by}")
    baseline = baseline if baseline is not None else next(iter(groups))
    if baseline not in groups:
        raise ValueError(f"基线 {baseline!r} 不在分组 {list(groups)} 中")

    def rate(k, n):
        lo, hi = wilson(k, n, z)
        return {"pass": k, "n": n, "rate": k / n, "ci": [lo, hi]}

    res = {
        "group_by": group_by, "baseline": baseline, "tolerance": tolerance, "alpha": alpha,
        "missing_group": missing,
        "groups": {g: rate(k, n) for g, (k, n) in groups.items()},
        "comparisons": {}, "scenes": {}, "regressions": [],
    }
    kb, nb = groups[baseline]
    for g, (k, n) in groups.items():
        if g == baseline:
            continue
        d, lo, hi = newcombe_diff(k, n, kb, nb, z)
        zs, p = two_prop_ztest(k, n, kb, nb)
        res["comparisons"][g] = {"diff": d, "ci": [lo, hi], "z": zs, "p_value": p}

    if scene_field:
        scenes = _tally(reviews, lambda r: None if group(r) is None else (scene_of(r), group(r)))
        for (scene, g), (k, n) in sorted(scenes.items()):
            entry = res["scenes"].setdefault(scene, {"groups": {}, "vs_baseline": {}})
            entry["groups"][g] = rate(k, n)
        for scene, entry in res["scenes"].items():
            base = entry["groups"].get(baseline)
            for g, st in entry["groups"].items():
                if g == baseline or base is None:
                    continue
                d, lo, hi = newcombe_diff(st["pass"], st["n"], base["pass"], base["n"], z)
                regress = d < -tolerance
                entry["vs_baseline"][g] = {"diff": d, "ci": [lo, hi], "regression": regress,
                                           "significant": hi < 0}
                if regress:
                    res["regressions"].append({"scene": scene, "group": g, "diff": d, "ci": [lo, hi],
                                               "n_new": st["n"], "n_old": base["n"]})
    res["regression_flag"] = bool(res["regressions"])
    return res


def ab_markdown(res: dict) -> str:
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    ci = lambda c: f"[{pct(c[0])}, {pct(c[1])}]"  # noqa: E731
    lines = [f"# A/B 通过率对比（分组字段 `{res['group_by']}`，基线 `{res['baseline']}`）", ""]
    if res["missing_group"]:
        lines.append(f"缺少分组字段的记录: {res['missing_group']} 条（未计入）\n")
    lines.append(md_table(["分组", "通过", "总数", "通过率", "95% CI (Wilson)"],
                          [[g, s["pass"], s["n"], pct(s["rate"]), ci(s["ci"])] for g, s in res["groups"].items()]))
    if res["comparisons"]:
        lines += ["", "## 与基线的差异（新 − 旧）", ""]
        lines.append(md_table(["分组", "差异", "95% CI (Newcombe)", "z", "p 值"],
                              [[g, pct(c["diff"]), ci(c["ci"]), fmt(c["z"], 2), fmt(c["p_value"], 4)]
                               for g, c in res["comparisons"].items()]))
    if res["scenes"]:
        lines += ["", "## 分场景", ""]
        rows = []
        for scene, e in res["scenes"].items():
            for g, s in e["groups"].items():
                vb = e["vs_baseline"].get(g)
                flag = "" if vb is None else ("⚠ 退步" if vb["regression"] else "")
                rows.append([scene, g, f"{s['pass']}/{s['n']}", pct(s["rate"]), ci(s["ci"]),
                             "-" if vb is None else pct(vb["diff"]), "-" if vb is None else ci(vb["ci"]), flag])
        lines.append(md_table(["场景", "分组", "通过/总数", "通过率", "95% CI", "差异", "差异 CI", "回归检查"], rows))
    tol = pct(res["tolerance"])
    if res["regression_flag"]:
        lines += ["", f"**回归警告**：{len(res['regressions'])} 个场景的新版本通过率比基线低超过 {tol}。"]
    else:
        lines += ["", f"回归检查通过：没有场景比基线低超过 {tol}。"]
    return "\n".join(lines) + "\n"
