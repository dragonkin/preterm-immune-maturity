"""
20g_celltype_rescore.py  —  P0-1: 细胞亚群内部重评 SigB（组成混淆校正）
问题: SigB 的训练集内早产/足月分离可能只是"细胞组成差异"冒充的。
方法: 在每个细胞亚群内部, 重新比较 早产vs足月 的 score_B（剔除组成效应）:
  1) 每亚群, 样本级 score_B 均值 → preterm_vs_term Cohen's d + Mann-Whitney
  2) 样本级置换 (打乱组标签) → 空分布 |d| → 实证 p
  3) 全谱打分: 样本级 score_B ~ (组成 PC) + 分组  的 partial effect
输入 data/gse271413/gse271413_scored.h5ad
输出 evidence/gse271413_celltype_rescore.json
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "gse271413" / "gse271413_scored.h5ad"
EVID = ROOT / "evidence"
rng = np.random.default_rng(20261126)
NPERM = 100


def cohens_d(x, y):
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((y.mean() - x.mean()) / sd) if sd > 0 else 0.0


def main():
    a = ad.read_h5ad(IN, backed="r")
    obs = a.obs[["sample", "group", "celltype_marker", "score_A", "score_B"]].copy()
    a.file.close()
    obs["group"] = obs["group"].astype(str)
    obs["celltype_marker"] = obs["celltype_marker"].astype(str)

    pt = (obs["group"] == "preterm_cb").values
    tm = (obs["group"] == "term_cb").values
    cb = pt | tm
    sample = obs["sample"].astype(str).values

    res = {"n_perm": NPERM, "by_celltype": {}, "composition_residual": {}}

    # ---- 全谱（不校正组成）对照 ----
    sm = obs[cb].groupby("sample")["score_B"].mean()
    meta = obs[cb].groupby("sample")["group"].first()
    x = sm[meta == "preterm_cb"].values
    y = sm[meta == "term_cb"].values
    d_all = cohens_d(x, y)
    smrows = sm.index.values

    # ---- 按亚群内部 ----
    ctypes = [c for c in obs["celltype_marker"].unique() if c != "nan"]
    n_cells_pt = obs[pt]["celltype_marker"].value_counts()
    n_cells_tm = obs[tm]["celltype_marker"].value_counts()
    min_cells = 50
    for c in sorted(ctypes):
        if n_cells_pt.get(c, 0) < min_cells or n_cells_tm.get(c, 0) < min_cells:
            print(f"  skip {c}: preterm={n_cells_pt.get(c,0)} term={n_cells_tm.get(c,0)}", flush=True)
            continue
        sub = obs[(obs["celltype_marker"] == c) & cb]
        smc = sub.groupby("sample")["score_B"].mean()
        met = sub.groupby("sample")["group"].first()
        x = smc[met == "preterm_cb"].values
        y = smc[met == "term_cb"].values
        # 随机化: 样本级打乱组标签, 记录 |d|
        smv = smc.values
        null = []
        for _ in range(NPERM):
            g = rng.permutation(met.values)
            xp, yp = smv[g == "preterm_cb"], smv[g == "term_cb"]
            if len(xp) >= 2 and len(yp) >= 2:
                null.append(abs(cohens_d(xp, yp)))
        null = np.array(null)
        d = cohens_d(x, y)
        p = float((null >= abs(d)).mean()) if len(null) else np.nan
        # Mann-Whitney
        p_mw = float(stats.mannwhitneyu(x, y).pvalue) if (len(x) and len(y)) else np.nan
        res["by_celltype"][c] = {
            "n_preterm_cells": int(n_cells_pt[c]), "n_term_cells": int(n_cells_tm[c]),
            "n_preterm_samples": int((met.values == "preterm_cb").sum()),
            "n_term_samples": int((met.values == "term_cb").sum()),
            "preterm_scoreB_mean": float(x.mean()), "term_scoreB_mean": float(y.mean()),
            "cohens_d": d, "null_absmax_d": float(null.max()) if len(null) else None,
            "p_empirical": p, "mannwhitney_p": p_mw,
        }
        print(f"  {c:16s} mid  d={d:+.3f} (preterm {x.mean():+.3f}/term {y.mean():+.3f}) "
              f"nullmax={null.max():.3f} p={p:.3f} mw={p_mw:.3f}", flush=True)

    res["by_celltype"]["_all_cells_unadjusted"] = {
        "n_preterm_samples": int((meta.values == "preterm_cb").sum()),
        "n_term_samples": int((meta.values == "term_cb").sum()),
        "cohens_d": d_all, "note": "全谱不校正组成的对照"}

    # ---- 组成校正回归: score_B ~ 组成PC + 分组 ----
    obs2 = obs[cb]
    # 样本 x 亚群比例
    tab = pd.crosstab(obs2["sample"], obs2["celltype_marker"])
    frac = tab.div(tab.sum(1), axis=0)
    sv = obs2.groupby("sample")["score_B"].mean()
    gv = obs2.groupby("sample")["group"].first()
    imp = frac.notna().all(axis=1)
    F = frac[imp].values
    sc = sv[imp].values
    gg = (gv[imp].values == "term_cb").astype(float)
    # 组成去均值 -> 主成分（取解释前 90% 的 PC）
    F0 = F - F.mean(0)
    U, S, Vt = np.linalg.svd(F0, full_matrices=False)
    cum = np.cumsum(S**2) / np.sum(S**2)
    k = int(np.searchsorted(cum, 0.90)) + 1
    PCs = U[:, :k] * S[:k]
    # 回归 score ~ PC + group
    X = np.column_stack([np.ones(len(sc)), PCs, gg])
    beta, *_ = np.linalg.lstsq(X, sc, rcond=None)
    r = sc - X @ beta
    # 分组 partial effect: 比较含组 vs 不含组的残差平方和
    X0 = np.column_stack([np.ones(len(sc)), PCs])
    beta0, *_ = np.linalg.lstsq(X0, sc, rcond=None)
    r0 = sc - X0 @ beta0
    f_stat = ((r0 @ r0 - r @ r) / 1) / (r @ r / (len(sc) - X.shape[1]))
    p_f = 1 - stats.f.cdf(f_stat, 1, len(sc) - X.shape[1])
    res["composition_residual"] = {
        "n_samples_used": int(len(sc)), "n_pc_90pct": int(k),
        "partial_group_coef": float(beta[-1]), "f_stat": float(f_stat), "f_pvalue": float(p_f),
        "note": "score_B ~ [分数组成主成分(90%)] + 分组; partial 系数为组效应在与'早产vs足月样本组成'正交后的剩余"}
    print(f"[20g] 组成校正回归: partial_group={beta[-1]:+.3f} F={f_stat:.2f} p={p_f:.4f} (k={k})", flush=True)

    with open(EVID / "gse271413_celltype_rescore.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("[20g] 完成", flush=True)


if __name__ == "__main__":
    main()