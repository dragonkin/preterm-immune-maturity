"""
20h_maturity_axis.py  —  P1: 连续免疫成熟轴实证
把 SigA 作为连续"成熟度轴"(脐血→成人), 检验:
  1) 出生后时间轨迹: GSE271413 内 postnatal 样本 scoreA 是否随 7→30→60天 上升(成熟),
     term_cb(0天) 与 postnatal 合并做 scoreA ~ age_days 置换检验
  2) 早产偏移: preterm_cb scoreA vs term_cb (SigA 能否把早产放到"更不成熟"端) — 已知阴性, 复查
  3) 外部连续梯度汇总: GSE302276(adult>cord) + GSE157007(cord<young<old) 用 SigA
输入 data/gse271413/gse271413_scored.h5ad + data/gse271413/signatures.json
输出 evidence/gse271413_maturity_axis.json
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
NPERM = 1000


def cohens_d(x, y):
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((y.mean() - x.mean()) / sd) if sd > 0 else 0.0


def main():
    a = ad.read_h5ad(IN, backed="r")
    obs = a.obs[["sample", "group", "age_days", "score_A"]].copy()
    a.file.close()
    obs["group"] = obs["group"].astype(str)
    res = {"n_perm": NPERM}

    # ---- 1) 出生后成熟时间轨迹: term_cb(0) + postnatal(7/30/60) ----
    sub = obs[obs["group"].isin(["term_cb", "postnatal", "adult"])]
    sm = sub.groupby("sample")[["score_A", "age_days"]].mean().dropna()
    sm = sm[pd.to_numeric(sm["age_days"], errors="coerce").notna()]
    age = sm["age_days"].values.astype(float)
    scA = sm["score_A"].values
    rho, p_sp = stats.spearmanr(age, scA)
    res["postnatal_maturation_trajectory"] = {
        "samples": int(len(sm)),
        "groups_note": "term_cb=0d, postnatal=7/30/60d, adult=NaN(排除)",
        "spearman_rho_age_vs_scoreA": float(rho), "spearman_p": float(p_sp)}
    print(f"[20h] 成熟时间轨迹 scoreA~age: rho={rho:+.3f} p={p_sp:.4f} n={len(sm)}", flush=True)
    # 样本级置换(打乱样本与age对应), |rho|
    ages = age.copy()
    null = []
    for _ in range(NPERM):
        g = rng.permutation(ages)
        null.append(abs(stats.spearmanr(g, scA).statistic))
    null = np.array(null)
    res["postnatal_maturation_trajectory"]["perm_absmax_rho"] = float(null.max())
    res["postnatal_maturation_trajectory"]["perm_p_2sided"] = float((null >= abs(rho)).mean())
    print(f"[20h]   置换: nullabsmax={null.max():.3f} p={(null>=abs(rho)).mean():.4f}", flush=True)
    # 分组均值展示
    for g in ["term_cb", "postnatal"]:
        subg = obs[obs["group"] == g]
        gm = subg.groupby("sample")["score_A"].mean()
        res.setdefault("group_mean_scoreA", {})[g] = {"mean": float(gm.mean()), "n": int(len(gm))}
    # postnatal 按日龄分桶
    buck = sm.groupby(np.round(age)).agg(m=("score_A", "mean"), n=("score_A", "size"))
    res["scoreA_by_age_day"] = {str(k): {"mean": float(v["m"]), "n": int(v["n"])}
                                for k, v in buck.iterrows()}
    print("[20h]   scoreA 随日龄均值:", {k: round(v["mean"], 3) for k, v in res["scoreA_by_age_day"].items()}, flush=True)

    # ---- 2) 早产偏移复查 (SigA 早产 vs 足月) ----
    pt = obs[obs["group"] == "preterm_cb"].groupby("sample")["score_A"].mean()
    tm = obs[obs["group"] == "term_cb"].groupby("sample")["score_A"].mean()
    d = cohens_d(pt.values, tm.values)   # y=term - x=preterm
    res["preterm_shift_SigA"] = {
        "preterm_scoreA": float(pt.mean()), "term_scoreA": float(tm.mean()),
        "cohens_d_gt0_means_preterm_higher": d,
        "note": "d>0 => 早产 scoreA 高于足月(偏更'成熟'/成人端); 预期早产应更低, 则此处为阴性"}
    u, p_mw = stats.mannwhitneyu(pt.values, tm.values)
    res["preterm_shift_SigA"]["mannwhitney_p"] = float(p_mw)
    print(f"[20h] 早产偏移: preterm={pt.mean():+.3f} term={tm.mean():+.3f} d={d:+.3f} MW_p={p_mw:.3f}", flush=True)

    # ---- 3) 外部连续梯度用 SigA (复用 20f 数值) ----
    ext_f = EVID / "gse271413_external_validation.json"
    if ext_f.exists():
        ex = json.load(open(ext_f, encoding="utf-8"))
        res["external_gradient_summary"] = {
            "GSE302276_adult_minus_cord": ex["GSE302276"].get("Adult_minus_Cord"),
            "GSE302276_MW_p": ex["GSE302276"].get("Adult_vs_Cord_MW_p"),
            "GSE157007_CordYoungOld_spearman": ex["GSE157007"].get("Cord_Young_Old_spearman"),
            "GSE157007_p": ex["GSE157007"].get("Cord_Young_Old_p"),
        }

    with open(EVID / "gse271413_maturity_axis.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("[20h] 完成", flush=True)


if __name__ == "__main__":
    main()