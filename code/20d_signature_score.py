"""
20d_signature_score.py  —  免疫成熟度签名构建 + 交叉验证评分

设计（避免循环论证）：
  签名A（成熟轴）: adult vs term cord blood  → 独立于"早产"问题构建
  签名B（早产轴）: term cord blood vs preterm cord blood

  交叉验证矩阵:
    Sig_A  → 必须能分开 preterm vs term CB（外部验证，非构建数据）
    Sig_B  → 必须能按出生后日龄排序（1week < 1month < 2month < adult）
  两者都做 100 次随机同尺寸基因集置换 → 经验 p

输入 : data/gse271413/gse271413_annot.h5ad
输出 : evidence/gse271413_signature.json
       data/gse271413/sample_scores.csv
       data/gse271413/signatures.json
"""
import json
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "gse271413" / "gse271413_annot.h5ad"
EVID = ROOT / "evidence"
OUTDIR = ROOT / "data" / "gse271413"
rng = np.random.default_rng(20261126)


def pseudobulk(adata, group_col="sample", layer=None):
    """按样本求和 counts → log CPM DataFrame(gene x sample)"""
    X = adata.layers["counts"] if layer else adata.X
    samples = adata.obs[group_col].astype(str).values
    uniq = sorted(set(samples))
    syms = adata.var["symbol"].astype(str).values
    rows = []
    M = X.tocsr() if hasattr(X, "tocsr") else X
    for s in uniq:
        idx = np.where(samples == s)[0]
        rows.append(np.asarray(M[idx].sum(0)).ravel())
    mat = np.vstack(rows).T  # gene x sample
    df = pd.DataFrame(mat, index=syms, columns=uniq)
    df = df.groupby(level=0).sum()  # 合并重名 symbol
    cpm = df / df.sum(0) * 1e6
    return np.log1p(cpm)


def de_ttest(pb, g1, g2, min_expr=1.0):
    """两组 t 检验，返回 (基因, 效应量log2FC, p)"""
    a = pb[g1].values
    b = pb[g2].values
    keep = (pb[g1].mean(1) > min_expr) | (pb[g2].mean(1) > min_expr)
    pbk = pb[keep]
    a, b = pbk[g1].values, pbk[g2].values
    lfc = a.mean(1) - b.mean(1)
    t, p = stats.ttest_ind(a, b, axis=1, equal_var=False)
    p = np.nan_to_num(p, nan=1.0)
    return pd.DataFrame({"gene": pbk.index, "lfc": lfc, "p": p}).set_index("gene")


def build_sig(df, n_top=200, p_cut=0.05):
    """取显著上调 + 下调各 n_top → (up_genes, down_genes, 说明)"""
    d = df[df["p"] < p_cut]
    up = d.sort_values("lfc", ascending=False).head(n_top).index.tolist()
    dn = d.sort_values("lfc", ascending=True).head(n_top).index.tolist()
    return up, dn


def score_samples(adata, up, dn, label):
    """每细胞 score = mean(up) - mean(dn)，用 sc.tl.score_genes 的 control 校正"""
    sym2ens = {}
    for e, s in zip(adata.var_names, adata.var["symbol"].astype(str)):
        sym2ens.setdefault(s, e)
    up_e = [sym2ens[g] for g in up if g in sym2ens]
    dn_e = [sym2ens[g] for g in dn if g in sym2ens]
    if len(up_e) < 10 or len(dn_e) < 10:
        return None, len(up_e), len(dn_e)
    sc.tl.score_genes(adata, up_e, score_name=f"tmp_up_{label}", use_raw=True,
                      ctrl_size=50, random_state=0)
    sc.tl.score_genes(adata, dn_e, score_name=f"tmp_dn_{label}", use_raw=True,
                      ctrl_size=50, random_state=0)
    s = adata.obs[f"tmp_up_{label}"].values - adata.obs[f"tmp_dn_{label}"].values
    adata.obs[f"score_{label}"] = s
    del adata.obs[f"tmp_up_{label}"]
    del adata.obs[f"tmp_dn_{label}"]
    return s, len(up_e), len(dn_e)


def sep_stat(scores, groups, a, b):
    """两组均值差 / 合并SD（Cohen's d）"""
    x = scores[groups == a]
    y = scores[groups == b]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((y.mean() - x.mean()) / sd) if sd > 0 else 0.0


def main():
    adata = ad.read_h5ad(IN)
    print(f"[20d] {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    grp = adata.obs["group"].astype(str).values

    pb = pseudobulk(adata)
    print(f"[20d] pseudobulk {pb.shape[1]} 样本 x {pb.shape[0]} 基因", flush=True)

    meta = adata.obs.groupby("sample", observed=True).agg(
        group=("group", "first"), age_days=("age_days", "first"), n=("gsm", "size")
    )
    samples = pb.columns.tolist()
    meas = meta.loc[samples]
    print(meas.to_string(), flush=True)

    g = lambda t: [s for s in samples if meas.loc[s, "group"] == t]

    # ---- 两条对比 ----
    de_A = de_ttest(pb, g("adult"), g("term_cb"))
    de_B = de_ttest(pb, g("term_cb"), g("preterm_cb"))
    print(f"[20d] DE_A(adult vs termCB) 显著 {int((de_A.p<0.05).sum())}", flush=True)
    print(f"[20d] DE_B(termCB vs pretermCB) 显著 {int((de_B.p<0.05).sum())}", flush=True)

    upA, dnA = build_sig(de_A)
    upB, dnB = build_sig(de_B)
    print(f"[20d] SigA up/dn = {len(upA)}/{len(dnA)}, SigB = {len(upB)}/{len(dnB)}", flush=True)

    res = {"n_cells": int(adata.n_obs), "samples": samples,
           "sample_meta": meas.reset_index().to_dict("records"),
           "sigA_genes": {"up": upA, "down": dnA},
           "sigB_genes": {"up": upB, "down": dnB}}

    # ---- 打分 ----
    sA, nA_u, nA_d = score_samples(adata, upA, dnA, "A")
    sB, nB_u, nB_d = score_samples(adata, upB, dnB, "B")
    print(f"[20d] SigA 命中 up/dn {nA_u}/{nA_d}, SigB {nB_u}/{nB_d}", flush=True)

    # ---- 样本级 ----
    ss = adata.obs.groupby("sample", observed=True).agg(
        group=("group", "first"), age_days=("age_days", "first"),
        n_cells=("gsm", "size"),
        scoreA=("score_A", "mean"), scoreA_sd=("score_A", "std"),
        scoreB=("score_B", "mean"), scoreB_sd=("score_B", "std"),
    )
    ss = ss.loc[samples]
    print("\n[20d] 样本级评分:\n" + ss.to_string(), flush=True)
    ss.to_csv(OUTDIR / "sample_scores.csv")

    # ---- 交叉验证 1: SigA 应分开 preterm vs term CB（SigA 未用早产数据构建）----
    d_A_preterm_vs_term = sep_stat(sA, grp, "preterm_cb", "term_cb")
    # ---- 交叉验证 2: SigB 应按出生后日龄排序 ----
    post = adata.obs["group"].astype(str).isin(["term_cb", "postnatal", "adult"]).values
    order_groups = ["term_cb", "postnatal", "adult"]
    grad_means = {k: float(sB[grp == k].mean()) for k in order_groups}
    # 日龄连续相关（仅 term_cb + postnatal，adult 无日龄）
    sub = (grp == "term_cb") | (grp == "postnatal")
    age = adata.obs["age_days"].values[sub].astype(float)
    sv = sB[sub]
    ok = ~np.isnan(age)
    rho_B_age = float(stats.spearmanr(age[ok], sv[ok]).statistic)
    rho_A_age = float(stats.spearmanr(age[ok], sA[sub][ok]).statistic)

    res["cv"] = {
        "SigA_preterm_vs_term_cohens_d": d_A_preterm_vs_term,
        "SigA_group_means": {k: float(sA[grp == k].mean()) for k in
                             ["preterm_cb", "term_cb", "postnatal", "adult"]},
        "SigB_group_means": {k: float(sB[grp == k].mean()) for k in
                             ["preterm_cb", "term_cb", "postnatal", "adult"]},
        "SigB_ordered_by_age": grad_means,
        "SigB_spearman_with_age_days": rho_B_age,
        "SigA_spearman_with_age_days": rho_A_age,
    }
    print(f"[20d] SigA preterm vs term CB: d={d_A_preterm_vs_term:+.3f}", flush=True)
    print(f"[20d] SigB vs age_days: rho={rho_B_age:+.3f}", flush=True)

    # ---- 置换检验: 随机同尺寸基因集 ----
    all_syms = adata.var["symbol"].astype(str).values
    all_syms = np.array([s for s in set(all_syms) if s and s != "nan"])
    n_perm = 100
    print(f"[20d] 置换检验 {n_perm} 次 ...", flush=True)
    nullA, nullB = [], []
    for i in range(n_perm):
        ru = list(rng.choice(all_syms, len(upA), replace=False))
        rd = list(rng.choice(all_syms, len(dnA), replace=False))
        sr, ok_u, ok_d = score_samples(adata, ru, rd, f"nullA{i}")
        if sr is None:
            continue
        nullA.append(sep_stat(sr, grp, "preterm_cb", "term_cb"))
        adata.obs.drop(columns=[f"score_nullA{i}"], inplace=True, errors="ignore")
        if i % 25 == 0:
            print(f"  perm {i}", flush=True)
    nullA = np.array(nullA)
    if len(nullA):
        pA = float((nullA >= d_A_preterm_vs_term).mean())
        res["perm_A"] = {"n": int(len(nullA)), "null_max": float(nullA.max()),
                         "null_mean": float(nullA.mean()), "obs": d_A_preterm_vs_term,
                         "p_empirical": pA}
        print(f"[20d] SigA 置换: null_max={nullA.max():+.3f} obs={d_A_preterm_vs_term:+.3f} p={pA:.3f}", flush=True)

    # SigB 置换：是否比随机集更好地随日龄单调
    print(f"[20d] SigB 置换 {n_perm} 次 ...", flush=True)
    for i in range(n_perm):
        ru = list(rng.choice(all_syms, len(upB), replace=False))
        rd = list(rng.choice(all_syms, len(dnB), replace=False))
        sr, ok_u, ok_d = score_samples(adata, ru, rd, f"nullB{i}")
        if sr is None:
            continue
        v = sr[sub][ok]
        if np.std(v) > 0:
            nullB.append(float(stats.spearmanr(age[ok], v).statistic))
        adata.obs.drop(columns=[f"score_nullB{i}"], inplace=True, errors="ignore")
        if i % 25 == 0:
            print(f"  perm {i}", flush=True)
    nullB = np.array(nullB)
    if len(nullB):
        pB = float((np.abs(nullB) >= abs(rho_B_age)).mean())
        res["perm_B"] = {"n": int(len(nullB)), "null_absmax": float(np.abs(nullB).max()),
                         "obs": rho_B_age, "p_empirical": pB}
        print(f"[20d] SigB 置换: null|max|={np.abs(nullB).max():.3f} obs={rho_B_age:+.3f} p={pB:.3f}", flush=True)

    adata.obs[["score_A", "score_B"]].to_csv(OUTDIR / "cell_scores.csv")
    adata.write_h5ad(IN.with_name("gse271413_scored.h5ad"), compression="gzip")
    with open(EVID / "gse271413_signature.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    with open(OUTDIR / "signatures.json", "w", encoding="utf-8") as fh:
        json.dump({"A_up": upA, "A_down": dnA, "B_up": upB, "B_down": dnB}, fh, indent=2)
    print("[20d] 完成", flush=True)


if __name__ == "__main__":
    main()
