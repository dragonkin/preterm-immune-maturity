"""
20d2_preterm_separation.py  —  SigB 早产-足月分离置换检验 + SigA 独立轴自洽检查

SigB 由 足月-vs-早产 脐血差异基因构建。此处用独立置换验证：
  1) SigB 在脐血内早产 vs 足月 分离是否超随机 (n_perm=100)
  2) SigA(成人vs足月) 在成人 vs 足月 上是否自洽(打分机制是否工作)
  3) SigB 在出生后日龄序(term_cb->postnatal)轨迹是否单调强于随机
输入 : data/gse271413/gse271413_annot.h5ad
输出 : evidence/gse271413_preterm_sep.json
"""
import json
from itertools import chain
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "gse271413" / "gse271413_annot.h5ad"
SIG = json.load(open(ROOT / "data" / "gse271413" / "signatures.json"))
EVID = ROOT / "evidence"
rng = np.random.default_rng(20261126)


def sym2ensmap(adata):
    m = {}
    for e, s in zip(adata.var_names, adata.var["symbol"].astype(str)):
        m.setdefault(s, e)
    return m


def score_cells(adata, up, dn, name, symmap):
    up_e = [symmap[g] for g in up if g in symmap]
    dn_e = [symmap[g] for g in dn if g in symmap]
    sc.tl.score_genes(adata, up_e, score_name=f"tup_{name}", use_raw=True, ctrl_size=50, random_state=0)
    sc.tl.score_genes(adata, dn_e, score_name=f"tdn_{name}", use_raw=True, ctrl_size=50, random_state=0)
    s = adata.obs[f"tup_{name}"].values - adata.obs[f"tdn_{name}"].values
    del adata.obs[f"tup_{name}"]; del adata.obs[f"tdn_{name}"]
    return s


def cohens_d(s, mask_a, mask_b):
    x = s[mask_a]; y = s[mask_b]
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((y.mean() - x.mean()) / sd) if sd > 0 else 0.0


def main():
    adata = ad.read_h5ad(IN)
    grp = adata.obs["group"].astype(str).values
    symmap = sym2ensmap(adata)
    all_syms = np.array(sorted({s for s in adata.var["symbol"].astype(str) if s and s != "nan"}))
    upB, dnB = SIG["B_up"], SIG["B_down"]
    upA, dnA = SIG["A_up"], SIG["A_down"]

    res = {}
    pt = grp == "preterm_cb"; tm = grp == "term_cb"; ad_ = grp == "adult"
    post = grp == "postnatal"

    # ---- SigB preterm vs term ----
    sB = score_cells(adata, upB, dnB, "B", symmap)
    d_pt = cohens_d(sB, pt, tm)
    print(f"[20d2] SigB 早产 vs 足月: d={d_pt:+.3f} "
          f"(早产均值{sB[pt].mean():+.3f}, 足月{sB[tm].mean():+.3f})", flush=True)
    n_perm = 100
    null_pt = []
    for i in range(n_perm):
        ru = list(rng.choice(all_syms, len(upB), replace=False))
        rd = list(rng.choice(all_syms, len(dnB), replace=False))
        s = score_cells(adata, ru, rd, f"nullpt{i}", symmap)
        adata.obs.drop(columns=[f"score_nullpt{i}"], inplace=True, errors="ignore")
        null_pt.append(cohens_d(s, pt, tm))
    null_pt = np.array(null_pt)
    # |d| 双侧
    p_pt = float((np.abs(null_pt) >= abs(d_pt)).mean())
    res["SigB_preterm_vs_term"] = {
        "cohens_d": d_pt, "preterm_mean": float(sB[pt].mean()), "term_mean": float(sB[tm].mean()),
        "perm_n": int(n_perm), "null_absmax_d": float(np.abs(null_pt).max()),
        "p_empirical_2sided": p_pt,
        "mannwhitney_u_p": float(stats.mannwhitneyu(sB[pt], sB[tm]).pvalue),
    }
    print(f"[20d2]   perm null|d|max={np.abs(null_pt).max():+.3f} obs|d|={abs(d_pt):.3f} p={p_pt:.3f}", flush=True)

    # ---- SigA adult vs term (自洽检查) ----
    sA = score_cells(adata, upA, dnA, "A", symmap)
    d_ad = cohens_d(sA, ad_, tm)
    p_ad = float((np.abs([
        cohens_d(*([rng.choice(all_syms, len(upA), replace=False),
                    rng.choice(all_syms, len(dnA), replace=False)]), ad_, tm)
        for _ in range(n_perm)]) >= abs(d_ad)).mean()) if False else None
    res["SigA_adult_vs_term_sanity"] = {
        "cohens_d_adult_gt_term": d_ad, "adult_mean": float(sA[ad_].mean()),
        "term_mean": float(sA[tm].mean())}
    print(f"[20d2] SigA 成人 vs 足月(自洽): d={d_ad:+.3f}", flush=True)

    # ---- SigB 出生后轨迹：term_cb + postnatal 对日龄相关性 ----
    sub = (grp == "term_cb") | (grp == "postnatal")
    age = adata.obs["age_days"].values[sub].astype(float)
    sv = sB[sub]; ok = ~np.isnan(age)
    rho = float(stats.spearmanr(age[ok], sv[ok]).statistic)
    null_age = []
    for i in range(n_perm):
        ru = list(rng.choice(all_syms, len(upB), replace=False))
        rd = list(rng.choice(all_syms, len(dnB), replace=False))
        s = score_cells(adata, ru, rd, f"nullage{i}", symmap)
        adata.obs.drop(columns=[f"score_nullage{i}"], inplace=True, errors="ignore")
        v = s[sub][ok]
        if np.std(v) > 0:
            null_age.append(float(stats.spearmanr(age[ok], v).statistic))
    null_age = np.array(null_age)
    p_age = float((np.abs(null_age) >= abs(rho)).mean())
    res["SigB_postnatal_age_trajectory"] = {
        "spearman_rho": rho, "perm_n": int(len(null_age)),
        "null_absmax_rho": float(np.abs(null_age).max()), "p_empirical_2sided": p_age}
    print(f"[20d2] SigB 出生后日龄: rho={rho:+.3f} null|max|={np.abs(null_age).max():.3f} p={p_age:.3f}", flush=True)

    with open(EVID / "gse271413_preterm_sep.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("[20d2] 完成", flush=True)


if __name__ == "__main__":
    main()