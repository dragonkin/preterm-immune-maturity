"""
20e_composition.py  —  细胞组成差异（RQ1）：早产 vs 足月，及出生后动态

输入 : data/gse271413/gse271413_scored.h5ad
输出 : evidence/gse271413_composition.json
       data/gse271413/composition_by_sample.csv
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
OUTDIR = ROOT / "data" / "gse271413"

adata = ad.read_h5ad(IN)
print(f"[20e] {adata.n_obs} cells", flush=True)

lab_col = "celltype_marker"
adata.obs[lab_col] = adata.obs[lab_col].astype(str)

# 样本 x 细胞类型 比例
tab = pd.crosstab(adata.obs["sample"], adata.obs[lab_col])
frac = tab.div(tab.sum(1), axis=0)
meta = adata.obs.groupby("sample", observed=True).agg(
    group=("group", "first"), age_days=("age_days", "first"), n=("gsm", "size")
)
frac = frac.join(meta)
frac.to_csv(OUTDIR / "composition_by_sample.csv")
print(frac.round(3).to_string(), flush=True)

groups = ["preterm_cb", "term_cb", "postnatal", "adult"]
res = {"celltypes": list(tab.columns), "sample_counts": tab.to_dict(),
       "group_means": {}}

for c in tab.columns:
    res["group_means"][c] = {
        g: float(frac.loc[frac["group"] == g, c].mean()) for g in groups
    }

# 早产 vs 足月 脐血：Mann-Whitney（样本级）
res["preterm_vs_term"] = {}
for c in tab.columns:
    a = frac.loc[frac["group"] == "preterm_cb", c].values
    b = frac.loc[frac["group"] == "term_cb", c].values
    if len(a) < 3 or len(b) < 3:
        continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    res["preterm_vs_term"][c] = {
        "preterm_mean": float(a.mean()), "term_mean": float(b.mean()),
        "diff": float(a.mean() - b.mean()), "p": float(p),
    }

sig = {k: v for k, v in res["preterm_vs_term"].items() if v["p"] < 0.10}
print("\n[20e] 早产 vs 足月 (p<0.10):", flush=True)
for k, v in sorted(sig.items(), key=lambda x: x[1]["p"]):
    print(f"  {k:22s} preterm={v['preterm_mean']:.3f} term={v['term_mean']:.3f} "
          f"diff={v['diff']:+.3f} p={v['p']:.4f}", flush=True)

# 中性粒细胞比例 vs 日龄（出生后）
if "Neutrophil" in frac.columns:
    sub = frac[frac["group"].isin(["term_cb"]) | (frac["group"] == "postnatal")]
    age = sub["age_days"].values.astype(float)
    ok = ~np.isnan(age)
    if ok.sum() > 3:
        r = stats.spearmanr(age[ok], sub["Neutrophil"].values[ok])
        res["neutrophil_vs_age"] = {"rho": float(r.statistic), "p": float(r.pvalue)}
        print(f"[20e] 中性粒比例 vs 日龄 rho={r.statistic:+.3f} p={r.pvalue:.4f}", flush=True)

# 评分 vs 细胞组成：成熟分数是不是只是中性粒比例的代理？
sc_cols = [c for c in ["score_A", "score_B"] if c in adata.obs]
res["score_confound"] = {}
for s in sc_cols:
    sm = adata.obs.groupby("sample", observed=True)[s].mean()
    for c in ["Neutrophil", "T_CD4", "Mono_classical", "NK", "B"]:
        if c not in frac.columns:
            continue
        r = stats.spearmanr(sm.loc[frac.index].values, frac[c].values)
        res["score_confound"][f"{s}_vs_{c}"] = {
            "rho": float(r.statistic), "p": float(r.pvalue)
        }
        print(f"[20e] {s} vs {c:16s} rho={r.statistic:+.3f} p={r.pvalue:.4f}", flush=True)

with open(EVID / "gse271413_composition.json", "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=2)
print("[20e] 完成", flush=True)
