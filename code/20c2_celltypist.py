"""
20c2_celltypist.py  —  CellTypist(gene symbol) + 共识标签

输入 : data/gse271413/gse271413_annot.h5ad
输出 : data/gse271413/gse271413_annot.h5ad 新增 celltypist* , celltype_consensus
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import celltypist
from celltypist import models

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "gse271413" / "gse271413_annot.h5ad"

adata = ad.read_h5ad(IN)
print(f"[20c2] {adata.n_obs} x {adata.n_vars}", flush=True)

ad2 = adata.copy()
sym = ad2.var["symbol"].astype(str).values
base = [f"{s}.{i}" if (not s or s == "nan") else s for i, s in enumerate(sym)]
seen, names = {}, []
for n in base:
    if n in seen:
        seen[n] += 1
        names.append(f"{n}.{seen[n]}")
    else:
        seen[n] = 0
        names.append(n)
ad2.var_names = names

model = models.Model.load(model="Immune_All_Low.pkl")
res = celltypist.annotate(ad2, model=model, majority_voting=False)
pred = res.predicted_labels["predicted_labels"].astype(str)
adata.obs["celltypist"] = pred.values

# 粗类映射
def coarse(v):
    v = v.lower()
    if "cd8" in v: return "T_CD8"
    if "nk" in v: return "NK"
    if "helper" in v or "cd4" in v or "treg" in v or v.startswith("t cell") or "cytotoxic t" in v: return "T_CD4"
    if "b cell" in v or "memory b" in v or "b_cell" in v: return "B"
    if "plasma" in v: return "Plasma"
    if "monoc" in v and "non" in v: return "Mono_nonclassical"
    if "monoc" in v: return "Mono_classical"
    if "neutrophil" in v: return "Neutrophil"
    if "dc" in v or "dendritic" in v: return "DC"
    if "erythr" in v: return "Erythrocyte"
    if "megakary" in v or "platelet" in v: return "Megakaryocyte"
    if "hspc" in v or "progenitor" in v or "stem cell" in v: return "HSPC"
    if "eosinophil" in v or "basophil" in v or "mast" in v or "granulocyte" in v: return "Granulocyte"
    if "doublet" in v or "unassigned" in v or "unknown" in v: return "Other"
    return "Other"

ct = pd.Series([coarse(x) for x in pred.values], index=adata.obs_names)

# 簇级 majority（更稳）：每 leiden 簇取 celltypist 粗类众数
clus = adata.obs["leiden"].astype(str).values
tab = pd.crosstab(adata.obs["leiden"].astype(str), ct)
cluster_majority = tab.idxmax(1)
adata.obs["celltype"] = [cluster_majority[c] for c in adata.obs["leiden"].astype(str).values]

print("\n[20c2] celltype 分布:", flush=True)
print(adata.obs["celltype"].value_counts().to_string(), flush=True)

print("\n[20c2] 簇→celltype 交叉:", flush=True)
print(pd.crosstab(adata.obs["leiden"].astype(str), adata.obs["celltype"]).to_string(), flush=True)

# celltypist 原始高频标签（供审阅）
print("\n[20c2] CellTypist 原始 top:", flush=True)
print(pred.value_counts().head(20).to_string(), flush=True)

adata.obs["celltype_consensus"] = adata.obs["celltype"].astype(str)
adata.write_h5ad(IN, compression="gzip")
print(f"[20c2] 写出 {IN}", flush=True)