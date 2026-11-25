"""
20c_annotate.py  —  CellTypist 自动注释 + marker-based 校准

输入 : data/gse271413/gse271413_proc.h5ad
输出 : data/gse271413/gse271413_annot.h5ad
       evidence/gse271413_annotation.json
"""
import json
from pathlib import Path

import anndata as ad
import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
from celltypist import models

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "gse271413" / "gse271413_proc.h5ad"
OUT = ROOT / "data" / "gse271413" / "gse271413_annot.h5ad"
EVID = ROOT / "evidence"

adata = ad.read_h5ad(IN)
print(f"[20c] {adata.n_obs} x {adata.n_vars}", flush=True)

# CellTypist 需要 log1p 归一化表达；已有 .X
markers = {
    "T_CD4": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],
    "T_CD8": ["CD3D", "CD8A", "CD8B", "GZMK", "NKG7"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "NK": ["NKG7", "GNLY", "KLRD1", "NCAM1", "FCGR3A", "KLRF1"],
    "B": ["CD79A", "MS4A1", "CD19", "CD37"],
    "Plasma": ["MZB1", "JCHAIN", "IGHG1", "SDC1"],
    "Mono_classical": ["CD14", "LYZ", "VCAN", "S100A8", "S100A9", "FCN1"],
    "Mono_nonclassical": ["FCGR3A", "MS4A7", "CDKN1C", "LST1"],
    "DC": ["FCER1A", "CD1C", "CLEC9A", "LILRA4", "IL3RA"],
    "Neutrophil": ["FCGR3B", "CSF3R", "CXCR2", "S100A8", "MNDA"],
    "Megakaryocyte": ["PPBP", "PF4", "ITGA2B"],
    "Erythrocyte": ["HBB", "HBA1", "HBA2"],
    "pDC": ["LILRA4", "IRF7", "GZMB"],
    "HSPC": ["CD34", "SPINK2", "PRSS57"],
}
present = {k: [g for g in v if g in adata.var["symbol"].values] for k, v in markers.items()}
present = {k: v for k, v in present.items() if v}

sym2ens = {}
for e, s in zip(adata.var_names, adata.var["symbol"]):
    sym2ens.setdefault(str(s), e)

# --- 手动 marker score（作为交叉校验，CellTypist 可能因环境无网络失败）---
score_cols = {}
for name, genes in present.items():
    ens = [sym2ens[g] for g in genes if g in sym2ens]
    if not ens:
        continue
    sc.tl.score_genes(adata, ens, score_name=f"sc_{name}", use_raw=True)
    score_cols[name] = f"sc_{name}"
print(f"[20c] marker 打分完成: {len(score_cols)} 个类型", flush=True)

score_df = adata.obs[list(score_cols.values())].copy()
score_df.columns = list(score_cols.keys())

# 每个 leiden 簇的平均分 → 定标签
cluster_scores = score_df.groupby(adata.obs["leiden"].values, observed=True).mean()
cluster_label = cluster_scores.idxmax(axis=1)
adata.obs["celltype_marker"] = adata.obs["leiden"].map(cluster_label).astype(str)
print("[20c] 簇 → marker 标签:", flush=True)
print(pd.crosstab(adata.obs["leiden"], adata.obs["celltype_marker"]).to_string(), flush=True)

# --- CellTypist（可选，离线模型）---
ct_ok = False
try:
    models.download_models(model="Immune_All_Low.pkl")
    model = models.Model.load(model="Immune_All_Low.pkl")
    pred = celltypist.annotate(adata, model=model, majority_voting=False)
    adata.obs["celltypist"] = pred.predicted_labels["predicted_labels"].values
    adata.obs["celltypist_majority"] = pred.predicted_labels["majority_voting"].values
    ct_ok = True
    print("[20c] CellTypist OK", flush=True)
    print(adata.obs["celltypist_majority"].value_counts().head(25).to_string(), flush=True)
except Exception as e:
    print(f"[20c] CellTypist 跳过: {type(e).__name__}: {str(e)[:200]}", flush=True)

adata.write_h5ad(OUT, compression="gzip")
print(f"[20c] 写出 {OUT}", flush=True)

rep = {
    "n_cells": int(adata.n_obs),
    "n_clusters": int(adata.obs["leiden"].nunique()),
    "cluster_to_marker_label": {str(k): str(v) for k, v in cluster_label.items()},
    "celltype_marker_counts": {str(k): int(v) for k, v in adata.obs["celltype_marker"].value_counts().items()},
    "celltypist_used": ct_ok,
}
if ct_ok:
    rep["celltypist_top"] = {str(k): int(v) for k, v in adata.obs["celltypist_majority"].value_counts().head(40).items()}
with open(EVID / "gse271413_annotation.json", "w", encoding="utf-8") as fh:
    json.dump(rep, fh, ensure_ascii=False, indent=2)
print("[20c] 完成", flush=True)
