"""
20b_qc_integrate.py  —  GSE271413 质控 + 归一化 + HVG + PCA + Harmony + Leiden + UMAP

输入 : data/gse271413/gse271413_raw.h5ad
输出 : data/gse271413/gse271413_proc.h5ad
       evidence/gse271413_qc.json
"""
import json
import os
from pathlib import Path

import anndata as ad
import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sc.settings.n_jobs = 8
ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "gse271413" / "gse271413_raw.h5ad"
OUT = ROOT / "data" / "gse271413" / "gse271413_proc.h5ad"
EVID = ROOT / "evidence"
EVID.mkdir(exist_ok=True)

report = {}

print("[20b] 读取 ...", flush=True)
adata = ad.read_h5ad(IN)
print(f"  raw {adata.n_obs} x {adata.n_vars}", flush=True)
report["raw_cells"] = int(adata.n_obs)

# ---------- QC ----------
adata.var["mt"] = adata.var["symbol"].astype(str).str.upper().str.startswith("MT-").values
adata.var["ribo"] = adata.var["symbol"].astype(str).str.upper().str.startswith(("RPS", "RPL")).values
adata.var["hb"] = adata.var["symbol"].astype(str).str.upper().str.match(r"^HB[ABDEGQMZ]\d?$").values

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=None, log1p=False)
qc = adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"]].describe()
print(qc.to_string(), flush=True)
report["qc_before"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in qc.items()}

n0 = adata.n_obs
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
adata = adata[adata.obs["n_genes_by_counts"] < 6000].copy()
print(f"[20b] QC 后 {adata.n_obs} cells (删 {n0 - adata.n_obs}, {100*(n0-adata.n_obs)/n0:.1f}%) x {adata.n_vars} genes", flush=True)
report["after_qc_cells"] = int(adata.n_obs)
report["after_qc_genes"] = int(adata.n_vars)
per_sample = adata.obs.groupby("sample", observed=True).size().sort_values()
report["per_sample_cells"] = {str(k): int(v) for k, v in per_sample.items()}

# ---------- normalize ----------
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

# ---------- HVG / PCA ----------
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="sample", flavor="seurat")
print(f"[20b] HVG {int(adata.var.highly_variable.sum())}", flush=True)
report["hvg"] = int(adata.var.highly_variable.sum())

adata_h = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata_h, max_value=10)
sc.tl.pca(adata_h, n_comps=50, svd_solver="arpack")
adata.obsm["X_pca"] = adata_h.obsm["X_pca"]

# ---------- Harmony ----------
adata.write_h5ad(OUT.with_name("gse271413_pca.h5ad"), compression="gzip")
print(f"[20b] PCA 中间件已存 gse271413_pca.h5ad", flush=True)

print("[20b] Harmony (batch=sample) ...", flush=True)
sc.external.pp.harmony_integrate(adata, key="sample", basis="X_pca",
                                adjusted_basis="X_pca_harmony", max_iter_harmony=10)
print(f"  harmony -> {adata.obsm['X_pca_harmony'].shape}", flush=True)
assert adata.obsm["X_pca_harmony"].shape[0] == adata.n_obs
print(f"  harmony -> {adata.obsm['X_pca_harmony'].shape}", flush=True)

# ---------- neighbors / leiden / umap ----------
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=20, n_pcs=50)
sc.tl.leiden(adata, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
print(f"[20b] leiden 簇数 {adata.obs['leiden'].nunique()}", flush=True)
report["n_clusters"] = int(adata.obs["leiden"].nunique())
sc.tl.umap(adata)

adata.write_h5ad(OUT, compression="gzip")
print(f"[20b] 写出 {OUT}", flush=True)

with open(EVID / "gse271413_qc.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)
print("[20b] 完成", flush=True)
