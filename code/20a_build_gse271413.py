"""
20a_build_gse271413.py
构建 GSE271413 全部 29 个 GEX 样本的合并 AnnData（原始 counts）。

输入 : data/gse271413/raw/*_{barcodes,features,matrix}.gz
       data/gse271413/series_matrix.txt.gz (stage 元数据)
输出 : data/gse271413/gse271413_raw.h5ad
       data/gse271413/sample_manifest.csv
"""
import gzip
import os
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "gse271413" / "raw"
SERIES = ROOT / "data" / "gse271413" / "series_matrix.txt.gz"
OUT = ROOT / "data" / "gse271413"


def read_stage_map():
    """从 series matrix 取 GSM -> stage / library type"""
    rows = {}
    with gzip.open(SERIES, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_"):
                parts = [p.strip('"') for p in line.rstrip("\n").split("\t")]
                rows.setdefault(parts[0], []).append(parts[1:])
    gsm = rows["!Sample_geo_accession"][0]
    title = rows["!Sample_title"][0]
    stage = rows["!Sample_characteristics_ch1"][0]
    lib = rows["!Sample_characteristics_ch1"][-1]
    df = pd.DataFrame(
        {
            "gsm": gsm,
            "title": title,
            "stage": [s.replace("Stage: ", "") for s in stage],
            "library": [s.replace("library type: ", "") for s in lib],
        }
    )
    return df[df["library"] == "GEX"].reset_index(drop=True)


STAGE_MAP = {
    "term cord blood": ("term_cb", 0.0),
    "preterm cord blood": ("preterm_cb", 0.0),
    "1 week": ("postnatal", 7.0),
    "1 month": ("postnatal", 30.0),
    "2 months": ("postnatal", 60.0),
    "adult": ("adult", np.nan),
}


def main():
    meta = read_stage_map()
    stems = sorted({p.name.replace("_matrix.mtx.gz", "") for p in RAW.glob("*_matrix.mtx.gz")})
    print(f"[20a] {len(stems)} GEX 样本, 元数据 {len(meta)} 行", flush=True)

    # GSM -> stem
    gsm2stem = {s.split("_")[0]: s for s in stems}
    meta["stem"] = meta["gsm"].map(gsm2stem)
    missing = meta[meta["stem"].isna()]
    if len(missing):
        print("[20a] 警告: 无矩阵的 GSM", missing["gsm"].tolist(), flush=True)

    adatas = []
    for _, row in meta.dropna(subset=["stem"]).iterrows():
        stem = row["stem"]
        bcs = gzip.open(RAW / f"{stem}_barcodes.tsv.gz", "rt").read().split()
        feats = pd.read_csv(
            RAW / f"{stem}_features.tsv.gz", sep="\t", header=None,
            names=["ensembl", "symbol", "type"],
        )
        mat = scipy.io.mmread(str(RAW / f"{stem}_matrix.mtx.gz")).tocsc().astype(np.int32)
        # mtx 为 (gene x cell)，转 cell x gene
        X = sp.csr_matrix(mat.T)
        a = ad.AnnData(X=X)
        a.obs_names = [f"{stem}_{b}" for b in bcs]
        a.var_names = feats["ensembl"].values
        a.var["symbol"] = feats["symbol"].values
        q = re.match(r"GSM\d+_(\d+)_(.+)$", stem)
        subj, visit = q.group(1), q.group(2)
        a.obs["gsm"] = row["gsm"]
        a.obs["sample"] = stem
        a.obs["title"] = row["title"]
        a.obs["stage_raw"] = row["stage"]
        a.obs["group"] = STAGE_MAP[row["stage"]][0]
        a.obs["age_days"] = STAGE_MAP[row["stage"]][1]
        a.obs["subject"] = f"S{subj}"
        a.obs["visit"] = visit
        a.obs["n_counts"] = np.asarray(X.sum(1)).ravel()
        a.obs["n_genes"] = np.asarray((X > 0).sum(1)).ravel()
        adatas.append(a)
        print(f"  {stem:32s} {row['stage']:20s} cells={a.n_obs:6d}", flush=True)

    adata = ad.concat(adatas, join="outer", merge="same", index_unique=None)
    adata.var_names_make_unique()
    adata.obs = adata.obs.astype(
        {c: "category" for c in ["gsm", "sample", "title", "stage_raw", "group", "subject", "visit"]}
        if all(c in adata.obs for c in ["gsm", "sample", "title", "stage_raw", "group", "subject", "visit"])
        else {}
    )
    print(f"[20a] 合并: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

    stage_tab = (
        adata.obs.groupby(["group", "stage_raw"], observed=True)
        .agg(n_cells=("gsm", "size"), n_samples=("sample", "nunique"))
        .reset_index()
    )
    print(stage_tab.to_string(index=False), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    meta.drop(columns=["stem"]).to_csv(OUT / "sample_manifest.csv", index=False)
    adata.write_h5ad(OUT / "gse271413_raw.h5ad", compression="gzip")
    print(f"[20a] 写出 {OUT/'gse271413_raw.h5ad'}", flush=True)


if __name__ == "__main__":
    main()
