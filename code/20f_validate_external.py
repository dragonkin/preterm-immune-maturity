"""
20f_validate_external.py  —  跨队列外部验证 (REWRITE, robust)
用 GSE271413 免疫成熟度签名在三个独立队列假bulk评分:
  GSE326288 : FT vs PT 首48h PBMC  → SigB 分开早产/足月
  GSE302276 : UCB vs Adult PB     → SigA 分开成人/脐血(成熟度)
  GSE157007 : Cord/Young/Old PBMC → SigA 随成熟度单调(Cord<Young<Old)
输入 data/gse271413/signatures.json ；输出 evidence/gse271413_external_validation.json
"""
import glob
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SIG = json.load(open(ROOT / "data" / "gse271413" / "signatures.json"))
EVID = ROOT / "evidence"
EVID.mkdir(exist_ok=True)


def load_single_mtx(dirp, stem):
    """按 stem 匹配 barcodes/features/matrix 三个文件(自动探测扩展名), 返回 (symbol_series)."""
    import scipy.io
    bcs_f = glob.glob(str(dirp / f"*{stem}*barcodes*"))
    feat_f = glob.glob(str(dirp / f"*{stem}*features*"))
    mat_f = [glob.glob(str(dirp / f"*{stem}*matrix.mtx.gz"))
             or glob.glob(str(dirp / f"*{stem}*matrix.tsv.gz"))
             or glob.glob(str(dirp / f"*{stem}*matrix*.gz"))]
    mat_f = [x for x in mat_f if x][0]
    if not (bcs_f and feat_f and mat_f):
        raise FileNotFoundError(f"{stem}: barcodes/features/matrix 缺失")
    feats = pd.read_csv(feat_f[0], sep="\t", header=None)
    genes = feats[1].astype(str).values  # symbol
    m = scipy.io.mmread(mat_f[0]).tocsc()
    gene_sum = np.asarray(m.sum(1)).ravel()
    s = pd.Series(gene_sum, index=genes).groupby(level=0).sum()
    return s


def load_single_h5(dirp, stem):
    """GSE326288 h5 文件, 返回 symbol 基因和序列."""
    import h5py
    from scipy.sparse import csc_matrix
    f = glob.glob(str(dirp / f"*{stem}*matrix*.h5")) or glob.glob(str(dirp / f"*{stem}*.h5"))
    if not f:
        raise FileNotFoundError(f"{stem}: h5 缺失")
    with h5py.File(f[0], "r") as h:
        g = h["matrix"]
        symb = [s.decode() if isinstance(s, bytes) else s for s in g["features"]["name"][:]]
        idx = g["indices"][:]; indptr = g["indptr"][:]; data = g["data"][:]
        n = g["shape"][0]
        M = csc_matrix((data, idx, indptr), shape=(n, g["shape"][1]))
    s = pd.Series(np.asarray(M.sum(1)).ravel(), index=symb).groupby(level=0).sum()
    return s


def sig_score(pb, up, dn):
    up = [g for g in up if g in pb.index]
    dn = [g for g in dn if g in pb.index]
    if not up or not dn:
        return pd.Series(np.nan, index=pb.columns)
    return pb.loc[up].mean(0) - pb.loc[dn].mean(0)


def main():
    sigA_up, sigA_dn = SIG["A_up"], SIG["A_down"]
    sigB_up, sigB_dn = SIG["B_up"], SIG["B_down"]
    print(f"[20f] SigA up/dn={len(sigA_up)}/{len(sigA_dn)}, SigB up/dn={len(sigB_up)}/{len(sigB_dn)}")
    res = {"signature_sizes": {"A_up": len(sigA_up), "A_dn": len(sigA_dn),
                               "B_up": len(sigB_up), "B_dn": len(sigB_dn)}}

    # ---- GSE326288 (h5) : FT vs PT ----
    dirp = ROOT / "data" / "gse326288" / "raw"
    ft = ["R01FT022", "R01FT024", "R01FT029", "R01FT030", "R01FT031", "R01FT032"]
    pt = ["R01PT016", "R01PT017", "R01PT018", "R01PT019", "R01PT020", "R01PT022", "PT023"]
    ha = ["R01HA008", "R01HA011", "R01HA012", "R01HA013", "R01HA014", "R01HA015"]
    pb_f, grp_f = {}, {}
    for stem in ft + pt + ha:
        g = "FT" if stem in ft else ("PT" if stem in pt else "HA")
        try:
            pb_f[stem] = load_single_h5(dirp, stem)
            grp_f[stem] = g
            print(f"  {stem:10s} {g} genes={len(pb_f[stem])}", flush=True)
        except FileNotFoundError as e:
            print(f"  !! {e}")
    pb = pd.DataFrame({k: np.log1p(v / v.sum() * 1e6) for k, v in pb_f.items()})
    sb = sig_score(pb, sigB_up, sigB_dn)
    res["GSE326288"] = {"sig": "B"}
    for s in pb.columns:
        res["GSE326288"][f"{s}_SigB"] = float(sb[s])
    pt_v = [sb[s] for s in pb.columns if grp_f[s] == "PT"]
    ft_v = [sb[s] for s in pb.columns if grp_f[s] == "FT"]
    ha_v = [sb[s] for s in pb.columns if grp_f[s] == "HA"]
    res["GSE326288"]["PT_mean"] = float(np.mean(pt_v))
    res["GSE326288"]["FT_mean"] = float(np.mean(ft_v))
    res["GSE326288"]["HA_mean"] = float(np.mean(ha_v))
    if len(pt_v) >= 2 and len(ft_v) >= 2:
        u, p = stats.mannwhitneyu(pt_v, ft_v)
        res["GSE326288"]["PT_vs_FT_MW_p"] = float(p)
        res["GSE326288"]["PT_vs_FT_mean_diff"] = float(np.mean(pt_v) - np.mean(ft_v))
        print(f"[20f] GSE326288 SigB PT({np.mean(pt_v):+.3f}) vs FT({np.mean(ft_v):+.3f}) "
              f"diff={np.mean(pt_v)-np.mean(ft_v):+.3f} MW_p={p:.4f}", flush=True)

    # ---- GSE302276 (mtx) : UCB vs Adult ----
    dirp = ROOT / "data" / "gse302276" / "raw"
    adult = ["PB1966", "PB1970", "PB1974"]
    cord = ["UCB1997", "UCB1998"]
    pb_f, grp_f = {}, {}
    for stem in adult + cord:
        g = "Adult" if stem in adult else "Cord"
        try:
            pb_f[stem] = load_single_mtx(dirp, stem)
            grp_f[stem] = g
            print(f"  {stem:8s} {g} genes={len(pb_f[stem])}", flush=True)
        except FileNotFoundError as e:
            print(f"  !! {e}")
    pb = pd.DataFrame({k: np.log1p(v / v.sum() * 1e6) for k, v in pb_f.items()})
    sa = sig_score(pb, sigA_up, sigA_dn)
    res["GSE302276"] = {"sig": "A"}
    adult_v = [sa[s] for s in pb.columns if grp_f[s] == "Adult"]
    cord_v = [sa[s] for s in pb.columns if grp_f[s] == "Cord"]
    res["GSE302276"]["Adult_mean"] = float(np.mean(adult_v))
    res["GSE302276"]["Cord_mean"] = float(np.mean(cord_v))
    if len(adult_v) >= 2 and len(cord_v) >= 2:
        u, p = stats.mannwhitneyu(adult_v, cord_v)
        res["GSE302276"]["Adult_vs_Cord_MW_p"] = float(p)
        res["GSE302276"]["Adult_minus_Cord"] = float(np.mean(adult_v) - np.mean(cord_v))
        print(f"[20f] GSE302276 SigA Adult({np.mean(adult_v):+.3f}) vs Cord({np.mean(cord_v):+.3f}) "
              f"diff={np.mean(adult_v)-np.mean(cord_v):+.3f} MW_p={p:.4f}", flush=True)

    # ---- GSE157007 (mtx) : Cord/Young/Old ----
    dirp = ROOT / "data" / "gse157007" / "raw"
    cord = ["F020", "F021", "F023", "F024"]
    young = ["F012", "F013", "F014", "F016", "F017"]
    old = ["OH14", "OH15", "OH17"]
    pb_f, grp_f = {}, {}
    for stem in cord + young + old:
        g = "Cord" if stem in cord else ("Young" if stem in young else "Old")
        try:
            pb_f[stem] = load_single_mtx(dirp, stem)
            grp_f[stem] = g
            print(f"  {stem:6s} {g} genes={len(pb_f[stem])}", flush=True)
        except FileNotFoundError as e:
            print(f"  !! {e}")
    pb = pd.DataFrame({k: np.log1p(v / v.sum() * 1e6) for k, v in pb_f.items()})
    sa = sig_score(pb, sigA_up, sigA_dn)
    res["GSE157007"] = {"sig": "A"}
    m = {g: [sa[s] for s in pb.columns if grp_f[s] == g] for g in ["Cord", "Young", "Old"]}
    m = {k: [x for x in v if not np.isnan(x)] for k, v in m.items()}
    for k, v in m.items():
        res["GSE157007"][f"{k}_mean"] = float(np.mean(v))
        res["GSE157007"][f"{k}_n"] = len(v)
        print(f"  {k}: n={len(v)} mean={float(np.mean(v)):+.3f}", flush=True)
    if all(len(v) >= 2 for v in m.values()):
        order = np.concatenate([[1] * len(m["Cord"]), [2] * len(m["Young"]), [3] * len(m["Old"])])
        vals = m["Cord"] + m["Young"] + m["Old"]
        rho, p = stats.spearmanr(order, vals)
        res["GSE157007"]["Cord_Young_Old_spearman"] = float(rho)
        res["GSE157007"]["Cord_Young_Old_p"] = float(p)
        print(f"[20f] GSE157007 SigA 成熟梯度 rho={rho:+.3f} p={p:.4f}", flush=True)

    with open(EVID / "gse271413_external_validation.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("\n[20f] 完成", flush=True)


if __name__ == "__main__":
    main()