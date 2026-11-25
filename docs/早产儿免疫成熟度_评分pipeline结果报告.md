# 早产儿免疫成熟度评分 — pipeline 落地结果报告

> 实际执行记录（2026-11，本地 Python/scanpy 路线，非计划稿）。
> 对应脚本 `code/20*.py`，中间产物 `data/gse271413/*.h5ad`，统计证据 `evidence/gse271413_*.json`。

---

## 1. 数据与流程

| 步骤 | 脚本 | 产出 | 说明 |
|---|---|---|---|
| 构建 | `20a_build_gse271413.py` | `gse271413_raw.h5ad` | 236,076 cells × 36,601 genes，29 个 GEX 样本 |
| QC/整合 | `20b_qc_integrate.py` | `gse271413_proc.h5ad` | 滤后 226,635 cells × 30,135 genes；Harmony 去批次 |
| 注释 | `20c_annotate.py` / `20c2_celltypist.py` | `gse271413_annot.h5ad` | CellTypist 12 类（T_CD4 8.4万…DC 1781） |
| 评分 | `20d_signature_score.py` | `gse271413_scored.h5ad`、`signatures.json` | SigA/SigB 各 200 up + 200 down |
| 分离检验 | `20d2_preterm_separation.py` | `gse271413_preterm_sep.json` | SigB/SigA 置换检验 |
| 组成混淆 | `20e_composition.py` | `gse271413_composition.json` | 细胞组成差异 + 评分混淆检验 |
| 外部验证 | `20f_validate_external.py` | `gse271413_external_validation.json` | 3 个独立队列假bulk验证 |

两个签名（假bulk logCPM + t-test top200）：
- **SigA（免疫成熟轴）**：成人 vs 足月脐血。
- **SigB（早产轴）**：足月 vs 早产脐血。
- 细胞级评分 = logCPM 平均（up 基因）− 平均（down 基因）。

---

## 2. 样本级评分（GSE271413 内部）

| 群组 | scoreA | scoreB |
|---|---|---|
| adult | +0.070 | −0.14 |
| term_cb | −0.024 | −0.084 |
| preterm_cb | ≈−0.005（略高于term） | **−0.245**（明显更负） |
| postnatal 7d/30d/60d | 渐近 0~+0.04 | −0.05~−0.34 |

---

## 3. 核心发现（含诚实阴性）

### 3.1 SigB 在训练集内强分离早产 vs 足月（置换显著）
- Cohen's d = **+1.43**（早产 −0.245 vs 足月 −0.084，Mann-Whitney p≈0）。
- 100 次随机基因集置换 `null |d| max = 0.48`，`p = 0.000` → **远超随机**。
- `SigB_postnatal_age_trajectory`：与日龄 rho=−0.168，`p=0.000`（对随机 null max 0.167，边缘显著）。

### 3.2 SigA 评分机制自洽，但不能捕获“早产未成熟”（诚实阴性）
- 自洽：成人 vs 足月 d = −1.29（评分机制工作正常）。
- 但 SigA 早产 vs 足月 d = −0.22，置换 `p=0.850` → **不显著且方向与预期相反**（早产 scoreA 略高于足月）。
- 结论：**成人成熟轴 ≠ 早产不成熟轴**。早产免疫状态并非单纯“沿成人成熟轴更靠向脐血/未成熟端”。

### 3.3 外部验证（3 个独立队列）
| 队列 | 签名 | 结果 | 判定 |
|---|---|---|---|
| GSE326288 (PBMC FT vs PT) | SigB | PT −0.015 vs FT −0.299，diff +0.285，MW p=0.18 | **方向反转，无法泛化** |
| GSE302276 (Adult vs Cord PB) | SigA | Adult +0.264 vs Cord −1.431，diff +1.69，p=0.20 | 方向正确但 n 小不显著 |
| GSE157007 (Cord/Young/Old) | SigA | rho=+0.35，p=0.26；Old +1.10 最高但 Cord +0.50 非最低 | 部分方向、不显著 |

**诚实验证结论：**
- SigA 捕捉"成人 vs 脐血"发育轴的**方向在两个独立队列一致**（成人/老年端更高），但统计不显著（n 小）。
- **SigB 的训练集内分离（d=1.43）无法跨队列复现**——在 GSE326288 PBMC 队列上方向直接反转。提示该早产 signature 是训练队列/组织特异性的，不是可移植的泛早产分类器。

### 3.4 组成混淆（关键警示）
`scoreB` 与细胞组成的强相关表明其分离信号部分由组成差异驱动：

| score | Neutrophil | T_CD4 | Mono_classical | NK |
|---|---|---|---|---|
| scoreB | +0.71 | **−0.76** | +0.73 | +0.07 |
| scoreA | −0.57 | +0.34 | −0.41 | +0.34 |

早产 vs 足月脐血组成差异（样本级 MW，p<0.10）：pDC（preterm 更低，p=0.008）、Neutrophil（更低 0.032）、Mono_classical（更低 0.095）。

> 即 `SigB` 对早产的强分离，很大程度是"早产样本里 T_CD4 少 / 髓系占比不同"这一个组成效应在起作用，而非纯粹的细胞内在转录成熟差异。报告成文时必须区分并做细胞亚群内部校正。

---

## 4. 结论与下一步

**主线结论**：
1. 免疫成熟度评分 pipeline 已完整落地（QC→整合→注释→评分→置换→组成→外部验证）。
2. SigA 较好地反映"成人/脐血"发育轴（跨队列方向一致）；但**不能**捕获早产-足月的未成熟差异。
3. SigB 在训练集内强分离早产/足月且超随机，但**被细胞组成深刻混淆**且**外部队列无法复现**。

**下一步（按优先级）**：
- [ ] 按细胞亚群内部重评 SigB（校正组成混淆后检验残余信号）。
- [ ] SigB 迈向可移植分类器需跨队列重训练（改用训练/验证严格分离）。
- [ ] 早产未成熟轴改用"成熟度连续化"（脐血→儿童→成人）而非成人 vs 脐血二分。
- [ ] 若追求可发表的阳结果，转向组成亚群（如髓系/T 细胞功能程序）而非整体 signature。