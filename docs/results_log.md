# Results Log

Every approach we try is recorded here with the same template (teacher's guidance: try various
approaches, and explain *why* classical wins if it does):

```
### <ID>: <approach name>
- Idea: what we are testing
- Why it might help:
- Setup: datasets, circuit/kernel, grid, splits
- Result: C-index vs best classical (table / link)
- Observed: what happened (numbers)
- Explanation: why, citing measurements R1–R7 (see docs/PLAN.md, Step 5b)
```

## Step 1: Classical reproduction of Van Belle et al. (2011): ✅ pipeline trusted

Run: `uv run python scripts/step1_reproduce.py --splits 20` (2026-10-07).
Full table: [results/step1_summary.md](../results/step1_summary.md). Per-split rows: `results/step1_splits.csv`.

Median test C-index over 20 splits (2/3 train, 1/3 test; inner 5-fold CV):

| Dataset | Kernel | Ranking | Hybrid | Regression | Cox | Paper (regr./hybrid) |
|---|---|---|---|---|---|---|
| VLC | linear | 0.721 | 0.720 | 0.721 | 0.718 | 0.69 (Cox 0.68) |
| VLC | clinical | 0.706 | 0.706 | 0.705 | | 0.69 |
| VLC | RBF | 0.720 | 0.711 | 0.707 | | |
| GBSG2 | linear | 0.681 | 0.681 | 0.678 | 0.677 | 0.67 (Cox 0.67) |
| GBSG2 | clinical | 0.692 | 0.691 | 0.684 | | 0.68 |
| GBSG2 | RBF | 0.683 | 0.682 | 0.682 | | |

**Verdict:**
- Regression and hybrid forms land within **+0.01 to +0.03** of the paper. Our Cox reference sits higher by the same margin,
  so the gap comes from the splits/tuning, not the SVM. The paper used 50 splits and coupled simulated annealing; we use 20 splits and a grid.
- **Ranking does not reproduce the paper's weakness** (paper 0.57–0.62, ours ≈ regression). Expected: the paper's
  RANKSVMC / Model 1 compare each patient only with its *nearest* comparable neighbour (O(n) constraints), while
  scikit-survival's ranking objective uses **all** comparable pairs. That is a known implementation difference
  (Pölsterl et al. 2015), noted in Methods. All three forms stay in the study.
- Hazard ratios have huge spread (normalized risk with outliers), as in the paper's tables. We report them but rely on the C-index.

**Implementation notes:**
- Kernel SVM trained as a linear SVM on the empirical kernel map `Φ` (`ΦΦᵀ = K`). This is exact
  (`tests/test_models.py`) and 20–60× faster than `FastKernelSurvivalSVM(kernel="precomputed")`.
- Shared `alpha` grid widened to 2^-14 … 2^4 after a scan showed CV optima across that whole range.

---

## Step 4: Quantum kernels on the paper's datasets (A1–A4, A6)

Run: `scripts/step4_qksm.py` (20 splits, 3 forms). Tables: [results/step4_comparison.md](../results/step4_comparison.md),
paired tests: `results/step4_wilcoxon.csv`. Median test C-index, **hybrid form** (other forms agree):

| Kernel | VLC | GBSG2 |
|---|---|---|
| linear (classical) | **0.720** | 0.681 |
| RBF (classical) | 0.711 | 0.682 |
| clinical (classical) | 0.706 | **0.691** |
| Cox-LASSO / RSF (baselines, Step 5) | 0.718 / 0.693 | 0.673 / 0.689 |
| A1 `Q-1L` (no entanglement) | 0.719 | 0.677 |
| A2 `Q-2L` | 0.715 | 0.679 |
| A3 `Q-3L` | 0.712 | 0.677 |
| A4 `Q-ZZ` | 0.718 | 0.685 |
| A6 `P-Q-2L` (projected) | 0.689 | 0.671 |

Paired Wilcoxon signed-rank over the 20 splits, vs the best classical kernel of each dataset:
- **VLC:** fidelity kernels −0.002 to −0.011, mostly not significant (p 0.03–0.45). It's a tie with linear.
- **GBSG2:** all quantum kernels significantly worse than clinical (−0.008 to −0.020, p ≤ 0.006).
- **Projected kernel:** significantly worse on both (−0.03, p < 0.001).
- **Entanglement vs the `Q-1L` control:** `Q-2L` / `Q-3L` not significant on either dataset.
  `Q-ZZ` > `Q-1L` on GBSG2 (p ≤ 0.001), the only such case. Ablation A4b (`Q-Z`, same circuit without ZZ gates) is running
  to see whether that gain is entanglement or encoding.

### A1: `Q-1L` product kernel (control)
- Idea: angle encoding with no entanglement, a classical `Π cos²` kernel (proved by unit test).
- Result: ties linear on VLC (0.719 vs 0.720), below clinical on GBSG2 (0.677 vs 0.691).
- Explanation: at its best `c` it is the same matrix as an RBF kernel on the angle features (alignment 1.000, R4).
  So it can only do what RBF does, and RBF ≈ linear on these datasets (R5).

### A2: `Q-2L` (encode → CNOT ring → encode)
- Idea: entanglement mixes features, which might capture interactions.
- Why it might help: interaction terms appear between qubits that the product kernel lacks.
- Result: no significant difference from `Q-1L` (VLC p = 0.06, GBSG2 p = 0.35); significantly below clinical on GBSG2.
- Explanation: the best `c` is the smallest one (0.1, and 0.01–0.03 gives the same), where the circuit's rotations are tiny and the
  CNOT-mixed state is still **99.3–100% aligned with an RBF kernel** on the same features (R4, `results/step3_small_c.csv`).
  At larger `c`, where entanglement really changes the kernel, the kernel concentrates (R1) and overfits (R2). So entanglement has
  no regime where it is both "on" and useful here.

### A3: `Q-3L` (deeper)
- Idea: more re-uploading layers mean a more expressive feature map.
- Result: slightly *worse* than `Q-2L` (VLC 0.712, GBSG2 0.677).
- Explanation: more layers concentrate the kernel faster. At c = 1: off-diagonal mean `Q-1L` 0.045 → `Q-2L` 0.036 →
  `Q-3L` 0.028 (VLC), effective rank fraction 0.25 → 0.33 → 0.40, train–test gap 0.21–0.22 (R1, R2). Expressivity adds variance, not signal.

### A4: `Q-ZZ` feature map (pairwise `x_j·x_k` entanglers)
- Idea: the standard QML feature map, with data-dependent entanglement.
- Result: the best quantum kernel on GBSG2 (0.685), but still below clinical (0.691, p = 0.006). Ties linear on VLC.
- Observed: at small `c` the pair term `c·x_j·x_k/π` is second order, so `Q-ZZ` ≈ a product kernel (alignment with matched RBF = 1.000).
- Explanation: **not entanglement**. See A4b.

### A4b: `Q-Z` = `Q-ZZ` without the ZZ gates (ablation, product state)
- Idea: `Q-ZZ` differs from `Q-1L` in two ways (H+RZ encoding *and* ZZ entanglers). Removing the ZZ gates isolates the entanglement effect.
- Result (hybrid): GBSG2 `Q-Z` 0.688 vs `Q-ZZ` 0.685, **no difference** (median diff 0.000, p = 0.90). `Q-Z` > `Q-1L` (+0.009, p = 1e-4).
  `Q-Z` still < clinical (−0.009, p = 0.008). VLC `Q-Z` 0.718 ≈ linear (p = 0.39).
- Explanation: the `Q-ZZ` gain over `Q-1L` comes entirely from its **single-qubit encoding**. H·RZ·H·RZ gives each feature a different
  (richer) one-qubit feature function than RY, and it is still a product kernel, i.e. classical. **Entanglement contributed nothing measurable.**

### A5: bandwidth (`c`) sweep, all 7 values recorded (`results/step3_configs.csv`, 5 splits)
- Every quantum kernel shows the same curve: as `c` grows 0.1 → 1, off-diagonal mean falls (VLC `Q-2L` 0.885 → 0.036),
  KTA falls (0.122 → 0.068), test C-index falls (0.698 → 0.623), train–test gap grows (0.087 → 0.172).
- So the tuned `c` always ends up at the bottom of the grid, where the kernel is most classical.

### A6: projected quantum kernel (`P-Q-2L`)
- Idea: Huang et al. (2021) propose it to avoid fidelity-kernel concentration.
- Result: the worst kernel on both datasets (VLC 0.689, GBSG2 0.671; −0.03, p < 0.001).
- Explanation: it avoids concentration (off-diagonal mean stays ≈ 0.4 for every `c`), but measuring single-qubit
  ⟨X⟩,⟨Y⟩,⟨Z⟩ maps each patient to 3·d bounded, periodic numbers (sin/cos of the angles plus entangled mixtures). That distorts the
  monotone covariate effects. KTA is the lowest of all kernels (VLC 0.113, GBSG2 0.025), so the pre-training diagnostic predicted the loss.

---

## Step 3: Pre-training diagnostics (Objective 2) and evidence R1–R4

Run: `scripts/step3_diagnostics.py` (5 splits × 44 kernel configs per dataset; hybrid SVM with alpha tuned per config).
Per-config means: `results/step3_configs.csv`.

**Survival KTA predicts the outcome before any training.** Spearman ρ between training-set KTA and test C-index across the 44 configs:

| | KTA (comparable pairs) | KTA (event-only cross-check) |
|---|---|---|
| VLC | **ρ = 0.82** (p = 9e-12) | 0.82 |
| GBSG2 | **ρ = 0.73** (p = 2e-8) | 0.68 |

- Both versions rank kernels the same way, so censored patients' lower-bound ranks don't distort the result.
- Permutation test (200 shuffles of (time, event)): p = 0.005 (the minimum possible) for every kernel except projected at c ≥ 0.68, so all kernels carry *some* survival information.
- **KTA ordering = outcome ordering:** linear / RBF have the highest KTA (VLC 0.18, GBSG2 0.068), and quantum kernels sit lower
  (best ≈ 0.12–0.13 / 0.028–0.029). The diagnostic says "quantum geometry is less aligned with survival" before any model is fit.

**R1 Concentration.** Off-diagonal mean of K at c = 0.1 → 1.0: `Q-2L` VLC 0.885 → 0.036, GBSG2 0.934 → 0.112. At large `c` every patient
looks "different" from every other, and deeper circuits (A3) concentrate faster.

**R2 Overfitting.** Train–test C-index gap: linear 0.049 (VLC) / 0.010 (GBSG2). Best quantum configs 0.07–0.09 / 0.03–0.04,
growing to 0.17–0.22 / 0.12–0.13 at c = 1. The effective rank grows at the same time (`Q-2L` VLC 1.4% → 33% of n).

**R3 Geometric difference vs RBF** at the best quantum configs: g ≈ 2.1–2.3 (VLC) and 2.8–4.0 (GBSG2), far below
√n ≈ 9.5 / 21. Per Huang et al. (2021), a classical RBF kernel can match these quantum kernels on this data, so there's no room for an advantage.
(g vs *linear* is huge only because the linear kernel has rank 8; it is not meaningful.)

**R4 Kernel alignment.** At the `c` where quantum kernels work best, they are 99.3–100% aligned with an RBF kernel
`exp(−‖θ−θ'‖²/4)` on the same angle features. This follows from `cos²(t/2) ≈ exp(−t²/4)` for small t: **the best-performing quantum kernel is an RBF kernel.**

---

## Step 5: Benchmark + synthetic dial (R5, R7)

Run: `scripts/step5_benchmark.py` (20 splits; kernels in hybrid form). Table: [results/step5_summary.md](../results/step5_summary.md).

| Model | synth-linear | synth-interaction | synth-smooth-periodic |
|---|---|---|---|
| linear SVM | 0.795 | 0.497 | 0.744 |
| Cox-LASSO | **0.796** | 0.500 | **0.745** |
| RBF | 0.792 | **0.771** | 0.737 |
| RSF | 0.773 | 0.670 | 0.738 |
| `Q-1L` / `Q-2L` / `Q-3L` | 0.793 / 0.795 / 0.793 | 0.770 / 0.769 / 0.765 | 0.739 / 0.734 / 0.737 |
| `Q-ZZ` / `Q-Z` | 0.793 / 0.793 | 0.768 / 0.767 | 0.737 / 0.735 |
| `P-Q-2L` | 0.789 | 0.758 | 0.744 |

- **R5 (how nonlinear the data is):** on VLC/GBSG2, RBF and RSF never beat linear Cox by more than 0.01–0.02. On synth-interaction
  RBF beats linear by **+0.27**. So the clinical datasets contain almost no nonlinear signal for *any* kernel to exploit.
- **Positive control works:** when interactions exist, nonlinear kernels win big. Quantum kernels win too, but only *tie* RBF (0.765–0.770 vs 0.771),
  consistent with R4 (they are RBF-like).
- **Design flaw found and fixed:** "smooth-periodic" `sin(2x₁)cos(2x₂) + sin(x₃+x₄)` is mostly monotone over the data range,
  so linear models still fit it (0.744). Kept for the record. Replaced by a high-frequency `periodic` set
  `sin(4x₁)cos(4x₂) + sin(4x₃)` (2 full oscillations).

### R7 / A5b: high-frequency periodic data and wrap-around kernels (`c` ∈ [1, 8])

| Model | VLC | GBSG2 | synth-interaction | synth-periodic |
|---|---|---|---|---|
| best classical | 0.720 (linear) | 0.691 (clinical) | 0.771 (RBF) | 0.686 (clinical) |
| RSF / Cox-LASSO | 0.693 / 0.718 | 0.689 / 0.673 | 0.670 / 0.500 | 0.670 / 0.569 |
| RBF / RBF-narrow (fairness check) | 0.711 / – | 0.682 / – | 0.771 / 0.743 | 0.547 / 0.534 |
| `Q-1L` (normal `c`) | 0.719 | 0.677 | 0.770 | 0.555 |
| **`Q-1L-wrap`** | 0.693 | 0.645 | 0.730 | **0.716** |
| `Q-2L-wrap` | 0.610 | 0.636 | 0.644 | 0.659 |
| `Q-ZZ-wrap` | 0.606 | 0.636 | 0.664 | 0.520 |

- **The only win of the project:** on synth-periodic, `Q-1L-wrap` beats every classical model: clinical +0.033 (p = 0.001),
  RSF +0.033 (19/20 splits, p = 1e-5), RBF-narrow +0.175 (20/20).
- **Fairness check:** RBF with much narrower bandwidths (2–128 × median heuristic) does *not* catch up (0.534), so the win is not a tuning artifact.
- **But it is not a quantum win:** `Q-1L-wrap` has **no entanglement**. It is the product-of-cosines kernel `Π cos²(c·Δx/2)`, a
  classical *periodic* kernel that could be computed in one line of NumPy. Adding entanglement makes it **worse**
  (`Q-2L-wrap` −0.062, p = 4e-6; `Q-ZZ-wrap` 0.520).
- **On real data the same wrap-around kernels hurt badly** (VLC −0.03 to −0.11, GBSG2 −0.05). The periodic shape makes patients with
  very different covariate values look similar, which contradicts risk that rises steadily with age, nodes, tumour size, etc.
- Explanation (R5 + R7): what wins is **the match between the kernel's shape and the true risk function** (inductive bias).
  Periodic kernel + periodic risk wins; periodic kernel + monotone clinical risk loses. Entanglement only ever made things worse here.

---

## Step 6: remaining approaches (A7, KPCA-Cox) and R6

Run: `scripts/step6_extra.py a7 kpca lc` (20 splits; learning curves 10 splits). Table: `results/step6_a7_kpca_lc.md`.

### A7: hybrid kernel `w·K_Q + (1−w)·K_linear` (c chosen by KTA before training)
- Idea: even if the quantum kernel loses alone, it may carry information the linear kernel lacks.
- Result (median C / median chosen w): VLC `H-Q-2L` 0.718 / **w = 0.17**; GBSG2 0.678 / 0.42; `H-Q-ZZ` the same.
  Never better than the best single classical kernel.
- Observed: on VLC, CV puts ~80% of the weight on the **linear** part. Mixing in the quantum kernel gives no complementary information.
- **KTA as a selector fails on interaction data:** synth-interaction `H-Q-2L` 0.657, vs 0.769 for `Q-2L` tuned by CV.
  Cause (`results/step6_a7_kta_choice.csv`): there, KTA *increases* with `c` (−0.003 → 0.029) while test C *decreases* (0.761 → 0.657),
  so KTA picks the worst `c`. On VLC/GBSG2, KTA and test C move together (both peak at small `c`).
- Explanation: KTA is computed on the training data, so it rewards flexible, concentrated kernels that "fit" the training ranks
  (the same effect as the R2 overfitting gap). When the true risk is non-monotone in the features, absolute KTA values are near zero,
  and this optimism dominates. **Lesson: KTA is good for screening kernel families, but CV is still needed for tuning.**

### KPCA-Cox (Kernel Cox version of the QKSM, Objective 1)

| Kernel | VLC | GBSG2 | synth-interaction |
|---|---|---|---|
| linear | **0.718** | 0.670 | 0.499 |
| RBF | 0.706 | 0.678 | **0.731** |
| clinical | 0.704 | **0.687** | 0.490 |
| `Q-1L` | 0.709 | 0.670 | **0.731** |
| `Q-2L` | 0.709 | 0.669 | 0.708 |
| `Q-ZZ` | 0.711 | 0.675 | 0.707 |

- Same pattern as the SVM: quantum ≤ best classical on real data; on interaction data the **product** `Q-1L` ties RBF, while the
  entangled `Q-2L`/`Q-ZZ` are lower (−0.023). The conclusion does not depend on which survival model sits on top of the kernel.

### R6: learning curves (hybrid SVM, 10 splits)

| GBSG2, training patients → | 91 (20%) | 183 | 274 | 366 | 457 (100%) |
|---|---|---|---|---|---|
| linear | **0.648** | **0.664** | 0.674 | 0.679 | 0.678 |
| RBF | 0.636 | 0.655 | 0.678 | 0.684 | 0.684 |
| `Q-1L` | 0.624 | 0.647 | 0.670 | 0.674 | 0.681 |
| `Q-2L` | 0.623 | 0.643 | 0.663 | 0.675 | 0.682 |
| `Q-ZZ` | 0.631 | 0.650 | 0.674 | 0.683 | 0.693 |

- With little data, linear wins clearly (+0.017 to +0.025 over quantum at 91 patients). Quantum kernels only catch up at the full training size.
  Flexible kernels need more data to estimate. Clinical cohorts are small (VLC has only 91 training patients), so this works against quantum kernels.
- On synth-interaction, quantum and RBF curves overlap at every size, and linear stays at 0.50. Again quantum = RBF.

---

## Step 7: TCGA-BRCA multi-omics (the problem statement's oncology setting)

Data: UCSC Xena TCGA hub. mRNA (HiSeqV2), copy number (GISTIC2), clinical, curated overall survival.
**1063 primary tumours, 148 deaths (86% censored).** Label-free pre-filter to the top 2000 variance genes per omic. Then, **inside each
training split**: top 1000 genes → z-score → PCA → 3 mRNA PCs + 3 CNV PCs + age + stage = **8 features = 8 qubits**, the same 8 for every
kernel model. Methylation (450K, 783 MB) not included.
Run: `scripts/step7_tcga.py` (20 splits, hybrid form). Tables: [results/step7_tcga_summary.md](../results/step7_tcga_summary.md),
tests: `results/step7_wilcoxon.txt`.

| Model | Median C-index | vs linear (paired) |
|---|---|---|
| `Q-1L` (no entanglement) | **0.766** | +0.006, 15/20 splits, p = 0.048 |
| `Q-Z` (no entanglement) | **0.766** | +0.008, p = 0.07 |
| linear | 0.765 | – |
| `Q-2L` | 0.762 | +0.001, p = 0.55 |
| `P-Q-2L` | 0.762 | p = 0.31 |
| `Q-ZZ` | 0.761 | p = 0.25 |
| RBF | 0.761 | p = 0.90 |
| clinical kernel | 0.754 | |
| Cox-LASSO (same 8 features) | 0.750 | |
| Cox-LASSO, **clinical only** (age, stage) | 0.746 | |
| Cox-LASSO, **all 4000 genes** + clinical | 0.724 | 8-feature Cox-LASSO better by +0.013, p = 0.001 |
| RSF | 0.715 | linear better by +0.030, p = 4e-6 |

**Observed:**
- All kernel SVMs are within 0.012 of each other. Unentangled product kernels (`Q-1L`, `Q-Z`) edge linear by +0.006–0.008. Only `Q-1L`
  reaches p < 0.05 (p = 0.048), which does **not** survive correction for the ~10 comparisons we made, so it is not a reliable improvement.
- **Entanglement again makes things worse:** `Q-2L` < `Q-1L` (15/20 splits, p = 0.03), `Q-ZZ` < `Q-Z` (14/20, p = 0.012).
- **Omics add almost nothing over clinical variables:** 8-feature Cox-LASSO vs clinical-only, p = 0.35; best kernel vs clinical-only Cox +0.02.
  Overall survival in BRCA is driven mostly by age and stage, and with only 148 deaths there is little signal left to find.
- **More features hurt:** Cox-LASSO on all genes (0.724) is worse than on 8 PCs (0.750). With 148 events, high-dimensional models overfit.
- Diagnostics (3 splits, 51 configs): KTA still predicts test C-index (Spearman **0.58**, p = 7e-6; event-only 0.60), weaker than on
  VLC/GBSG2 (0.82 / 0.73). Best quantum configs again sit close to RBF (alignment 0.92–0.93, g vs RBF 2.1–2.7 ≪ √n ≈ 27).
  The best `c` is larger here (0.32–0.46), so the quantum kernels are less concentrated at their best on this dataset.

**Explanation:**
- **Why the product kernels slightly beat RBF/linear (hypothesis, labelled as such):** they use min-max scaling to [0, π] and a bounded
  similarity, so patients with extreme PCA scores (omics PCs are heavy-tailed) influence the model less than under z-scoring. This is a
  property of the *encoding*, not of quantum mechanics: the kernel is the classical `Π cos²` product.
- **Why entanglement doesn't help:** same as before (R1/R2): it adds flexibility, and with 148 events extra flexibility is variance, not signal (R6).
- **Why multi-omics doesn't change the picture:** the outcome is dominated by smooth, monotone clinical effects (R5). The omic PCs add
  little, so there are no complex feature interactions for any nonlinear kernel, quantum or classical, to exploit.

---

## Why classical performs better: summary of the evidence

| # | Reason | Evidence |
|---|---|---|
| 1 | **The clinical data has almost no nonlinear signal.** Survival risk rises steadily with each covariate, so linear models are already near the ceiling. | RBF and RSF never beat linear Cox by > 0.02 on VLC/GBSG2/TCGA-BRCA (RSF is worse on TCGA, −0.03) (R5). Multi-omics PCs add nothing significant over age + stage on TCGA (p = 0.35). When nonlinear signal exists (synth-interaction), nonlinear kernels gain +0.27. |
| 2 | **In the regime where quantum kernels work best, they *are* classical RBF kernels.** | At the best `c`, 99.3–100% alignment with `exp(−‖θ−θ'‖²/4)` on the same angles, since `cos²(t/2) ≈ exp(−t²/4)` (R4). Geometric difference vs RBF only 2–4 ≪ √n (R3). |
| 3 | **When the circuit becomes "really quantum" (larger angles, entanglement, depth), the kernel concentrates and overfits.** | Off-diagonal similarity 0.9 → 0.03, effective rank ×20, train–test gap ×2–3, test C falls (R1, R2, A3, A5). |
| 4 | **Entanglement never helped.** Every apparent gain came from the single-qubit encoding. | `Q-2L`/`Q-3L` ≈ or < `Q-1L` everywhere; `Q-ZZ` = `Q-Z` on GBSG2 (p = 0.90, A4b); on TCGA, periodic and interaction data the entangled versions are significantly *worse* (e.g. TCGA `Q-2L` < `Q-1L` p = 0.03, `Q-ZZ` < `Q-Z` p = 0.012). The small TCGA edge of `Q-1L` over linear (+0.006, p = 0.048, not robust to multiple testing) is also an unentangled, classical kernel. |
| 5 | **Inductive-bias mismatch:** angle encodings make similarity periodic, which is wrong for monotone clinical risk. | Wrap-around kernels lose 0.03–0.11 on VLC/GBSG2 but win +0.03 on truly periodic data (A5b/R7). |
| 6 | **Small cohorts favour simple kernels.** | Learning curves: linear leads by ~0.02 at 91 patients; quantum only catches up at full size (R6). |
| 7 | **The pre-training diagnostic saw it coming.** | Survival KTA ranks kernels like test C-index (ρ = 0.82 VLC / 0.73 GBSG2 / 0.58 TCGA-BRCA), and quantum kernels had lower KTA than linear/RBF before training (Objective 2). Limitation: as a tuner of `c` on non-monotone data it fails (A7). |

**Conditions under which a quantum-circuit kernel *could* help** (from R7): the true risk must be **periodic / oscillating** in the
features, and even then the winning kernel was the *unentangled* product kernel, which is classically computable. Any real quantum advantage
would need a structure that is both (a) present in the data and (b) not reproducible by a classical kernel (large geometric difference, R3).
Nothing we measured on clinical data meets (a), and nothing we built meets (b).
