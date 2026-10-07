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
- Explanation: pending A4b.

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
  so linear models still fit it (0.744). Kept for the record. Replaced by a high-frequency `periodic` set (2 full oscillations). Running.
