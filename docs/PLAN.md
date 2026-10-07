# Final Project Plan: Quantum Kernel Methods for Survival Analysis

**Status:** ✅ FINAL, revised for teacher's guidance (2026-10-07) · **Team:** Ansari, Tarun · **Timeline:** 4 weeks (lean scope) · **Anchor paper:** Van Belle et al. 2011

## Teacher's guidance (2026-10-07)

> A better quantum kernel is **not** expected. Apply **various approaches**. An improvement is welcome,
> but if classical performs better (the likely case), give **proper reasons and reasoning for why**.

What this changes:
1. **Breadth over tuning one model:** we try a catalogue of quantum-kernel approaches (Step 2, A1–A8),
   each designed to test one idea about where a quantum advantage *could* come from.
2. **Every approach is recorded** in [results_log.md](results_log.md) with the same template:
   *idea → why it might help → setup → result → what we observed → explanation*.
3. **New Step 5b, "Why classical wins":** each explanation must be backed by a **measurement**
   (R1–R7), not opinion. The report's main contribution is this evidence-based reasoning.

## Core idea

Van Belle's survival SVMs take the kernel as a plug-in part. We keep their models and
their testing protocol, and change **only the kernel**: linear / RBF / clinical → quantum.
Same models, same splits, same metric, so the comparison is fair and easy to defend.

## Decisions (fixed)

| Topic | Choice |
|---|---|
| Survival SVM | `scikit-survival` Survival SVM on a precomputed kernel, `rank_ratio` ∈ {1 = ranking, 0.5 = hybrid, 0 = regression}. Trained as a linear SVM on the empirical kernel map `Φ` (`ΦΦᵀ = K`): exact and tested, 20–60× faster |
| Quantum simulator | Qiskit (`Statevector`), 4–8 qubits |
| Quantum circuits | **≥ 2 encoding layers** (encode → CNOTs → encode). 1-layer = labelled "no entanglement" control only |
| Kernel Cox | Kernel PCA on the (quantum) kernel → penalized Cox on the components |
| Multi-omics | TCGA-BRCA (UCSC Xena): mRNA + methylation + CNV + clinical |
| Splits | **20** repeated 2/3 train – 1/3 test splits (paper: 50; reduced for time) |
| Main metric | Harrell's C-index; significance via **paired Wilcoxon signed-rank** test over splits |
| Tuning budget | Same `alpha` grid for every model; quantum `c` grid has the **same size** (7 values) as the RBF bandwidth grid |
| Reporting unit | Each kernel/circuit is its **own row**. Only `c`/bandwidth and `alpha` are tuned inside a row; the circuit is never auto-selected |

**Honesty notes for the report:**
- `scikit-survival`'s Survival SVM (Pölsterl et al.) *follows* Van Belle's ranking/regression
  idea but is not identical: it uses a squared hinge loss and compares all comparable pairs, not just the nearest neighbour. We say this in Methods.
- Kernel PCA + Cox is an approximation of Kernel Cox (Li & Luan 2003). We name it "KPCA-Cox".
- Survival KTA was first proposed by Colak et al. 2026 (ref [5]). We cite it and present ours as
  an extension: a target built from the paper's comparable pairs, a shuffled-label test, and a check of whether KTA predicts the C-index.
- The paper used the (unpaired) Wilcoxon rank-sum test. We use the **paired signed-rank** test, because every
  model is scored on the same splits. Repeated splits overlap in their training data, so they are not independent,
  which makes p-values optimistic. We say this and treat them as indicative only.
- In the survival-rank target, a censored patient's rank is only a **lower bound** on their true rank (they survived *at least* that long).

---

## Step 1: Reproduce the paper classically (baseline)

- **Data:** `load_veterans_lung_cancer()` (VLC, n=137), `load_gbsg2()` (GBSG2, n=686).
  One-hot encode categorical features and standardize continuous ones (fit on the train part only).
- **Models:** Survival SVM × `rank_ratio` {1, 0.5, 0} × kernels {linear, RBF, clinical (Eq. 12–13)}.
- **Protocol:** 20 splits. Inside each training part, use 5-fold CV over `alpha` (and RBF bandwidth),
  scored by C-index. Refit on the full training part and score on the test part.
- ⚠️ **Sign trap:** when `rank_ratio < 1` the model predicts survival *time*, so negate it before computing the C-index.
- ⚠️ Set **`fit_intercept=True` when `rank_ratio < 1`**, because the regression part needs the bias term `b` (paper Eq. 14).
- **Check:** median C-index near the paper's: **VLC ≈ 0.69, GBSG2 ≈ 0.67** (regression/hybrid).
  Ranking-only should come out lower (paper: ≈ 0.58–0.62). If our numbers match, the pipeline is trustworthy.
- ✅ **Done.** Regression/hybrid within +0.01–0.03 of the paper (the Cox reference shifts by the same amount).
  Ranking does *not* reproduce the paper's weakness, because scikit-survival uses all comparable pairs. Details in [results_log.md](results_log.md).
  Note: on VLC/GBSG2, RBF never beats linear. There is little nonlinear signal to find, which feeds R5.

## Step 2: Build the quantum kernels (approaches A1–A9)

- ⚠️ **Why 1 layer is not quantum.** Gates that don't depend on the data and come *after* the last
  encoding cancel out of the overlap. For `U(x) = C·R(x)`:
  `⟨ψ(x)|ψ(x')⟩ = ⟨0|R(x)† C†C R(x')|0⟩ = ⟨0|R(x)† R(x')|0⟩`, so the CNOTs vanish and the kernel reduces to
  `Π_k cos²(c·(x_k − x'_k)/2)`, a classical product-of-cosines kernel.
  **Rule:** entangling gates count only if more data encoding comes *after* them.
- **Circuits** (syllabus gates only):

  | Name | Circuit | Role |
  |---|---|---|
  | `Q-1L` (control) | `RY(c·x)` on every qubit, **no CNOTs** | labelled **"no entanglement" control** (= classical cos² kernel) |
  | `Q-2L` (main) | `RY(c·x)` → CNOT ring → `RY(c·x)` | **minimum genuinely quantum map** (data re-uploaded after entanglement) |
  | `Q-3L` | `RY(c·x)` → CNOT ring → `RY(c·x)` → CNOT ring → `RY(c·x)` | deeper variant |
  | `Q-ZZ` | 2 reps of `H` → `RZ(c·x_k)` → `ZZ(c·x_j·x_k/π)` on ring pairs | standard ZZ feature map (data-dependent entanglers) |

  - Never end a circuit with a CNOT ring, since it has no effect. Every circuit ends with an encoding layer.
  - **Key comparison:** `Q-2L` / `Q-3L` / `Q-ZZ` vs `Q-1L`. If the entangled maps don't beat the
    no-entanglement control, entanglement isn't helping, and we report that.
- **Unit tests** (catch this bug automatically):
  1. `Q-1L` kernel == `Π cos²(c·Δx/2)` computed analytically (to 1e-10).
  2. Appending a CNOT ring to the end of any circuit leaves `K` unchanged.
  3. `Q-2L` kernel ≠ product-of-cosines (the entanglement actually changes `K`).
- **Qubits = features** (VLC ≈ 8, GBSG2 ≈ 8 after encoding; TCGA reduced to 8).
- **Kernel:** compute each patient's statevector once → `K[i,j] = |⟨ψ_i|ψ_j⟩|²` for the whole matrix.
  Test-vs-train kernel `K[test, train]` is computed the same way.
- **Feature scaling for angles:** min-max rescale each feature to **[0, π]** (fit on the train part only, and clip
  test values into the range), *then* multiply by `c`. With `c ≤ 1`, angles stay in [0, π] and don't wrap around
  (RY is periodic, so wrapped angles would make distant patients look similar).
  - For `Q-ZZ`, the pair term is divided by π (`c·x_j·x_k/π`). Its maximum is then `c·π ≤ π`, the same range
    as the single-qubit angles. So every circuit uses the **same [0, π] scaling and the same `c` grid**, and the
    equal-tuning-budget rule holds without exceptions.
- **Bandwidth `c`** scales the input angles. It is the single most important knob. Grid of 7 values
  (e.g. 0.1–1.0, log-spaced), the same size as the RBF bandwidth grid.

### Approaches catalogue

Each approach tests **one idea** about where a quantum advantage could come from. Each is its own
table row and gets its own entry in [results_log.md](results_log.md).

| ID | Approach | Idea being tested | Priority |
|---|---|---|---|
| **A1** | `Q-1L` product kernel (control) | baseline: angle encoding *without* entanglement (= classical cos² kernel) | core |
| **A2** | `Q-2L` re-uploading + CNOT ring | does **entanglement** add useful feature interactions? | core |
| **A3** | `Q-3L` deeper circuit | does **more depth / expressivity** help, or does it hurt (concentration)? | core |
| **A4** | `Q-ZZ` feature map | do **data-dependent entanglers** (pairwise `x_j·x_k` terms, standard in literature) help? | core |
| **A4b** | `Q-Z` = `Q-ZZ` **without** the ZZ gates (product state, classical) | ablation: is a `Q-ZZ` gain due to entanglement or to its H+RZ encoding? | core (added after Step 4) |
| **A5** | **Bandwidth sweep** of `c` (recorded for all 7 values, not just the tuned one) | how kernel quality changes from "everyone similar" to "everyone different" | core |
| **A6** | **Projected quantum kernel** (Huang et al. 2021): measure ⟨X⟩,⟨Y⟩,⟨Z⟩ on each qubit, RBF on those | does a kernel that avoids exponential concentration do better than the fidelity kernel? | core |
| **A7** | **Hybrid kernel** `w·K_Q + (1−w)·K_linear` | does the quantum kernel carry information **complementary** to the linear one? (`w` grid reported separately, since it adds tuning) | core |
| **A8** | **Trained kernel**: one trainable rotation layer between encodings, optimized to maximize KTA on training data | can we *learn* a better quantum feature map instead of fixing it? | optional |
| **A9** | **Finite shots** (Qiskit sampler, 1024 shots) for the best circuit | what real-hardware sampling noise would cost | optional |

## Step 3: Diagnostics before training (Objective 2)

1. **Health check** (per kernel, train part only):
   - Off-diagonal mean & std. Values near 0 mean "everyone looks different" (K ≈ identity); values near 1 mean "everyone looks the same".
   - Effective rank of K. Flag the kernel as unhealthy if it is too close to either extreme.
2. **Survival KTA:**
   - Target similarity from survival ranks: `T[i,j] = 1 − |r_i − r_j| / (n−1)`.
   - Alignment computed **only over comparable pairs** (paper Eq. 16), using centered matrices:
     `KTA = Σ_comp K̃_ij T̃_ij / √(Σ_comp K̃_ij² · Σ_comp T̃_ij²)`.
   - Compared against (a) a tuned RBF kernel and (b) **shuffled labels**: permute (time, event) together 200×,
     which gives a null distribution and a p-value.
   - **Cross-check:** also compute KTA on **event-only patients** (all pairs comparable, with exact ranks). If both
     versions rank the kernels the same way, the censored lower-bound ranks aren't distorting the result.
3. **Does KTA work?** For every kernel configuration (maps × `c` × layers, plus RBF × bandwidths),
   record KTA (pre-training) and test C-index (post-training) → scatter plot + Spearman ρ.
   Configurations: {`Q-1L`, `Q-2L`, `Q-3L`, `Q-ZZ`} × `c` grid, plus RBF × bandwidth grid.
4. **Entanglement check:** report KTA of each entangled map minus KTA of `Q-1L`, to see whether entanglement improves alignment before any training.

## Step 4: Quantum Kernel Survival Model (Objective 1)

- **QKSM-SVM:** the Step-1 Survival SVM with the quantum `K` as a precomputed kernel, in all three `rank_ratio` forms.
  Run `Q-2L`, `Q-3L`, `Q-ZZ` (quantum) and `Q-1L` (control) **each as a separate model / table row**.
  Inside each row, inner CV tunes only `c` and `alpha`. The circuit is never picked automatically, so results stay comparable per circuit.
- **QKSM-KPCA-Cox:** kernel PCA on the training `K` (centered) → keep the top m components
  (m tuned, e.g. 3–10) → ridge-penalized Cox. Test patients are projected using `K[test, train]`.
- The same KPCA-Cox is also run with the RBF kernel, for a fair comparison.

## Step 5: Benchmark (Objective 3)

- **Baselines:** Cox-LASSO (`CoxnetSurvivalAnalysis`), Random Survival Forest, classical Survival SVM (linear/RBF/clinical).
- **Datasets:**
  1. VLC, GBSG2 (paper's)
  2. **TCGA-BRCA multi-omics:** per omic layer, keep top-variance genes, then PCA → about 2 components per layer,
     plus age and stage → **8 features = 8 qubits**. All selection and PCA are fitted **inside each training fold** (no leakage).
     **All kernel models (classical and quantum) get the same 8 features.**
     **Reality check:** Cox-LASSO on the *full* omic feature set (no reduction), to show what the 8-feature cut costs.
     Caveat: BRCA has few deaths (~14%), so C-index will be noisy. We report the number of events.
  3. **Synthetic:** 8 features. Hazard depends on a nonlinear interaction (e.g. `sin(x1·x2) + x3·x4`),
     exponential survival times, about 40% random censoring. This is a positive control for the claim
     **"nonlinear kernels beat linear"**. RBF will likely capture the pattern too, so this is not a claim that quantum wins.
- **Report:** median ± IQR C-index over 20 splits, plus paired Wilcoxon signed-rank test of QKSM vs each baseline
  **and vs the `Q-1L` no-entanglement control**.

## Step 5b: Why classical wins (evidence toolkit)

The teacher wants **reasons**. Every claim in the report must point to one of these measurements.

| ID | Measurement | What it shows | Expected if classical wins |
|---|---|---|---|
| **R1** | **Kernel concentration:** off-diagonal mean/variance of `K` vs `c`, depth, number of qubits | quantum kernels drift toward `K ≈ I` (all patients "different"), so there's nothing to learn from | variance shrinks fast with depth/qubits |
| **R2** | **Spectrum & effective dimension** of `K`; **train vs test C-index gap** | a flat eigenvalue spectrum = too many directions = fitting noise on small n (91–457 patients) | quantum: flatter spectrum, bigger train–test gap (overfitting) |
| **R3** | **Geometric difference** `g(K_classical ‖ K_Q)` (Huang et al. 2021, "Power of data in QML") | if `g` is small, a classical kernel can provably do at least as well on *this* data | small `g` vs RBF |
| **R4** | **Kernel–kernel alignment** `A(K_Q, K_RBF)`, `A(K_Q, K_linear)` | if the quantum kernel is nearly the same matrix as a classical one, it cannot behave differently | high alignment for shallow circuits |
| **R5** | **Nonlinearity in the data:** gain of RBF / RSF over linear Cox | if no classical nonlinear model beats linear, risk is ~monotone in the covariates; quantum encodings are *periodic* (cos²), a mismatched **inductive bias** | gain ≈ 0 on VLC/GBSG2 (already seen in Step 1) |
| **R6** | **Learning curves:** C-index vs training size (20 → 100% of train) | whether quantum kernels need more data than classical | quantum curve below / flatter |
| **R7** | **Synthetic dial:** vary the true risk from linear → interaction → periodic | maps *when* each kernel wins; locates conditions under which quantum could help | quantum competitive only for periodic structure |

Each approach's results_log entry ends with an **"Explanation"** line that cites R-numbers, e.g.
*"A3 lost to A2 because off-diagonal variance fell 10× (R1) and the train–test gap doubled (R2)."*

## Step 6: Report

- Tables in the paper's format (Tables 2–5 style): datasets × models → C-index.
- **Approaches table:** A1–A9 × (what we tried, result vs best classical, explanation with R-numbers).
- One page: **did KTA predict which kernel did better?** (scatter + Spearman).
- **Chapter "Why classical performs better"**: the R1–R7 evidence, figure by figure, ending in a short list of
  conditions under which a quantum kernel *could* help (from R7).
- Honest conclusion. Improvement → reported with its evidence. No improvement → the "impactful negative result", with the reasons.

---

## Repo layout

```
src/qksm/
  data.py         # VLC, GBSG2, TCGA-BRCA, synthetic loaders + fold-safe preprocessing
  kernels.py      # linear, RBF, clinical, quantum (Qiskit statevectors)
  diagnostics.py  # health check, survival KTA, permutation test, R1-R4 (concentration, spectrum, g, alignment)
  models.py       # Survival SVM wrapper (sign fix), KPCA-Cox, baselines
  evaluate.py     # repeated splits, inner CV, C-index, Wilcoxon
scripts/          # one runnable script per step/approach (step1_reproduce.py, ...)
notebooks/        # explanation + figures
results/          # CSV tables + figures
docs/             # research notes, this plan, results_log (every approach recorded), report
```

## Schedule (4 weeks)

| Week | Ansari | Tarun | Milestone |
|---|---|---|---|
| **1** | ✅ setup, data loaders, evaluation loop, Step 1 | clinical kernel, Survival SVM wrapper | ✅ **Step 1 numbers ≈ paper** |
| **2** | quantum kernels A1–A6 (Qiskit) + unit tests, health check | KTA + permutation test, R1/R4 diagnostics | A1–A6 results on VLC/GBSG2, logged |
| **3** | A7 hybrid, KPCA-Cox, R2/R3/R6 | TCGA-BRCA pipeline, synthetic dial (R7), baselines (Cox-LASSO, RSF) | all approaches on all datasets, logged |
| **4** | final 20-split runs, Wilcoxon, tables; A8/A9 if time | "Why classical wins" chapter (R1–R7 figures), KTA-vs-C-index | **report draft** |

Swap the names freely. The split is by module so you don't edit the same files.

## Risks

| Risk | Fix |
|---|---|
| Quantum K ≈ identity | lower `c`, fewer layers; health check catches it |
| "Quantum" kernel is secretly classical (trailing CNOTs cancel) | minimum 2 layers with re-encoding after the CNOTs; unit tests 1–3 in Step 2; `Q-1L` reported only as a labelled control |
| Our Step-1 numbers ≠ paper | fix before any quantum work; the paper used different tuning (coupled simulated annealing, 10-fold), so a small gap is OK |
| TCGA download/merging takes long | start in week 2, in parallel; fall back to mRNA + clinical only |
| Running time | statevectors once per split; 8 qubits = 256 amplitudes, which is trivial |
| Quantum kernel gets an unfair tuning advantage | equal-size grids for `c` and RBF bandwidth; same `alpha` grid; circuit fixed per row |
| Angle wrap-around makes distant patients look alike | rescale to [0, π] before `c`, clip test values |
| Explanations sound like opinion | every "why" must cite a measurement R1–R7; no R-number, no claim |
| Too many approaches for 4 weeks | A1–A7 core; A8/A9 only if week 4 has room |
