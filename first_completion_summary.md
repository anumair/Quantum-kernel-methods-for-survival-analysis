# First Completion Summary: Quantum Kernel Methods for Survival Analysis

| | |
|---|---|
| **Project title** | Quantum Kernel Methods for Survival Analysis |
| **Team** | Ansari, Tarun |
| **Repository** | https://github.com/anumair/Quantum-kernel-methods-for-survival-analysis |
| **Anchor paper** | V. Van Belle, K. Pelckmans, S. Van Huffel, J. A. K. Suykens, *"Support vector methods for survival analysis: a comparison between ranking and regression approaches"*, Artificial Intelligence in Medicine 53(2), 2011 |
| **Work period covered** | Project start → first completion (2026-10-07) |
| **Status** | All three objectives run end-to-end with results, figures and evidence-based reasoning. Remaining: the formal report, optional extensions (methylation, A8, A9). |

This document records **everything done in the project from scratch**: the brief, the literature, every planning decision
(including the mistakes we caught), the environment, the code, every experiment and its numbers, the engineering
problems and how they were fixed, and the final reasoning for why classical kernels win.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [The brief: problem statement, objectives, teacher's guidance](#2-the-brief)
3. [Chronological log of the work](#3-chronological-log-of-the-work)
4. [Literature review (main paper + references 2–5)](#4-literature-review)
5. [Planning: how the plan was built and corrected](#5-planning-how-the-plan-was-built-and-corrected)
6. [Environment and repository](#6-environment-and-repository)
7. [Methods in full detail](#7-methods-in-full-detail)
8. [Engineering problems found and fixed](#8-engineering-problems-found-and-fixed)
9. [Results, step by step](#9-results-step-by-step)
10. [Catalogue of approaches tried (A1–A9)](#10-catalogue-of-approaches-tried-a1a9)
11. [Why classical performs better: the reasoning](#11-why-classical-performs-better-the-reasoning)
12. [Objectives scorecard](#12-objectives-scorecard)
13. [Limitations and honesty notes](#13-limitations-and-honesty-notes)
14. [What remains](#14-what-remains)
15. [How to reproduce everything](#15-how-to-reproduce-everything)
16. [File index](#16-file-index)
17. [Glossary](#17-glossary)
18. [References](#18-references)

---

## 1. Executive summary

**What we built.** A hybrid framework, the *Quantum Kernel Survival Model (QKSM)*, that keeps the survival models of
Van Belle et al. (2011) exactly as they are and swaps only the **kernel**: from linear / RBF / clinical to **quantum
kernels** computed by simulating small quantum circuits in Qiskit. The quantum kernel is plugged into the Survival SVM (in
its ranking, hybrid and regression forms) and into a Kernel Cox model (KPCA-Cox). We also built a pre-training diagnostic,
**survival Kernel-Target Alignment (KTA)**, and benchmarked against Cox-LASSO, Random Survival Forest and classical
Survival SVMs on 2 clinical datasets from the paper, a TCGA-BRCA multi-omics cohort and 4 synthetic datasets.

**What we found.**
- **No genuine quantum advantage.** On real data (VLC, GBSG2, TCGA-BRCA) the quantum kernels tie or lose to the best classical kernel.
- **Entanglement never helped.** In every dataset, adding entangling gates left C-index unchanged or made it significantly worse.
- **The one "win"** (high-frequency periodic synthetic data, +0.033 C-index over every classical model) came from the
  **unentangled** product kernel. That is a classical periodic kernel, not a quantum effect.
- **Objective 2 works:** survival KTA, computed before any training, predicts which kernel configuration will do better
  (Spearman ρ = 0.82 VLC, 0.73 GBSG2, 0.58 TCGA-BRCA). It also flagged quantum kernels as weaker before training. One limitation was found: as a *tuner* of the angle scale on non-monotone data it picks the wrong value.

**Why classical wins (each point measured, not assumed):**
1. The clinical data has almost no nonlinear signal.
2. In the regime where quantum kernels work best, they *are* RBF kernels (99.3–100% alignment).
3. Making the circuit "more quantum" concentrates the kernel and causes overfitting.
4. Entanglement adds variance, not signal.
5. Angle encodings make similarity periodic, which is the wrong shape for monotone clinical risk.
6. Small cohorts favour simple kernels.
7. A small geometric difference shows that classical kernels can match these quantum kernels on this data.

This is the "impactful negative result" the problem statement allows for, explained by evidence, as the teacher asked.

---

## 2. The brief

### 2.1 Problem statement (from the teacher)

> Clinical survival analysis requires handling right-censored data (patients still alive at the end of the study or lost
> to follow-up) and complex, non-linear multi-omics interactions. In clinical oncology, it needs to predict time-to-event
> outcomes like overall survival rather than binary outcomes. Classical kernel survival models (e.g., Kernel Cox, Survival
> SVM [1, 2]) implicitly map features into a high-dimensional reproducing kernel Hilbert space, where non-linear survival
> patterns become separable. However, QML has almost ignored survival analysis. Existing literature is dominated by binary
> classification tasks (tumor vs normal) [3], and some have deployed Variational quantum circuits for survival analysis
> [4, 5]. No framework uses quantum in Survival SVM.

### 2.2 Objectives

1. **QKSM:** formulate a hybrid framework that replaces classical kernels with quantum kernels in established survival models (Survival SVM, Kernel Cox).
2. **Pre-training diagnostic:** develop a survival KTA metric to test whether the quantum feature map is aligned with the censored survival target *before any optimization*.
3. **Benchmarking:** evaluate against classical baselines (Cox-LASSO, Random Survival Forests, Survival SVM) on multi-omics datasets, using Harrell's C-index.

### 2.3 Possible outcomes (from the brief)

- An improved C-index in multi-omics oncology by capturing non-linear feature interactions, **or**
- If QKSM underperforms the classical baselines, an **impactful negative result**.

### 2.4 Teacher's later guidance (received mid-project)

> A better quantum kernel is not expected. Apply various approaches. An improvement is welcome, but if classical performs
> better (the likely case), give proper reasons and reasoning for why.

This changed the project from "tune one quantum model" to "**try many approaches, record each, and explain the outcome with evidence**" (see §5.6).

### 2.5 Course context

The course syllabus covers: introduction to quantum computing; the mathematics (Hilbert spaces, Dirac notation, tensor
products, density operators, measurement); qubits, the Bloch sphere and gates (NOT, Hadamard, T, CNOT, Toffoli, Z); quantum
information (Bell states, teleportation, no-cloning, QKD, error correction); QFT, phase estimation, quantum walks;
Deutsch-Jozsa, Grover, Simon, Shor; and quantum programming libraries. Quantum machine learning is *not* in the syllabus, so
the quantum part was kept within these concepts:

| Project component | Syllabus concept |
|---|---|
| Encoding a patient into a quantum state | Module 3: qubits, rotation gates (RY, RZ), Hadamard, circuit design |
| Linking features | Module 3: CNOT, entanglement |
| Quantum kernel `|⟨ψ(x)|ψ(x')⟩|²` | Module 2: inner products, measurement probabilities |
| Projected kernel (⟨X⟩, ⟨Y⟩, ⟨Z⟩ of each qubit) | Module 2: density operators, measurement in bases |
| Simulation in Qiskit | Module 7: development libraries |

---

## 3. Chronological log of the work

| # | Phase | What happened | Commit |
|---|---|---|---|
| 1 | Kick-off | Empty `Quantum/` folder; user created the empty GitHub repo. Shared the course syllabus and said to keep the quantum depth manageable. | – |
| 2 | Literature | Read the main paper (12 pages, full text) and references [2]–[5] (full texts for [2], [5]; abstract + details for [3], [4]; [3] via the university browser). Notes → `docs/research_notes.md`. | – |
| 3 | First plan | Wrote a plan anchored on the main paper (re-implement its models, change only the kernel). | – |
| 4 | Plan merge | The user brought a second plan draft. Merged the two; flagged that `scikit-survival`'s Survival SVM is close to but not identical to the paper's, that Kernel PCA + Cox is an approximation, and that KTA already exists in ref [5]. User chose: scikit-survival, KPCA-Cox, under-1-month timeline, TCGA-BRCA. | – |
| 5 | Red flag #1 | User caught that a 1-layer "RY + CNOT ring" circuit is not quantum: trailing CNOTs cancel in the overlap. Verified mathematically; made 2 encoding layers the minimum and kept 1-layer only as a labelled control. | – |
| 6 | Red flag #2 + 6 additions | Conflict: circuits must not be auto-selected if we report per circuit. Added equal tuning grids, `fit_intercept`, [0, π] scaling, censored-rank note, same TCGA features for all models, paired Wilcoxon, reworded the synthetic claim. Decided the `Q-ZZ` wrap-around fix (divide the pair term by π). Plan marked FINAL. | – |
| 7 | Setup | Git (local identity `anumair <ansariumairnanded@gmail.com>`), Python 3.12 via `uv`, Qiskit 2.5.2, scikit-survival 0.28, lifelines 0.30, scikit-learn 1.9. Smoke-tested datasets (VLC 137/128 events, GBSG2 686/299 match the paper). | `da27a12` |
| 8 | Step 1 | Classical reproduction. Fixed: slow kernel SVM (→ exact empirical kernel map, 20–60× faster), α grid too narrow, CPU oversubscription. Matched the paper within +0.01–0.03. | `afe8147` |
| 9 | Teacher guidance | Plan revised: approaches catalogue A1–A9, evidence toolkit R1–R7, results-log template. | `28175bf` |
| 10 | Step 2/4 | Qiskit circuits, unit tests for the trailing-CNOT bug, diagnostics module, baselines, synthetic data. First 20-split quantum benchmark on VLC/GBSG2. | `456e959` |
| 11 | Step 3 | Diagnostics on 44 kernel configs × 5 splits: KTA predicts C-index; concentration, overfitting, geometric difference; small-c check proving best quantum kernels ≈ RBF. | `1389369` |
| 12 | Ablations | A4b (`Q-Z`) proved the `Q-ZZ` gain is encoding, not entanglement. Fixed the flawed "periodic" synthetic set. A5b wrap-around kernels, RBF-narrow fairness check, A7 hybrid kernel, KPCA-Cox, learning curves. Found the KTA-as-tuner limitation. | `bd6474b` |
| 13 | TCGA-BRCA | Downloaded mRNA, CNV, clinical, survival (~70 MB) from UCSC Xena. 1063 patients / 148 deaths. Fold-safe PCA to 8 features. Fixed a slow / unstable high-dimensional Cox-LASSO. | `674972b` |
| 14 | Figures | Six report figures (F1–F6), each checked visually and fixed for overlaps. | `055e1bd` |
| 15 | This document | Full record of the project to first completion. | – |

---

## 4. Literature review

### 4.1 [1] Van Belle et al. 2011: the main paper

**Setting.** Data `{x_i, y_i, δ_i}`: covariates, observed time, event indicator (`δ = 1` event, `δ = 0` right-censored).
Models output a **prognostic index** `u(x) = wᵀφ(x)`, a risk score rather than a time.

**Comparable pairs (Eq. 16).** The order of two patients is known only if both had events, or the earlier one had an
event and the later one was censored after it. Two censored patients can never be ordered. This rule underlies both the C-index and our KTA.

**Models compared.**

| Model | Idea | Constraints | Notes |
|---|---|---|---|
| RANKSVMC (Eq. 17) | penalize misordered comparable pairs | ranking | O(n²) constraints; a simplified version compares only with the nearest comparable neighbour (O(n)) |
| Model 1 (Eq. 19) | ranking with margin `y_i − y_j̄(i)` | ranking | |
| SVCR (Eq. 14, Shivaswamy) | SVR where censored patients are penalized only if predicted too early | regression | 1 hyperparameter; best on high-dimensional data |
| SVRC (Khan & Zubek) | 4 different penalties | regression | too many hyperparameters |
| **Model 2 (Eq. 22, new)** | ranking + regression constraints | both | reduces to Model 1 or SVCR as special cases |

**Key property for us:** every model uses the data only through `k(x_i, x_j)` (Eqs. 10, 15, 18, 23). **Swap the kernel and the model is unchanged.**

**Kernels in the paper:** linear, RBF, polynomial, and the **clinical kernel** (Eq. 12–13): per-feature
`(range − |x_p − z_p|)/range` for continuous/ordinal features, `1/0` match for nominal ones, averaged.

**Metrics (§4.2):** C-index (Eq. 24), logrank χ² (median split of the prognostic index), hazard ratio (univariate Cox on the index rescaled to [0, 1]).

**Protocol (§4.3):** 50 random splits (2/3 train, 1/3 test); hyperparameters tuned by 10-fold CV with coupled simulated
annealing on the C-index; Wilcoxon rank-sum tests. Datasets: VLC (137), leukemia (129), prostate (483), Mayo lung (167),
GBSG (686); high-dimensional: DBCD, NSBCD, DLBCL.

**Findings:** ranking-only models were significantly worse and unstable; models with regression constraints were best;
on high-dimensional data SVCR won; linear Cox ≈ SVMs on clinical data. Typical C-index 0.60–0.78 (GBSG ≈ 0.67–0.68, VLC ≈ 0.69).

### 4.2 [2] Li & Luan 2003: Kernel Cox regression

- Model `λ(t|x) = λ₀(t)·exp(f(x))`, with `f` in an RKHS; loss = negative Cox partial likelihood + `λ‖f‖²`.
- Representer theorem: `f(x) = Σ a_i K(x, x_i)`, so the problem becomes finite-dimensional in `a` (needs only `K`).
- Newton–Raphson (Nelder–Mead fallback); λ by leave-one-out CV.
- **Only the linear kernel was used.** Tiny datasets (DLBCL n = 40, lung n = 22, breast n = 49) with univariate-Cox gene pre-selection.
  Nonlinear kernels were described as easy but never tried, which leaves room for a quantum kernel.

### 4.3 [3] Maheshwari et al. 2022: QML in biomedicine (systematic review, IEEE Access)

30 papers (2013–2021) using QSVM, VQC, QNN, QCNN, annealing and quantum-inspired methods. The tasks are almost all
**classification / diagnosis** (COVID X-ray, diabetes, cancer vs normal, EEG); **no survival analysis**. Classical models
often match or slightly beat quantum ones. PCA / feature selection is used to fit the few available qubits. This supports the brief's claim that QML has ignored survival analysis.

### 4.4 [4] Novák et al. 2025: QNNs for propensity scores (arXiv:2506.19973)

1,177 colorectal cancer patients; 4 covariates on 4 qubits; Qiskit + sQUlearn; ZFeatureMap; CMA-ES optimizer. The QNN
estimates **propensity scores only**; survival itself (Kaplan–Meier, Cox, Aalen) is classical. No quantum kernels.

### 4.5 [5] Colak et al. 2026: QResid-Boost + KTA-Survival (J. Clin. Med.): the closest prior work

- Cox-LASSO plus a variational quantum circuit (6 qubits, 5 layers, PennyLane) trained on martingale residuals; final
  risk `η_Cox + sigmoid(α)·VQC(x)`.
- **Already proposes "KTA-Survival".** Their "quantum kernel" there is an RBF on Pauli-Z expectation values (a projected
  kernel), *not* a fidelity kernel. Their target kernel marks comparable pairs; they report ΔKTA vs a tuned RBF, with no go/no-go threshold.
- Datasets: GBSG2, FLChain, WHAS500, synthetic. The quantum gate α stays small and the model collapses to Cox-LASSO; it only helps on synthetic periodic data.
- **They do not use quantum kernels inside a Survival SVM or Kernel Cox.** So our objective 1 is new, and our KTA must be presented as an *extension* of theirs (cited).

### 4.6 What the literature implied for our design

- The gap claimed in the brief is real: no survival SVM with a quantum kernel exists.
- KTA must be credited to [5]. Our additions: a true fidelity kernel, a target built from the paper's comparable-pairs rule,
  a shuffled-label test, an event-only cross-check, and a test of whether KTA predicts the C-index.
- [5] already found that quantum barely helps on low-dimensional clinical data, so a negative result was likely.

Full notes: [docs/research_notes.md](docs/research_notes.md).

---

## 5. Planning: how the plan was built and corrected

### 5.1 First plan

Re-implement the paper's models, add a quantum kernel, KTA, benchmark, report. Proposed the own-QP implementation in `cvxpy`.

### 5.2 Merging the user's plan draft

The user's draft proposed `scikit-survival` with `rank_ratio` (1 = ranking, 0 = regression, between = hybrid), Qiskit,
KPCA + Cox for Kernel Cox, KTA with shuffled labels, fold-internal TCGA reduction and a synthetic positive control. I kept
all of that and flagged three corrections:
1. `scikit-survival`'s Survival SVM (Pölsterl et al.) is *close to* but not identical to Van Belle's: squared hinge loss,
   all comparable pairs. With `rank_ratio < 1` it predicts **time**, so the sign must be flipped before computing the C-index.
2. Kernel PCA + Cox is an *approximation* of Kernel Cox (we name it "KPCA-Cox").
3. KTA was already proposed in [5] and must be cited.

**Decisions taken by the user:** scikit-survival engine; KPCA-Cox; timeline under 1 month (so 20 splits instead of 50); TCGA-BRCA as the multi-omics cohort.

### 5.3 Red flag #1: the 1-layer circuit is not quantum (caught by the user)

For `U(x) = C·R(x)`, with `C` a data-independent CNOT ring:
`⟨ψ(x)|ψ(x')⟩ = ⟨0|R(x)† C†C R(x')|0⟩ = ⟨0|R(x)† R(x')|0⟩`, because `C†C = I`. The kernel reduces to
`Π_k cos²(c·(x_k − x'_k)/2)`, a classical product of cosines.

**Rule adopted:** entangling gates count only if more data encoding comes *after* them. Minimum quantum circuit = encode →
CNOT ring → encode. The 1-layer version is kept as the labelled **"no entanglement" control** `Q-1L`. Three unit tests
enforce this (§7.5.4). Saved as a permanent memory note so it is never repeated.

### 5.4 Red flag #2 and six additions (from the user's review)

| Issue | Fix in the plan |
|---|---|
| Auto-selecting the circuit in CV conflicts with reporting per circuit | each circuit is its own table row; only `c` and α are tuned inside a row |
| Unequal tuning chances | same 7-value α grid for every model; quantum `c` grid has 7 values like the RBF bandwidth grid |
| `fit_intercept` | `True` when `rank_ratio < 1` (the regression part needs the bias `b`) |
| Angle wrap-around | min-max rescale each feature to [0, π] on training data, clip test values, then multiply by `c ≤ 1` |
| Censored ranks | noted as lower bounds; KTA also reported on event-only patients |
| TCGA fairness | all kernel models get the same 8 features; Cox-LASSO on all genes as a reality check |
| Statistics | paired Wilcoxon signed-rank (each model scored on the same splits); note that splits aren't independent |
| Synthetic claim | reworded to "nonlinear kernels beat linear" (RBF should handle it too) |

### 5.5 `Q-ZZ` wrap-around decision (taken by me on request)

The ZZ pair term `c²·x_j·x_k` reaches `c²π² ≈ 9.9` with features in [0, π]. Options were to rescale `Q-ZZ` features to [0, 1] or
divide the pair term by π. **Chosen:** `ZZ(c·x_j·x_k/π)`. Its maximum is `c·π ≤ π`, so every circuit keeps the same [0, π]
scaling and the same `c` grid, and the equal-budget rule holds without exceptions. ([0, 1] would have squeezed the single-qubit angles below 1 rad.)

### 5.6 Revision for the teacher's guidance

- **Approaches catalogue A1–A9:** each tests one idea about where a quantum advantage could come from (§10).
- **Evidence toolkit R1–R7:** every "why" must cite a measurement:
  R1 concentration, R2 spectrum & train–test gap, R3 geometric difference, R4 kernel alignment, R5 nonlinearity in the data, R6 learning curves, R7 synthetic dial.
- **Results-log template** for every approach: *idea → why it might help → setup → result → observed → explanation*.

Final plan: [docs/PLAN.md](docs/PLAN.md).

---

## 6. Environment and repository

### 6.1 Tools

| Tool | Version | Use |
|---|---|---|
| macOS (Apple Silicon, 10 cores) | Darwin 25.2 | development machine |
| Python | 3.12.15 (installed by `uv`; the system 3.9 was too old) | everything |
| uv | – | environment + dependency lock (`uv.lock`) |
| Qiskit | 2.5.2 | quantum circuits, exact statevector simulation |
| scikit-survival | 0.28.0 | Survival SVM, Cox-LASSO (Coxnet), Random Survival Forest, C-index, clinical kernel |
| lifelines | 0.30.3 | logrank test, univariate Cox for the hazard ratio |
| scikit-learn | 1.9.1 | splits, CV, scaling, PCA |
| NumPy / SciPy / pandas / matplotlib / joblib | – | maths, stats, tables, figures, parallelism |
| pytest | – | 30 unit tests |

Git identity is set for this repo only: `anumair <ansariumairnanded@gmail.com>`. Tarun is to be added as a collaborator by the repo owner.

### 6.2 Repository layout

```
Quantum/
├── first_completion_summary.md   ← this document
├── README.md                     setup + layout
├── pyproject.toml, uv.lock, .python-version
├── docs/
│   ├── PLAN.md                   final plan (approaches A1–A9, evidence R1–R7, schedule)
│   ├── research_notes.md         literature notes for refs [1]–[5]
│   └── results_log.md            every approach recorded with the template
├── src/qksm/                     library (≈ 2000 lines incl. tests and scripts)
│   ├── data.py                   VLC, GBSG2 loaders; fold-safe Standardizer
│   ├── kernels.py                linear, RBF (+narrow), clinical, quantum, projected, hybrid kernels
│   ├── quantum.py                Qiskit circuits, fidelity kernel, Pauli expectations, AngleScaler
│   ├── models.py                 Survival SVM wrapper (sign fix, intercept, empirical kernel map)
│   ├── kpca_cox.py               KPCA-Cox
│   ├── baselines.py              Cox-LASSO, RSF, high-dimensional Cox-LASSO
│   ├── diagnostics.py            health check, survival KTA, permutation test, alignment, geometric difference
│   ├── evaluate.py               C-index, logrank, hazard ratio, repeated splits + inner CV, Wilcoxon
│   ├── synthetic.py              synthetic dial (linear / interaction / smooth-periodic / periodic)
│   └── tcga.py                   TCGA-BRCA loader + fold-safe OmicsReducer
├── tests/                        test_models.py, test_quantum.py, test_diagnostics.py (30 tests)
├── scripts/                      step1_reproduce, step3_diagnostics, step4_qksm, step5_benchmark,
│                                 step6_extra, step7_tcga, make_figures
├── results/                      all CSV/Markdown result tables + figures/
└── data/raw/                     downloaded TCGA files (not committed)
```

---

## 7. Methods in full detail

### 7.1 Datasets

| Dataset | Source | Patients | Events | Censored | Features (encoded) | Role |
|---|---|---|---|---|---|---|
| **VLC**: Veterans' lung cancer | `sksurv.datasets` (paper's dataset) | 137 | 128 | 7% | 8: age, Karnofsky, months from diagnosis, cell type (3 dummies), prior therapy, treatment | paper reproduction + quantum test |
| **GBSG2**: German breast cancer | `sksurv.datasets` (paper's dataset) | 686 | 299 | 56% | 8: age, ER, PR, nodes, tumour size, grade (ordinal), hormonal therapy, menopausal status | paper reproduction + quantum test |
| **TCGA-BRCA**: multi-omics breast cancer | UCSC Xena TCGA hub | 1063 | 148 | 86% | 8 per split: 3 mRNA PCs, 3 CNV PCs, age, stage | the brief's multi-omics oncology setting |
| synth-linear | `synthetic.py` | 500 | 300 | 40% | 8 uniform in [−π/2, π/2] | positive control: linear risk |
| synth-interaction | `synthetic.py` | 500 | 300 | 40% | 8 | risk `x₁x₂ + x₃x₄` |
| synth-smooth-periodic | `synthetic.py` | 500 | 300 | 40% | 8 | first periodic attempt (turned out mostly monotone) |
| synth-periodic | `synthetic.py` | 500 | 300 | 40% | 8 | risk `sin(4x₁)cos(4x₂) + sin(4x₃)`, 2 full oscillations |

The VLC and GBSG2 sizes and event counts match the paper exactly.

**Synthetic generator.** `X ~ Uniform(−π/2, π/2)⁸`; risk `η` standardized and scaled to the same strength (×1.5) for every kind; times from a Cox model with exponential baseline `T = −log U / (0.1·e^η)`; independent uniform censoring with its range found by bisection to give 40% censored.

**TCGA-BRCA construction.**
- Files: `HiSeqV2.gz` (mRNA log2(RSEM+1), 62 MB), `Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz` (3 MB),
  `BRCA_clinicalMatrix` (2 MB), `BRCA_survival.txt` (curated TCGA-CDR overall survival, 68 KB).
- Cohort: primary tumours (`-01`) present in all files with OS available and `OS.time > 0` → **1063 patients, 148 deaths, median follow-up 860 days**.
- Stage mapped I→1, II→2, III→3, IV→4 (Stage X / missing / discrepancy → NaN, 21 patients, imputed per split).
- Only label-free step on the whole cohort: keep the top 2000 variance genes per omic (to keep the data small).
- **Inside each training split** (`OmicsReducer`): top 1000 variance genes per omic → z-score → PCA → 3 components per omic, plus age and stage, with training-median imputation. **8 features = 8 qubits**, identical for every kernel model.
- Methylation (450K array, 783 MB) was **not** included (see §13).

### 7.2 Preprocessing

- **Classical kernels / baselines:** nominal → one-hot (drop first), ordinal → integer codes, then z-scoring fitted on the training part only.
- **Quantum kernels:** same encoding, then **min-max to [0, π]** fitted on training data (test values clipped), then multiplied by the angle scale `c`.
- **Clinical kernel:** applied to the raw features (it handles nominal / ordinal / continuous variables itself).

### 7.3 Survival SVM (the paper's models)

- Engine: scikit-survival `FastSurvivalSVM` with `rank_ratio` ∈ {**1** = ranking, **0.5** = hybrid (≈ Model 2), **0** = regression (≈ SVCR)}.
- **Sign fix:** with `rank_ratio < 1` the model predicts (log-)survival *time*, so we negate it to get a risk score. Unit-tested: every form gives C > 0.55 on VLC.
- **Intercept:** `fit_intercept=True` when `rank_ratio < 1` (scikit-survival refuses it for pure ranking).
- **Empirical kernel map (speed fix, exact).** With `K_tr = VΛVᵀ`, the features `Φ = VΛ^{1/2}` satisfy `ΦΦᵀ = K_tr`,
  and test features are `Φ_te = K_te·VΛ^{-1/2}`. Training the fast *linear* SVM on `Φ` gives the same solution as the kernel
  SVM (representer theorem). Verified: correlation 1.000000 with `FastKernelSurvivalSVM(kernel="precomputed")`, max prediction
  difference ~1e-6, and 20–60× faster. Eigenvalues below 1e-10 × max are dropped.
- **Regularization grid:** `α ∈ {2⁻¹⁴, 2⁻¹¹, 2⁻⁸, 2⁻⁵, 2⁻², 2¹, 2⁴}` (7 values), shared by every kernel. (In scikit-survival, α weights the loss, so smaller α means stronger regularization.)

### 7.4 Classical kernels

| Kernel | Formula | Tuned |
|---|---|---|
| linear | `x·z` (z-scored) | – |
| RBF | `exp(−γ‖x−z‖²)`, `γ = m / median‖xᵢ−xⱼ‖²`, `m ∈ 2^{−3…3}` | m (7 values) |
| RBF-narrow (fairness check) | same, `m ∈ 2^{1…7}` | m (7 values) |
| clinical (paper Eq. 12–13) | per-feature `(range−|Δ|)/range` or match, averaged (scikit-survival `ClinicalKernelTransform`) | – |

### 7.5 Quantum kernels

#### 7.5.1 Encoding and kernels

Each patient's d features become d qubit rotation angles `θ = c·x̃`, with `x̃ ∈ [0, π]`. The circuit `U(θ)` prepares
`|ψ(x)⟩ = U(θ)|0…0⟩`, simulated exactly with Qiskit `Statevector` (8 qubits = 256 amplitudes). All statevectors are computed once, and the whole kernel matrix is one matrix product.

- **Fidelity kernel:** `K[i,j] = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²` (an inner product followed by a measurement probability).
- **Projected kernel** (Huang et al. 2021): measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on each qubit (from the single-qubit reduced density
  matrices), giving 3d numbers per patient, then an RBF on those with γ fixed by the median heuristic (so only `c` is tuned).

#### 7.5.2 Circuits

| Name | Circuit | Entangled? | Role |
|---|---|---|---|
| `Q-1L` | RY(θ) on every qubit | no | **A1** "no entanglement" control (= classical `Π cos²`) |
| `Q-2L` | RY(θ) → CNOT ring → RY(θ) | yes | **A2** minimum genuinely quantum map |
| `Q-3L` | RY → ring → RY → ring → RY | yes | **A3** deeper |
| `Q-ZZ` | 2 × [H → RZ(θ_k) → RZZ(θ_j θ_k / π) on ring pairs] | yes | **A4** standard ZZ feature map |
| `Q-Z` | `Q-ZZ` without the RZZ gates | no | **A4b** ablation (product state, classical) |
| `P-Q-2L` | projected kernel on `Q-2L` | yes | **A6** |
| `*-wrap` | `Q-1L`, `Q-2L`, `Q-ZZ` with `c ∈ [1, 8]` | – | **A5b** angles wrap around (periodic regime) |
| `H-Q-2L`, `H-Q-ZZ` | `w·K_Q + (1−w)·K_linear` | – | **A7** hybrid kernel |

No circuit ends with a CNOT ring (it would have no effect).

#### 7.5.3 Angle scale grid

`c ∈ geomspace(0.1, 1, 7) = {0.100, 0.147, 0.215, 0.316, 0.464, 0.681, 1.000}`, the same size as the RBF bandwidth grid.
A5b uses `c ∈ geomspace(1, 8, 7)` in separately labelled rows.

#### 7.5.4 Unit tests guarding the trailing-CNOT bug (`tests/test_quantum.py`)

1. `Q-1L` kernel equals the analytic `Π cos²((a−b)/2)` to 1e-10.
2. Appending a CNOT ring to the end of **any** circuit leaves the kernel unchanged (to 1e-10).
3. `"RY then CNOT ring"` is still exactly the product-of-cosines kernel (the original bug).
4. `Q-2L`, `Q-3L`, `Q-ZZ` differ from the product kernel (max difference > 0.01).
5. Every fidelity kernel has unit diagonal, is symmetric, positive semi-definite, and lies in [0, 1].
6. `pauli_expectations` gives ⟨X⟩ = sin θ, ⟨Y⟩ = 0, ⟨Z⟩ = cos θ for RY product states.
7. Kernel interface shapes on VLC. (`Q-Z` was also verified separately to factorize over qubits.)

### 7.6 Hybrid kernel (A7)

`K = w·K_Q + (1−w)·K_linear`, both scaled to unit mean diagonal; `w ∈ {0, 1/6, …, 1}` (7 values; w = 0 is pure linear,
w = 1 pure quantum). The quantum angle scale `c` is chosen **before training** as the `c` with the highest survival KTA on the
training part, which uses Objective 2 in practice. Labels for that are looked up by training-row index only.

### 7.7 KPCA-Cox (Kernel Cox version, A-Objective 1)

Centered kernel PCA on the training kernel → top m components (`m ∈ {2, 3, 5, 8, 12, 20, 30}`, 7 values) → standardized →
ridge-penalized Cox (`CoxPHSurvivalAnalysis(alpha=0.1)`). Test patients are projected with the centered `K[test, train]`.
Same splits and inner CV as the SVM; the grid is kernel param × m.

### 7.8 Baselines

| Baseline | Implementation | Grid (7 values) |
|---|---|---|
| Cox-LASSO | `CoxnetSurvivalAnalysis(l1_ratio=1)` on z-scored features | α ∈ geomspace(1e-3, 0.5) |
| Random Survival Forest | 300 trees, `max_features="sqrt"` | min_samples_leaf ∈ {3, 5, 10, 15, 20, 30, 50} |
| Cox-LASSO, all genes (TCGA) | same, on ~4000 genes + clinical | α ∈ geomspace(0.01, 0.5) (smaller α was slow and diverged) |
| Cox-LASSO, clinical only (TCGA) | age + stage | as Cox-LASSO |
| Linear Cox (`phlinear` in the paper, Step 1) | `CoxPHSurvivalAnalysis(alpha=1e-4)` | – |

### 7.9 Survival KTA and the diagnostics (Objective 2, R1–R4)

- **Comparable mask** `M[i,j]` (Eq. 16, symmetrized): the earlier time is an observed event.
- **Target kernel:** `T[i,j] = 1 − |rᵢ − rⱼ|/(n−1)`, with r = rank of the survival time (patients with similar survival are "similar").
- **Survival KTA:** centered alignment over comparable pairs only:
  `KTA = Σ_M K̃ᵢⱼ T̃ᵢⱼ / √(Σ_M K̃ᵢⱼ² · Σ_M T̃ᵢⱼ²)`, with `K̃ = HKH`, `H = I − 11ᵀ/n`. Computed on the **training data only, without fitting any model**.
- **Event-only cross-check:** KTA on patients with an observed event (exact ranks, all pairs comparable).
- **Permutation test:** shuffle (time, event) pairs together 200×; p = fraction of shuffles with KTA ≥ observed.
- **R1 health / concentration:** off-diagonal mean and std of K; effective rank `(Σλ)²/Σλ²`.
- **R2 overfitting:** train C-index − test C-index; eigenvalue spectrum.
- **R3 geometric difference** (Huang et al. 2021): `g(K_C‖K_Q) = √‖√K_Q (K_C + 10⁻³I)⁻¹ √K_Q‖_∞`, both traces normalized to n. `g ≈ 1` means a classical kernel can learn anything the quantum one can; a quantum advantage is only *possible* if g approaches √n.
- **R4 kernel alignment:** centered alignment between two kernels (1 = same geometry).

### 7.10 Evaluation protocol and statistics

- **20 repeated random splits**, 2/3 train / 1/3 test, stratified on the event indicator (the paper used 50; reduced for time).
- **Inner 5-fold CV on the training part** chooses the hyperparameters by C-index; refit on the full training part; score on the test part.
  The kernel is computed once on the outer training part and inner folds reuse slices of it (scaling sees the whole training part, never the test part).
- **Metrics:** Harrell's C-index (main), logrank χ² (median split), hazard ratio (univariate Cox on the [0, 1]-rescaled risk).
- **Reporting:** median ± IQR over splits, as in the paper.
- **Tests:** paired **Wilcoxon signed-rank** over the 20 splits. Splits share training data, so p-values are indicative only.
- **Parallelism:** joblib/loky over all (model, split) jobs on 10 cores, one BLAS thread per worker.

---

## 8. Engineering problems found and fixed

| # | Problem | Symptom | Fix | Evidence it worked |
|---|---|---|---|---|
| 1 | scikit-survival's kernel SVM was slow | 1–10 s per fit on GBSG2, hitting iteration limits; hours per kernel | exact empirical kernel map + linear SVM (§7.3) | identical predictions (corr 1.000000), 20–60× faster; unit test |
| 2 | α grid too narrow | every kernel picked the grid edge | scanned 2⁻¹⁸…2⁴; widened shared grid to 2⁻¹⁴…2⁴ | optima now inside the grid |
| 3 | CPU oversubscription | first full run killed at the 10-min limit with no output | `parallel_config(inner_max_num_threads=1)`; progress logging; longer limits | full Step 1 in ~11 min |
| 4 | output buffering (`grep` in a pipe) | empty logs while running | logs written to files; read at the end | – |
| 5 | censoring bisection reversed | synthetic data 100% censored | flipped the bisection direction | 40% censored on all sets |
| 6 | first "periodic" synthetic set mostly monotone | linear models still scored 0.744 | kept as `smooth-periodic`; new high-frequency `periodic` | linear 0.552 on the new set |
| 7 | `Q-ZZ` gain ambiguous (entanglement *and* encoding changed) | – | ablation `Q-Z` (A4b) | `Q-Z` = `Q-ZZ`, p = 0.90 |
| 8 | quantum best `c` at the grid edge (0.1) | could hide gains below 0.1 | extra check at c = 0.01, 0.03 | no further gain; plateau |
| 9 | RBF might lose on periodic data just from its bandwidth grid | fairness concern | `rbf-narrow` (2–128 × median) | still 0.534, so the A5b win is not a tuning artifact |
| 10 | Cox-LASSO on 4000 genes | 109 s per fit at tiny α; `ArithmeticError` divergence | separate `cox-lasso-hd` grid from 0.01 (keeps ~160 genes); failed fits scored as −∞ | 79 s per split for the whole TCGA set |
| 11 | high-dimensional baseline also ran on 8 features | spurious row | restricted 8-feature baselines to Cox-LASSO and RSF | – |
| 12 | a loky worker restarted during TCGA | warning | checked: all 12 models × 20 splits present, no NaN | verified |
| 13 | figure layout | legend over data (F1), colliding line labels (F3, F6), footnote overlap and slight overclaim in the title (F5) | moved legends, removed colliding direct labels, reworded | re-rendered and inspected |

---

## 9. Results, step by step

All numbers are **median test C-index over 20 splits** unless stated. Per-split CSVs are in `results/`.

### 9.1 Step 1: classical reproduction of the paper

| Dataset | Kernel | Ranking | Hybrid | Regression | Cox | Paper (regr./hybrid) |
|---|---|---|---|---|---|---|
| VLC | linear | 0.721 | 0.720 | 0.721 | 0.718 | 0.69 (Cox 0.68) |
| VLC | clinical | 0.706 | 0.706 | 0.705 | | 0.69 |
| VLC | RBF | 0.720 | 0.711 | 0.707 | | |
| GBSG2 | linear | 0.681 | 0.681 | 0.678 | 0.677 | 0.67 (Cox 0.67) |
| GBSG2 | clinical | 0.692 | 0.691 | 0.684 | | 0.68 |
| GBSG2 | RBF | 0.683 | 0.682 | 0.682 | | |

- Regression / hybrid are within **+0.01 to +0.03** of the paper, and our Cox reference shifts by the same amount, so the gap comes from the splits and tuning, not the SVM. **Pipeline trusted.**
- **Ranking does not reproduce the paper's weakness** (paper 0.57–0.62, ours ≈ regression). The paper's ranking models
  compare only nearest comparable neighbours; scikit-survival compares all comparable pairs (a known later improvement).
- Logrank χ² and hazard ratios behave like the paper's (huge HR spread, e.g. GBSG2 linear hybrid 946 ± 393019), so we rely on the C-index.
- Early observation: **RBF never beats linear** on VLC/GBSG2. Little nonlinear signal (later R5).
- Runtime: VLC 65 s, GBSG2 616 s. Figure: `results/figures/f1_reproduction.png`.

### 9.2 Step 4: quantum kernels on the paper's datasets (A1–A4, A4b, A6)

| Kernel | VLC rank. | VLC hyb. | VLC regr. | GBSG2 rank. | GBSG2 hyb. | GBSG2 regr. |
|---|---|---|---|---|---|---|
| linear | **0.721** | **0.720** | **0.721** | 0.681 | 0.681 | 0.678 |
| RBF | 0.720 | 0.711 | 0.707 | 0.683 | 0.682 | 0.682 |
| clinical | 0.706 | 0.706 | 0.705 | **0.692** | **0.691** | 0.684 |
| `Q-1L` | 0.716 | 0.719 | 0.715 | 0.678 | 0.677 | 0.676 |
| `Q-2L` | 0.715 | 0.715 | 0.718 | 0.675 | 0.679 | 0.678 |
| `Q-3L` | 0.713 | 0.712 | 0.716 | 0.679 | 0.677 | 0.678 |
| `Q-ZZ` | 0.709 | 0.718 | 0.716 | 0.684 | 0.685 | 0.682 |
| `Q-Z` (ablation) | 0.714 | 0.718 | 0.717 | 0.685 | **0.688** | **0.686** |
| `P-Q-2L` | 0.690 | 0.689 | 0.675 | 0.673 | 0.671 | 0.663 |

Paired Wilcoxon (`results/step4_wilcoxon.csv`), vs the best classical kernel:
- **VLC:** fidelity kernels −0.002 to −0.011, mostly not significant (p 0.03–0.45). A tie with linear.
- **GBSG2:** all quantum kernels significantly below clinical (−0.008 to −0.020, p ≤ 0.006).
- **Projected kernel:** significantly worse on both (−0.023 to −0.036, p < 0.001).
- **vs the `Q-1L` control:** `Q-2L`/`Q-3L` not significant; `Q-ZZ` > `Q-1L` on GBSG2 (p ≤ 0.001), explained by A4b below.
- Runtime: VLC 281 s, GBSG2 3017 s (+ `Q-Z` 53 s / 413 s).

### 9.3 A4b: is the `Q-ZZ` gain entanglement?

| GBSG2 (hybrid) | median diff | splits won | p |
|---|---|---|---|
| `Q-Z` vs `Q-ZZ` | 0.000 | 10/20 | 0.90 |
| `Q-Z` vs `Q-1L` | +0.009 | 17/20 | 1e-4 |
| `Q-Z` vs clinical | −0.009 | 4/20 | 0.008 |

**No.** The same circuit without the ZZ entanglers scores the same. The gain over `Q-1L` comes from the H·RZ single-qubit encoding (a different, still classical, one-qubit feature function). It still loses to the clinical kernel.

### 9.4 Step 3: pre-training diagnostics (44 configs × 5 splits per dataset)

**KTA predicts the outcome before training:**

| Dataset | Spearman(KTA, test C) | event-only cross-check | configs |
|---|---|---|---|
| VLC | **0.821** (p = 9e-12) | 0.815 | 44 |
| GBSG2 | **0.730** (p = 2e-8) | 0.675 | 44 |
| TCGA-BRCA (3 splits) | **0.583** (p = 7e-6) | 0.602 | 51 |

- The permutation p-value is 0.005 (the minimum possible with 200 shuffles) for every kernel except the projected kernel at large `c`.
- **KTA ordering = outcome ordering:** linear/RBF have the highest KTA (VLC 0.18, GBSG2 0.068), and the best quantum kernels sit lower (VLC 0.12–0.13, GBSG2 0.028–0.029). The diagnostic called the result before any model was fit.

**R1 concentration and R2 overfitting along the `c` sweep** (VLC, `Q-2L`, means over 5 splits):

| c | 0.100 | 0.147 | 0.215 | 0.316 | 0.464 | 0.681 | 1.000 |
|---|---|---|---|---|---|---|---|
| off-diagonal mean | 0.885 | 0.767 | 0.565 | 0.313 | 0.136 | 0.068 | 0.036 |
| effective rank / n | 0.014 | 0.018 | 0.031 | 0.068 | 0.147 | 0.239 | 0.328 |
| KTA | 0.122 | 0.123 | 0.122 | 0.119 | 0.102 | 0.073 | 0.068 |
| test C-index | 0.698 | 0.694 | 0.683 | 0.664 | 0.664 | 0.636 | 0.623 |
| train − test gap | 0.087 | 0.100 | 0.111 | 0.119 | 0.149 | 0.173 | 0.172 |

- Linear gap for reference: 0.049 (VLC), 0.010 (GBSG2). GBSG2 `Q-2L` gap 0.043 → 0.123.
- Deeper circuits concentrate faster. At c = 1 the off-diagonal mean is `Q-1L` 0.045 → `Q-2L` 0.036 → `Q-3L` 0.028 (VLC).

**R3 geometric difference vs RBF** at the best quantum configs: 2.1–2.3 (VLC) and 2.8–4.0 (GBSG2), far below √n ≈ 9.5 / 21.
(g vs *linear* is large only because the linear kernel has rank 8; it is not meaningful.)

**R4: the best quantum kernels are RBF kernels.** The best `c` is always the smallest grid value. An extra check below it
(`results/step3_small_c.csv`, 5 splits) showed no further gain, and **alignment with an RBF kernel `exp(−‖θ−θ'‖²/4)` on the same angle features**:

| Dataset | Kernel | c = 0.01 | c = 0.03 | c = 0.10 |
|---|---|---|---|---|
| VLC | `Q-1L` / `Q-2L` / `Q-ZZ` | 1.000 / 1.000 / 1.000 | 1.000 / 0.9996 / 1.000 | 1.000 / 0.9961 / 1.000 |
| GBSG2 | `Q-1L` / `Q-2L` / `Q-ZZ` | 1.000 / 0.9999 / 1.000 | 1.000 / 0.9992 / 1.000 | 1.000 / 0.9928 / 1.000 |

The reason is mathematical: `cos²(t/2) ≈ exp(−t²/4)` for small t.

Figures: `f3_concentration_sweep.png`, `f4_kta_vs_cindex.png`.

### 9.5 Step 5: benchmark + synthetic dial (R5, R7)

| Model (hybrid / baseline) | synth-linear | synth-interaction | synth-smooth-periodic | synth-periodic |
|---|---|---|---|---|
| linear | 0.795 | 0.497 | 0.744 | 0.552 |
| RBF | 0.792 | **0.771** | 0.737 | 0.547 |
| RBF-narrow | – | 0.743 | – | 0.534 |
| clinical | 0.788 | 0.475 | 0.740 | 0.686 |
| Cox-LASSO | **0.796** | 0.500 | **0.745** | 0.569 |
| RSF | 0.773 | 0.670 | 0.738 | 0.670 |
| `Q-1L` / `Q-Z` | 0.793 / 0.793 | 0.770 / 0.767 | 0.739 / 0.735 | 0.555 / 0.548 |
| `Q-2L` / `Q-3L` / `Q-ZZ` | 0.795 / 0.793 / 0.793 | 0.769 / 0.765 / 0.768 | 0.734 / 0.737 / 0.737 | 0.550 / 0.556 / 0.543 |
| `P-Q-2L` | 0.789 | 0.758 | 0.744 | 0.586 |
| **`Q-1L-wrap`** | – | 0.730 | 0.739 | **0.716** |
| `Q-2L-wrap` / `Q-ZZ-wrap` | – | 0.644 / 0.664 | 0.715 / 0.717 | 0.659 / 0.520 |

Baselines on the real data: VLC Cox-LASSO 0.718, RSF 0.693; GBSG2 Cox-LASSO 0.673, RSF 0.689.

- **R5:** on synth-interaction, nonlinear kernels gain **+0.27** over linear. On VLC/GBSG2 neither RBF nor RSF beats linear Cox by more than 0.02. So the real data has little nonlinear signal.
- On interaction data, quantum kernels **tie RBF** (`Q-1L` vs RBF: diff 0.000, p = 0.29) and entanglement is slightly **harmful** (`Q-2L` < `Q-1L` in 18/20 splits, p = 6e-5).

### 9.6 A5b: wrap-around kernels (c ∈ [1, 8])

| Model | VLC | GBSG2 | synth-interaction | synth-periodic |
|---|---|---|---|---|
| best classical | 0.720 | 0.691 | 0.771 | 0.686 |
| `Q-1L` (normal c) | 0.719 | 0.677 | 0.770 | 0.555 |
| `Q-1L-wrap` | 0.693 | 0.645 | 0.730 | **0.716** |
| `Q-2L-wrap` | 0.610 | 0.636 | 0.644 | 0.659 |
| `Q-ZZ-wrap` | 0.606 | 0.636 | 0.664 | 0.520 |

On synth-periodic, `Q-1L-wrap` vs clinical +0.033 (15/20, p = 0.001); vs RSF +0.033 (19/20, p = 1e-5); vs RBF-narrow +0.175 (20/20, p = 2e-6); vs `Q-2L-wrap` +0.062 (19/20, p = 4e-6).
**The only win of the project, but by an unentangled (classical) periodic kernel.** On real data the same wrap-around kernels lose 0.03–0.11.

### 9.7 A7: hybrid kernel `w·K_Q + (1−w)·K_linear`

| Dataset | `H-Q-2L` C / median w | `H-Q-ZZ` C / median w |
|---|---|---|
| VLC | 0.718 / **0.17** | 0.717 / 0.17 |
| GBSG2 | 0.678 / 0.42 | 0.682 / 0.33 |
| synth-interaction | 0.657 / 1.00 | 0.664 / 1.00 |
| synth-periodic | 0.549 / 0.83 | 0.553 / 0.33 |

- On VLC, CV puts ~80% of the weight on the linear part. The quantum kernel adds no complementary information.
- **KTA-as-tuner limitation:** on synth-interaction the hybrid scores 0.657, vs 0.769 for `Q-2L` tuned by CV. Investigation
  (`results/step6_a7_kta_choice.csv`, `Q-2L`, 3 splits): KTA *rises* with c (−0.003 → 0.029) while test C *falls* (0.761 → 0.657),
  so KTA picks c = 1, the worst value. On VLC/GBSG2 the two move together (both peak at small c). KTA on training data rewards flexible,
  concentrated kernels when the risk is non-monotone and absolute KTA is near zero. **Lesson: use KTA to screen kernel families, and CV to tune.**

### 9.8 KPCA-Cox (Kernel Cox version of the QKSM)

| Kernel | VLC | GBSG2 | synth-interaction |
|---|---|---|---|
| linear | **0.718** | 0.670 | 0.499 |
| RBF | 0.706 | 0.678 | **0.731** |
| clinical | 0.704 | **0.687** | 0.490 |
| `Q-1L` | 0.709 | 0.670 | **0.731** |
| `Q-2L` | 0.709 | 0.669 | 0.708 |
| `Q-ZZ` | 0.711 | 0.675 | 0.707 |

The same pattern as the SVM: the conclusion does not depend on which survival model sits on top of the kernel.

### 9.9 R6: learning curves (hybrid SVM, 10 splits)

| GBSG2, training patients | 91 | 183 | 274 | 366 | 457 |
|---|---|---|---|---|---|
| linear | **0.648** | **0.664** | 0.674 | 0.679 | 0.678 |
| RBF | 0.636 | 0.655 | **0.678** | **0.684** | 0.684 |
| `Q-1L` | 0.624 | 0.647 | 0.670 | 0.674 | 0.681 |
| `Q-2L` | 0.623 | 0.643 | 0.663 | 0.675 | 0.682 |
| `Q-ZZ` | 0.631 | 0.650 | 0.674 | 0.683 | **0.693** |

With 91 patients linear leads by 0.017–0.025; the quantum kernels catch up only at full size. On synth-interaction the quantum and RBF curves overlap at every size (e.g. 0.765/0.762/0.764 vs RBF 0.767 at full size), while linear stays at 0.50.
Step 6 total runtime: 2712 s. Figure: `f6_learning_curves.png`.

### 9.10 Step 7: TCGA-BRCA multi-omics

| Model | Median C | IQR | vs linear (paired) |
|---|---|---|---|
| `Q-Z` | **0.766** | 0.047 | +0.008, p = 0.07 |
| `Q-1L` | **0.766** | 0.039 | +0.006, 15/20, p = 0.048 |
| linear | 0.765 | 0.049 | – |
| `Q-2L` | 0.762 | 0.036 | +0.001, p = 0.55 |
| `P-Q-2L` | 0.762 | 0.040 | p = 0.31 |
| `Q-ZZ` | 0.761 | 0.037 | p = 0.25 |
| RBF | 0.761 | 0.042 | p = 0.90 |
| clinical kernel | 0.754 | 0.055 | |
| Cox-LASSO (8 features) | 0.750 | 0.036 | |
| Cox-LASSO, clinical only | 0.746 | 0.043 | 8-feature Cox-LASSO vs this: p = 0.35 |
| Cox-LASSO, all ~4000 genes | 0.724 | 0.036 | 8-feature better, +0.013, p = 0.001 |
| RSF | 0.715 | 0.057 | linear better, +0.030, p = 4e-6 |

- All kernel SVMs are within 0.012 of each other.
- The unentangled product kernels edge linear by +0.006–0.008. Only `Q-1L` reaches p < 0.05 (0.048), which does **not** survive correction for ~10 comparisons.
- **Entanglement again worse:** `Q-2L` < `Q-1L` (p = 0.03), `Q-ZZ` < `Q-Z` (p = 0.012).
- **Omics add almost nothing over age + stage** (p = 0.35); all genes overfit; RSF is worst.
- Diagnostics: best quantum configs at c = 0.32–0.46 (less concentrated than on VLC/GBSG2), alignment with RBF 0.92–0.93, g vs RBF 2.1–2.7 ≪ √n ≈ 27.
- Hypothesis (labelled as such) for the small edge of the product kernels: min-max scaling and bounded similarity reduce the influence of heavy-tailed PCA scores. That is a property of the encoding, not of quantum mechanics.
- Runtime: benchmark 1461 s, diagnostics 289 s.

### 9.11 Entanglement effect across everything (Figure F5)

Per-split difference (entangled − same circuit without entanglement): median, with paired Wilcoxon p:

| Dataset | `Q-2L` − `Q-1L` | `Q-ZZ` − `Q-Z` | `Q-2L-wrap` − `Q-1L-wrap` |
|---|---|---|---|
| VLC | −0.006 (p = 0.059) | +0.000 (p = 0.38) | **−0.068** (p = 6e-6) |
| GBSG2 | −0.001 (p = 0.35) | +0.000 (p = 0.90) | **−0.008** (p = 0.014) |
| TCGA-BRCA | **−0.002** (p = 0.030) | **−0.001** (p = 0.012) | – |
| synth-interaction | **−0.002** (p = 6e-5) | +0.000 (p = 0.96) | **−0.089** (p = 2e-6) |
| synth-periodic | +0.002 (p = 0.35) | −0.000 (p = 0.49) | **−0.062** (p = 4e-6) |

Bold = significant (p < 0.05), all of them negative. There is no significant positive effect anywhere: entanglement never helped.

### 9.12 Figures

| File | Content |
|---|---|
| `results/figures/f1_reproduction.png` | our C-index vs the paper's (Step 1) |
| `results/figures/f2_benchmark.png` | C-index boxplots for every model, 5 datasets, coloured by kernel family |
| `results/figures/f3_concentration_sweep.png` | off-diagonal similarity, train–test gap, test C-index vs c |
| `results/figures/f4_kta_vs_cindex.png` | pre-training KTA vs post-training C-index (3 datasets) |
| `results/figures/f5_entanglement_effect.png` | entangled − unentangled C-index per split |
| `results/figures/f6_learning_curves.png` | C-index vs training size |

Colour code (same in every figure): classical = blue, quantum without entanglement = orange, quantum entangled = aqua, projected / baselines = gray.

---

## 10. Catalogue of approaches tried (A1–A9)

| ID | Approach | Idea tested | Result vs best classical | Verdict & explanation |
|---|---|---|---|---|
| A1 | `Q-1L` product kernel | angle encoding, no entanglement | tie (VLC), −0.014 (GBSG2), +0.001 vs linear (TCGA) | classical `Π cos²` kernel; ≈ RBF at its best c (R4) |
| A2 | `Q-2L` encode–CNOT–encode | does entanglement add useful interactions? | ≤ `Q-1L` everywhere | at the best c it is still 99.3–100% RBF-aligned; at larger c it concentrates (R1) and overfits (R2) |
| A3 | `Q-3L` deeper | does depth/expressivity help? | slightly worse than A2 | concentrates fastest; gap 0.21–0.22 at c = 1 |
| A4 | `Q-ZZ` | data-dependent entanglers | best quantum on GBSG2 (0.685) but < clinical | gain fully explained by its encoding (A4b) |
| A4b | `Q-Z` (ablation) | entanglement vs encoding | = `Q-ZZ` (p = 0.90) | entanglement contributed nothing |
| A5 | c sweep (all 7 values recorded) | how kernel quality varies with c | best at smallest c | more "quantum" = worse |
| A5b | wrap-around (c ∈ [1, 8]) | the periodic regime | **wins on periodic data (+0.033)**, loses 0.03–0.11 on real data | the winner is unentangled (classical periodic kernel); shape matching, not quantum |
| A6 | projected kernel | avoids concentration | worst on VLC/GBSG2 (−0.03) | lowest KTA; bounded periodic Pauli features distort monotone effects |
| A7 | hybrid `w·K_Q + (1−w)·K_lin` | complementary information? | never better | CV puts most weight on linear; KTA-chosen c fails on interaction data |
| – | KPCA-Cox | Kernel Cox version | same pattern as SVM | conclusion is model-independent |
| A8 | trained (KTA-optimized) circuit | learn the feature map | **not done** (optional) | – |
| A9 | finite shots (1024) | hardware sampling noise | **not done** (optional) | – |

Full entries with the template: [docs/results_log.md](docs/results_log.md).

---

## 11. Why classical performs better: the reasoning

Every reason is tied to a measurement (R-number) and a result section.

| # | Reason | Evidence |
|---|---|---|
| 1 | **The data has almost no nonlinear signal.** Clinical survival risk rises steadily (monotonically) with age, stage, nodes, tumour size, performance score. Linear models are already near the ceiling, so no nonlinear kernel can gain. | R5: RBF/RSF never beat linear by more than 0.02 on VLC/GBSG2/TCGA (RSF is −0.03 on TCGA). Omics PCs add nothing over age + stage (p = 0.35). When nonlinear signal does exist (synth-interaction), nonlinear kernels gain +0.27 (§9.5). |
| 2 | **Where quantum kernels work best, they are RBF kernels.** Small rotation angles make `cos²(t/2) ≈ exp(−t²/4)`. | R4: 99.3–100% alignment with RBF on the same angles at the best c (§9.4). R3: geometric difference vs RBF only 2–4 ≪ √n (9.5–27), so a classical kernel can match them (Huang et al. 2021). |
| 3 | **Making the circuit "more quantum" makes it worse.** Larger angles, more layers and entanglement spread the states apart in the 2⁸-dimensional Hilbert space: every patient looks equally dissimilar (exponential concentration), and the SVM memorizes the training data. | R1: off-diagonal similarity 0.885 → 0.036 (VLC `Q-2L`). R2: effective rank ×23, train–test gap 0.087 → 0.172, test C 0.698 → 0.623 (§9.4). A3 concentrates fastest. |
| 4 | **Entanglement never helped.** Every apparent gain came from the single-qubit encoding. | A4b: `Q-ZZ` = `Q-Z` (p = 0.90). `Q-2L` ≤ `Q-1L` everywhere; significantly worse on TCGA (p = 0.03), interaction (p = 6e-5), periodic wrap (p = 4e-6); `Q-ZZ` < `Q-Z` on TCGA (p = 0.012) (§9.11, F5). |
| 5 | **Inductive-bias mismatch.** Angle encodings make similarity *periodic*: patients with very different values can look alike once angles wrap. That suits periodic risk and contradicts monotone clinical risk. | A5b: wrap-around kernels lose 0.03–0.11 on VLC/GBSG2 but win +0.033 on truly periodic data (§9.6). |
| 6 | **Small cohorts favour simple kernels.** Flexible kernels need more data; clinical cohorts are small (VLC has 91 training patients; TCGA has only 148 deaths). | R6: linear leads by ~0.02 at 91 patients; quantum only catches up at full size (§9.9). All-genes Cox-LASSO overfits on TCGA (−0.026 vs 8 features). |
| 7 | **The pre-training diagnostic saw it coming.** | KTA ranks kernels like test C-index (ρ = 0.82 / 0.73 / 0.58), and quantum kernels had lower KTA than linear/RBF before any training (§9.4, F4). Limitation: as a tuner of c on non-monotone data it fails (§9.7). |

**When could a quantum-circuit kernel help?** (from R7) Only when the true risk is periodic / oscillating in the features, and even then the winner here was the *unentangled* product kernel, which is classically computable in one line. A real quantum advantage would need a structure that is (a) present in the data and (b) not reproducible by a classical kernel (a large geometric difference). Nothing we measured on clinical data meets (a), and nothing we built meets (b).

---

## 12. Objectives scorecard

| Objective | Delivered | Outcome |
|---|---|---|
| **1. QKSM:** quantum kernels in Survival SVM and Kernel Cox | ✅ 6 quantum kernel variants (+ wrap + hybrid) in the Survival SVM (3 forms) and in KPCA-Cox, on 7 datasets | works end-to-end; no genuine quantum improvement |
| **2. Survival KTA** pre-training diagnostic | ✅ comparable-pairs KTA + event-only cross-check + permutation test + health check; validated across 44–51 configs | predicts test C-index (ρ 0.58–0.82); flagged quantum as weaker before training; limitation found as a tuner |
| **3. Benchmark** vs Cox-LASSO, RSF, Survival SVM on multi-omics, C-index | ✅ VLC, GBSG2, TCGA-BRCA (mRNA + CNV + clinical), 4 synthetic sets; 20 splits; paired Wilcoxon | classical ≥ quantum on all real data |
| Possible outcome | – | the **impactful negative result**, with seven evidence-backed reasons (§11) |

---

## 13. Limitations and honesty notes

- **Survival SVM implementation:** scikit-survival (Pölsterl et al.) uses a squared hinge loss and all comparable pairs, so it is close to, not identical to, Van Belle's models. This is why our ranking form does not reproduce the paper's weakness.
- **KPCA-Cox** approximates Kernel Cox (Li & Luan 2003); it is not the exact partial-likelihood RKHS solution.
- **KTA prior art:** survival KTA was first proposed by Colak et al. 2026 [5]; ours is an extension.
- **Splits:** 20 instead of the paper's 50. Tuning by grid, not coupled simulated annealing.
- **Statistics:** paired Wilcoxon over overlapping splits is optimistic, so p-values are indicative. No multiple-comparison correction is applied in the tables; we note where a result would not survive it (TCGA `Q-1L` p = 0.048).
- **Inner CV reuses the outer-train kernel:** scaling, ranges and bandwidth heuristics (and TCGA PCA inside the outer split) see the whole training part during inner CV. The test part is never touched.
- **TCGA:** label-free top-2000 variance pre-filter on the whole cohort (no labels used). **Methylation not included** (783 MB file), so 2 omic layers instead of 3. Only 148 deaths, so C-index is noisy (IQR ≈ 0.04–0.05).
- **Simulation only:** exact noiseless statevectors; no hardware, no shot noise (A9 not done).
- **Synthetic data** is designed by us; the periodic set deliberately favours periodic kernels (like [5]'s positive control).
- **Censored ranks** are only lower bounds; the event-only KTA cross-check agrees, so they don't distort the conclusions.
- **Projected kernel** uses a fixed median-heuristic γ (so its tuning budget equals the others').

---

## 14. What remains

| Item | Status | Notes |
|---|---|---|
| Formal report (Abstract / Intro / Methods / Results / Why classical wins / Conclusion) | **to do** | all content, tables and figures are ready (this document + `docs/results_log.md` + `results/figures/`) |
| README "how to run each step" | small | commands in §15 |
| Methylation (450K) for TCGA | optional | 783 MB; or state as a limitation |
| A8: trained quantum kernel (optimize one rotation layer for KTA) | optional | ~1–2 h |
| A9: finite shots (Qiskit sampler, 1024 shots) | optional, recommended | ~1 h; shows the cost of hardware noise |
| Explanatory notebooks | optional | for Tarun / teacher |
| Add Tarun as GitHub collaborator | owner action | Settings → Collaborators |

---

## 15. How to reproduce everything

```bash
uv sync
```

```bash
uv run pytest
```

| Step | Command | Runtime (10 cores) | Output |
|---|---|---|---|
| 1 Classical reproduction | `uv run python scripts/step1_reproduce.py --splits 20` | ~11 min | `results/step1_*` |
| 4 Quantum kernels (VLC, GBSG2) | `uv run python scripts/step4_qksm.py --splits 20` | ~55 min | `results/step4_*` |
| 4 Q-Z ablation only | `uv run python scripts/step4_qksm.py --only Q-Z` | ~8 min | updates `step4_splits.csv` |
| 3 Diagnostics | `uv run python scripts/step3_diagnostics.py --splits 5 --perm 200` | ~15 min | `results/step3_*` |
| 5 Benchmark + synthetic | `uv run python scripts/step5_benchmark.py --splits 20` | ~45 min | `results/step5_*` |
| 5 A5b wrap + RBF-narrow | `uv run python scripts/step5_benchmark.py --datasets synth-periodic synth-interaction synth-smooth-periodic vlc gbsg2 --wrap --no-baselines` | ~60 min | updates `step5_splits.csv` |
| 6 A7 + KPCA-Cox + learning curves | `uv run python scripts/step6_extra.py a7 kpca lc` | ~45 min | `results/step6_*` |
| 7 TCGA-BRCA | download the 4 files listed in `scripts/step7_tcga.py` into `data/raw/`, then `uv run python scripts/step7_tcga.py --splits 20` | ~30 min | `results/step7_*` |
| Figures | `uv run python scripts/make_figures.py` | seconds | `results/figures/` |

All randomness is seeded (splits `random_state=0`, CV folds seeded per split, RSF `random_state=0`, synthetic `seed=0`).

---

## 16. File index

| File | What it contains |
|---|---|
| `docs/PLAN.md` | final plan: decisions, steps, approaches A1–A9, evidence R1–R7, schedule, risks |
| `docs/research_notes.md` | literature notes for refs [1]–[5] |
| `docs/results_log.md` | every approach with the template + the "why classical wins" summary |
| `results/step1_summary.md` / `.csv`, `step1_splits.csv` | Step 1 tables and per-split rows |
| `results/step4_comparison.md`, `step4_splits.csv`, `step4_wilcoxon.csv` | quantum kernels on VLC/GBSG2 |
| `results/step3_configs.csv`, `step3_diagnostics.csv`, `step3_kta_vs_cindex.txt`, `step3_small_c.csv` | diagnostics |
| `results/step5_summary.md`, `step5_splits.csv` | benchmark + synthetic + wrap + RBF-narrow |
| `results/final_wilcoxon.txt` | A4b, A5b and synthetic paired tests |
| `results/step6_a7_kpca_lc.md`, `step6_a7_splits.csv`, `step6_kpca_splits.csv`, `step6_learning_curves.csv`, `step6_a7_kta_choice.csv` | A7, KPCA-Cox, R6, KTA-tuner check |
| `results/step7_tcga_summary.md`, `step7_tcga_splits.csv`, `step7_tcga_diagnostics.csv`, `step7_wilcoxon.txt` | TCGA-BRCA |
| `results/figures/f1…f6_*.png` | report figures |

---

## 17. Glossary

| Term | Meaning |
|---|---|
| Right-censored | we only know the patient survived *at least* until their last follow-up |
| C-index (Harrell) | fraction of comparable patient pairs whose predicted risks are in the right order; 0.5 = random, 1 = perfect |
| Comparable pair | a pair whose true order is known (the earlier time is an observed event) |
| Prognostic index / risk score | the model's output; higher = expected to fail sooner |
| Kernel | a similarity function `k(x, x')`; kernel methods only ever use these similarities |
| RKHS | the (possibly infinite-dimensional) feature space a kernel implicitly maps into |
| Fidelity quantum kernel | `|⟨ψ(x)|ψ(x')⟩|²`, the overlap of two quantum states |
| Projected quantum kernel | a classical kernel on measured single-qubit expectation values |
| Product state | a state with no entanglement; its kernel is a product over qubits and is classically easy |
| Exponential concentration | as circuits get larger/deeper, kernel values between different inputs shrink toward a constant, so the kernel stops distinguishing patients |
| KTA | kernel-target alignment: how similar the kernel's geometry is to an ideal kernel built from the labels |
| Geometric difference g | how much "room" a quantum kernel has over a classical one on given data (Huang et al. 2021) |
| Survival SVM forms | ranking (order pairs), regression (predict time with censoring handled), hybrid (both) |
| KPCA-Cox | kernel PCA features fed into a Cox model, an approximation of Kernel Cox |
| Cox-LASSO | Cox proportional hazards with an L1 penalty (feature selection) |
| RSF | Random Survival Forest |
| Paired Wilcoxon signed-rank | non-parametric test of whether one model's per-split scores are systematically higher than another's |

---

## 18. References

1. V. Van Belle, K. Pelckmans, S. Van Huffel, J. A. K. Suykens, "Support vector methods for survival analysis: a comparison between ranking and regression approaches," *Artificial Intelligence in Medicine*, 53(2):107–118, 2011.
2. H. Li, Y. Luan, "Kernel Cox regression models for linking gene expression profiles to censored survival data," *Pacific Symposium on Biocomputing* 8:65–76, 2003.
3. D. Maheshwari, B. Garcia-Zapirain, D. Sierra-Sosa, "Quantum machine learning applications in the biomedical domain: A systematic review," *IEEE Access*, 10:80463–80484, 2022.
4. V. Novák, I. Zelinka, L. Přibylová, L. Martínek, "Quantum Neural Networks for Propensity Score Estimation and Survival Analysis in Observational Biomedical Studies," arXiv:2506.19973, 2025.
5. C. Colak et al., "Quantum Computing in a Diagnostic-First Quantum Residual Boosting Framework for Clinical Survival Analysis in Oncology and Cardiology," *Journal of Clinical Medicine*, 15(14):5387, 2026.
6. H.-Y. Huang et al., "Power of data in quantum machine learning," *Nature Communications*, 12:2631, 2021 (projected kernel, geometric difference).
7. S. Pölsterl, N. Navab, A. Katouzian, "Fast training of support vector machines for survival analysis," *ECML PKDD*, 2015 (the scikit-survival Survival SVM).
8. A. Daemen, B. De Moor, "Development of a kernel function for clinical data," *IEEE EMBC*, 2009 (clinical kernel).
9. S. Thanasilp, S. Wang, M. Cerezo, Z. Holmes, "Exponential concentration in quantum kernel methods," *Nature Communications*, 15:5200, 2024 (concentration, R1).
10. UCSC Xena TCGA hub: TCGA-BRCA HiSeqV2, GISTIC2 copy number, clinical matrix, curated survival (TCGA-CDR).
