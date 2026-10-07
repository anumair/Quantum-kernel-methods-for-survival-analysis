# Research Notes: Quantum Kernel Methods for Survival Analysis

Notes from reading the main paper and references [1]-[5] of the problem statement.

---

## The one idea that makes the project work

Every model in [1] and [2] uses the data only through the kernel matrix
`K[i, j] = k(x_i, x_j)`. Change the kernel and the model stays the same.

So a **Quantum Kernel Survival Model (QKSM)** works like this:

1. Encode each patient `x` into a quantum state `|ψ(x)⟩` with a small circuit
   (rotation gates + CNOTs, i.e. Module 3 of the syllabus).
2. Quantum (fidelity) kernel: `k_Q(x, x') = |⟨ψ(x)|ψ(x')⟩|²`
   (an inner product plus a measurement probability, i.e. Module 2).
3. Hand that `n × n` matrix to a Survival SVM or a Kernel Cox model in place of the RBF or linear kernel.

Practical tip: on a simulator, compute the `n` statevectors once and get
the whole Gram matrix as `|Ψ Ψ†|²`. That is cheap for ≤ 12–16 qubits.

---

## [1] Van Belle et al. 2011 — Support vector methods for survival analysis (main paper)

**Setup:** data `{x_i, y_i, δ_i}`, where `δ = 1` means the event was observed and `δ = 0` means right-censored.
The model outputs a **prognostic index** `u(x) = wᵀφ(x)`, which is a risk score, not a time.

**Comparable pair** (Eq. 16): we can only say "i failed before j" when both are
events, or when i is an event and j was censored *later*. Two censored patients can never be compared.

**Models compared:**

| Model | Idea | Constraint type | Notes |
|---|---|---|---|
| RANKSVMC (Eq. 17) | penalize misordered comparable pairs | ranking | O(n²) constraints. A simplified version compares each point only with its nearest comparable neighbour, giving O(n) |
| Model 1 (Eq. 19) | ranking, but the margin is the time gap `y_i − y_j̄(i)` | ranking | |
| SVCR (Eq. 14, Shivaswamy) | SVR, but for censored points only penalize predictions that are *too early* | regression | 1 hyperparameter (γ). Best on high-dim data |
| SVRC (Khan & Zubek) | 4 different penalties | regression | too many hyperparameters |
| **Model 2 (Eq. 22, new)** | ranking + regression constraints | both | 2 hyperparameters. Reduces to Model 1 (γ→0) or SVCR (γ_rank→0) |

**Prediction** (all models) is a weighted sum of kernel values, e.g. Eq. 15:
`u(x*) = Σ (α_i − δ_i α*_i) k(x_i, x*) + b`. This is where the quantum kernel plugs in.

**Kernels used:** linear, RBF, polynomial, plus a "clinical kernel" (Eq. 12–13):
an additive per-feature kernel `(range − |x_p − z_p|)/range` for continuous features and `1/0` match for categorical ones.

**Metrics:**
- **C-index** (Eq. 24): concordant pairs / comparable pairs. This is the project's main metric.
- Logrank χ²: split patients at the median risk score and test whether the two groups' survival differs.
- Hazard ratio from a univariate Cox fit on the normalized risk score.

**Protocol (copy this):** 50 random splits (2/3 train, 1/3 test), tune hyperparameters with
10-fold CV using the C-index, report median ± spread, and test significance with
the Wilcoxon rank-sum test.

**Datasets:** clinical sets VLC (veterans lung, n=137), leukemia LD/LCR (n=129),
PC (prostate, n=483), MLC (Mayo lung, n=167), GBSG (breast, n=686). High-dimensional
gene-expression sets: DBCD, NSBCD, DLBCL.

**Findings:**
- Pure ranking models (Model 1, RANKSVMC) are significantly worse and unstable (Figs. 3–4).
- Models that include regression constraints (SVCR, Model 2) are best.
- On high-dimensional data, SVCR wins. Linear Cox is roughly equal to the SVMs on clinical data.
- C-index values on clinical data are mostly 0.60–0.78. GBSG ≈ 0.67–0.68.

**For us:** implement the quantum kernel inside **SVCR** (the best performer) and/or
a ranking survival SVM. `scikit-survival`'s `FastKernelSurvivalSVM(kernel="precomputed")` takes a precomputed kernel matrix directly.

---

## [2] Li & Luan 2003 — Kernel Cox regression

- Model: `λ(t|x) = λ₀(t) exp(f(x))`, with `f` in an RKHS.
- Loss: negative Cox partial likelihood plus `λ‖f‖²`.
- **Representer theorem:** `f(x) = Σ a_i K(x, x_i)`. The problem becomes a finite one:
  `R(a) = −δᵀ(Ka) + Σ_{i:δ_i=1} log Σ_{j∈R_i} exp((Ka)_j) + λ aᵀKa`, where `R_i` is the risk set (patients still at risk at time `t_i`).
- Solved with Newton–Raphson (falling back to Nelder–Mead), with λ chosen by leave-one-out CV.
- They used **only the linear kernel**. Datasets were tiny (DLBCL n=40, lung n=22, breast n=49),
  with genes pre-filtered by univariate Cox p<0.05. They state that nonlinear kernels are "easy" but never tried them, which leaves room for our quantum kernel.
- Evaluation: Kaplan–Meier curves for score>0 vs score<0 groups. No C-index.

**For us:** Kernel Cox is about 20 lines (minimize `R(a)` with `scipy.optimize`
or PyTorch). It needs only `K`, so the quantum kernel drops straight in.

---

## [3] Maheshwari et al. 2022 — QML in the biomedical domain (systematic review)

- 30 papers (2013–2021). Methods: QSVM, VQC, QNN, QCNN, quantum annealing, quantum-inspired methods.
- Tasks are almost entirely **classification/diagnosis** (COVID X-ray, diabetes, cancer
  vs normal, EEG). No survival or time-to-event work. This supports the "QML has ignored survival analysis" claim in the problem statement.
- Common finding: classical models match or slightly beat quantum ones (e.g. diabetes QSVM/VQC).
- Common preprocessing: feature selection/PCA to fit the small number of qubits.

---

## [4] Novák et al. 2025 — QNNs for propensity score + survival (arXiv:2506.19973)

- 1,177 colorectal cancer patients, 4 covariates (age, sex, stage, BMI), so 4 qubits.
- Qiskit + sQUlearn. ZFeatureMap (H + RZ(x)), SummedPaulis observable, CMA-ES optimizer, variance regularization.
- The QNN predicts **propensity scores only** (laparoscopic vs open surgery). Survival
  itself is done classically (KM, Cox, Aalen) after matching.
- QNN AUC up to 0.75 on small samples. **No quantum kernels.** Limitation: more features means deeper, noisier circuits.

---

## [5] Colak et al. 2026 — QResid-Boost + KTA-Survival (J. Clin. Med. 15:5387) ⚠️ most relevant

- **Model:** Cox-LASSO plus a variational quantum circuit (6 qubits, 5 layers, angle-Y
  re-uploading, strongly-entangling, PennyLane) trained on Cox **martingale residuals**.
  Final risk = `η_Cox + sigmoid(α)·VQC(x)`.
- **KTA-Survival (already proposed here!):**
  - "Quantum kernel" `K_Q(i,j) = exp(−‖z_i − z_j‖²/2N)` on the 6 Pauli-Z expectation values.
    This is an RBF kernel on measured outputs (a "projected" kernel), **not** a true
    fidelity kernel `|⟨ψ_i|ψ_j⟩|²`.
  - Target kernel `K_T(i,j) = 1` if the pair is comparable, else 0, with a zero diagonal.
  - Centered alignment `A(K_Q, K_T) = ⟨K_Q,K_T⟩_F / (‖K_Q‖_F ‖K_T‖_F)`.
  - Reported **ΔKTA** = KTA_quantum − best KTA over an RBF-bandwidth grid.
  - Results: ΔKTA ≈ +0.02–0.03 (significant by permutation test), but **no fixed go/no-go threshold**.
- **Datasets:** GBSG2 (686), FLChain (1500), WHAS500 (external), synthetic Weibull. All low-dimensional (6 features).
- **Results:** GBSG2 test C: RSF 0.719, Cox-LASSO 0.702, QResid-Boost 0.702, pure VQC
  ≈ 0.56. The quantum gate α stays small, so the model collapses back to Cox. Calibration (IBS) is worse.
  It only helps on the synthetic periodic data (+0.04 C).
- **Their stated future work:** higher-dimensional omics, richer classical kernels in the KTA comparison, nested CV.
  They do **not** use quantum kernels inside a Survival SVM or Kernel Cox.

---

## What this means for our project

**The gap is real.** No reference puts a quantum kernel inside a Survival SVM or Kernel Cox:
[3] is all classification, [4] and [5] use variational circuits, and [5]'s "kernel" appears only in the diagnostic.

**Be careful with objective 2 (KTA):** [5] already published a survival KTA. We must cite it and
present ours as an extension or improvement. Concrete ways to do that:
1. Use the **true fidelity kernel** (and optionally a projected kernel) instead of an RBF on Pauli-Z outputs.
2. Use a **better target kernel**. [5]'s `K_T` as described marks a pair "comparable" but does not
   encode *who is at higher risk*. That makes it symmetric in direction, so it rewards geometry
   unrelated to risk order. Alternatives: a rank-based target `K_T(i,j) = 1 − |r_i − r_j|/n` on
   comparable pairs only, or a martingale-residual target `m mᵀ`. (Verify against [5]'s full text before claiming this.)
3. **Validate the diagnostic:** across many feature maps/bandwidths, check whether higher
   KTA actually predicts higher test C-index (scatter plot + Spearman correlation). [5] did not do this.
4. Benchmark against **richer classical kernels** (RBF with a tuned bandwidth, the clinical kernel from [1]).

**Benchmarking (objective 3):** Cox-LASSO (`sksurv` Coxnet), RSF (`sksurv`), classical
Survival SVM (RBF/linear). Report C-index using [1]'s protocol (repeated splits + Wilcoxon).
Adding IBS would be good practice, but the C-index is the required metric.

**Multi-omics data:** quantum circuits need few features (≈ 4–16 qubits), so reduce each omic layer
(e.g. mRNA, methylation, CNV) with PCA or univariate-Cox selection ([2]'s approach), then concatenate. Candidate sources:
TCGA cohorts (e.g. BRCA, KIRC, LGG via UCSC Xena / LinkedOmics) or METABRIC. Start with GBSG2 as a sanity check,
since it is used in both [1] and [5], so we can compare numbers directly.

**A negative result is acceptable:** the problem statement explicitly says that underperforming classical
baselines is an "impactful negative result". [5] already found that quantum barely helps on low-dimensional clinical data.
