# Quantum Kernel Methods for Survival Analysis

Course project (Ansari, Tarun). We replace the classical kernels in the survival SVMs of
Van Belle et al. (2011) with **quantum fidelity kernels** and test whether they help predict
time-to-event outcomes with right-censored data.

- **Objective 1:** Quantum Kernel Survival Model (QKSM): quantum kernel inside Survival SVM and KPCA-Cox
- **Objective 2:** Survival Kernel-Target Alignment (KTA), a diagnostic computed before training
- **Objective 3:** Benchmark vs Cox-LASSO, Random Survival Forest, and classical Survival SVM (C-index)

See [docs/PLAN.md](docs/PLAN.md) for the full plan and [docs/research_notes.md](docs/research_notes.md) for the literature notes.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Layout

```
src/qksm/     library code (data, kernels, diagnostics, models, evaluate)
tests/        unit tests
notebooks/    one notebook per step
results/      tables and figures
data/         raw / processed data (not committed)
docs/         plan, notes, report
```
