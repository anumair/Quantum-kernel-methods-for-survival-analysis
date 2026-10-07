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
