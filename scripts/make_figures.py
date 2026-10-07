"""Report figures from the saved results (no re-computation).

    uv run python scripts/make_figures.py      -> results/figures/*.png

Colour = kernel family, the same in every figure:
  classical (linear/RBF/clinical) blue, quantum without entanglement orange,
  quantum with entanglement aqua, projected quantum / baselines neutral gray.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "figures"
OUT.mkdir(exist_ok=True)

# Reference palette (dataviz skill, light mode): first three categorical slots + neutrals.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHTGRAY = "#8a8985", "#d9d8d4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"

FAMILY = {
    "linear": "classical", "rbf": "classical", "clinical": "classical", "rbf-narrow": "classical",
    "Q-1L": "quantum, no entanglement", "Q-Z": "quantum, no entanglement", "Q-1L-wrap": "quantum, no entanglement",
    "Q-2L": "quantum, entangled", "Q-3L": "quantum, entangled", "Q-ZZ": "quantum, entangled",
    "Q-2L-wrap": "quantum, entangled", "Q-ZZ-wrap": "quantum, entangled",
    "P-Q-2L": "projected quantum", "cox-lasso": "baseline", "rsf": "baseline",
}
COLOR = {"classical": BLUE, "quantum, no entanglement": ORANGE, "quantum, entangled": AQUA,
         "projected quantum": GRAY, "baseline": GRAY}
LABEL = {"linear": "linear", "rbf": "RBF", "clinical": "clinical", "rbf-narrow": "RBF (narrow)",
         "cox-lasso": "Cox-LASSO", "rsf": "RSF", "P-Q-2L": "projected"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "legend.frameon": False, "text.color": INK,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def family_legend(ax, families, loc="lower center", **kw):
    handles = [plt.Line2D([], [], marker="s", ls="", ms=8, color=COLOR[f], label=f) for f in families]
    ax.legend(handles=handles, loc=loc, ncol=len(families), **kw)


# ---------------------------------------------------------------- data


def load_hybrid():
    s1 = pd.read_csv(R / "step1_splits.csv")
    s4 = pd.read_csv(R / "step4_splits.csv")
    s5 = pd.read_csv(R / "step5_splits.csv")
    s7 = pd.read_csv(R / "step7_tcga_splits.csv")
    df = pd.concat([s1, s4, s5, s7], ignore_index=True)
    df = df[df.form.isin(["hybrid", "baseline"])]
    return df.drop_duplicates(["dataset", "kernel", "form", "split"], keep="last")


# ---------------------------------------------------------------- F1 reproduction


def fig_reproduction():
    s = pd.read_csv(R / "step1_summary.csv")
    s = s[s.kernel.isin(["linear", "clinical"]) & s.paper.notna()].copy()
    s["ours"] = s.cindex.str.split(" ").str[0].astype(float)
    s["row"] = s.dataset.str.upper() + " · " + s.kernel + " · " + s.form
    s = s.sort_values(["dataset", "kernel", "form"], ascending=[False, True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(s))
    ax.hlines(y, s[["ours", "paper"]].min(1), s[["ours", "paper"]].max(1), color=LIGHTGRAY, lw=2)
    ax.plot(s.paper, y, "o", ms=8, color=ORANGE, label="Van Belle et al. (2011)", mec="#fcfcfb", mew=1.5)
    ax.plot(s.ours, y, "o", ms=8, color=BLUE, label="this project (20 splits)", mec="#fcfcfb", mew=1.5)
    ax.set_yticks(y, s.row)
    ax.set_xlabel("median test C-index")
    ax.set_title("Step 1: classical reproduction of the paper", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.text(0.0, -0.17, "Regression / hybrid match within +0.01-0.03. Ranking is higher than the paper because\n"
            "scikit-survival compares all comparable pairs, not only the nearest neighbour.",
            transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    save(fig, "f1_reproduction.png")


# ---------------------------------------------------------------- F2 benchmark


def fig_benchmark(df):
    panels = [("vlc", "VLC (lung, n=137)"), ("gbsg2", "GBSG2 (breast, n=686)"),
              ("tcga-brca", "TCGA-BRCA multi-omics (n=1063)"),
              ("synth-interaction", "Synthetic: interaction"), ("synth-periodic", "Synthetic: high-frequency periodic")]
    order = ["linear", "rbf", "rbf-narrow", "clinical", "Q-1L", "Q-Z", "Q-2L", "Q-3L", "Q-ZZ", "P-Q-2L",
             "Q-1L-wrap", "Q-2L-wrap", "Q-ZZ-wrap", "cox-lasso", "rsf"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 12.5))
    for ax, (ds, title) in zip(axes, panels):
        d = df[df.dataset == ds]
        ks = [k for k in order if k in set(d.kernel)]
        data = [d[d.kernel == k].cindex.to_numpy() for k in ks]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        for patch, k in zip(bp["boxes"], ks):
            patch.set(facecolor=COLOR[FAMILY[k]], edgecolor="#fcfcfb", alpha=0.9, lw=1)
        best = max(np.median(x) for k, x in zip(ks, data) if FAMILY[k] in ("classical", "baseline"))
        ax.axvline(best, color=BLUE, ls=(0, (3, 3)), lw=1)
        ax.set_yticks(range(1, len(ks) + 1), [LABEL.get(k, k) for k in ks], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("test C-index over 20 splits (Survival SVM hybrid form; baselines tuned the same way)")
    family_legend(axes[0], ["classical", "quantum, no entanglement", "quantum, entangled", "baseline"],
                  loc="upper center", bbox_to_anchor=(0.5, 1.45), fontsize=8)
    fig.text(0.01, 0.005, "Dashed line = best classical / baseline median. Gray = projected quantum kernel or non-kernel baseline.",
             fontsize=8, color=INK2)
    fig.tight_layout(h_pad=1.6)
    save(fig, "f2_benchmark.png")


# ---------------------------------------------------------------- F3 concentration sweep


def fig_concentration():
    c = pd.read_csv(R / "step3_configs.csv")
    circ = [("Q-1L", ORANGE, "-"), ("Q-2L", AQUA, "-"), ("Q-3L", AQUA, (0, (4, 2))), ("Q-ZZ", AQUA, (0, (1, 1.5)))]
    rows = [("offdiag_mean", "mean off-diagonal\nsimilarity (R1)"), ("gap", "train - test\nC-index gap (R2)"),
            ("test_cindex", "test C-index")]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.2), sharex=True)
    for j, ds in enumerate(["vlc", "gbsg2"]):
        g = c[c.dataset == ds]
        lin = g[g.kernel == "linear"].iloc[0]
        for i, (col, lab) in enumerate(rows):
            ax = axes[i, j]
            for k, color, ls in circ:
                h = g[g.kernel == k].sort_values("param")
                ax.plot(h.param, h[col], color=color, ls=ls, lw=2, marker="o", ms=4)
            if col != "offdiag_mean":
                ax.axhline(lin[col], color=BLUE, lw=1.2, ls=(0, (3, 3)))
                if j == 0:
                    ax.text(0.1, lin[col], " linear", color=INK2, fontsize=8, va="bottom")
            ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title({"vlc": "VLC", "gbsg2": "GBSG2"}[ds])
    for ax in axes[-1]:
        ax.set_xlabel("angle scale c (log)")
    fig.suptitle("A5 sweep: larger angles concentrate the quantum kernel and hurt the model", x=0.02, ha="left",
                 fontweight="bold", fontsize=10)
    fig.text(0.02, 0.945, "Q-ZZ and Q-1L nearly coincide: at small c the ZZ pair term is negligible.",
             fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], color=col, ls=ls, lw=2, label=k) for k, col, ls in circ]
    handles.append(plt.Line2D([], [], color=BLUE, ls=(0, (3, 3)), lw=1.2, label="linear kernel"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, "f3_concentration_sweep.png")


# ---------------------------------------------------------------- F4 KTA vs C-index


def fig_kta():
    c3 = pd.read_csv(R / "step3_configs.csv")
    c7 = pd.read_csv(R / "step7_tcga_diagnostics.csv").groupby(["kernel", "param"], dropna=False) \
        .mean(numeric_only=True).reset_index().assign(dataset="tcga-brca")
    allc = pd.concat([c3, c7], ignore_index=True)
    marker = {"classical": "o", "quantum, no entanglement": "^", "quantum, entangled": "s", "projected quantum": "D"}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, (ds, title) in zip(axes, [("vlc", "VLC"), ("gbsg2", "GBSG2"), ("tcga-brca", "TCGA-BRCA")]):
        g = allc[allc.dataset == ds]
        for fam in marker:
            h = g[g.kernel.map(FAMILY) == fam]
            ax.scatter(h.kta, h.test_cindex, s=34, marker=marker[fam], color=COLOR[fam], edgecolor="#fcfcfb",
                       linewidth=1, label=fam, zorder=3)
        rho, p = spearmanr(g.kta, g.test_cindex)
        ax.set_title(f"{title}   Spearman ρ = {rho:.2f}", loc="left")
        ax.set_xlabel("survival KTA (training data, before fitting)")
    axes[0].set_ylabel("test C-index (after fitting)")
    handles = [plt.Line2D([], [], marker=marker[f], ls="", ms=7, color=COLOR[f], label=f) for f in marker]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.suptitle("Objective 2: the pre-training KTA diagnostic predicts which kernel configuration does better",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "f4_kta_vs_cindex.png")


# ---------------------------------------------------------------- F5 entanglement effect


def fig_entanglement(df):
    pairs = [("vlc", "VLC"), ("gbsg2", "GBSG2"), ("tcga-brca", "TCGA-BRCA"),
             ("synth-interaction", "synth: interaction"), ("synth-periodic", "synth: periodic")]
    comps = [("Q-2L", "Q-1L", "Q-2L − Q-1L  (add CNOT entanglement)", AQUA),
             ("Q-ZZ", "Q-Z", "Q-ZZ − Q-Z  (add ZZ entanglement)", ORANGE),
             ("Q-2L-wrap", "Q-1L-wrap", "Q-2L-wrap − Q-1L-wrap  (large angles)", GRAY)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = 0
    ticks, labels = [], []
    for ds, title in pairs:
        d = df[df.dataset == ds]
        for a, b, lab, color in comps:
            if not ({a, b} <= set(d.kernel)):
                continue
            x = d[d.kernel == a].sort_values("split").cindex.to_numpy() - d[d.kernel == b].sort_values("split").cindex.to_numpy()
            q1, med, q3 = np.percentile(x, [25, 50, 75])
            ax.scatter(x, np.full_like(x, y) + np.random.default_rng(0).uniform(-0.18, 0.18, len(x)),
                       s=8, color=color, alpha=0.35, lw=0, zorder=2)
            ax.hlines(y, q1, q3, color=color, lw=4, zorder=3)
            ax.plot(med, y, "o", ms=8, color=color, mec=INK, mew=1, zorder=4)
            ticks.append(y)
            labels.append(f"{title}: {a} vs {b}")
            y += 1
        y += 0.6
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("difference in test C-index per split (entangled − same circuit without entanglement)")
    ax.set_title("Reason 4: entanglement never helped. Medians are ≈ 0 or negative", loc="left")
    ax.text(0.0, -0.14, "dot = median, bar = IQR, faint dots = the 20 splits. Right of 0 would mean entanglement helps.",
            transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    save(fig, "f5_entanglement_effect.png")


# ---------------------------------------------------------------- F6 learning curves


def fig_learning():
    lc = pd.read_csv(R / "step6_learning_curves.csv")
    series = [("linear", BLUE, "-"), ("rbf", BLUE, (0, (3, 2))), ("Q-1L", ORANGE, "-"),
              ("Q-2L", AQUA, "-"), ("Q-ZZ", AQUA, (0, (1, 1.5)))]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, (ds, title) in zip(axes, [("gbsg2", "GBSG2"), ("synth-interaction", "Synthetic: interaction")]):
        g = lc[lc.dataset == ds].groupby(["kernel", "n_train"]).cindex.median().reset_index()
        for k, color, ls in series:
            h = g[g.kernel == k]
            ax.plot(h.n_train, h.cindex, color=color, ls=ls, lw=2, marker="o", ms=4)
        ax.set_title(title, loc="left")
        ax.set_xlabel("training patients")
    axes[0].set_ylabel("median test C-index (10 splits)")
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2, label=LABEL.get(k, k)) for k, c, ls in series]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle("R6: on GBSG2 linear wins with few patients and quantum kernels only catch up at full size;\n"
                 "on interaction data quantum kernels track RBF at every size",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    save(fig, "f6_learning_curves.png")


if __name__ == "__main__":
    df = load_hybrid()
    fig_reproduction()
    fig_benchmark(df)
    fig_concentration()
    fig_kta()
    fig_entanglement(df)
    fig_learning()
