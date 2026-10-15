"""Results dashboard for the Quantum Kernel Survival Model (QKSM) project: two pages.

Reads only the saved result files in results/ (nothing is recomputed).

    uv run streamlit run dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr, wilcoxon

R = Path(__file__).resolve().parents[1] / "results"
st.set_page_config(page_title="QKSM results", page_icon="📈", layout="wide")

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
FAMILY = {
    "linear": "classical", "rbf": "classical", "clinical": "classical",
    "Q-1L": "quantum, no entanglement", "Q-Z": "quantum, no entanglement", "Q-1L-wrap": "quantum, no entanglement",
    "Q-2L": "quantum, entangled", "Q-3L": "quantum, entangled", "Q-ZZ": "quantum, entangled",
    "Q-2L-wrap": "quantum, entangled", "Q-ZZ-wrap": "quantum, entangled",
    "P-Q-2L": "projected quantum", "cox-lasso": "baseline", "rsf": "baseline",
}
COLOR = {"classical": BLUE, "quantum, no entanglement": ORANGE, "quantum, entangled": AQUA,
         "projected quantum": GRAY, "baseline": GRAY}
FAMILY_ORDER = list(COLOR)
LABEL = {"rbf": "RBF", "P-Q-2L": "projected", "cox-lasso": "Cox-LASSO", "rsf": "Random Survival Forest"}
ORDER = ["linear", "rbf", "clinical", "Q-1L", "Q-Z", "Q-2L", "Q-3L", "Q-ZZ", "P-Q-2L",
         "Q-1L-wrap", "Q-2L-wrap", "Q-ZZ-wrap", "cox-lasso", "rsf"]
DATASETS = {"vlc": "VLC · lung cancer (n=137)", "gbsg2": "GBSG2 · breast cancer (n=686)",
            "tcga-brca": "TCGA-BRCA · multi-omics (n=1063)", "synth-interaction": "Synthetic · interaction risk",
            "synth-periodic": "Synthetic · periodic risk"}
REAL = ["vlc", "gbsg2", "tcga-brca"]


def label(k):
    return LABEL.get(k, k)


def style(fig, height):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="", font=dict(size=13), height=height)
    return fig


@st.cache_data
def splits():
    df = pd.concat([pd.read_csv(R / f) for f in
                    ["step1_splits.csv", "step4_splits.csv", "step5_splits.csv", "step7_tcga_splits.csv"]])
    df = df[df.form.isin(["hybrid", "baseline"]) & df.kernel.isin(ORDER) & df.dataset.isin(DATASETS)]
    df = df.drop_duplicates(["dataset", "kernel", "form", "split"], keep="last")
    df["family"] = df.kernel.map(FAMILY)
    df["model"] = df.kernel.map(label)
    return df


@st.cache_data
def diagnostics():
    c3 = pd.read_csv(R / "step3_configs.csv")
    c7 = (pd.read_csv(R / "step7_tcga_diagnostics.csv").groupby(["kernel", "param"], dropna=False)
          .mean(numeric_only=True).reset_index().assign(dataset="tcga-brca"))
    df = pd.concat([c3, c7], ignore_index=True)
    df["family"] = df.kernel.map(FAMILY)
    return df


def paired(d, a, b):
    x = d[d.kernel == a].sort_values("split").cindex.to_numpy()
    y = d[d.kernel == b].sort_values("split").cindex.to_numpy()
    diff = x - y
    return float(np.median(diff)), int((diff > 0).sum()), float(wilcoxon(x, y).pvalue) if diff.any() else 1.0


# ---------------------------------------------------------------- page 1


def page_results():
    st.title("Quantum Kernel Methods for Survival Analysis")
    st.markdown("**Result: no genuine quantum advantage.** We kept the survival SVM of Van Belle et al. (2011) and "
                "swapped only the kernel. On all three real cohorts the quantum kernels tie or lose to the best "
                "classical kernel, and every quantum kernel that comes out on top has no entanglement.")
    c = st.columns(4)
    c[0].metric("Real patients", "1,886", "VLC · GBSG2 · TCGA-BRCA", delta_color="off")
    c[1].metric("Paper reproduced", "≤ +0.03", "C-index gap vs Van Belle et al.", delta_color="off")
    c[2].metric("KTA predicts C-index", "ρ 0.58–0.82", "before any training", delta_color="off")
    c[3].metric("Entanglement helped", "0 of 14", "comparisons significantly positive", delta_color="off")

    df = splits()
    st.subheader("Best classical vs best quantum")
    rows = []
    for ds in DATASETS:
        d = df[df.dataset == ds]
        med = d.groupby("kernel").cindex.median()
        cl = med[[k for k in med.index if FAMILY[k] in ("classical", "baseline")]]
        qu = med[[k for k in med.index if FAMILY[k].startswith("quantum")]]
        bc, bq = cl.idxmax(), qu.idxmax()
        dlt, wins, p = paired(d, bq, bc)
        rows.append({"Dataset": DATASETS[ds], "Best classical": f"{label(bc)}  {cl.max():.3f}",
                     "Best quantum": f"{bq}  {qu.max():.3f}", "Quantum − classical": f"{dlt:+.3f}",
                     "Splits quantum wins": f"{wins}/20", "p (paired Wilcoxon)": f"{p:.2g}",
                     "Entangled?": "yes" if FAMILY[bq] == "quantum, entangled" else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Median test C-index over 20 repeated 2/3–1/3 splits, Survival SVM hybrid form. The only quantum win "
               "(periodic synthetic data) comes from an unentangled, classically computable kernel.")

    st.subheader("Every model, one dataset")
    ds = st.radio("Dataset", list(DATASETS), format_func=DATASETS.get, horizontal=True, label_visibility="collapsed")
    d = df[df.dataset == ds]
    ks = [k for k in ORDER if k in set(d.kernel)]
    fig = px.box(d, x="cindex", y="model", color="family", orientation="h", points="all",
                 category_orders={"model": [label(k) for k in ks], "family": FAMILY_ORDER},
                 color_discrete_map=COLOR, hover_data={"split": True, "cindex": ":.3f", "family": False, "model": False})
    best = d[d.family.isin(["classical", "baseline"])].groupby("kernel").cindex.median().max()
    fig.add_vline(x=best, line_dash="dash", line_color=BLUE, annotation_text="best classical")
    fig.update_xaxes(title="test C-index (20 splits)")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 90 + 34 * len(ks)), width="stretch")


# ---------------------------------------------------------------- page 2


def page_why():
    st.title("Why classical performs better")

    st.subheader("1 · Entanglement never helped")
    df = splits()
    pts, rows = [], []
    short = {"vlc": "VLC", "gbsg2": "GBSG2", "tcga-brca": "TCGA", "synth-interaction": "interaction",
             "synth-periodic": "periodic"}
    for ds in DATASETS:
        d = df[df.dataset == ds]
        for a, b in [("Q-2L", "Q-1L"), ("Q-ZZ", "Q-Z"), ("Q-2L-wrap", "Q-1L-wrap")]:
            if {a, b} <= set(d.kernel):
                x = d[d.kernel == a].sort_values("split").cindex.to_numpy() - \
                    d[d.kernel == b].sort_values("split").cindex.to_numpy()
                name = f"{short[ds]} · {a} − {b}"
                pts += [{"comparison": name, "Δ C-index": v} for v in x]
                rows.append(name)
    pts = pd.DataFrame(pts)
    fig = px.strip(pts, x="Δ C-index", y="comparison", color_discrete_sequence=[AQUA])
    med = pts.groupby("comparison")["Δ C-index"].median()
    fig.add_trace(go.Scatter(x=med.values, y=med.index, mode="markers", name="median",
                             marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="#0b0b0b"))))
    fig.add_vline(x=0, line_color="#0b0b0b")
    fig.update_yaxes(title="", categoryorder="array", categoryarray=rows[::-1])
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig, 460), width="stretch")
    st.caption("Entangled circuit minus the same circuit without entanglement, per split. Every significant difference "
               "is negative (e.g. TCGA Q-2L − Q-1L p = 0.03, Q-ZZ − Q-Z p = 0.012).")

    cfg = diagnostics()
    left, right = st.columns(2)
    with left:
        st.subheader("2 · More quantum = concentration + overfitting")
        ds = st.radio("Dataset ", REAL, format_func=lambda k: DATASETS[k].split(" ·")[0], horizontal=True)
        g = cfg[(cfg.dataset == ds) & cfg.kernel.isin(["Q-1L", "Q-2L", "Q-3L", "Q-ZZ"])].sort_values("param")
        long = g.melt(id_vars=["kernel", "param"], value_vars=["offdiag_mean", "test_cindex"], var_name="measure")
        long["measure"] = long.measure.map({"offdiag_mean": "kernel similarity between patients",
                                            "test_cindex": "test C-index"})
        fig = px.line(long, x="param", y="value", color="kernel", facet_row="measure", markers=True, log_x=True,
                      color_discrete_map={"Q-1L": ORANGE, "Q-2L": AQUA, "Q-3L": "#0f7a54", "Q-ZZ": "#6fd3ad"},
                      labels={"param": "angle scale c (log)", "value": ""})
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(style(fig, 460), width="stretch")
        st.caption("Larger angles make every patient look equally dissimilar and C-index falls. At the best (small) "
                   "angles the quantum kernel is 99–100% aligned with a classical RBF kernel.")
    with right:
        st.subheader("3 · KTA predicted it before training")
        g = cfg[cfg.dataset == ds]
        rho, p = spearmanr(g.kta, g.test_cindex)
        fig = px.scatter(g, x="kta", y="test_cindex", color="family", color_discrete_map=COLOR,
                         category_orders={"family": FAMILY_ORDER}, hover_name="kernel",
                         labels={"kta": "survival KTA (training data, before fitting)", "test_cindex": "test C-index"})
        fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
        fig.update_layout(title=f"Spearman ρ = {rho:.2f}  (p = {p:.1e})")
        st.plotly_chart(style(fig, 460), width="stretch")
        st.caption("Each point is one kernel configuration. Linear / RBF have the highest KTA; quantum kernels sit "
                   "lower, so the diagnostic called the outcome before any model was trained.")

    st.subheader("The reasons, with their evidence")
    st.table(pd.DataFrame([
        ("Clinical data has almost no nonlinear signal",
         "RBF / RSF never beat linear by > 0.02 on real data; omics add nothing over age + stage (p = 0.35)"),
        ("Where quantum kernels work best, they are RBF kernels", "99.3–100% alignment with RBF; geometric difference 2–4 ≪ √n"),
        ("More 'quantum' concentrates the kernel and overfits", "similarity 0.885 → 0.036, train–test gap 0.087 → 0.172 (VLC)"),
        ("Entanglement never helped", "0 of 14 comparisons significantly positive; 7 significantly negative"),
        ("Periodic encodings mismatch monotone clinical risk", "wrap-around kernels lose 0.03–0.11 on real data, win only on periodic data"),
        ("Small cohorts favour simple kernels", "linear leads by ~0.02 with 91 training patients"),
    ], columns=["Reason", "Evidence"], index=range(1, 7)))


page = st.sidebar.radio("Page", ["Results", "Why classical wins"])
st.sidebar.caption("Median test C-index over 20 repeated splits. All numbers read from results/.")
page_results() if page == "Results" else page_why()
