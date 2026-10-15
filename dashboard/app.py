"""Results dashboard for the Quantum Kernel Survival Model (QKSM) project.

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

# ---------------------------------------------------------------- vocabulary

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
FAMILY = {
    "linear": "classical", "rbf": "classical", "clinical": "classical", "rbf-narrow": "classical",
    "Q-1L": "quantum, no entanglement", "Q-Z": "quantum, no entanglement", "Q-1L-wrap": "quantum, no entanglement",
    "Q-2L": "quantum, entangled", "Q-3L": "quantum, entangled", "Q-ZZ": "quantum, entangled",
    "Q-2L-wrap": "quantum, entangled", "Q-ZZ-wrap": "quantum, entangled",
    "P-Q-2L": "projected quantum", "H-Q-2L": "hybrid", "H-Q-ZZ": "hybrid",
    "cox-lasso": "baseline", "rsf": "baseline", "cox-lasso-all-genes": "baseline", "cox-lasso-clinical-only": "baseline",
}
COLOR = {"classical": BLUE, "quantum, no entanglement": ORANGE, "quantum, entangled": AQUA,
         "projected quantum": GRAY, "hybrid": GRAY, "baseline": GRAY}
LABEL = {"linear": "linear", "rbf": "RBF", "rbf-narrow": "RBF (narrow)", "clinical": "clinical",
         "P-Q-2L": "projected (P-Q-2L)", "cox-lasso": "Cox-LASSO", "rsf": "Random Survival Forest",
         "cox-lasso-all-genes": "Cox-LASSO, all genes", "cox-lasso-clinical-only": "Cox-LASSO, clinical only"}
ORDER = ["linear", "rbf", "rbf-narrow", "clinical", "Q-1L", "Q-Z", "Q-2L", "Q-3L", "Q-ZZ", "P-Q-2L",
         "Q-1L-wrap", "Q-2L-wrap", "Q-ZZ-wrap", "cox-lasso", "rsf", "cox-lasso-clinical-only", "cox-lasso-all-genes"]
DATASETS = {"vlc": "VLC: veterans' lung cancer (n=137)", "gbsg2": "GBSG2: German breast cancer (n=686)",
            "tcga-brca": "TCGA-BRCA multi-omics (n=1063)", "synth-linear": "Synthetic: linear risk",
            "synth-interaction": "Synthetic: interaction risk", "synth-smooth-periodic": "Synthetic: smooth periodic",
            "synth-periodic": "Synthetic: high-frequency periodic"}
REAL = ["vlc", "gbsg2", "tcga-brca"]
FAMILY_ORDER = ["classical", "quantum, no entanglement", "quantum, entangled", "projected quantum", "baseline"]


def label(k):
    return LABEL.get(k, k)


def style(fig, height=None):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="",
                      font=dict(size=13), height=height)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))   # "dataset=vlc" -> "vlc"
    return fig


# ---------------------------------------------------------------- data


@st.cache_data
def splits():
    parts = [pd.read_csv(R / f) for f in
             ["step1_splits.csv", "step4_splits.csv", "step5_splits.csv", "step7_tcga_splits.csv"]]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["dataset", "kernel", "form", "split"], keep="last")
    df["family"] = df.kernel.map(FAMILY).fillna("classical")
    df["model"] = df.kernel.map(label)
    return df


@st.cache_data
def csv(name):
    return pd.read_csv(R / name)


@st.cache_data
def diag_configs():
    c3 = csv("step3_configs.csv")
    c7 = (csv("step7_tcga_diagnostics.csv").groupby(["kernel", "param"], dropna=False)
          .mean(numeric_only=True).reset_index().assign(dataset="tcga-brca"))
    df = pd.concat([c3, c7], ignore_index=True)
    df["family"] = df.kernel.map(FAMILY)
    return df


def paired(df, a, b):
    x = df[df.kernel == a].sort_values("split").cindex.to_numpy()
    y = df[df.kernel == b].sort_values("split").cindex.to_numpy()
    if len(x) != len(y) or len(x) == 0:
        return np.nan, np.nan, np.nan
    d = x - y
    p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    return float(np.median(d)), int((d > 0).sum()), float(p)


def form_view(df, ds):
    d = df[df.dataset == ds]
    forms = [f for f in ["hybrid", "ranking", "regression"] if f in set(d.form)]
    return d, forms


# ---------------------------------------------------------------- pages


def page_overview():
    st.title("Quantum Kernel Methods for Survival Analysis")
    st.caption("Final implementation · Ansari & Tarun · anchor paper: Van Belle et al. (2011), "
               "*Support vector methods for survival analysis*")
    st.markdown(
        "**Result: no genuine quantum advantage.** On all three real cohorts the quantum kernels tie or lose to the best "
        "classical kernel, entanglement never improves the C-index, and the pre-training KTA diagnostic predicted this "
        "before any model was trained.")

    df = splits()
    c = st.columns(5)
    c[0].metric("Datasets", "7", "3 real + 4 synthetic", delta_color="off")
    c[1].metric("Real patients", "1,886", "VLC · GBSG2 · TCGA-BRCA", delta_color="off")
    c[2].metric("Quantum kernel variants", "9", "incl. controls and wrap-around", delta_color="off")
    c[3].metric("KTA → C-index (Spearman ρ)", "0.58–0.82", "before training", delta_color="off")
    c[4].metric("Model fits evaluated", f"{len(df):,}", "test C-index rows, 20 splits each", delta_color="off")

    st.subheader("Best classical vs best quantum (hybrid SVM form)")
    rows = []
    for ds in REAL + ["synth-interaction", "synth-periodic"]:
        d = df[(df.dataset == ds) & df.form.isin(["hybrid", "baseline"])]
        med = d.groupby("kernel").cindex.median()
        cl = med[[k for k in med.index if FAMILY.get(k) in ("classical", "baseline")]]
        qu = med[[k for k in med.index if str(FAMILY.get(k, "")).startswith("quantum")]]
        bc, bq = cl.idxmax(), qu.idxmax()
        dlt, wins, p = paired(d, bq, bc)
        rows.append({"Dataset": DATASETS[ds], "Best classical / baseline": f"{label(bc)} · {cl.max():.3f}",
                     "Best quantum": f"{bq} · {qu.max():.3f}", "Quantum − classical (median, paired)": round(dlt, 3),
                     "Splits quantum wins": f"{wins}/20", "Paired Wilcoxon p": round(p, 4),
                     "Best quantum is entangled?": "yes" if FAMILY[bq] == "quantum, entangled" else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("Every quantum kernel that comes out on top is an unentangled (classical) product kernel.")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Architecture")
        st.image(str(R / "figures" / "architecture.png"), width='stretch')
    with right:
        st.subheader("Objectives scorecard")
        st.dataframe(pd.DataFrame([
            {"Objective": "1 · QKSM: quantum kernels in Survival SVM + Kernel Cox",
             "Delivered": "9 quantum kernels in the SVM (ranking, hybrid, regression) and KPCA-Cox",
             "Outcome": "works end-to-end; no genuine quantum improvement"},
            {"Objective": "2 · Survival KTA pre-training diagnostic",
             "Delivered": "comparable-pairs KTA, event-only check, permutation test, health check",
             "Outcome": "predicts test C-index (ρ 0.58–0.82); fails as a tuner on non-monotone data"},
            {"Objective": "3 · Benchmark vs Cox-LASSO, RSF, Survival SVM",
             "Delivered": "VLC, GBSG2, TCGA-BRCA + 4 synthetic sets, 20 splits, paired Wilcoxon",
             "Outcome": "classical ≥ quantum on all real data"},
        ]), hide_index=True, width='stretch')
        st.subheader("How to read the colours")
        for fam in FAMILY_ORDER:
            st.markdown(f"<span style='color:{COLOR[fam]};font-size:20px'>■</span> {fam}", unsafe_allow_html=True)


def page_reproduction():
    st.title("Paper reproduction (Step 1)")
    st.markdown("Our classical Survival SVM against the numbers published by Van Belle et al. (2011). "
                "Regression and hybrid forms match within +0.01 to +0.03, so the pipeline is trusted.")
    s = csv("step1_summary.csv")
    s = s[s.paper.notna()].copy()
    s["ours"] = s.cindex.str.split(" ").str[0].astype(float)
    s["row"] = s.dataset.str.upper() + " · " + s.kernel + " · " + s.form
    s = s.sort_values(["dataset", "kernel", "form"])
    fig = go.Figure()
    for _, r in s.iterrows():
        fig.add_shape(type="line", x0=r.paper, x1=r.ours, y0=r.row, y1=r.row, line=dict(color="#d9d8d4", width=3))
    fig.add_trace(go.Scatter(x=s.paper, y=s.row, mode="markers", name="Van Belle et al. (2011)",
                             marker=dict(size=12, color=ORANGE)))
    fig.add_trace(go.Scatter(x=s.ours, y=s.row, mode="markers", name="this project (20 splits)",
                             marker=dict(size=12, color=BLUE)))
    fig.update_xaxes(title="median test C-index")
    st.plotly_chart(style(fig, 520), width='stretch')
    st.info("Ranking is higher than in the paper: scikit-survival compares all comparable pairs, while the paper's "
            "ranking models compare each patient only with its nearest comparable neighbour.")
    st.dataframe(csv("step1_summary.csv").rename(columns={"cindex": "C-index (median ± IQR)", "paper": "paper C-index",
                                                          "logrank": "logrank χ²", "hazard_ratio": "hazard ratio"}),
                 hide_index=True, width='stretch')


def page_benchmark():
    st.title("Benchmark: every model, every dataset")
    df = splits()
    c1, c2 = st.columns([2, 1])
    ds = c1.selectbox("Dataset", list(DATASETS), format_func=DATASETS.get)
    d, forms = form_view(df, ds)
    form = c2.selectbox("Survival SVM form", forms, help="ranking = rank_ratio 1, hybrid = 0.5, regression = 0")
    d = d[d.form.isin([form, "baseline"])]
    ks = [k for k in ORDER if k in set(d.kernel)]
    d = d[d.kernel.isin(ks)]

    fig = px.box(d, x="cindex", y="model", color="family", orientation="h", points="all",
                 category_orders={"model": [label(k) for k in ks], "family": FAMILY_ORDER},
                 color_discrete_map=COLOR, hover_data={"split": True, "cindex": ":.3f"})
    best = d[d.family.isin(["classical", "baseline"])].groupby("kernel").cindex.median().max()
    fig.add_vline(x=best, line_dash="dash", line_color=BLUE, annotation_text="best classical median")
    fig.update_xaxes(title="test C-index over 20 splits")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 80 + 34 * len(ks)), width='stretch')

    st.subheader("Medians and paired tests")
    ref = st.selectbox("Compare every model against", ks,
                       index=ks.index(d[d.family.isin(["classical", "baseline"])].groupby("kernel").cindex.median().idxmax()),
                       format_func=label)
    rows = []
    for k in ks:
        x = d[d.kernel == k].cindex
        dlt, wins, p = paired(d, k, ref) if k != ref else (0.0, None, None)
        rows.append({"Model": label(k), "Family": FAMILY[k], "Median C-index": round(x.median(), 3),
                     "IQR": round(x.quantile(.75) - x.quantile(.25), 3), f"Δ vs {label(ref)}": round(dlt, 3),
                     "Splits won": "–" if wins is None else f"{wins}/20", "Paired Wilcoxon p": p})
    t = pd.DataFrame(rows).sort_values("Median C-index", ascending=False)
    t["Paired Wilcoxon p"] = t["Paired Wilcoxon p"].map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.3g}")
    st.dataframe(t.style.format({"Median C-index": "{:.3f}", "IQR": "{:.3f}", f"Δ vs {label(ref)}": "{:+.3f}"}),
                 hide_index=True, width='stretch')
    st.caption("Splits share training data, so p-values are indicative only.")


def page_entanglement():
    st.title("Entanglement effect")
    st.markdown("Each point is one split: C-index of the **entangled** circuit minus the **same circuit without "
                "entanglement**. Right of 0 would mean entanglement helps. **No comparison is significantly positive.**")
    df = splits()
    comps = [("Q-2L", "Q-1L", "add CNOT entanglement"), ("Q-ZZ", "Q-Z", "add ZZ entanglement"),
             ("Q-2L-wrap", "Q-1L-wrap", "add CNOTs at large angles")]
    rows, pts = [], []
    for ds in ["vlc", "gbsg2", "tcga-brca", "synth-interaction", "synth-periodic"]:
        d = df[(df.dataset == ds) & df.form.isin(["hybrid"])]
        for a, b, what in comps:
            if not {a, b} <= set(d.kernel):
                continue
            x = d[d.kernel == a].sort_values("split").cindex.to_numpy() - d[d.kernel == b].sort_values("split").cindex.to_numpy()
            dlt, wins, p = paired(d, a, b)
            short = {"vlc": "VLC", "gbsg2": "GBSG2", "tcga-brca": "TCGA-BRCA",
                     "synth-interaction": "synth interaction", "synth-periodic": "synth periodic"}[ds]
            name = f"{short} · {a} − {b}"
            rows.append({"Comparison": name, "What changes": what, "Median Δ C-index": round(dlt, 3),
                         "Splits entangled wins": f"{wins}/20", "Paired Wilcoxon p": f"{p:.2g}",
                         "Verdict": "significantly worse" if p < 0.05 and dlt < 0 else
                         ("significantly better" if p < 0.05 and dlt > 0 else "no difference")})
            pts += [{"Comparison": name, "Δ C-index": v, "What changes": what} for v in x]
    pts = pd.DataFrame(pts)
    fig = px.strip(pts, x="Δ C-index", y="Comparison", color="What changes",
                   color_discrete_sequence=[AQUA, ORANGE, GRAY], stripmode="overlay")
    med = pts.groupby("Comparison")["Δ C-index"].median().reset_index()
    fig.add_trace(go.Scatter(x=med["Δ C-index"], y=med.Comparison, mode="markers", name="median",
                             marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="#0b0b0b"))))
    fig.add_vline(x=0, line_color="#0b0b0b")
    fig.update_yaxes(title="", categoryorder="array", categoryarray=list(reversed([r["Comparison"] for r in rows])))
    st.plotly_chart(style(fig, 560), width='stretch')
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


def page_kta():
    st.title("KTA: the pre-training diagnostic (Objective 2)")
    st.markdown("Survival KTA is computed on the **training data only, before any model is fitted**. "
                "Each point is one kernel configuration (kernel × angle scale or bandwidth).")
    cfg = diag_configs()
    ds = st.radio("Dataset", REAL, format_func=lambda k: DATASETS[k].split(" (")[0], horizontal=True)
    g = cfg[cfg.dataset == ds].copy()
    g["config"] = g.kernel + g.param.map(lambda p: "" if pd.isna(p) else f" · {p:.3g}")
    rho, p = spearmanr(g.kta, g.test_cindex)
    rho_e, _ = spearmanr(g.kta_event_only, g.test_cindex)
    c = st.columns(3)
    c[0].metric("Spearman ρ (KTA vs test C-index)", f"{rho:.2f}", f"p = {p:.1e}", delta_color="off")
    c[1].metric("Event-only cross-check ρ", f"{rho_e:.2f}", "same ranking of kernels", delta_color="off")
    c[2].metric("Configurations", f"{len(g)}", "kernel × parameter", delta_color="off")
    fig = px.scatter(g, x="kta", y="test_cindex", color="family", symbol="family", hover_name="config",
                     color_discrete_map=COLOR, category_orders={"family": FAMILY_ORDER},
                     hover_data={"kta": ":.3f", "test_cindex": ":.3f", "family": False})
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.update_xaxes(title="survival KTA (training data, before fitting)")
    fig.update_yaxes(title="test C-index (after fitting)")
    st.plotly_chart(style(fig, 480), width='stretch')

    l, r = st.columns(2)
    with l:
        st.subheader("Best KTA per kernel")
        b = g.loc[g.groupby("kernel").kta.idxmax(), ["kernel", "param", "kta", "test_cindex"]].copy()
        b["param"] = b.param.map(lambda p: "–" if pd.isna(p) else f"{p:.3g}")
        st.dataframe(b.sort_values("kta", ascending=False).round(3), hide_index=True, width='stretch')
        st.caption("Linear / RBF have the highest KTA, and quantum kernels sit lower: the diagnostic called the "
                   "outcome before training.")
    with r:
        st.subheader("Best quantum kernels ≈ RBF (R4)")
        sc = csv("step3_small_c.csv").groupby(["dataset", "kernel", "c"]).mean(numeric_only=True).reset_index()
        st.dataframe(sc[["dataset", "kernel", "c", "align_matched_rbf", "kta", "test_cindex"]]
                     .rename(columns={"align_matched_rbf": "alignment with RBF on the same angles"}).round(4),
                     hide_index=True, width='stretch')
        st.caption("cos²(t/2) ≈ exp(−t²/4) for small angles, so the best-performing quantum kernel is an RBF kernel.")
    st.warning("Limitation: on the synthetic interaction data, KTA rises with the angle scale while test C-index falls, "
               "so KTA alone picks the worst c. Use KTA to screen kernel families and CV to tune.")
    k = csv("step6_a7_kta_choice.csv").melt(id_vars=["dataset", "c"], value_vars=["kta", "test"], var_name="measure")
    k["measure"] = k.measure.map({"kta": "survival KTA (training)", "test": "test C-index"})
    fig = px.line(k, x="c", y="value", facet_col="dataset", facet_row="measure", markers=True, log_x=True,
                  color="measure", color_discrete_map={"survival KTA (training)": ORANGE, "test C-index": BLUE},
                  labels={"value": "", "c": "angle scale c (Q-2L)"})
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig, 460), width='stretch')


def page_sweep():
    st.title("Angle sweep: why 'more quantum' is worse (R1–R4)")
    cfg = diag_configs()
    metrics = {"offdiag_mean": "mean off-diagonal similarity (R1)", "eff_rank_frac": "effective rank / n (R2)",
               "gap": "train − test C-index gap (R2)", "test_cindex": "test C-index", "kta": "survival KTA",
               "align_rbf": "alignment with RBF (R4)", "g_vs_rbf": "geometric difference vs RBF (R3)"}
    c1, c2 = st.columns([1, 2])
    ds = c1.radio("Dataset", REAL, format_func=lambda k: DATASETS[k].split(" (")[0])
    chosen = c2.multiselect("Metrics", list(metrics), default=["offdiag_mean", "gap", "test_cindex"],
                            format_func=metrics.get)
    g = cfg[(cfg.dataset == ds) & cfg.kernel.isin(["Q-1L", "Q-Z", "Q-2L", "Q-3L", "Q-ZZ", "P-Q-2L"])]
    lin = cfg[(cfg.dataset == ds) & (cfg.kernel == "linear")]
    for m in chosen:
        fig = px.line(g.sort_values("param"), x="param", y=m, color="kernel", markers=True, log_x=True,
                      color_discrete_map={"Q-1L": ORANGE, "Q-Z": "#f2a07f", "Q-2L": AQUA, "Q-3L": "#0f7a54",
                                          "Q-ZZ": "#6fd3ad", "P-Q-2L": GRAY})
        if m in ("gap", "test_cindex", "kta") and len(lin):
            fig.add_hline(y=float(lin[m].iloc[0]), line_dash="dash", line_color=BLUE, annotation_text="linear kernel")
        fig.update_layout(title=metrics[m])
        fig.update_xaxes(title="angle scale c (log)")
        fig.update_yaxes(title="")
        st.plotly_chart(style(fig, 330), width='stretch')
    st.caption("As c grows, kernels concentrate (patients look equally dissimilar), the effective rank and the "
               "train–test gap grow, and test C-index falls. The geometric difference vs RBF stays far below √n "
               "(VLC ≈ 9.5, GBSG2 ≈ 21, TCGA ≈ 27) at the best settings.")


def page_synthetic():
    st.title("Synthetic dial and wrap-around kernels (R5, R7, A5b)")
    st.markdown("Synthetic datasets with a **known** risk shape show when each kernel can win. "
                "Wrap-around kernels (c from 1 to 8) make similarity periodic.")
    df = splits()
    d = df[df.dataset.str.startswith("synth") & df.form.isin(["hybrid", "baseline"])]
    piv = d.groupby(["kernel", "dataset"]).cindex.median().unstack()
    piv = piv.reindex([k for k in ORDER if k in piv.index])[[c for c in DATASETS if c in piv.columns]]
    fig = px.imshow(piv.values, x=[DATASETS[c].replace("Synthetic: ", "") for c in piv.columns],
                    y=[label(k) for k in piv.index], color_continuous_scale="Blues", text_auto=".3f", aspect="auto",
                    labels=dict(color="median C-index"))
    st.plotly_chart(style(fig, 620), width='stretch')
    st.markdown("- **Interaction risk:** linear models are at chance (≈ 0.50); RBF and quantum kernels reach ≈ 0.77, "
                "so quantum only ties RBF.\n- **High-frequency periodic risk:** the unentangled `Q-1L-wrap` wins "
                "(0.716 vs 0.686 for the best classical model), but it is a classical periodic kernel; adding "
                "entanglement makes it worse.")
    real = df[df.dataset.isin(["vlc", "gbsg2"]) & (df.form == "hybrid") &
              df.kernel.isin(["linear", "clinical", "Q-1L", "Q-1L-wrap", "Q-2L-wrap", "Q-ZZ-wrap"])]
    fig = px.box(real, x="model", y="cindex", color="family", facet_col="dataset", color_discrete_map=COLOR,
                 category_orders={"family": FAMILY_ORDER})
    fig.update_layout(title="The same wrap-around kernels on real clinical data")
    fig.update_xaxes(title="")
    st.plotly_chart(style(fig, 420), width='stretch')
    st.caption("Periodic similarity contradicts clinical risk that rises steadily with each covariate.")


def page_extra():
    st.title("Hybrid kernel, KPCA-Cox and learning curves")
    a7 = csv("step6_a7_splits.csv")
    st.subheader("A7 · hybrid kernel w·K_quantum + (1 − w)·K_linear")
    t = a7.groupby(["dataset", "kernel"]).agg(median_C=("cindex", "median"), median_w=("param", "median")).reset_index()
    l, r = st.columns(2)
    l.dataframe(t.round(3), hide_index=True, width='stretch')
    fig = px.box(a7, x="dataset", y="param", color="kernel", points="all",
                 color_discrete_map={"H-Q-2L": AQUA, "H-Q-ZZ": "#6fd3ad"},
                 labels={"param": "chosen w (0 = linear, 1 = quantum)", "dataset": ""})
    r.plotly_chart(style(fig, 340), width='stretch')
    st.caption("On VLC cross-validation puts about 80% of the weight on the linear part: the quantum kernel adds no "
               "complementary information.")

    st.subheader("KPCA-Cox (Kernel Cox version of the QKSM)")
    kp = csv("step6_kpca_splits.csv")
    kp["family"] = kp.kernel.map(FAMILY)
    fig = px.box(kp, x="kernel", y="cindex", color="family", facet_col="dataset", color_discrete_map=COLOR,
                 category_orders={"kernel": ["linear", "rbf", "clinical", "Q-1L", "Q-2L", "Q-ZZ"],
                                  "family": FAMILY_ORDER})
    fig.update_xaxes(title="")
    fig.update_yaxes(matches=None, showticklabels=True)
    st.plotly_chart(style(fig, 400), width='stretch')

    st.subheader("R6 · learning curves (hybrid SVM, 10 splits)")
    lc = csv("step6_learning_curves.csv").groupby(["dataset", "kernel", "n_train"]).cindex.median().reset_index()
    fig = px.line(lc, x="n_train", y="cindex", color="kernel", facet_col="dataset", markers=True,
                  color_discrete_map={"linear": BLUE, "rbf": "#86b6ef", "Q-1L": ORANGE, "Q-2L": AQUA, "Q-ZZ": "#6fd3ad"},
                  labels={"n_train": "training patients", "cindex": "median test C-index"})
    fig.update_xaxes(matches=None)
    st.plotly_chart(style(fig, 380), width='stretch')
    st.caption("With 91 patients the linear kernel leads by about 0.02; quantum kernels only catch up at full size.")


def page_tcga():
    st.title("TCGA-BRCA multi-omics")
    st.markdown("1063 primary breast tumours, 148 deaths (86% censored). Per split, mRNA and copy number are reduced "
                "inside the training part to 3 + 3 principal components, plus age and stage = **8 features = 8 qubits**, "
                "the same for every kernel model.")
    df = splits()
    d = df[df.dataset == "tcga-brca"]
    c = st.columns(4)
    c[0].metric("Patients", "1,063")
    c[1].metric("Deaths", "148", "86% censored", delta_color="off")
    c[2].metric("Best model", "Q-1L / Q-Z · 0.766", "+0.006 vs linear, p = 0.048", delta_color="off")
    c[3].metric("Omics vs clinical only", "p = 0.35", "no significant gain", delta_color="off")
    ks = [k for k in ORDER if k in set(d.kernel)]
    fig = px.box(d, x="cindex", y="model", color="family", orientation="h", points="all",
                 category_orders={"model": [label(k) for k in ks], "family": FAMILY_ORDER}, color_discrete_map=COLOR)
    fig.update_xaxes(title="test C-index over 20 splits")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 520), width='stretch')
    rows = []
    for a, b in [("Q-1L", "linear"), ("Q-Z", "linear"), ("Q-2L", "linear"), ("Q-ZZ", "linear"), ("rbf", "linear"),
                 ("Q-2L", "Q-1L"), ("Q-ZZ", "Q-Z"), ("cox-lasso", "cox-lasso-clinical-only"),
                 ("cox-lasso", "cox-lasso-all-genes"), ("linear", "rsf")]:
        dlt, wins, p = paired(d, a, b)
        rows.append({"Comparison": f"{label(a)} vs {label(b)}", "Median Δ": round(dlt, 3), "Splits won": f"{wins}/20",
                     "Paired Wilcoxon p": f"{p:.2g}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("The edge of Q-1L over linear (p = 0.048) does not survive correction for ~10 comparisons, and Q-1L is "
               "an unentangled, classical kernel. Using all ~4000 genes overfits.")


def page_reasons():
    st.title("Why classical performs better")
    st.markdown("Each reason is backed by a measurement, not an assumption.")
    st.table(pd.DataFrame([
        {"#": 1, "Reason": "The clinical data has almost no nonlinear signal",
         "Evidence": "RBF / RSF never beat linear by > 0.02 on VLC, GBSG2, TCGA; omics add nothing over age + stage "
                     "(p = 0.35); with real interactions (synthetic) nonlinear kernels gain +0.27"},
        {"#": 2, "Reason": "Where quantum kernels work best, they are RBF kernels",
         "Evidence": "99.3–100% alignment with RBF on the same angles; geometric difference vs RBF 2–4 ≪ √n"},
        {"#": 3, "Reason": "Making the circuit 'more quantum' concentrates the kernel and overfits",
         "Evidence": "off-diagonal similarity 0.885 → 0.036, train–test gap 0.087 → 0.172, C-index 0.698 → 0.623 (VLC Q-2L)"},
        {"#": 4, "Reason": "Entanglement never helped",
         "Evidence": "Q-ZZ = Q-Z (p = 0.90); entangled versions significantly worse on TCGA, interaction and periodic data"},
        {"#": 5, "Reason": "Periodic angle encodings mismatch monotone clinical risk",
         "Evidence": "wrap-around kernels lose 0.03–0.11 on real data, win +0.03 only on truly periodic data"},
        {"#": 6, "Reason": "Small cohorts favour simple kernels",
         "Evidence": "linear leads by ~0.02 with 91 training patients; all-genes Cox-LASSO overfits on TCGA"},
        {"#": 7, "Reason": "The pre-training diagnostic saw it coming",
         "Evidence": "KTA ranks kernels like test C-index (ρ 0.82 / 0.73 / 0.58); quantum KTA below linear / RBF"},
    ]).set_index("#"))
    st.success("When could a quantum kernel help? Only if the true risk is periodic in the features, and even then "
               "the winner here was the unentangled product kernel, which is classically computable.")


PAGES = {
    "Overview": page_overview, "Paper reproduction": page_reproduction, "Benchmark": page_benchmark,
    "Entanglement effect": page_entanglement, "KTA diagnostic": page_kta, "Angle sweep": page_sweep,
    "Synthetic and wrap-around": page_synthetic, "Hybrid, KPCA-Cox, learning curves": page_extra,
    "TCGA-BRCA": page_tcga, "Why classical wins": page_reasons,
}
with st.sidebar:
    st.header("QKSM results")
    choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.caption("All numbers are read from results/. Median test C-index over 20 repeated 2/3–1/3 splits unless stated.")
PAGES[choice]()
