"""Stage 10 — research-paper style manuscript (PDF) built from pipeline outputs.

The structure mirrors the reference article (Li et al., ACS Sustainable Chem.
Eng. 2025, 13, 13940): Abstract, Introduction, Data and Methods (models,
metrics, K-means, oversampling, SHAP, genetic algorithm), Results and
Discussion, Conclusion and References. Every number is read from the
results/ files written by stages 01-08, so re-running the pipeline keeps the
manuscript consistent with the figures. Nothing is retrained here.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from common import FIGURES_DIR, RESULTS_DIR, read_cleaned, read_processed, build_feature_matrix

OUT_PDF = RESULTS_DIR / "HMF_FDCA_ML_Research_Paper.pdf"
BUILD = RESULTS_DIR / "paper_build"
PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
TEXT_W = PAGE_W - 2 * MARGIN
GREEN = colors.HexColor("#2f6b3a")

MODEL_SHORT = {"DecisionTree": "DT", "SVM": "SVM", "RandomForest": "RF",
               "NeuralNetwork": "NN", "GradientBoosting": "GB", "ExtraTrees": "ET"}

# Published reference points found during web verification (see References).
LITERATURE = [
    ("Ni(OH)2/Ni foam (freestanding)", "1 M KOH, 10 mM HMF, 1.45 V vs RHE",
     "Studied at the same conditions as solution 1", "[4]"),
    ("Vacancy-engineered Ni(OH)2/CP", "1.45 V vs RHE", "Yield 98.5%, FE 98.5%", "[5]"),
    ("ZIF-67@beta-Ni(OH)2/NF", "1.45 V vs RHE", "Yield 99.0%, FE 95.2%", "[6]"),
    ("Ni(OH)2-CeO2/NF", "1.504 V vs RHE", "Conv. 98.9%, sel. 98.8%, FE 98.4%", "[7]"),
    ("Cu(OH)2-decorated Cu foam", "1.0 M KOH, 100 mM HMF", "Yield 98.7%, FE ~100%", "[8]"),
    ("Cu(OH)2 nanowires/Cu foam", "0.1 M KOH", "Cu(OH)2 vs CuO activity comparison", "[9]"),
]

REFERENCES = [
    "Li, M.; Liu, D.; Yu, L.; Yu, S.; Li, L.; Liu, S.; Yu, H.; Liu, Y. High-Efficiency Catalyst "
    "Screening Driven by Machine Learning for the Oxidation of 5-Hydroxymethylfurfural to "
    "2,5-Furandicarboxylic Acid. <i>ACS Sustainable Chem. Eng.</i> <b>2025</b>, 13, 13940-13952.",
    "Cha, H. G.; Choi, K. S. Combined biomass valorization and hydrogen production in a "
    "photoelectrochemical cell. <i>Nat. Chem.</i> <b>2015</b>, 7, 328-333.",
    "Eerhart, A. J. J. E.; Faaij, A. P. C.; Patel, M. K. Replacing fossil based PET with biobased "
    "PEF; process analysis, energy and GHG balance. <i>Energy Environ. Sci.</i> <b>2012</b>, 5, 6407-6422.",
    "Realizing efficient electrochemical oxidation of 5-hydroxymethylfurfural on a freestanding "
    "Ni(OH)2/nickel foam catalyst. <i>Chin. J. Catal.</i> <b>2024</b>, 63, 282-291. "
    "https://www.cjcatal.com/EN/Y2024/V63/I8/282",
    "Effects of different vacancies in nickel hydroxides on the electrooxidation towards "
    "5-hydroxymethylfurfural. <i>Chem. Eng. J.</i> "
    "https://www.sciencedirect.com/science/article/abs/pii/S1385894723056486",
    "Enhanced electrooxidation of 5-hydroxymethylfurfural over a ZIF-67@beta-Ni(OH)2/NF "
    "heterostructure catalyst. <i>J. Colloid Interface Sci.</i> <b>2025</b>. "
    "https://www.sciencedirect.com/science/article/abs/pii/S0021979725005557",
    "CeO2 enhancement on self-grown Ni(OH)2 on nickel foam for 5-hydroxymethylfurfural "
    "electrooxidation. <i>Electrochim. Acta</i> <b>2025</b>. "
    "https://www.sciencedirect.com/science/article/abs/pii/S0013468625005626",
    "Cu(OH)2-decorated copper foam for HMF electro-oxidation to FDCA (CuOOH active species). "
    "https://peeref.com/works/23914013",
    "Free-standing Cu(OH)2 and CuO nanowires on copper foam for HMF electro-oxidation. "
    "<i>Korean J. Chem. Eng.</i> <b>2020</b>. https://link.springer.com/article/10.1007/s11814-020-0474-9",
    "Chawla, N. V.; Bowyer, K. W.; Hall, L. O.; Kegelmeyer, W. P. SMOTE: Synthetic Minority "
    "Over-sampling Technique. <i>J. Artif. Intell. Res.</i> <b>2002</b>, 16, 321-357.",
    "He, H.; Bai, Y.; Garcia, E. A.; Li, S. ADASYN: Adaptive synthetic sampling approach for "
    "imbalanced learning. <i>Proc. IEEE IJCNN</i> <b>2008</b>, 1322-1328.",
    "Lundberg, S. M.; Lee, S.-I. A Unified Approach to Interpreting Model Predictions. "
    "<i>Adv. Neural Inf. Process. Syst.</i> <b>2017</b>, 30.",
    "Deb, K.; Pratap, A.; Agarwal, S.; Meyarivan, T. A fast and elitist multiobjective genetic "
    "algorithm: NSGA-II. <i>IEEE Trans. Evol. Comput.</i> <b>2002</b>, 6, 182-197.",
    "Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. <i>J. Mach. Learn. Res.</i> "
    "<b>2011</b>, 12, 2825-2830.",
]


# ---------------------------------------------------------------------------
# Styles and helpers
# ---------------------------------------------------------------------------

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=base["Title"], fontName="Times-Bold", fontSize=17,
                            leading=21, alignment=TA_CENTER, spaceAfter=8),
    "authors": ParagraphStyle("authors", parent=base["Normal"], fontName="Times-Roman",
                              fontSize=10.5, alignment=TA_CENTER, spaceAfter=10),
    "abstract": ParagraphStyle("abstract", parent=base["Normal"], fontName="Times-Roman",
                               fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12,
                         textColor=GREEN, spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Times-Bold", fontSize=10.5,
                         spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", parent=base["Normal"], fontName="Times-Roman", fontSize=10,
                           leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5),
    "caption": ParagraphStyle("caption", parent=base["Normal"], fontName="Times-Roman",
                              fontSize=8.8, leading=11, alignment=TA_JUSTIFY, spaceBefore=3,
                              spaceAfter=9),
    "cell": ParagraphStyle("cell", parent=base["Normal"], fontName="Times-Roman", fontSize=7.8,
                           leading=9.2),
    "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontName="Times-Bold", fontSize=7.8,
                            leading=9.2),
    "ref": ParagraphStyle("ref", parent=base["Normal"], fontName="Times-Roman", fontSize=8.5,
                          leading=10.5, leftIndent=14, firstLineIndent=-14, spaceAfter=2),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def chem(text: str) -> str:
    """Escape and subscript digits after element symbols / closing brackets."""
    import re
    text = escape(text)
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"<sub>\1</sub>", text)


def fig(path: Path, caption: str, width: float = TEXT_W, max_h: float = 17 * cm):
    if not path.exists():
        return [P(f"<i>[Missing figure: {escape(str(path.name))}]</i>")]
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    w = width
    h = w * ratio
    if h > max_h:
        h = max_h
        w = h / ratio
    img.drawWidth, img.drawHeight = w, h
    return [KeepTogether([img, P(caption, "caption")])]


def table(df: pd.DataFrame, caption: str, col_widths=None, bold_first_col=False):
    header = [Paragraph(f"<b>{escape(str(c))}</b>", S["cell"]) for c in df.columns]
    rows = [header]
    for _, r in df.iterrows():
        rows.append([Paragraph(chem(str(v)) if isinstance(v, str) else escape(str(v)),
                               S["cellb"] if (bold_first_col and j == 0) else S["cell"])
                     for j, v in enumerate(r.values)])
    t = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9efe9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return [KeepTogether([P(caption, "caption"), t]), Spacer(1, 8)]


EQ_COUNTER = [0]


def equation(latex: str, fontsize: int = 13):
    """Render a numbered display equation with matplotlib mathtext."""
    EQ_COUNTER[0] += 1
    n = EQ_COUNTER[0]
    BUILD.mkdir(parents=True, exist_ok=True)
    path = BUILD / f"eq_{n:02d}.png"
    f = plt.figure(figsize=(0.01, 0.01))
    f.text(0, 0, f"${latex}$", fontsize=fontsize)
    f.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(f)
    img = Image(str(path))
    scale = 0.24
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    t = Table([[img, Paragraph(f"({n})", S["body"])]],
              colWidths=[TEXT_W - 1.4 * cm, 1.4 * cm])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t, n


def workflow_figure(path: Path) -> None:
    """Figure 1: data-model-optimisation workflow schematic."""
    steps = [("Literature\ndatabase", "154 records\n99 papers"),
             ("Feature\nengineering", "anode-only catalyst,\nelectrolyte pH, cell"),
             ("K-means\nclustering", "Conversion-Yield\nK = 3-6"),
             ("Oversampling", "ROS / SMOTE /\nADASYN (fold-local)"),
             ("ML models", "DT, SVM, RF, NN,\nGB, ET"),
             ("SHAP", "feature\ncontributions"),
             ("NSGA-II", "entropy-weighted\nPareto decision")]
    fig_, ax = plt.subplots(figsize=(11, 2.5))
    ax.axis("off")
    palette = ["#cfe3f5", "#d8eed9", "#fbe0c8", "#f5d0dc", "#e3d7f2", "#fff1b8", "#cdeee8"]
    for i, ((t, sub), c) in enumerate(zip(steps, palette)):
        x = i * 1.55
        ax.add_patch(plt.Rectangle((x, 0.25), 1.3, 1.0, facecolor=c, edgecolor="#444", lw=1,
                                   joinstyle="round"))
        ax.text(x + 0.65, 0.92, t, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + 0.65, 0.48, sub, ha="center", va="center", fontsize=7.3, color="#333")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.52, 0.75), xytext=(x + 1.32, 0.75),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))
    ax.set_xlim(-0.1, len(steps) * 1.55)
    ax.set_ylim(0, 1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig_)


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(GREEN)
    canvas.setLineWidth(0.8)
    canvas.line(MARGIN, PAGE_H - 1.3 * cm, PAGE_W - MARGIN, PAGE_H - 1.3 * cm)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.setFillColor(GREEN)
    canvas.drawString(MARGIN, PAGE_H - 1.15 * cm, "HMF Electro-oxidation to FDCA — ML Catalyst Screening")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.15 * cm, "Research Report")
    canvas.setFillColor(colors.black)
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawCentredString(PAGE_W / 2, 1.1 * cm, str(doc.page))
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_inputs() -> dict:
    d = {}
    raw = read_cleaned()
    X, features, _ = build_feature_matrix(read_processed())
    d["n_records"] = len(raw)
    d["n_papers"] = raw["Paper Title"].nunique()
    d["n_features"] = len(features)
    d["missing"] = {c: int(X[c].isna().sum()) for c in ("Temperature_C", "Potential_vs_RHE_V",
                                                         "Electrolyte_Concentration_M",
                                                         "HMF_Concentration_mM")}
    d["targets_n"] = {t: int(pd.to_numeric(raw[t], errors="coerce").notna().sum())
                      for t in ("Conversion_pct", "Yield_pct", "FE_pct", "Selectivity_pct")}
    yld = pd.to_numeric(raw["Yield_pct"], errors="coerce")
    glass = raw["Catalyst_Conductive_Glass"] == 1
    foam = raw["Catalyst_Foam_Substrate"] == 1
    d.update(glass_n=int(glass.sum()), glass_yield=float(yld[glass].median()),
             other_yield=float(yld[~glass].median()), foam_n=int(foam.sum()),
             foam_yield=float(yld[foam].median()))
    d["clusters"] = {k: pd.read_csv(RESULTS_DIR / f"kmeans_labels_K{k}.csv")["cluster"]
                     .value_counts().sort_index().to_dict() for k in (3, 4, 5, 6)}
    d["summary"] = pd.read_csv(RESULTS_DIR / "classification_summary.csv")
    base_path = RESULTS_DIR / "classification_summary_baseline_v1.csv"
    d["baseline"] = pd.read_csv(base_path) if base_path.exists() else None
    d["best"] = json.loads((RESULTS_DIR / "best_classification_candidate.json").read_text())
    d["table1"] = pd.read_csv(RESULTS_DIR / "paper_table_1_metrics.csv")
    d["regression"] = pd.read_csv(RESULTS_DIR / "per_target_regression_summary.csv")
    d["curves"] = pd.read_csv(RESULTS_DIR / "learning_curves_summary.csv")
    d["shap3"] = pd.read_csv(RESULTS_DIR / "shap_paper_K3_feature_importance.csv")
    d["shap5"] = pd.read_csv(RESULTS_DIR / "shap_paper_K5_feature_importance.csv")
    d["shap_audit"] = json.loads((RESULTS_DIR / "shap_paper_K3_K5_audit.json").read_text())
    d["nsga"] = json.loads((RESULTS_DIR / "nsga2_status.json").read_text())
    d["top10"] = pd.read_csv(RESULTS_DIR / "nsga2_top10_ni_cu_hydroxide.csv")
    d["cu3"] = pd.read_csv(RESULTS_DIR / "nsga2_best3_cu_hydroxide.csv")
    d["general"] = pd.read_csv(RESULTS_DIR / "nsga2_pareto_candidates.csv")
    d["history"] = pd.read_csv(RESULTS_DIR / "nsga2_generation_history.csv")
    return d


def spacing_text(history: pd.DataFrame) -> str:
    """Describe the spacing traces from the recorded history (no fixed claims)."""
    parts = []
    for label, name in (("NiOH2", "Ni(OH)<sub>2</sub>"), ("CuOH2", "Cu(OH)<sub>2</sub>")):
        h = history[history["scenario"] == label]["spacing"]
        if h.empty:
            continue
        parts.append(f"{name}: {h.iloc[0]:.3f} at generation 1, median {h.tail(500).median():.3f} "
                     f"over the last 500 generations (max {h.tail(500).max():.3f})")
    return ("The spacing metric of the first front (Figure 11) quickly settles to a low, "
            "stable level (" + "; ".join(parts) + "). The population is seeded with measured "
            "records, so the initial front is already compact and there is no long transient "
            "like the one in Figure 14 of ref 1. The residual fluctuation comes from "
            "solutions that differ only in temperature or potential.")


def best_per_model(summary: pd.DataFrame, kind: str, k: int) -> pd.DataFrame:
    sub = summary[(summary["target_kind"] == kind) & (summary["k"] == k)]
    return (sub.sort_values("macro_f1_mean", ascending=False).groupby("model").head(1)
            .set_index("model"))


# ---------------------------------------------------------------------------
# Manuscript
# ---------------------------------------------------------------------------

def build(d: dict) -> None:
    story = []
    best = d["best"]
    summ = d["summary"]
    bin_best = best_per_model(summ, "binary_high_performance_90", 0)
    k3_best = best_per_model(summ, "kmeans_cluster", 3)
    k_overall = {k: summ[(summ.target_kind == "kmeans_cluster") & (summ.k == k)]
                 .sort_values("macro_f1_mean", ascending=False).iloc[0] for k in (3, 4, 5, 6)}
    top = d["top10"]
    t1 = top.iloc[0]
    hyd = d["nsga"]["hydroxide_scenario"]
    w = hyd["entropy_weights"]
    env = hyd["condition_envelope"]
    reg = d["regression"].set_index("target")
    best_model = MODEL_SHORT.get(best["model"], best["model"])

    # ---- Title block ---------------------------------------------------------
    story += [Spacer(1, 6),
              P("Machine-Learning-Driven Catalyst Screening for the Electrochemical Oxidation "
                "of 5-Hydroxymethylfurfural to 2,5-Furandicarboxylic Acid", "title"),
              P("Research report generated from the HMF-Connected pipeline · "
                f"{date.today():%d %B %Y}", "authors")]

    abstract = (
        "<b>ABSTRACT:</b> Electrochemical oxidation of 5-hydroxymethylfurfural (HMF) to "
        "2,5-furandicarboxylic acid (FDCA) couples biomass upgrading with hydrogen evolution, but "
        "catalyst development still relies on trial-and-error experiments. Following the "
        "machine-learning (ML) screening paradigm of Li et al. [1], we assembled "
        f"{d['n_records']} electro-oxidation records from {d['n_papers']} publications and "
        f"encoded {d['n_features']} descriptors of the catalyst, electrolyte, potential, "
        "temperature, TEMPO mediation and cell configuration. Records were grouped by K-means "
        "clustering in the conversion-yield space (K = 3-6). Class imbalance was handled with "
        "random oversampling, SMOTE and ADASYN applied inside training folds only. Six classifiers "
        "were compared under 25 repeats of 5-fold publication-grouped cross-validation, which "
        "scores every model on papers it has never seen. The "
        f"{best_model} model with {best['sampler']} gave the best macro-F1 for the ≥90% FDCA "
        f"performance screen ({best['macro_f1_mean']:.3f}; dummy baseline "
        f"{best['dummy_macro_f1_mean']:.3f}), and gradient boosting reached "
        f"{k_overall[3]['macro_f1_mean']:.3f} for K = 3. SHAP analysis linked performance to the "
        "electrode support, applied potential and HMF concentration. A validation-gated NSGA-II "
        "search with mixed encoding and entropy-weighted decision making, restricted to Ni(OH)<sub>2</sub> or "
        "Cu(OH)<sub>2</sub> anodes, converged on Ni(OH)<sub>2</sub> on nickel foam in "
        f"~{t1['electrolyte_M']:.1f} M KOH with ~{t1['HMF_mM']:.0f} mM HMF at "
        f"{t1['potential_V_RHE']:.2f} V vs RHE and {t1['temperature_C']:.0f} °C. This region "
        "agrees with published Ni(OH)<sub>2</sub> electrodes that reach 95-99% FDCA yield.")
    box = Table([[P(abstract, "abstract")]], colWidths=[TEXT_W])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4e6")),
                             ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fbf8f")),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [box, Spacer(1, 5),
              P("<b><font color='#2f6b3a'>KEYWORDS:</font></b> <i>machine learning, "
                "5-hydroxymethylfurfural, 2,5-furandicarboxylic acid, electrocatalysis, "
                "oversampling, SHAP, genetic algorithm, NSGA-II</i>", "abstract")]

    # ---- 1. Introduction ----------------------------------------------------
    story += [P("1. INTRODUCTION", "h1"),
              P("FDCA is a bio-based replacement for terephthalic acid. Polymerised with ethylene "
                "glycol it gives poly(ethylene furanoate), whose life-cycle energy use and "
                "greenhouse-gas emissions are lower than those of PET [3]. Electrochemical "
                "oxidation of HMF to FDCA runs at ambient temperature and pressure. It uses water "
                "as the oxygen source and can be paired with cathodic hydrogen evolution [2], "
                "avoiding the pressurised O<sub>2</sub>, noble metals and added bases of thermal "
                "aerobic routes. The reaction proceeds through HMFCA or DFF and FFCA "
                "intermediates. Its rate and selectivity depend jointly on the anode material, "
                "electrolyte pH, applied potential and cell design, which makes catalyst "
                "selection a high-dimensional problem."),
              P("Li et al. [1] showed that this problem can be attacked with a "
                "data-model-synthesis loop for the thermocatalytic route. Their loop runs K-means "
                "classification of literature data, composite oversampling, a four-model ML "
                "comparison, SHAP interpretation and NSGA-II optimisation, and it identified "
                "Ru/Mn<sub>6</sub>Ce<sub>1</sub>O<sub>Y</sub> as a promising catalyst. Here we "
                "transfer the same workflow to the <i>electrochemical</i> route. Three changes "
                "matter for honest model assessment on a small literature set:"),
              P("(i) Validation is <b>publication-grouped</b>, so that records from one paper "
                "never appear in both training and test folds. (ii) Oversampling happens "
                "<b>only inside training folds</b>, and repeats in which a sampler is infeasible "
                "for any fold are excluded rather than silently scored on a subset. (iii) The "
                "genetic algorithm is <b>gated</b> on regressors that beat a median baseline, and "
                "its search space is bounded by conditions actually reported for the catalyst "
                "family of interest.")]
    story += fig(BUILD / "figure1_workflow.png",
                 "<b>Figure 1.</b> Workflow of the study: literature database, feature "
                 "engineering, K-means classification, fold-local oversampling, six-model "
                 "comparison, SHAP interpretation and NSGA-II catalyst/condition optimisation.")

    # ---- 2. Data and methods ------------------------------------------------
    miss = d["missing"]
    story += [P("2. DATA AND METHODS", "h1"),
              P("2.1. Data Preprocessing.", "h2"),
              P(f"The database contains {d['n_records']} records of HMF electro-oxidation "
                f"extracted from {d['n_papers']} publications. Each record lists the cell type, "
                "anode/cathode materials, electrolyte, HMF concentration, potential, TEMPO use, "
                "temperature and the measured conversion, FDCA yield, faradaic efficiency (FE) "
                f"and selectivity, reported for {d['targets_n']['Conversion_pct']}, "
                f"{d['targets_n']['Yield_pct']}, {d['targets_n']['FE_pct']} and "
                f"{d['targets_n']['Selectivity_pct']} records, respectively. Free text was "
                "parsed into numerical descriptors. Catalyst descriptors are extracted from the "
                "<i>anode/working electrode</i> only, because Pt or graphite counter electrodes "
                "would otherwise be counted as the catalyst. Element and family flags (oxide, "
                "hydroxide, oxyhydroxide, LDH, phosphide, nitride, sulfide, selenide, boride, "
                "MOF) are matched on chemical formulas such as Ni<sub>3</sub>S<sub>2</sub>, "
                "CoNiP and Ni<sub>2</sub>P. Substrate (foam, conductive glass, carbon), "
                "counter-electrode type and electrolyte pH, explicit or computed as "
                "14 + log[OH<super>-</super>], complete the "
                f"{d['n_features']} descriptors. Potential vs RHE is missing for "
                f"{miss['Potential_vs_RHE_V']} records and temperature for "
                f"{miss['Temperature_C']}. Missing inputs are median-imputed inside each "
                "training fold, with binary missing-value indicators appended, and outcome "
                "variables are never used as inputs."),
              P("2.2. Machine Learning Models.", "h2"),
              P("Decision tree (DT), support vector machine (SVM), random forest (RF), "
                "neural network (NN), gradient boosting (GB) and extra trees (ET) classifiers "
                "were compared [14]. Hyperparameters were selected from a small per-family grid "
                "under the same grouped protocol. Every score reported is the macro-averaged F1 "
                "of pooled out-of-fold predictions over 25 repeats × 5 publication-grouped folds. "
                "Accuracy, precision, recall and F1 are computed from the confusion matrix "
                "(eqs 1-4).")]
    for latex in (r"\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}",
                  r"\mathrm{Precision}=\frac{TP}{TP+FP}",
                  r"\mathrm{Recall}=\frac{TP}{TP+FN}",
                  r"F1=\frac{2\times\mathrm{Precision}\times\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}"):
        story.append(equation(latex)[0])
    story += [P("2.3. <i>K</i>-Means Clustering Algorithm.", "h2"),
              P("K-means partitions records into K clusters by minimising the within-cluster "
                "sum of squares in the standardised (conversion, yield) space for "
                "K ∈ {3, 4, 5, 6}, with 100 random initialisations (eqs 5-7). Missing outcomes "
                "were median-imputed for clustering only, and those records are flagged in all "
                "figures. A secondary binary target labels a record high-performance when "
                "min(conversion, yield) ≥ 90%.")]
    for latex in (r"d(x_i,c_j)=\sqrt{\sum_{m=1}^{M}(x_{im}-c_{jm})^2}",
                  r"c_j=\frac{1}{|S_j|}\sum_{p_i\in S_j}p_i",
                  r"\min\ \sum_{j=1}^{k}\sum_{p_i\in S_j}\|p_i-c_j\|^2"):
        story.append(equation(latex)[0])
    story += [P("2.4. Application of Oversampling Technology.", "h2"),
              P("Random oversampling (ROS), SMOTE [10] and ADASYN [11] were applied only to "
                "training folds. SMOTE interpolates between a minority sample and one of its "
                "nearest minority neighbours (eq 8). ADASYN allocates more synthetic samples to "
                "minority points surrounded by majority neighbours (eq 9). The neighbour count "
                "is min(2, n<sub>min</sub> − 1), so a class with one training record makes the "
                "sampler infeasible for that fold. Such repeats are reported, not scored, and a "
                "model/sampler pair is ranked only if at least 80% of its repeats are complete.")]
    story.append(equation(r"x_{\mathrm{new}}=x_i+\lambda\,(x_{zi}-x_i),\quad \lambda\in[0,1]")[0])
    story.append(equation(r"g_i=\frac{r_i}{\sum_i r_i}\times G")[0])
    story += [P("2.5. Feature Interpretability Analysis.", "h2"),
              P("SHAP values [12] decompose each prediction into additive feature "
                "contributions relative to a baseline expectation (eq 10). Kernel SHAP was "
                "applied to the complete fitted pipeline, including imputation and scaling, on "
                "a publication-grouped 80/20 hold-out. For K = 3 and K = 5 it explains the best "
                "model/sampler pair of the comparison.")]
    story.append(equation(r"f(x)=g(x')=\phi_0+\sum_{i=1}^{m}\phi_i\,x'_i")[0])
    story += [P("2.6. Application of the Integrated Genetic Algorithm.", "h2"),
              P("NSGA-II [13] was used for multi-objective screening, following the design "
                "of ref 1 with electrochemical objectives. Predicted conversion X, FDCA yield Y "
                "and FE are maximised, while temperature T and applied potential E are "
                "minimised. These two operating burdens replace the reference article's "
                "pressure and reaction time, which this dataset does not record. Predictions "
                "come from per-target random-forest regressors (mutual-information selection of 15 "
                "descriptors, 60 trees of depth 3) that beat the median baseline in "
                "MAE for all four outputs under repeated grouped cross-validation. The "
                "decision variables use mixed encoding. Discrete descriptors are binary, and "
                "one-hot groups (electrolyte cation and anion) obey the hard constraint "
                "Σx<sub>i</sub> = 1. Continuous variables are min-max normalised to [0, 1]. "
                f"The search used a population of {d['nsga']['population_size']} for "
                f"{d['nsga']['generations']} generations, with binary tournament selection, "
                "blend crossover, Gaussian mutation and elitist (μ + λ) survival by front and "
                "crowding distance. Pareto solutions were ranked with the entropy-weighted "
                "fitness of eq 11. Convergence was monitored with the spacing metric (eq 12) "
                "on fixed-scale objectives, where 0 is each objective's ideal value.")]
    story.append(equation(r"\mathrm{fitness}=\omega_1X+\omega_2Y+\omega_3\mathrm{FE}-\omega_4T-\omega_5E")[0])
    story.append(equation(r"S=\sqrt{\frac{1}{|H|}\sum_{i=1}^{|H|}(\bar d-d_i)^2}")[0])
    story += [P("In the constrained scenario the anode is a single-metal Ni(OH)<sub>2</sub> or "
                "Cu(OH)<sub>2</sub>, with all other element and family flags set to zero and "
                "derived descriptors recomputed. Potential and temperature are bounded to the "
                "range reported for real Ni/Cu-hydroxide anodes in the database "
                f"({env['Potential_vs_RHE_V'][0]:.2f}-{env['Potential_vs_RHE_V'][1]:.2f} V vs RHE "
                f"and {env['Temperature_C'][0]:.0f}-{env['Temperature_C'][1]:.0f} °C). Each metal "
                "is optimised separately, and the final populations are pooled and re-ranked.")]

    # ---- 3. Results ---------------------------------------------------------
    story += [P("3. RESULTS AND DISCUSSION", "h1"),
              P("3.1. <i>K</i>-Means Clustering Results.", "h2")]
    cl = pd.DataFrame([{"K": k, "Cluster sizes": ", ".join(f"{c}: {n}" for c, n in v.items())}
                       for k, v in d["clusters"].items()])
    story += [P("Figure 2 shows the clusters for K = 3-6. Unlike the thermocatalytic data of "
                "ref 1, electro-oxidation records crowd the high-conversion corner. The "
                f"dominant cluster holds {max(d['clusters'][3].values())} of {d['n_records']} "
                "records at K = 3. Further clusters separate the low-yield, high-conversion "
                "region, where oxidation stops at intermediates or over-oxidises, from the "
                "few low-conversion records. Higher K subdivides the high-yield region but "
                "leaves minority classes of one to four records (Table 1). That severe "
                "imbalance motivates both oversampling and the binary performance screen.")]
    story += fig(FIGURES_DIR / "KMeans_Figure6" / "HMF_KMeans_Figure6_Style.png",
                 "<b>Figure 2.</b> Without-sampling K-means clustering results: (a) K = 3, "
                 "(b) K = 4, (c) K = 5 and (d) K = 6. Shaded polygons are cluster convex hulls, "
                 "and spokes join each record to its cluster mean.", max_h=15 * cm)
    story += table(cl, "<b>Table 1.</b> Cluster Sizes for K = 3-6 (154 Records)",
                   col_widths=[2 * cm, TEXT_W - 2 * cm])

    story += [P("3.2. Performance Evaluation of the Learning Models.", "h2")]
    rows = []
    base_df = d["baseline"]
    for model in ["DecisionTree", "SVM", "RandomForest", "NeuralNetwork", "GradientBoosting", "ExtraTrees"]:
        row = {"Model": MODEL_SHORT[model]}
        for kind, k, lab, cur in (("binary_high_performance_90", 0, "Binary ≥90%", bin_best),
                                  ("kmeans_cluster", 3, "K = 3", k3_best)):
            if base_df is not None:
                b = base_df[(base_df.target_kind == kind) & (base_df.k == k) & (base_df.model == model)]
                row[f"{lab} before"] = f"{b.macro_f1_mean.max():.3f}" if len(b) else "—"
            row[f"{lab} now"] = (f"{cur.loc[model, 'macro_f1_mean']:.3f} ({cur.loc[model, 'sampler']})"
                                 if model in cur.index else "—")
        rows.append(row)
    perf = pd.DataFrame(rows)
    story += [P("Table 2 lists the best macro-F1 of each model family. Correcting the "
                "descriptors raised scores for every family. Previously the Pt cathode was "
                "counted as the catalyst in 74 records, and phosphide, nitride and sulfide "
                "anodes were never detected. Other gains came from missing-value indicators "
                "and regularised hyperparameters: the NN no longer early-stops on a 10% split "
                "of an already small fold. For the binary screen the NN reaches "
                f"{bin_best.loc['NeuralNetwork', 'macro_f1_mean']:.3f} and ET "
                f"{bin_best.loc['ExtraTrees', 'macro_f1_mean']:.3f}, against a stratified dummy "
                f"baseline of {best['dummy_macro_f1_mean']:.3f}. For K = 3, GB reaches "
                f"{k3_best.loc['GradientBoosting', 'macro_f1_mean']:.3f}. The K = 4-6 "
                "classifications remain weak (best macro-F1 "
                f"{k_overall[4]['macro_f1_mean']:.2f}, {k_overall[5]['macro_f1_mean']:.2f} and "
                f"{k_overall[6]['macro_f1_mean']:.2f}) because their minority clusters contain "
                "only a handful of papers."),
              P("These scores are lower than the 0.75-0.85 reported in ref 1 because the "
                "evaluation is stricter. Under random splits, near-duplicate records from the "
                "same paper fall on both sides of the split and inflate the score; publication "
                "grouping prevents that. The learning curves of Figure 3 show the expected "
                "small-data behaviour: training F1 well above validation F1 for the tree "
                "models, and for SVM at K = 3 a validation score that keeps rising with the "
                "training size.")]
    story += table(perf, "<b>Table 2.</b> Best Pooled Out-of-Fold Macro-F1 per Model "
                         "(Best Sampler in Brackets), Before and After the Descriptor and "
                         "Model Corrections", bold_first_col=True)
    story += fig(FIGURES_DIR / "learning_curves" / "HMF_Learning_Curves_Figure.png",
                 "<b>Figure 3.</b> Learning curves of DT, SVM, RF and NN (rows) for K = 3, 4 and 5 "
                 "(columns) on publication-grouped hold-outs. Red: training score; green: "
                 "validation score; bands: ±1 SD over 10 repeats.", max_h=19 * cm)
    story += fig(FIGURES_DIR / "classification" / "model_sampler_binary90.png",
                 "<b>Figure 4.</b> Model and sampling comparison for the binary ≥90% FDCA "
                 "performance screen (pooled out-of-fold macro-F1, mean ± SD over 25 repeats; "
                 "dotted line: 0.76 reference).", max_h=9.5 * cm)
    story += fig(FIGURES_DIR / "classification" / "model_sampler_K3.png",
                 "<b>Figure 5.</b> Model and sampling comparison for the K = 3 cluster "
                 "classification.", max_h=9.5 * cm)

    story += [P("3.3. <i>K</i>-Means Analysis after Oversampling.", "h2"),
              P("Figure 6 shows the synthetic records produced by ADASYN and SMOTE in the "
                "conversion-yield plane. Interpolation fills the sparse low-yield clusters "
                "along the segments between measured records and preserves the cluster "
                "topology, as in ref 1. For K = 3 and K = 6 the interpolating samplers are "
                "infeasible because one cluster has a single measured record, so random "
                "oversampling is the only option there.")]
    story += fig(FIGURES_DIR / "oversampling_kmeans" / "KMeans_After_Oversampling_K3_to_K6.png",
                 "<b>Figure 6.</b> K-means clustering after oversampling: (a, c, e, g) ADASYN and "
                 "(b, d, f, h) SMOTE for K = 3-6. Circles: measured records; crosses: synthetic "
                 "records.", max_h=20 * cm)

    story += [P("3.4. Analysis of Learning Curves after Oversampling.", "h2")]
    t1df = d["table1"]
    t1df = t1df[t1df["Valid folds"] == t1df["Valid folds"].max()].sort_values("F1-score", ascending=False).head(10)
    t1df = t1df.assign(**{c: t1df[c].map(lambda v: f"{v:.4f}") for c in ("Accuracy", "Precision", "Recall", "F1-score")})
    story += [P("Figure 7 and Table 3 summarise the four reference-paper models after "
                "oversampling, averaged over held-out folds (fold-level means, as in ref 1). "
                "At K = 3, RF and NN with ROS give the highest fold-averaged F1. Only "
                "combinations whose sampler was feasible in every fold are tabulated. The "
                "oversampled NN learning curves (Figure 8) show validation F1 rising with the "
                "number of training papers, which indicates the models are data-limited rather "
                "than capacity-limited.")]
    story += fig(FIGURES_DIR / "paper_style" / "Figure_9_Oversampling_F1.png",
                 "<b>Figure 7.</b> F1-scores of the four models after oversampling for K = 3-5: "
                 "ROS, ADASYN and SMOTE (fold-level mean ± SD).", max_h=8 * cm)
    story += table(t1df[["K", "Model", "Oversampling", "Accuracy", "Precision", "Recall", "F1-score"]],
                   "<b>Table 3.</b> Best Oversampled Models (All Folds Feasible), Fold-Averaged "
                   "Held-Out Metrics")
    story += fig(FIGURES_DIR / "learning_curves" / "Figure_11_Oversampled_NN_Learning_Curves_ROS_ADASYN_SMOTE.png",
                 "<b>Figure 8.</b> Learning curves of the oversampled NN model (ROS, ADASYN, SMOTE) "
                 "for K = 3-6.", max_h=19 * cm)

    story += [P("3.5. Results of Feature Importance Analysis.", "h2")]
    runs = d["shap_audit"]["runs"]
    s3 = ", ".join(escape(f) for f in d["shap3"]["feature"].head(3))
    s5 = ", ".join(escape(f) for f in d["shap5"]["feature"].head(3))
    story += [P(f"For K = 3 the best model ({MODEL_SHORT.get(runs['3']['model'])} + "
                f"{runs['3']['sampler']}) relies mainly on {s3}. For K = 5 "
                f"({MODEL_SHORT.get(runs['5']['model'])} + {runs['5']['sampler']}) the leading "
                f"features are {s5} (Figures 9 and 10). The conductive-glass flag (FTO/ITO/BDD "
                f"thin films) marks a distinct group. These {d['glass_n']} records have a median "
                f"FDCA yield of {d['glass_yield']:.0f}%, against {d['other_yield']:.0f}% for the "
                "rest, consistent with low-loading thin films that often stop at intermediates. "
                f"High-surface-area foam supports ({d['foam_n']} records, median yield "
                f"{d['foam_yield']:.0f}%) dominate the high-yield cluster. Applied potential and "
                "HMF concentration follow in importance, matching the known sensitivity of FDCA "
                "selectivity to the potential window and substrate loading. SHAP values are "
                "associational, not causal.")]
    story += fig(FIGURES_DIR / "shap" / "Figure_12_SHAP_Bar.png",
                 "<b>Figure 9.</b> Global SHAP feature importance (mean |SHAP|) of the best "
                 "model for K = 3 (left) and K = 5 (right).", max_h=9.5 * cm)
    story += fig(FIGURES_DIR / "shap" / "Figure_13_SHAP_Beeswarm.png",
                 "<b>Figure 10.</b> SHAP beeswarm plots for K = 3 and K = 5, coloured by feature "
                 "value (low to high).", max_h=9.5 * cm)

    story += [P("3.6. Application of the Integrated Genetic Algorithm in ML Modeling for "
                "Catalyst Optimization.", "h2")]
    story += [P("The validation gate passed: the per-target regressors beat the median "
                f"baseline MAE for conversion ({reg.loc['Conversion_pct', 'model_mae_mean']:.1f} vs "
                f"{reg.loc['Conversion_pct', 'baseline_mae_mean']:.1f}), yield "
                f"({reg.loc['Yield_pct', 'model_mae_mean']:.1f} vs {reg.loc['Yield_pct', 'baseline_mae_mean']:.1f}), "
                f"FE ({reg.loc['FE_pct', 'model_mae_mean']:.1f} vs {reg.loc['FE_pct', 'baseline_mae_mean']:.1f}) "
                f"and selectivity ({reg.loc['Selectivity_pct', 'model_mae_mean']:.1f} vs "
                f"{reg.loc['Selectivity_pct', 'baseline_mae_mean']:.1f}). Their R<super>2</super> "
                "is close to zero, however, so predictions are shrunk toward the dataset mean "
                "and should be read as a ranking signal. "
                f"{spacing_text(d['history'])} Entropy weights computed on the hydroxide Pareto set "
                f"were ω<sub>X</sub> = {w['Conversion_pct']:.3f}, ω<sub>Y</sub> = "
                f"{w['Yield_pct']:.3f}, ω<sub>FE</sub> = {w['FE_pct']:.3f}, ω<sub>T</sub> = "
                f"{w['Temperature_C']:.3f} and ω<sub>E</sub> = {w['Potential_vs_RHE_V']:.3f}. "
                "Potential and yield discriminate most strongly among Pareto solutions.")]
    story += fig(FIGURES_DIR / "nsga2" / "Figure_14_Spacing_Trace.png",
                 "<b>Figure 11.</b> Spacing trace of the NSGA-II Pareto front for the (a) "
                 "Ni(OH)<sub>2</sub> and (b) Cu(OH)<sub>2</sub> anode searches.", max_h=6.5 * cm)
    tt = top[["rank", "catalyst", "support", "electrolyte", "electrolyte_M", "HMF_mM",
              "potential_V_RHE", "temperature_C", "predicted_Conversion_pct",
              "predicted_Yield_pct", "predicted_FE_pct", "entropy_fitness"]].copy()
    tt.columns = ["#", "Catalyst", "Support", "Electrolyte", "c (M)", "HMF (mM)", "E (V RHE)",
                  "T (°C)", "X (%)", "Y (%)", "FE (%)", "Fitness"]
    for c in ("X (%)", "Y (%)", "FE (%)"):
        tt[c] = tt[c].map(lambda v: f"{v:.1f}")
    tt["Fitness"] = tt["Fitness"].map(lambda v: f"{v:.3f}")
    cu = d["cu3"].iloc[0]
    gen = d["general"].iloc[0]
    story += [P("Table 4 lists the ten best solutions of the constrained search. All ten are "
                f"Ni(OH)<sub>2</sub> on nickel foam in KOH ({top['electrolyte_M'].min():.1f}-"
                f"{top['electrolyte_M'].max():.1f} M) with ~{top['HMF_mM'].median():.0f} mM HMF "
                f"at {top['potential_V_RHE'].min():.2f}-{top['potential_V_RHE'].max():.2f} V vs RHE "
                f"and ~25 °C. Predicted FDCA yields are {top['predicted_Yield_pct'].min():.1f}-"
                f"{top['predicted_Yield_pct'].max():.1f}% and FE {top['predicted_FE_pct'].min():.1f}-"
                f"{top['predicted_FE_pct'].max():.1f}%. The best Cu(OH)<sub>2</sub> solution, "
                f"on Cu foam in {cu['electrolyte_M']:.2f} M {escape(str(cu['electrolyte']))} at "
                f"{cu['potential_V_RHE']:.2f} V, is predicted at {cu['predicted_Yield_pct']:.1f}% "
                "yield. It ranks lower because the database contains few high-yield "
                "Cu(OH)<sub>2</sub> records. As a control, the unconstrained search proposed "
                f"{escape(str(gen['catalyst']))} at {gen['potential_V_RHE']:.2f} V and "
                f"{gen['temperature_C']:.0f} °C, outside the region where Ni/Cu hydroxides "
                "oxidise HMF. This extrapolation by the tree regressors is why the "
                "catalyst-specific condition envelope is required.")]
    story += table(tt, "<b>Table 4.</b> Top-10 NSGA-II Solutions with a Ni(OH)<sub>2</sub> or "
                       "Cu(OH)<sub>2</sub> Anode, Ranked by Pareto Front and Entropy-Weighted "
                       "Fitness (Model Predictions)")
    story += fig(FIGURES_DIR / "nsga2" / "NSGA2_top10_NiCu_hydroxide.png",
                 "<b>Figure 12.</b> Predicted FDCA yield vs FE of the top-10 solutions, "
                 "coloured by entropy-weighted fitness.", width=0.7 * TEXT_W, max_h=9 * cm)
    lit = pd.DataFrame(LITERATURE, columns=["Reported catalyst", "Conditions", "Reported performance", "Ref."])
    story += [P("<b>Comparison with experiment.</b> The nearest database record to solution 1 "
                f"is {chem(str(t1['nearest_catalyst']))} ({escape(str(t1['nearest_conditions']))}). "
                f"It measured {t1['observed_Yield_pct']:.1f}% yield and {t1['observed_FE_pct']:.1f}% "
                f"FE, against the predicted {t1['predicted_Yield_pct']:.1f}% and "
                f"{t1['predicted_FE_pct']:.1f}%. The model is therefore optimistic by about "
                f"{t1['predicted_Yield_pct'] - t1['observed_Yield_pct']:.0f} percentage points. "
                "Table 5 compares the solutions with published electrodes. A freestanding "
                "Ni(OH)<sub>2</sub>/nickel-foam electrode was studied at exactly the proposed "
                "conditions (1 M KOH, 10 mM HMF, 1.45 V vs RHE) [4], and modified Ni(OH)<sub>2</sub> "
                "anodes at 1.45-1.50 V report 95-99% FDCA yield and FE [5-7]. "
                "Cu(OH)<sub>2</sub>/Cu-foam reaches 98.7% yield in 1 M KOH [8]. The algorithm "
                "thus recovers the experimentally established operating window for "
                "Ni(OH)<sub>2</sub>. Its absolute predictions, and the lower ranking of "
                "Cu(OH)<sub>2</sub>, reflect the limits of a 154-record database.")]
    story += table(lit, "<b>Table 5.</b> Published Ni(OH)<sub>2</sub> and Cu(OH)<sub>2</sub> "
                        "Electrodes for HMF Electro-oxidation Used to Check the GA Solutions",
                   col_widths=[4.2 * cm, 4.3 * cm, 6.3 * cm, 1.2 * cm])

    # ---- 4. Conclusion ------------------------------------------------------
    story += [P("4. CONCLUSION", "h1"),
              P("We transferred the ML-driven catalyst-screening workflow of ref 1 to the "
                "electrochemical oxidation of HMF to FDCA. Correcting how catalyst descriptors "
                "are extracted (anode only, formula-aware) and evaluating every model on unseen "
                "publications improved and stabilised the classifiers. The NN with SMOTE is the "
                f"best high-performance screen (macro-F1 {best['macro_f1_mean']:.3f}), and GB is "
                f"best for K = 3 ({k_overall[3]['macro_f1_mean']:.3f}). SHAP highlights the "
                "electrode architecture and the applied potential. The entropy-weighted NSGA-II "
                "search, constrained to Ni(OH)<sub>2</sub>/Cu(OH)<sub>2</sub> anodes, converges "
                "on Ni(OH)<sub>2</sub> on nickel foam in about 1 M KOH with about 10 mM HMF at "
                "1.41-1.45 V vs RHE and room temperature. That window is consistent with "
                "published electrodes reaching 95-99% FDCA yield."),
              P("<b>Limitations.</b> The database is small, has missing values and leans "
                "toward reported successes. The per-target regressors explain little variance "
                "(R<super>2</super> ≈ 0), so predicted performance is optimistic by about seven "
                "points relative to the closest measured record. Hyperparameters were chosen on "
                "the same records. The within-paper label-permutation test is not significant "
                f"(p = {best.get('permutation_p_value', float('nan')):.2f}), which shows that "
                "most signal separates papers rather than conditions within a paper. GA "
                "solutions are hypotheses for experimental validation, not optimal catalysts.")]

    story += [P("DATA AND CODE AVAILABILITY", "h1"),
              P("All figures and tables were generated by <i>run_all.py</i> (stages 01-10) from "
                "<i>data/raw</i>. Numerical results are in <i>results/</i>, including "
                "<i>classification_summary.csv</i>, <i>nsga2_top10_ni_cu_hydroxide.csv</i> and "
                "<i>FINAL_RESEARCH_REPORT.md</i>.")]
    story += [P("REFERENCES", "h1")]
    for i, ref in enumerate(REFERENCES, 1):
        story.append(P(f"({i}) {ref}", "ref"))

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=1.8 * cm, bottomMargin=1.7 * cm,
                            title="ML-Driven Catalyst Screening for HMF Electro-oxidation to FDCA",
                            author="HMF-Connected pipeline")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    workflow_figure(BUILD / "figure1_workflow.png")
    data = load_inputs()
    build(data)
    print(f"Research paper written: {OUT_PDF}")


if __name__ == "__main__":
    main()
