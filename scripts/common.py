"""Shared data contracts and leakage-safe helpers for the HMF/FDCA workflow.

The active pipeline uses one row identity across preprocessing, clustering,
classification, regression and optimisation. This module is deliberately the
single place where paths, feature exclusions and paper-group definitions live.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              ExtraTreesRegressor, GradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PREPROCESSED_FILE = DATA_DIR / "HMF_ML_Preprocessed.xlsx"
CLEANED_FILE = DATA_DIR / "HMF_Cleaned_Full_Dataset.xlsx"
TARGETS = ["Conversion_pct", "Yield_pct", "FE_pct", "Selectivity_pct"]
TARGET_LABELS = {
    "Conversion_pct": "Conversion (%)",
    "Yield_pct": "FDCA yield (%)",
    "FE_pct": "Faradaic efficiency (%)",
    "Selectivity_pct": "Selectivity (%)",
}
RANDOM_STATE = 42

# Explicitly remove every measured outcome and any variable calculated from it.
# Remaining numeric columns are engineered descriptors of the eight agreed
# experimental input groups: TEMPO, temperature, monomer, concentration,
# catalyst, electrolyte, potential and cell type.
FORBIDDEN_TOKENS = (
    "conversion", "yield", "selectivity", "faradaic", "fe_pct",
    "fdca_target", "performance", "cluster", "class_label", "target",
    "above_100", "greater_than", "paper", "major_product", "product_sel",
)


def ensure_directories() -> None:
    """Create only project output directories; never overwrite source data."""
    for path in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_processed() -> pd.DataFrame:
    """Read the single preprocessed feature table created by stage 01."""
    if not PREPROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PREPROCESSED_FILE}. Run 01_preprocess_data.py first."
        )
    frame = pd.read_excel(PREPROCESSED_FILE)
    if len(frame) != 154:
        warnings.warn(f"Expected 154 literature records; found {len(frame)}.")
    return frame


def read_cleaned() -> pd.DataFrame:
    """Read cleaned literature metadata aligned row-for-row to processed data."""
    if not CLEANED_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CLEANED_FILE}. Run 01_preprocess_data.py first."
        )
    frame = pd.read_excel(CLEANED_FILE)
    if len(frame) != len(read_processed()):
        raise ValueError("Cleaned and preprocessed records are not row-aligned.")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select only the eight agreed experimental input groups and descriptors.

    The numeric columns are expanded descriptors, not a claim that the study
    has only eight raw columns. Current/current-density and cell-voltage fields
    are excluded because they were not among the agreed eight inputs.
    """
    selected: list[str] = []
    for column in frame.columns:
        name = str(column)
        low = name.lower()
        if any(token in low for token in FORBIDDEN_TOKENS):
            continue
        allowed = (
            name in {"HMF_Concentration_mM", "Potential_vs_RHE_V", "Potential_vs_RHE",
                     "Temperature_C", "TEMPO_binary", "Electrolyte_Concentration_M",
                     "Alkaline_Electrolyte", "Divided_Cell", "Undivided_Cell",
                     "Flow_Cell", "Three_Electrode"}
            or name.startswith("Catalyst_")
            or name.startswith("Counter_Electrode_")
            or name.startswith("Electrolyte_")
            or name.startswith("Reactant_")
        )
        if not allowed or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
            continue
        selected.append(column)
    if not selected:
        raise ValueError("No usable descriptors from the eight agreed input groups.")
    return selected


def build_feature_matrix(frame: pd.DataFrame | None = None):
    """Return encoded inputs, feature names and source frame with stable row IDs.

    The cleaned literature workbook contributes the Reactant monomer one-hot
    group. If all records use HMF, this descriptor is constant and correctly
    excluded from fitting rather than being assigned artificial variation.
    Catalyst, electrolyte and cell-type chemical descriptors come from stage 01.
    """
    frame = read_processed() if frame is None else frame.copy()
    cleaned = read_cleaned()
    if "Reactant" in cleaned.columns:
        reactant = cleaned["Reactant"].fillna("Not reported").astype(str).str.strip()
        reactant = reactant.str.replace(r"\s+", " ", regex=True).str.lower()
        # Harmonise spelling/purity annotations for the same HMF monomer.
        reactant = reactant.where(
            ~reactant.str.contains(r"hmf|hydroxymethylfurfural", regex=True, na=False),
            "hmf",
        )
        reactant_features = pd.get_dummies(reactant, prefix="Reactant", dtype=int)
        frame = pd.concat([frame.reset_index(drop=True), reactant_features], axis=1)
    columns = feature_columns(frame)
    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X.index = np.arange(len(X), dtype=int)
    return X, columns, frame


def paper_groups() -> np.ndarray:
    """Return publication identifiers for leakage-aware grouped validation."""
    raw = read_cleaned()
    candidates = [c for c in raw.columns if str(c).strip().lower() == "paper title"]
    if candidates:
        groups = raw[candidates[0]].fillna("UNKNOWN_PAPER").astype(str).str.strip()
    else:
        groups = pd.Series([f"record_{i}" for i in range(len(raw))])
    return groups.to_numpy()


HIGH_PERFORMANCE_THRESHOLD = 90.0


def binary_high_performance_labels(raw: pd.DataFrame | None = None):
    """Return (row_ids, y) for the secondary ">= 90 %" FDCA screen.

    Single definition shared by stages 04, 06 and 08. Rows must report FDCA as
    the major product and a measured yield inside 0-100. Performance is
    min(conversion, yield) when a valid conversion is reported, else the yield.
    Because yield <= conversion physically, this equals the yield in all but a
    handful of records where the source reports yield > conversion. Values
    outside 0-100 are treated as missing, never clipped (same rule as
    regression).
    """
    raw = read_cleaned() if raw is None else raw
    if "Major_Product_Clean" in raw.columns:
        fdca = raw["Major_Product_Clean"].astype(str).str.contains(
            r"FDCA|FURANDICARBOXYLIC", case=False, regex=True, na=False)
    else:
        fdca = pd.Series(True, index=raw.index)
    yld = pd.to_numeric(raw["Yield_pct"], errors="coerce")
    conv = pd.to_numeric(raw["Conversion_pct"], errors="coerce")
    keep = fdca & yld.between(0, 100)
    row_ids = np.flatnonzero(keep.to_numpy())
    yl = yld.iloc[row_ids].to_numpy(dtype=float)
    cv = conv.where(conv.between(0, 100)).iloc[row_ids].to_numpy(dtype=float)
    performance = np.where(np.isfinite(cv), np.minimum(cv, yl), yl)
    return row_ids, (performance >= HIGH_PERFORMANCE_THRESHOLD).astype(int)


def make_classifier(name: str, seed: int = RANDOM_STATE):
    """Construct deterministic models tuned for literature-scale grouped CV.

    Hyperparameters were chosen from a small per-family grid evaluated with
    the same 5-fold publication-grouped protocol as stage 04 (binary >=90%
    screen and K=3 clusters). Settings favour strong regularisation because
    only ~120-154 records from ~99 papers are available:
    - DecisionTree: depth 6 (depth 5 under-fit the corrected descriptors).
    - SVM: C=0.3, gamma=0.01 (smoother RBF boundary than gamma='scale').
    - RandomForest: 300 trees, depth 6.
    - NeuralNetwork: one 32-unit layer, alpha=0.1, NO early stopping. The
      former early-stopping split held out 10% of an already tiny, imbalanced
      training fold and frequently stopped on a majority-class solution.
    - GradientBoosting: 100 decision stumps (depth 1) with shrinkage.
    - ExtraTrees: added family; randomised splits are robust on sparse
      one-hot catalyst descriptors.
    """
    models = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=2, class_weight="balanced",
            random_state=seed,
        ),
        "SVM": SVC(
            C=0.3, kernel="rbf", gamma=0.01, class_weight="balanced",
            probability=True, random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(32,), alpha=0.1, max_iter=2000,
            early_stopping=False, random_state=seed,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=1, learning_rate=0.1,
            min_samples_leaf=3, subsample=0.8,
            random_state=seed,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=1,
        ),
        # Soft vote of a randomised-tree model and a strongly regularised linear
        # model: their errors differ (non-linear vs additive), which helps on
        # ~100 records. Hyperparameters were set in a small exploratory
        # comparison on the same records, so scores are mildly optimistic.
        "Ensemble": VotingClassifier(
            estimators=[
                ("et", ExtraTreesClassifier(
                    n_estimators=400, min_samples_leaf=2,
                    class_weight="balanced_subsample", random_state=seed, n_jobs=1)),
                ("lr", LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=2000)),
            ],
            voting="soft",
        ),
    }
    if name not in models:
        raise KeyError(f"Unknown classifier: {name}")
    return models[name]


def make_regressor_single(seed: int = RANDOM_STATE) -> Pipeline:
    """Per-target regressor used by stages 06-08: median imputation + a mean of
    Huber gradient boosting (depth-2 trees) and extra-trees.

    In a pooled out-of-fold, paper-grouped comparison this beat the earlier
    depth-3 random forest with mutual-information feature selection (R2 about
    0.3-0.4 instead of about 0.1 for Yield/FE/Selectivity). The imputer is the
    first pipeline step because stage 09 reuses it.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", VotingRegressor([
            ("gb", GradientBoostingRegressor(
                n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.7,
                loss="huber", random_state=seed)),
            ("et", ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=4, max_features=0.5,
                random_state=seed, n_jobs=1)),
        ])),
    ])


def balance_with_singleton_fallback(name: str, X, y, seed: int = RANDOM_STATE):
    """SMOTE / ADASYN for display-oriented analyses (stage 03 figure, stage 05 curves).

    SMOTE and ADASYN need at least two members in every class. When a class has a
    single record, that record is first duplicated (random over-sampling to two),
    after which no interpolation is possible for that class: its "synthetic" points
    coincide with the measured one. ADASYN neighbours are widened from 2 up to 10
    (never above the smallest class size minus one) until the algorithm finds majority-class neighbours. Returns
    (X_resampled, y_resampled, info) with info listing the duplicated classes and
    the neighbour count used. Raises ValueError if no setting works.

    Stage 04 does NOT use this fallback: its classification scores treat these
    combinations as infeasible, so headline F1 values are unaffected.
    """
    from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
    y = np.asarray(y)
    X = np.asarray(X)
    classes, counts = np.unique(y, return_counts=True)
    singles = [int(c) for c, n in zip(classes, counts) if n < 2]
    n_orig = len(y)
    if singles:
        ros = RandomOverSampler(sampling_strategy={c: 2 for c in singles}, random_state=seed)
        X, y = ros.fit_resample(X, y)
    min_count = int(np.unique(y, return_counts=True)[1].min())
    if min_count < 2:
        raise ValueError("a class has no records")
    info = {"duplicated_singleton_classes": singles, "n_before": n_orig}
    if name == "SMOTE":
        sampler = SMOTE(k_neighbors=min(2, min_count - 1), random_state=seed)
        X_res, y_res = sampler.fit_resample(X, y)
        info["neighbours"] = min(2, min_count - 1)
        return X_res, y_res, info
    if name == "ADASYN":
        last = None
        for nn in range(min(2, min_count - 1), min(10, min_count - 1) + 1):  # needs n_neighbors <= class size - 1
            try:
                X_res, y_res = ADASYN(n_neighbors=nn, random_state=seed).fit_resample(X, y)
                info["neighbours"] = nn
                return X_res, y_res, info
            except (ValueError, RuntimeError) as exc:
                last = exc
        raise ValueError(f"ADASYN could not be fitted: {last}")
    raise ValueError(name)


def grouped_splits(y: Iterable, groups: Iterable, seed: int, n_splits: int = 3):
    """Create paper-grouped folds; reduce folds only when group support requires."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("Labels and publication-group arrays have different lengths.")
    if len(y) < 4:
        raise ValueError("Too few records for publication-grouped validation.")
    unique_groups = len(np.unique(groups))
    if unique_groups < 2:
        raise ValueError("At least two independent papers are required for grouped CV.")
    # Never create folds that omit a class from the training partition. Test
    # folds may legitimately lack an extremely rare class; metrics use the
    # observed labels and zero_division=0, with support recorded downstream.
    for folds in range(min(n_splits, unique_groups), 1, -1):
        try:
            splitter = StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=seed
            )
            splits = list(splitter.split(np.zeros(len(y)), y, groups))
            if all(len(tr) > 0 and len(te) > 0 and len(np.unique(y[tr])) >= 2
                   for tr, te in splits):
                return splits
        except ValueError:
            continue
    raise ValueError("Insufficient class representation across independent papers.")


def classification_preprocessing_steps() -> list:
    """Fold-local preprocessing shared by every classification stage (04-06).

    ``add_indicator=True`` appends one binary column per descriptor that has
    missing values in the training fold (e.g. temperature or potential not
    reported). Whether a paper reports a condition is an input-side property,
    not an outcome, and median imputation alone would erase it.
    """
    from sklearn.feature_selection import VarianceThreshold
    return [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ]


def base_preprocessor(scale: bool = True) -> Pipeline:
    """Fold-local imputation and scaling shared by classification and regression."""
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def write_json(path: Path, payload) -> None:
    """Write JSON with NumPy/Pandas scalar support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def stage_status(stage: str, status: str, details: dict | None = None) -> None:
    """Append a durable stage marker consumed by the final report."""
    ensure_directories()
    path = RESULTS_DIR / "pipeline_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"stages": {}}
    manifest["stages"][stage] = {
        "status": status,
        "details": details or {},
        "pid": os.getpid(),
    }
    write_json(path, manifest)


def load_cluster_labels(k: int) -> pd.DataFrame:
    """Load K-means output from stage 02; never silently refit it downstream."""
    path = RESULTS_DIR / f"kmeans_labels_K{k}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run stage 02 before stage 03/04.")
    labels = pd.read_csv(path)
    required = {"source_row", "cluster"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Invalid K={k} cluster file: expected {required}.")
    return labels


def save_figure(fig, path: Path) -> None:
    """Save a high-resolution figure and close it to keep memory bounded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def confidence_interval_95(values) -> tuple[float, float, float, float]:
    """Return (mean, sd, ci_low, ci_high) for an array of scores."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else float("nan")
        return m, float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return mean, sd, float(ci[0]), float(ci[1])


def mutual_info_scored(X, y):
    """Pickleable feature scoring function with fixed random seed."""
    from sklearn.feature_selection import mutual_info_regression
    return mutual_info_regression(X, y, random_state=RANDOM_STATE)
