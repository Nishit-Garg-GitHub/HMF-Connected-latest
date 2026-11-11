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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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


def make_classifier(name: str, seed: int = RANDOM_STATE):
    """Construct small, deterministic models for fast literature-scale CV."""
    models = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=2, class_weight="balanced",
            random_state=seed,
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
            probability=True, random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.02, max_iter=300,
            early_stopping=True, n_iter_no_change=15, random_state=seed,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=3, subsample=0.8,
            random_state=seed,
        ),
    }
    if name not in models:
        raise KeyError(f"Unknown classifier: {name}")
    return models[name]


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
