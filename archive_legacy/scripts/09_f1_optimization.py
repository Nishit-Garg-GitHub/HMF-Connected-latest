# ============================================================
# 09_f1_optimization.py
#
# HMF ELECTROOXIDATION -> FDCA : PERFORMANCE-CLASS PREDICTION
#
# OBJECTIVE (following Li et al., ACS Sustain. Chem. Eng. 2025):
#   Predict the catalytic performance class of an experiment
#   (defined from HMF conversion and FDCA yield) from catalyst /
#   reaction-condition descriptors, and maximise the macro F1.
#
# WHAT IS DIFFERENT FROM 04/05/06:
#   * Classes come from conversion/yield (no paper-count rescaling).
#   * Several class definitions are evaluated and ALL are reported.
#   * Oversampling is applied ONLY to training folds (imblearn
#     pipeline) - validation folds are never oversampled.
#   * Rows from the same paper never appear in both train and test
#     (StratifiedGroupKFold on "Paper Title").
#   * Selection of model/sampler/hyper-parameters is done with
#     NESTED cross-validation so the headline number is not a
#     best-of-many-trials estimate.
#   * Conversion, Yield, FE, Selectivity are never input features.
#
# USAGE:
#   python 09_f1_optimization.py [stage1|nested|all]
# ============================================================

import re
import signal
import sys
import time
from contextlib import contextmanager
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "Nishith_Summary Table of different cell types.xlsx"
ML_FILE = ROOT / "HMF_ML_Preprocessed.xlsx"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_JOBS = -1

# ============================================================
# 1. CLASS DEFINITIONS
# ============================================================
# "perf" = min(conversion, yield): the paper's "dual indicator".
# Yield can never exceed conversion physically, so min() is the
# binding indicator when both are reported.


def with_perf(df):
    """Rows usable for classification. perf = min(conversion, yield).
    Yield cannot exceed conversion, so when only the yield is reported
    perf = yield exactly; this adds the rows that have a yield but no
    conversion. Rows with a conversion but no yield stay excluded
    (perf is only bounded above)."""
    sub = df.dropna(subset=["Yield_pct"]).copy().reset_index(drop=True)
    conv = sub["Conversion_pct"].clip(0, 100)
    yld = sub["Yield_pct"].clip(0, 100)
    sub["Perf"] = np.where(conv.notna(), np.minimum(conv, yld), yld)
    sub["Yld"] = yld
    return sub


def label_scheme(df, scheme):
    perf = df["Perf"]
    yld = df["Yld"]

    if scheme == "K3_paper":
        # Paper text: high >=80, middle 50-80, low <50 (dual indicators)
        return pd.cut(perf, [-1, 50, 80, 101], right=False, labels=False)

    if scheme == "K3_yield":
        # Yield tiers: <80, 80-95, >=95
        return pd.cut(yld, [-1, 80, 95, 101], right=False, labels=False)

    if scheme == "K4_yield":
        return pd.cut(yld, [-1, 50, 80, 95, 101], right=False, labels=False)

    if scheme == "K5_paper":
        # Paper: high-yield class is split in two when K grows.
        return pd.cut(
            perf, [-1, 50, 80, 90, 95, 101], right=False, labels=False
        )

    if scheme == "K2_high":
        # High-performance (both >=90) vs rest
        return (perf >= 90).astype(int)

    raise ValueError(scheme)


SCHEMES = ["K2_high", "K3_paper", "K3_yield", "K4_yield", "K5_paper"]

# ============================================================
# 2. FEATURES
# ============================================================

TARGET_LIKE = {
    "Conversion_pct", "Yield_pct", "FE_pct", "Selectivity_pct",
    "FDCA_Target", "Yield_gt_Conversion", "Conversion_gt_100",
    "Yield_gt_100", "FE_gt_100", "Potential_vs_RHE",
}


def text_flags(raw):
    """Descriptor flags from the free-text columns that DESCRIBE THE
    SETUP (catalyst, electrolyte, cell). 'Key Feature', 'Advantages'
    and 'Limitations' are NOT used: they describe the outcome."""
    cat = raw["Catalysts"].astype(str).str.lower()
    ele = raw["Electrolyte"].astype(str).str.lower()
    cel = raw["Cell Type"].astype(str).str.lower()
    anode = cat.str.extract(r"anode[^;,]*?:?\s*([^;]*)", expand=False).fillna(cat)

    f = pd.DataFrame(index=raw.index)

    def has(s, pat):
        return s.str.contains(pat, regex=True, na=False).astype(int)

    f["Sup_NickelFoam"] = has(cat, r"nickel foam|\bnf\b|ni foam")
    f["Sup_CarbonPaperCloth"] = has(cat, r"carbon (paper|cloth|fiber|felt)|\bcp\b|\bcc\b")
    f["Sup_FTO_Ti"] = has(cat, r"\bfto\b|titanium|\bti\b|ti foil|ti mesh")
    f["Sup_CuFoam"] = has(cat, r"cu foam|copper foam")
    f["Sup_Foam"] = has(cat, r"foam")
    f["Bifunctional"] = has(cat, r"bifunctional|both electrodes|anode and cathode")
    f["Nano_Struct"] = has(cat, r"nanosheet|nanowire|nanorod|nanoarray|nanotube|nanoparticle|nanocone|nanoflower")
    f["Hetero"] = has(cat, r"heterojunction|heterostructure|interface")
    f["Doped"] = has(cat, r"doped|doping|-doped|dopant")
    f["Vacancy"] = has(cat, r"vacanc")
    f["Amorphous"] = has(cat, r"amorphous")
    f["Bimetal"] = has(anode, r"nife|nico|cofe|nimn|nicu|conimo|nimo|fe-|-fe|nico|niv|nifev|nicr")
    f["Pt_Cathode"] = has(cat, r"cathode[^;]*\bpt\b")
    f["Pt_Anode"] = has(anode, r"\bpt\b|pt/|pt-")
    f["N_Elements_Anode"] = anode.apply(
        lambda s: len(set(re.findall(r"\b(?:Ni|Co|Fe|Cu|Pt|Pd|Au|Ru|Ir|Mn|Mo|W|Ce|Zn|Ag|V|Cr|Ti|Rh|Sn|Bi|Cd|La)\b", s, flags=re.I)))
    )
    f["Elec_Borate_Phosphate"] = has(ele, r"borate|phosphate|buffer")
    f["Elec_Carbonate"] = has(ele, r"carbonate|hco3|k2co3|khco3|nahco3")
    f["Elec_NaOH"] = has(ele, r"naoh")
    f["Elec_KOH"] = has(ele, r"koh")
    f["Elec_Acidic_Neutral"] = has(ele, r"h2so4|hclo4|neutral|ph 7|acid")
    f["Elec_Cosolvent"] = has(ele, r"acetonitrile|dmso|ethanol|methanol|organic")
    f["Cell_HType"] = has(cel, r"h-type|h-shaped|h-cell|h cell")
    f["Cell_Membrane_Free"] = has(cel, r"membrane-free|undivided|single")
    f["Cell_Electrolyzer"] = has(cel, r"electrolyzer|electrolyser|two-electrode|mea|zero-gap")
    f["Cell_ThreeElec"] = has(cel, r"three")
    f["Cell_Flow"] = has(cel, r"flow")
    f["Cell_Unspecified"] = has(cel, r"not specified")
    return f


def engineered(df):
    """Ratios / interactions that tree models cannot form easily."""
    e = pd.DataFrame(index=df.index)
    hmf = df["HMF_Concentration_mM"]
    ele = df["Electrolyte_Concentration_M"] * 1000.0
    e["Log_HMF_mM"] = np.log10(hmf.clip(lower=0.01))
    e["HMF_to_Electrolyte"] = np.log10((hmf / ele.replace(0, np.nan)).clip(lower=1e-4))
    e["Pot_x_Alkaline"] = df["Potential_V"] * df["Alkaline_Electrolyte"]
    e["Pot_x_TEMPO"] = df["Potential_V"] * df["TEMPO_binary"]
    e["Log_Electrolyte_M"] = np.log10(df["Electrolyte_Concentration_M"].clip(lower=1e-3))
    e["Noble_Metal"] = df[["Catalyst_Pt", "Catalyst_Pd", "Catalyst_Au", "Catalyst_Ru", "Catalyst_Ir", "Catalyst_Ag"]].max(axis=1)
    e["Base_Metal"] = df[["Catalyst_Ni", "Catalyst_Co", "Catalyst_Fe", "Catalyst_Cu", "Catalyst_Mn"]].max(axis=1)
    e["N_Metals"] = df[[c for c in df.columns if c.startswith("Catalyst_") and c[9:] in
                        ("Ni", "Co", "Fe", "Cu", "Pt", "Pd", "Au", "Ru", "Ir", "Mn", "Mo", "W", "Ce", "Zn", "Ag")]].sum(axis=1)
    return e


def load_data(extra_text=True):
    raw = pd.read_excel(RAW_FILE)
    ml = pd.read_excel(ML_FILE)
    assert len(raw) == len(ml)
    df = ml.copy()
    df["Paper"] = raw["Paper Title"].astype(str).str.strip().str.lower()
    if extra_text:
        flags = text_flags(raw)
        for c in flags.columns:
            df[c] = flags[c].values
        eng = engineered(df)
        for c in eng.columns:
            df[c] = eng[c].values
    return df


def feature_columns(df, extra_text=True):
    cols = []
    for c in df.columns:
        if c in TARGET_LIKE or c == "Paper":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().sum() == 0 or df[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols


# ============================================================
# 3. SAFE OVERSAMPLERS (tiny classes inside CV folds)
# ============================================================


class SafeOver(BaseEstimator):
    """SMOTE-family sampler that degrades gracefully when a class has
    very few training samples (k_neighbors < minority size)."""

    def __init__(self, kind="smote", k=3, random_state=SEED):
        self.kind = kind
        self.k = k
        self.random_state = random_state

    def fit_resample(self, X, y):
        counts = np.bincount(y)
        counts = counts[counts > 0]
        n_min = counts.min()
        if len(counts) < 2 or n_min < 2:
            return RandomOverSampler(random_state=self.random_state).fit_resample(X, y)
        k = int(max(1, min(self.k, n_min - 1)))
        try:
            if self.kind == "smote":
                s = SMOTE(k_neighbors=k, random_state=self.random_state)
            elif self.kind == "bsmote":
                s = BorderlineSMOTE(k_neighbors=k, m_neighbors=max(2, min(5, len(y) - 1)),
                                    random_state=self.random_state)
            elif self.kind == "adasyn":
                s = ADASYN(n_neighbors=k, random_state=self.random_state)
            else:
                s = RandomOverSampler(random_state=self.random_state)
            return s.fit_resample(X, y)
        except Exception:
            return RandomOverSampler(random_state=self.random_state).fit_resample(X, y)

    def get_params(self, deep=True):
        return {"kind": self.kind, "k": self.k, "random_state": self.random_state}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


# ============================================================
# 4. MODEL ZOO  (each entry = one hyper-parameter setting)
# ============================================================
# name -> (estimator, needs_scaling, supports_class_weight_variant)


def model_zoo():
    z = {}
    for leaf in (1, 3):
        z[f"DT_l{leaf}"] = (DecisionTreeClassifier(min_samples_leaf=leaf, random_state=SEED), False)
    for mf in ("sqrt", 0.5):
        for leaf in (1, 2):
            z[f"RF_{mf}_l{leaf}"] = (RandomForestClassifier(
                n_estimators=300, max_features=mf, min_samples_leaf=leaf,
                random_state=SEED, n_jobs=1), False)
            z[f"RFbal_{mf}_l{leaf}"] = (RandomForestClassifier(
                n_estimators=300, max_features=mf, min_samples_leaf=leaf,
                class_weight="balanced_subsample", random_state=SEED, n_jobs=1), False)
            z[f"ET_{mf}_l{leaf}"] = (ExtraTreesClassifier(
                n_estimators=400, max_features=mf, min_samples_leaf=leaf,
                random_state=SEED, n_jobs=1), False)
            z[f"ETbal_{mf}_l{leaf}"] = (ExtraTreesClassifier(
                n_estimators=400, max_features=mf, min_samples_leaf=leaf,
                class_weight="balanced", random_state=SEED, n_jobs=1), False)
    for d in (1, 2, 3):
        z[f"GB_d{d}"] = (GradientBoostingClassifier(
            n_estimators=150, max_depth=d, learning_rate=0.05, subsample=0.8,
            random_state=SEED), False)
    for C in (1, 10, 100):
        z[f"SVM_C{C}"] = (SVC(C=C, gamma="scale", random_state=SEED), True)
        z[f"SVMbal_C{C}"] = (SVC(C=C, gamma="scale", class_weight="balanced", random_state=SEED), True)
    for C in (0.03, 0.3, 3):
        z[f"LR_C{C}"] = (LogisticRegression(C=C, max_iter=3000), True)
        z[f"LRbal_C{C}"] = (LogisticRegression(C=C, max_iter=3000, class_weight="balanced"), True)
    for k in (3, 7):
        z[f"KNN_k{k}"] = (KNeighborsClassifier(n_neighbors=k, weights="distance"), True)
    for hl in ((32,), (64, 32)):
        for a in (1e-3, 1e-1):
            z[f"NN_{'x'.join(map(str, hl))}_a{a}"] = (MLPClassifier(
                hidden_layer_sizes=hl, alpha=a, max_iter=1500, random_state=SEED), True)
    return z


SAMPLERS = {"none": None, "smote": "smote", "bsmote": "bsmote", "adasyn": "adasyn", "ros": "ros"}
TOPK = (0, 20)  # 0 = keep all features, else univariate top-k (fit inside each fold)


def build_pipeline(model_name, sampler_name, topk=0):
    base, needs_scale = model_zoo()[model_name]
    steps = [("imp", SimpleImputer(strategy="median", keep_empty_features=True))]
    if needs_scale:
        steps.append(("sc", StandardScaler()))
    if topk:
        steps.append(("sel", SelectKBest(f_classif, k=topk)))
    if SAMPLERS[sampler_name] is not None:
        steps.append(("os", SafeOver(kind=SAMPLERS[sampler_name])))
    steps.append(("clf", clone(base)))
    return Pipeline(steps)


def candidate_list(n_features):
    """(model, sampler, topk). Class-weighted models ('bal') replace
    oversampling, so they only run with sampler 'none'."""
    cands = []
    for m in model_zoo():
        samplers = ["none"] if "bal" in m else list(SAMPLERS)
        for s in samplers:
            for k in TOPK:
                if k and k >= n_features:
                    continue
                cands.append((m, s, k))
    return cands


# ============================================================
# 5. CROSS-VALIDATION HELPERS
# ============================================================


class CandidateTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Wall-clock guard so one pathological candidate (e.g. a sampler
    stuck on a 2-sample class) cannot stall a whole parallel run."""
    def _raise(*_):
        raise CandidateTimeout()
    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def grouped_splits(y, groups, n_splits, seed):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def fit_predict(pipe, Xtr, ytr, Xte):
    pipe = clone(pipe)
    pipe.fit(Xtr, ytr)
    return pipe.predict(Xte)


def cv_f1(pipe, X, y, groups, seeds, n_splits=5):
    """Repeated grouped CV; F1 on the pooled out-of-fold predictions of
    each repeat (stable for tiny classes)."""
    macro, weighted = [], []
    for seed in seeds:
        preds = np.full(len(y), -1)
        for tr, te in grouped_splits(y, groups, n_splits, seed):
            preds[te] = fit_predict(pipe, X[tr], y[tr], X[te])
        macro.append(f1_score(y, preds, average="macro"))
        weighted.append(f1_score(y, preds, average="weighted"))
    return np.array(macro), np.array(weighted)


def prepare(df, features, scheme):
    sub = with_perf(df)
    X = sub[features].to_numpy(dtype=float)
    y = label_scheme(sub, scheme).to_numpy().astype(int)
    return sub, X, y, sub["Paper"].to_numpy()


# ============================================================
# 6. STAGE 1 : SCREEN ALL CANDIDATES, ALL SCHEMES
# ============================================================


def stage1(df, features, seeds):
    rows = []
    for scheme in SCHEMES:
        sub, X, y, groups = prepare(df, features, scheme)
        print(f"\n[{scheme}] n={len(y)} class counts={np.bincount(y).tolist()}", flush=True)

        def run(c):
            mac, wtd = cv_f1(build_pipeline(*c), X, y, groups, seeds)
            return dict(scheme=scheme, model=c[0], sampler=c[1], topk=c[2],
                        macro_f1=mac.mean(), macro_sd=mac.std(), weighted_f1=wtd.mean())

        t = time.time()
        res = Parallel(n_jobs=N_JOBS)(delayed(run)(c) for c in candidate_list(X.shape[1]))
        rows += res
        base = f1_score(y, np.full(len(y), np.bincount(y).argmax()), average="macro")
        print(f"  {len(res)} candidates in {time.time()-t:.0f}s; majority-class macro-F1={base:.3f}")
        for b in sorted(res, key=lambda r: -r["macro_f1"])[:3]:
            print(f"  top: {b['model']:16s} {b['sampler']:7s} k={b['topk']:<3d} "
                  f"macro={b['macro_f1']:.3f}+-{b['macro_sd']:.3f} weighted={b['weighted_f1']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "stage1_screen.csv", index=False)
    return out


# ============================================================
# 7. STAGE 2 : NESTED CV  (honest estimate of "pick the best")
# ============================================================


def nested_cv(df, features, scheme, outer_seeds):
    """Outer: grouped 5-fold. Inner: the best candidate out of the
    WHOLE pool is selected with grouped 3-fold CV on the outer-train
    part only; outer-test data never influences the choice."""
    sub, X, y, groups = prepare(df, features, scheme)
    cands = candidate_list(X.shape[1])

    def one_outer(seed, tr, te):
        Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
        best, best_score = None, -1.0
        for c in cands:
            try:
                with time_limit(30):
                    mac, _ = cv_f1(build_pipeline(*c), Xtr, ytr, gtr, seeds=[seed + 1000], n_splits=3)
            except CandidateTimeout:
                print(f"TIMEOUT inner {scheme} {c}", flush=True)
                continue
            if mac[0] > best_score:
                best, best_score = c, mac[0]
        try:
            with time_limit(120):
                pred = fit_predict(build_pipeline(*best), Xtr, ytr, X[te])
        except CandidateTimeout:
            print(f"TIMEOUT final {scheme} {best}", flush=True)
            pred = np.full(len(te), np.bincount(ytr).argmax())
        return te, pred, best

    macro, weighted, chosen = [], [], []
    for seed in outer_seeds:
        splits = grouped_splits(y, groups, 5, seed)
        res = Parallel(n_jobs=N_JOBS)(delayed(one_outer)(seed, tr, te) for tr, te in splits)
        preds = np.full(len(y), -1)
        for te, pred, best in res:
            preds[te] = pred
            chosen.append(best)
        macro.append(f1_score(y, preds, average="macro"))
        weighted.append(f1_score(y, preds, average="weighted"))
    return np.array(macro), np.array(weighted), chosen


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    from collections import Counter

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    df = load_data()
    feats = feature_columns(df)
    print(f"Features: {len(feats)}   rows: {len(df)}  usable rows (have yield): "
          f"{df['Yield_pct'].notna().sum()}")

    if mode in ("stage1", "all"):
        stage1(df, feats, seeds=list(range(5)))

    if mode in ("nested", "all"):
        rows = []
        for scheme in SCHEMES:
            t = time.time()
            mac, wtd, chosen = nested_cv(df, feats, scheme, outer_seeds=list(range(100, 102)))
            top = Counter(chosen).most_common(2)
            print(f"[{scheme}] NESTED macro-F1={mac.mean():.3f}+-{mac.std():.3f} "
                  f"weighted-F1={wtd.mean():.3f}  chosen={top}  ({time.time()-t:.0f}s)", flush=True)
            rows.append(dict(scheme=scheme, nested_macro_f1=mac.mean(), nested_macro_sd=mac.std(),
                             nested_weighted_f1=wtd.mean(), most_chosen=str(top)))
        pd.DataFrame(rows).to_csv(OUT_DIR / "nested_results.csv", index=False)
