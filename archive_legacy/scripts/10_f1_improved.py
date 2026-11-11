# ============================================================
# 10_f1_improved.py
#
# HMF ELECTROOXIDATION -> FDCA : IMPROVED PERFORMANCE-CLASS F1
#
# Builds on 09_f1_optimization.py (same data, class schemes,
# publication-grouped splits and nested protocol) and adds:
#   * Decision rules tuned for macro-F1 inside the inner folds:
#     per-class probability weights for classifiers, and class
#     cut-offs for regressors.
#   * Regress-then-bin candidates: predict the continuous
#     performance (or yield) and cut it into classes, so the
#     model learns from the full outcome scale.
#   * Soft-vote ensemble of the three best inner candidates.
#   * Cleaned descriptors: Pt counts only when it is on the anode
#     (the raw Pt flag mostly marks Pt counter electrodes), plus
#     reporting flags for temperature, potential and current.
#
# Outer test folds never influence model choice, weights or
# cut-offs. Outer seeds 100 and 101 give the same outer splits as
# 09_f1_optimization.py, so the scores compare directly.
#
# USAGE:
#   python 10_f1_improved.py [--record] [--rhe] [--text] [scheme ...]
#   schemes: K2_high K3_paper K3_yield K4_yield K5_paper KM3 KM4 KM5
# ============================================================

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f1base", ROOT / "09_f1_optimization.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

SEED = 42
OUTER_SEEDS = [100, 101, 102, 103, 104]
INNER_SEEDS = [1000, 1001]
OUT_DIR = ROOT / "results"

# "group": records of one publication never straddle train/test (default).
# "record": stratified record-level splits, as in the reference study;
#           records of the same publication can appear on both sides.
SPLIT = "group"

# Regression target per scheme (the quantity the classes are cut from)
REG_TARGET = {
    "K2_high": "Perf", "K3_paper": "Perf", "K5_paper": "Perf",
    "K3_yield": "Yld", "K4_yield": "Yld",
    "KM3": "Perf", "KM4": "Perf", "KM5": "Perf",
}


def splits(y, groups, n_splits, seed):
    if SPLIT == "record":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y))
    return base.grouped_splits(y, groups, n_splits, seed)


def prepare(df, feats, scheme):
    """Threshold schemes come from 09. KM<k> schemes follow the reference
    study: K-means on standardized conversion and yield of the records
    that report both, clusters numbered by increasing mean performance."""
    if not scheme.startswith("KM"):
        return base.prepare(df, feats, scheme)
    sub = base.with_perf(df)
    sub = sub[sub["Conversion_pct"].notna()].reset_index(drop=True)
    Z = sub[["Conversion_pct", "Yield_pct"]].to_numpy(dtype=float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    km = KMeans(n_clusters=int(scheme[2:]), n_init=50, random_state=SEED).fit(Z)
    order = np.argsort([sub["Perf"][km.labels_ == c].mean() for c in range(km.n_clusters)])
    y = np.argsort(order)[km.labels_]
    return sub, sub[feats].to_numpy(dtype=float), y, sub["Paper"].to_numpy()

# ============================================================
# 1. FEATURES
# ============================================================


def anode_text(raw):
    cat = raw["Catalysts"].astype(str)
    return cat.str.split(r"[Cc]athode").str[0].str.replace(r"(?i)anode\s*:?", "", regex=True)


def load_features(extra=()):
    """extra: "rhe" adds the potential-reference flag; "text" adds 12
    latent components of character n-grams of the anode catalyst name
    (fitted on the descriptor text only, never on outcomes)."""
    df = base.load_data()
    feats = base.feature_columns(df)
    # Pt: anode only. Noble metal: recomputed without counter electrodes.
    df["Noble_Metal"] = df[["Pt_Anode", "Catalyst_Pd", "Catalyst_Au", "Catalyst_Ru",
                            "Catalyst_Ir", "Catalyst_Ag"]].max(axis=1)
    df["N_Metals"] = df["N_Metals"] - df["Catalyst_Pt"] + df["Pt_Anode"]
    for col, name in (("Temperature_C", "Reported_Temperature"),
                      ("Potential_V", "Reported_Potential"),
                      ("Current_mA", "Reported_Current")):
        df[name] = df[col].notna().astype(int)
    feats = [f for f in feats if f not in ("Catalyst_Pt", "Pt_Cathode")]
    feats += ["Reported_Temperature", "Reported_Potential", "Reported_Current"]
    if "rhe" in extra:
        feats.append("Potential_vs_RHE")
    if "text" in extra:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        raw = pd.read_excel(base.RAW_FILE)
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=False,
                             min_df=2, sublinear_tf=True).fit_transform(anode_text(raw))
        Z = TruncatedSVD(n_components=12, random_state=SEED).fit_transform(tf)
        for i in range(Z.shape[1]):
            df[f"AnodeText_{i}"] = Z[:, i]
            feats.append(f"AnodeText_{i}")
    return df, feats


# ============================================================
# 2. CANDIDATES
# ============================================================
# name -> (estimator, needs_scaling, kind) ; kind = "clf" | "reg"


def zoo():
    z = {}
    for leaf in (1, 3):
        z[f"RFbal_l{leaf}"] = (RandomForestClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=leaf,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=1), False, "clf")
        z[f"ETbal_l{leaf}"] = (ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=leaf,
            class_weight="balanced", random_state=SEED, n_jobs=1), False, "clf")
    z["GB_d2"] = (GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
        random_state=SEED), False, "clf")
    for C in (0.03, 0.3, 3):
        z[f"LRbal_C{C}"] = (LogisticRegression(C=C, max_iter=3000, class_weight="balanced"), True, "clf")
    for C in (1, 10):
        z[f"SVMbal_C{C}"] = (SVC(C=C, gamma="scale", class_weight="balanced", probability=True,
                                 random_state=SEED), True, "clf")
    z["KNN_k7"] = (KNeighborsClassifier(n_neighbors=7, weights="distance"), True, "clf")
    z["NN_32_a0.1"] = (MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=1500,
                                     random_state=SEED), True, "clf")
    for leaf in (1, 3):
        z[f"RFreg_l{leaf}"] = (RandomForestRegressor(
            n_estimators=400, max_features=0.5, min_samples_leaf=leaf,
            random_state=SEED, n_jobs=1), False, "reg")
        z[f"ETreg_l{leaf}"] = (ExtraTreesRegressor(
            n_estimators=400, max_features=0.5, min_samples_leaf=leaf,
            random_state=SEED, n_jobs=1), False, "reg")
    z["GBreg_d2"] = (GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
        loss="huber", random_state=SEED), False, "reg")
    z["Ridge_a10"] = (Ridge(alpha=10.0), True, "reg")
    z["SVR_C10"] = (SVR(C=10.0, epsilon=2.0), True, "reg")
    return z


ZOO = zoo()


def candidate_list():
    cands = []
    for m, (_, _, kind) in ZOO.items():
        for k in (0, 20):
            if kind == "clf" and "bal" not in m:
                for s in ("none", "smote"):
                    cands.append((m, s, k))
            else:
                cands.append((m, "none", k))
    return cands


def build(model, sampler, topk):
    est, scale, kind = ZOO[model]
    steps = [("imp", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("sc", StandardScaler()))
    if topk:
        steps.append(("sel", SelectKBest(f_classif, k=topk)))
    if sampler != "none":
        steps.append(("os", base.SafeOver(kind=sampler)))
    steps.append(("est", clone(est)))
    return Pipeline(steps), kind


# ============================================================
# 3. SCORES AND MACRO-F1 DECISION RULES
# ============================================================


def fit_scores(model, sampler, topk, Xtr, ytr, rtr, Xte, n_classes):
    """Class-probability matrix (clf) or continuous prediction (reg).
    Feature selection for regressors uses the class labels."""
    pipe, kind = build(model, sampler, topk)
    if kind == "clf":
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)
        full = np.zeros((len(Xte), n_classes))
        full[:, pipe.classes_] = p
        return full
    if topk:
        pre = Pipeline(pipe.steps[:-1]).fit(Xtr, ytr)
        est = clone(pipe.steps[-1][1]).fit(pre.transform(Xtr), rtr)
        return est.predict(pre.transform(Xte))
    pipe.fit(Xtr, rtr)
    return pipe.predict(Xte)


def is_reg(model):
    return ZOO[model][2] == "reg"


def apply_rule(scores, rule, reg):
    if reg:
        return np.searchsorted(rule, scores, side="right")
    return np.argmax(scores * rule, axis=1)


def tune_weights(P, y, n_classes):
    """Coordinate ascent on per-class probability weights."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 19))
    w = np.ones(n_classes)
    best = f1_score(y, np.argmax(P * w, axis=1), average="macro")
    for _ in range(3):
        for c in range(1, n_classes):
            for g in grid:
                trial = w.copy()
                trial[c] = g
                s = f1_score(y, np.argmax(P * trial, axis=1), average="macro")
                if s > best + 1e-9:
                    best, w = s, trial
    return w, best


def tune_cuts(r, y, n_classes):
    """Coordinate ascent on ordered cut-offs of a continuous prediction."""
    qs = np.unique(np.quantile(r, np.linspace(0.02, 0.98, 49)))
    # start from class-frequency quantiles
    cum = np.cumsum(np.bincount(y, minlength=n_classes))[:-1] / len(y)
    cuts = np.quantile(r, cum)
    best = f1_score(y, np.searchsorted(cuts, r, side="right"), average="macro")
    for _ in range(3):
        for j in range(n_classes - 1):
            lo = cuts[j - 1] if j > 0 else -np.inf
            hi = cuts[j + 1] if j < n_classes - 2 else np.inf
            for q in qs:
                if not lo <= q <= hi:
                    continue
                trial = cuts.copy()
                trial[j] = q
                s = f1_score(y, np.searchsorted(trial, r, side="right"), average="macro")
                if s > best + 1e-9:
                    best, cuts = s, trial
    return cuts, best


def tune(scores, y, n_classes, reg):
    return tune_cuts(scores, y, n_classes) if reg else tune_weights(scores, y, n_classes)


# ============================================================
# 4. NESTED CV
# ============================================================


def inner_oof(c, X, y, r, groups, n_classes):
    """Out-of-fold scores over INNER_SEEDS repeats, stacked row-wise."""
    out = []
    for s in INNER_SEEDS:
        oof = np.zeros((len(y), n_classes)) if not is_reg(c[0]) else np.zeros(len(y))
        for tr, te in splits(y, groups, 3, s):
            oof[te] = fit_scores(*c, X[tr], y[tr], r[tr], X[te], n_classes)
        out.append(oof)
    return np.concatenate(out), np.tile(y, len(INNER_SEEDS))


def one_outer(X, y, r, groups, tr, te, n_classes, cands):
    Xtr, ytr, rtr, gtr = X[tr], y[tr], r[tr], groups[tr]
    inner = {}
    for c in cands:
        oof, yy = inner_oof(c, Xtr, ytr, rtr, gtr, n_classes)
        rule, score = tune(oof, yy, n_classes, is_reg(c[0]))
        inner[c] = (score, oof, rule)

    # soft-vote ensemble of the three best classifier candidates
    clf_rank = sorted((c for c in inner if not is_reg(c[0])), key=lambda c: -inner[c][0])[:3]
    ens_oof = np.mean([inner[c][1] for c in clf_rank], axis=0)
    ens_rule, ens_score = tune_weights(ens_oof, yy, n_classes)

    best = max(inner, key=lambda c: inner[c][0])
    if ens_score > inner[best][0]:
        P = np.mean([fit_scores(*c, Xtr, ytr, rtr, X[te], n_classes) for c in clf_rank], axis=0)
        return te, apply_rule(P, ens_rule, reg=False), ("ENS3",) + tuple(m for m, _, _ in clf_rank)
    s = fit_scores(*best, Xtr, ytr, rtr, X[te], n_classes)
    return te, apply_rule(s, inner[best][2], is_reg(best[0])), best


def nested(df, feats, scheme):
    sub, X, y, groups = prepare(df, feats, scheme)
    r = sub[REG_TARGET[scheme]].to_numpy(dtype=float)
    n_classes = int(y.max()) + 1
    cands = candidate_list()
    jobs = [(seed, tr, te) for seed in OUTER_SEEDS
            for tr, te in splits(y, groups, 5, seed)]
    res = Parallel(n_jobs=-1)(
        delayed(one_outer)(X, y, r, groups, tr, te, n_classes, cands) for _, tr, te in jobs)
    per_seed = {}
    for (seed, _, _), (te, pred, chosen) in zip(jobs, res):
        per_seed.setdefault(seed, [np.full(len(y), -1), []])
        per_seed[seed][0][te] = pred
        per_seed[seed][1].append(chosen)
    rows = []
    for seed, (pred, chosen) in per_seed.items():
        rows.append(dict(scheme=scheme, seed=seed,
                         macro_f1=f1_score(y, pred, average="macro"),
                         weighted_f1=f1_score(y, pred, average="weighted"),
                         per_class_f1=np.round(f1_score(y, pred, average=None), 3).tolist(),
                         chosen=chosen))
    return rows


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    from collections import Counter

    args = sys.argv[1:]
    if "--record" in args:
        SPLIT = "record"
        args.remove("--record")
    extra = [a[2:] for a in args if a in ("--rhe", "--text")]
    args = [a for a in args if a not in ("--rhe", "--text")]
    schemes = args or base.SCHEMES
    df, feats = load_features(extra)
    print(f"Features: {len(feats)}  candidates: {len(candidate_list())}", flush=True)
    all_rows = []
    for scheme in schemes:
        t = time.time()
        rows = nested(df, feats, scheme)
        all_rows += rows
        r = pd.DataFrame(rows)
        same = r[r.seed.isin([100, 101])]
        top = Counter(c for row in rows for c in row["chosen"]).most_common(3)
        print(f"[{scheme}] macro-F1 seeds100-101={same.macro_f1.mean():.3f} "
              f"all seeds={r.macro_f1.mean():.3f}+-{r.macro_f1.std(ddof=0):.3f} "
              f"weighted={r.weighted_f1.mean():.3f} per-class={rows[0]['per_class_f1']} "
              f"chosen={top} ({time.time()-t:.0f}s)", flush=True)
        name = "improved_nested_results" + ("_record" if SPLIT == "record" else "")
        name += "".join("_" + e for e in extra)
        if args:
            name += "_" + "_".join(args)
        pd.DataFrame(all_rows).to_csv(OUT_DIR / f"{name}.csv", index=False)
