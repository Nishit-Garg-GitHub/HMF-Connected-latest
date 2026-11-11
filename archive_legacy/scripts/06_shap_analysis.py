# HMF ELECTROOXIDATION — SHAP Figure 12
# Generates paper-style SHAP bar charts for SMOTE + NN, K=3 and K=5.

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import shap

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(ROOT, "HMF_ML_Preprocessed.xlsx")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "KMeans_Oversampling", "SHAP")
RANDOM_STATE = 42
K_VALUES = [3, 5]
TEST_SIZE = 0.20
BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 30
SHAP_NSAMPLES = 200
EXPLAIN_CLASS = 0

# Paper Table S2 raw class counts; proportions are scaled to this dataset.
PAPER_COUNTS = {
    3: np.array([231, 115, 40], dtype=float),
    5: np.array([170, 85, 75, 34, 22], dtype=float),
}
os.makedirs(OUTPUT_DIR, exist_ok=True)


def scaled_counts(n, paper_counts):
    p = paper_counts / paper_counts.sum()
    raw = p * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[:n - counts.sum()]:
        counts[i] += 1
    # SMOTE requires >=2 per class.
    for i in range(len(counts)):
        if counts[i] >= 2:
            continue
        need = 2 - counts[i]
        counts[i] = 2
        for donor in np.argsort(-counts):
            if donor == i:
                continue
            take = min(need, max(0, counts[donor] - 2))
            counts[donor] -= take
            need -= take
            if need == 0:
                break
        if need:
            raise ValueError('Unable to construct valid class counts.')
    return counts


def make_labels(score, k):
    counts = scaled_counts(len(score), PAPER_COUNTS[k])
    order = np.argsort(score, kind='mergesort')
    y = np.empty(len(score), dtype=int)
    start = 0
    for cls, count in enumerate(counts):
        y[order[start:start + count]] = cls
        start += count
    return y


def load_data():
    df = pd.read_excel(DATA_FILE)
    mask = df['Conversion_pct'].notna() & df['Yield_pct'].notna()
    data = df.loc[mask].reset_index(drop=True)
    score = 0.5 * np.clip(data['Conversion_pct'].astype(float).to_numpy(), 0, 100) + 0.5 * np.clip(data['Yield_pct'].astype(float).to_numpy(), 0, 100)

    targets = {'Conversion_pct','Yield_pct','FE_pct','Selectivity_pct','FDCA_Target',
               'Yield_Greater_Than_Conversion','Yield_gt_Conversion','FE_Above_100','FE_gt_100',
               'Conversion_Above_100','Conversion_gt_100','Yield_Above_100','Yield_gt_100'}
    cols = [c for c in data.columns if c not in targets and data[c].notna().any() and data[c].nunique(dropna=True) > 1]
    X = data[cols].select_dtypes(include=[np.number]).copy()
    for c in list(X.columns):
        v = X[c].dropna()
        if len(v) == 0:
            X.drop(columns=c, inplace=True)
        elif v.nunique() <= 2 and (v.mean() < 0.02 or v.mean() > 0.98):
            X.drop(columns=c, inplace=True)
    return X, score


def train_model(X, score, k):
    y = make_labels(score, k)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    min_class = int(pd.Series(ytr).value_counts().min())
    kn = max(1, min(3, min_class - 1))
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=kn)),
        ('model', MLPClassifier(hidden_layer_sizes=(32,16), alpha=0.001,
                                learning_rate_init=0.001, max_iter=2000,
                                early_stopping=True, validation_fraction=0.15,
                                random_state=RANDOM_STATE))
    ])
    pipe.fit(Xtr, ytr)
    return pipe, Xtr.reset_index(drop=True), Xte.reset_index(drop=True), y


def shap_for_model(pipe, Xtr, Xte):
    background = Xtr.sample(min(BACKGROUND_SIZE, len(Xtr)), random_state=RANDOM_STATE)
    explain = Xte.sample(min(EXPLAIN_SIZE, len(Xte)), random_state=RANDOM_STATE)

    def predict_class(z):
        if not isinstance(z, pd.DataFrame):
            z = pd.DataFrame(z, columns=Xtr.columns)
        return pipe.predict_proba(z)[:, EXPLAIN_CLASS]

    explainer = shap.KernelExplainer(predict_class, background)
    vals = np.asarray(explainer.shap_values(explain, nsamples=SHAP_NSAMPLES, l1_reg='num_features(10)'))
    if vals.ndim == 3:
        vals = vals[:, :, 0] if vals.shape[-1] == 1 else vals[:, :, EXPLAIN_CLASS]
    if vals.ndim != 2:
        raise RuntimeError(f'Unexpected SHAP shape: {vals.shape}')
    return vals, explain


def panel(ax, vals, explain, k, label):
    imp = np.mean(np.abs(vals), axis=0)
    idx = np.argsort(imp)[-10:]
    names = explain.columns.to_numpy()[idx]
    values = imp[idx]
    order = np.argsort(values)
    names, values = names[order], values[order]
    colors = plt.cm.Reds(np.linspace(0.35, 0.85, len(values)))
    y = np.arange(len(values))
    bars = ax.barh(y, values, color=colors, height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Mean (average impact on model output magnitude)', fontsize=7)
    ax.tick_params(axis='x', labelsize=6.5); ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.18, linewidth=0.6); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    mx = max(float(values.max()), 1e-12)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + mx*0.012, b.get_y()+b.get_height()/2, f'+{v:.2f}', va='center', fontsize=6.5)
    ax.set_xlim(0, mx*1.22)
    ax.text(0.98, 0.02, label, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(f'NN model under SMOTE — K = {k}', fontsize=8.5, pad=5)
    return names, values


def main():
    X, score = load_data()
    print('Valid observations:', len(X))
    print('ML features:', len(X.columns))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.65))
    all_top = {}
    for ax, k, label in zip(axes, K_VALUES, ['(a)', '(b)']):
        print(f'\nK={k}')
        pipe, Xtr, Xte, y = train_model(X, score, k)
        print('Class distribution:', pd.Series(y).value_counts().sort_index().to_dict())
        vals, explain = shap_for_model(pipe, Xtr, Xte)
        pd.DataFrame(vals, columns=explain.columns).to_csv(os.path.join(OUTPUT_DIR, f'SHAP_values_K{k}.csv'), index=False)
        names, values = panel(ax, vals, explain, k, label)
        all_top[k] = (names, values)
        # Individual panel.
        fig1, ax1 = plt.subplots(figsize=(4.9, 3.25))
        panel(ax1, vals, explain, k, label)
        fig1.tight_layout()
        fig1.savefig(os.path.join(OUTPUT_DIR, f'Figure_12_SHAP_Bar_K{k}.png'), dpi=600, bbox_inches='tight')
        plt.close(fig1)
        print('Top features:')
        for n, v in zip(names[::-1], values[::-1]):
            print(f'  {n}: {v:.6f}')
    fig.text(0.5, 0.005, 'Figure 12. Bar chart of the NN model under the SMOTE (a) K = 3 and (b) K = 5.', ha='center', va='bottom', fontsize=7.5)
    plt.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.16, wspace=0.34)
    combined = os.path.join(OUTPUT_DIR, 'Figure_12_SHAP_Bar_Combined.png')
    fig.savefig(combined, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('\nSaved:', combined)


if __name__ == '__main__':
    main()
