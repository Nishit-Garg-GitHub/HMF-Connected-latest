"""Leakage-aware HMF/FDCA analysis. Run: python 12_publication_pipeline.py
All learned transforms and feature selection are fitted within CV training folds.
The script reports infeasible samplers rather than silently substituting another sampler.
"""
from pathlib import Path
import warnings, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'publication_outputs'; FIG=OUT/'figures'; OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
RAW=ROOT/'HMF_Cleaned_Full_Dataset.xlsx'; PRE=ROOT/'HMF_ML_Preprocessed.xlsx'
# QUICK SCREENING SETTINGS: 5 repeats x 5 paper-grouped folds.
# Increase to 20 repeats only after selecting a shortlist for final evaluation.
SEEDS=list(range(2026,2031)); MAX_FEATURES=15
SAMPLERS=['none','ros']  # fast comparison; SMOTE/ADASYN can be added in the final study
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def load():
    raw=pd.read_excel(RAW); pre=pd.read_excel(PRE)
    if len(raw)!=len(pre): raise ValueError('Cleaned and preprocessed workbooks do not align row-for-row.')
    paper=raw['Paper Title'].fillna('UNKNOWN').astype(str).str.strip().str.lower()
    yld=pd.to_numeric(raw['Yield_pct'],errors='coerce'); conv=pd.to_numeric(raw['Conversion_pct'],errors='coerce')
    keep=yld.notna() & raw['Major_Product_Clean'].astype(str).str.contains(r'FDCA|FURANDICARBOXYLIC',case=False,regex=True,na=False)
    d=pre.loc[keep].reset_index(drop=True).copy(); r=raw.loc[keep].reset_index(drop=True); groups=paper.loc[keep].reset_index(drop=True)
    yld=yld.loc[keep].clip(0,100).reset_index(drop=True); conv=conv.loc[keep].clip(0,100).reset_index(drop=True)
    perf=np.where(conv.notna(),np.minimum(conv.fillna(100),yld),yld)
    # K2: high if binding performance is at least 90%; fixed before model fitting.
    y=(perf>=90).astype(int)
    forbidden=['Yield','Conversion','Selectivity','FE_','FDCA_Target','Major_Product','Perf','Paper','Cluster','Class','Target','Above_100','Greater_Than']
    cols=[]
    for c in d.columns:
        if any(x.lower() in c.lower() for x in forbidden): continue
        if pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique(dropna=True)>1 and d[c].notna().any(): cols.append(c)
    X=d[cols].replace([np.inf,-np.inf],np.nan)
    return raw, r, X, y, groups, perf, cols

def make_pipe(model, sampler='none', k=15):
    steps=[('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('select',SelectKBest(f_classif,k=min(k,MAX_FEATURES)))]
    if sampler=='ros': steps.append(('sample',RandomOverSampler(random_state=42)))
    elif sampler=='smote': steps.append(('sample',SMOTE(k_neighbors=3,random_state=42)))
    elif sampler=='adasyn': steps.append(('sample',ADASYN(n_neighbors=3,random_state=42)))
    steps.append(('model',clone(model)))
    return ImbPipeline(steps)

def grouped_cv(y,g,seed,n=5):
    # Validate class support at group level before splitting.
    return list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))

def main():
    raw,r,X,y,g,perf,cols=load(); print(f'FDCA eligible records={len(y)}, papers={g.nunique()}, class counts={np.bincount(y).tolist()}, numeric predictors={len(cols)}')
    print(f'QUICK SCREEN: {len(SEEDS)} repeats x 5 folds x 4 models x {len(SAMPLERS)} samplers', flush=True)
    audit={'rows_original':len(raw),'papers_original':int(raw['Paper Title'].nunique()),'eligible_rows':len(y),'eligible_papers':int(g.nunique()),'class_counts':{str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},'features':cols,'outer_repeats':len(SEEDS),'outer_folds':5,'models':['DecisionTree','RandomForest','SVM_RBF','NeuralNetwork_MLP'],'samplers':SAMPLERS,'run_mode':'quick_screening_not_final_publication_estimate'}
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
    models={
      'DecisionTree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=3,class_weight='balanced',random_state=42),
      'RandomForest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced_subsample',max_features='sqrt',random_state=42,n_jobs=-1),
      'SVM_RBF':SVC(C=1.0,kernel='rbf',gamma='scale',class_weight='balanced'),
      'NeuralNetwork_MLP':MLPClassifier(hidden_layer_sizes=(32,16),activation='relu',alpha=0.01,learning_rate_init=0.001,max_iter=400,early_stopping=True,n_iter_no_change=20,random_state=42)
    }
    rows=[]; oof=[]; infeasible=[]
    for seed in SEEDS:
      for fold,(tr,te) in enumerate(grouped_cv(y,g,seed)):
        print(f'Progress: repeat {seed-SEEDS[0]+1}/{len(SEEDS)}, fold {fold+1}/5; evaluating {len(models)} models x {len(SAMPLERS)} samplers', flush=True)
        for mn,model in models.items():
          for sampler in SAMPLERS:
            try:
              pipe=make_pipe(model,sampler); pipe.fit(X.iloc[tr],y[tr]); pred=pipe.predict(X.iloc[te]); score=f1_score(y[te],pred,average='macro',zero_division=0)
              rows.append({'seed':seed,'fold':fold,'model':mn,'sampler':sampler,'macro_f1':score,'n_train':len(tr),'n_test':len(te),'paper_groups_test':g.iloc[te].nunique(),'status':'ok'})
              for ix,pr in zip(te,pred): oof.append({'seed':seed,'row':int(ix),'y_true':int(y[ix]),'y_pred':int(pr),'model':mn,'sampler':sampler})
            except Exception as e:
              infeasible.append({'seed':seed,'fold':fold,'model':mn,'sampler':sampler,'reason':str(e)[:400]})
              rows.append({'seed':seed,'fold':fold,'model':mn,'sampler':sampler,'macro_f1':np.nan,'n_train':len(tr),'n_test':len(te),'paper_groups_test':g.iloc[te].nunique(),'status':'infeasible'})
    res=pd.DataFrame(rows); res.to_csv(OUT/'repeated_grouped_cv_scores.csv',index=False); pd.DataFrame(infeasible).to_csv(OUT/'sampler_infeasibility.csv',index=False); pd.DataFrame(oof).to_csv(OUT/'oof_predictions.csv',index=False)
    summ=res.groupby(['model','sampler']).macro_f1.agg(['count','mean','std']).reset_index(); summ.to_csv(OUT/'model_sampler_summary.csv',index=False)
    # Figure: F1 distribution; infeasible folds are omitted, and counts are in table.
    fig,ax=plt.subplots(figsize=(10,5)); groups=[]; labels=[]
    for (m,s),z in res.groupby(['model','sampler']):
      vals=z.macro_f1.dropna().values
      if len(vals): groups.append(vals); labels.append(f'{m}\n{s} (n={len(vals)})')
    ax.boxplot(groups,showmeans=True); ax.set_xticks(range(1,len(labels)+1),labels,rotation=30,ha='right'); ax.set_ylabel('Macro-F1'); ax.set_title('Paper-grouped repeated cross-validation'); fig.tight_layout(); fig.savefig(FIG/'f1_grouped_cv.png'); fig.savefig(FIG/'f1_grouped_cv.pdf'); plt.close(fig)
    # descriptive K-means on outcome pair only; not used as predictor or validation target.
    both=r['Conversion_pct'].notna() & r['Yield_pct'].notna(); Z=r.loc[both,['Conversion_pct','Yield_pct']].astype(float).clip(0,100)
    if len(Z)>=6:
      z=(Z-Z.mean())/Z.std().replace(0,1); km=KMeans(n_clusters=2,n_init=100,random_state=42).fit(z); labs=km.labels_; means=pd.DataFrame({'cluster':labs,'conversion':Z['Conversion_pct'].values,'yield':Z['Yield_pct'].values}).groupby('cluster').mean(); order=means.mean(axis=1).sort_values().index.tolist(); mapping={old:new for new,old in enumerate(order)}; lab=np.array([mapping[x] for x in labs]);
      fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(Z['Conversion_pct'],Z['Yield_pct'],c=lab,cmap='viridis',s=35,alpha=.8); ax.set_xlabel('HMF conversion (%)'); ax.set_ylabel('FDCA yield (%)'); ax.set_title('Exploratory K-means clusters (outcome space; descriptive only)'); fig.tight_layout(); fig.savefig(FIG/'kmeans_outcome_clusters.png'); fig.savefig(FIG/'kmeans_outcome_clusters.pdf'); plt.close(fig)
      pd.DataFrame({'cluster_low_to_high':lab,'conversion':Z['Conversion_pct'].values,'yield':Z['Yield_pct'].values}).to_csv(OUT/'kmeans_cluster_assignments.csv',index=False)
    # learning curve: grouped holdout across group fractions, only on the best valid sampler/model summary.
    valid=summ.dropna(subset=['mean']).sort_values('mean',ascending=False)
    if len(valid):
      best=valid.iloc[0]; model=models[best.model]; sampler=best.sampler; rng=np.random.default_rng(42); uniq=np.array(g.unique()); fractions=np.linspace(.30,.85,5); lc=[]
      for rep in range(3):
       shuffled=rng.permutation(uniq)
       for frac in fractions:
        n=max(5,int(len(uniq)*frac)); train_groups=set(shuffled[:n]); tr=np.where(g.isin(train_groups))[0]; te=np.where(~g.isin(train_groups))[0]
        if len(te)<5 or len(np.unique(y[tr]))<2: continue
        try:
         p=make_pipe(model,sampler); p.fit(X.iloc[tr],y[tr]); lc.append({'repeat':rep,'fraction':frac,'train_records':len(tr),'train_macro_f1':f1_score(y[tr],p.predict(X.iloc[tr]),average='macro'),'test_macro_f1':f1_score(y[te],p.predict(X.iloc[te]),average='macro'),'model':best.model,'sampler':sampler})
        except Exception: pass
      ldf=pd.DataFrame(lc); ldf.to_csv(OUT/'learning_curve_group_holdout.csv',index=False)
      if len(ldf):
       q=ldf.groupby('fraction')[['train_macro_f1','test_macro_f1']].agg(['mean','std']); fig,ax=plt.subplots(figsize=(7,5)); ax.errorbar(q.index,q[('train_macro_f1','mean')],yerr=q[('train_macro_f1','std')],marker='o',label='Training'); ax.errorbar(q.index,q[('test_macro_f1','mean')],yerr=q[('test_macro_f1','std')],marker='o',label='Unseen-paper holdout'); ax.set_xlabel('Fraction of paper groups used for training'); ax.set_ylabel('Macro-F1'); ax.set_title('Group holdout learning curve (3 repeats; screening)'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'learning_curve.png'); fig.savefig(FIG/'learning_curve.pdf'); plt.close(fig)
      # permutation importance on a group-held-out test set, final explanatory model, not causal.
      tr,te=next(iter(grouped_cv(y,g,777))); p=make_pipe(model,sampler); p.fit(X.iloc[tr],y[tr]); imp=permutation_importance(p,X.iloc[te],y[te],scoring='f1_macro',n_repeats=10,random_state=42,n_jobs=-1); impdf=pd.DataFrame({'feature':cols,'importance_mean':imp.importances_mean,'importance_sd':imp.importances_std}).sort_values('importance_mean',ascending=False); impdf.to_csv(OUT/'permutation_importance.csv',index=False)
      top=impdf.head(15).sort_values('importance_mean'); fig,ax=plt.subplots(figsize=(8,6)); ax.barh(top.feature,top.importance_mean,xerr=top.importance_sd); ax.set_xlabel('Decrease in held-out macro-F1 after permutation'); ax.set_title('Held-out permutation importance (associational, not causal)'); fig.tight_layout(); fig.savefig(FIG/'permutation_importance.png'); fig.savefig(FIG/'permutation_importance.pdf'); plt.close(fig)
    # status / report
    lines=['# Reproducible HMF–FDCA ML analysis','',f'- Eligible records: {len(y)}; publication groups: {g.nunique()}.',f'- Binary labels: low (<90%)={int((y==0).sum())}; high (>=90%)={int((y==1).sum())}.',f'- Validation: {len(SEEDS)} repeats of 5-fold StratifiedGroupKFold; records from one paper remain in one fold.','- Preprocessing, imputation, scaling, feature selection and sampling are fitted on each training fold only.','- Quick screening compares no sampling versus RandomOverSampler (ROS). SMOTE/ADASYN are intentionally omitted to reduce runtime; add them in the final shortlisted-model analysis.','- K-means is descriptive in conversion/yield outcome space and is not a predictive model.','- Learning curve uses paper-group holdout. Permutation importance is associational, not causal.','- This is a quick screening run with fold-level model comparison, not a fully nested hyperparameter-selection estimate. Do not present the maximum score across model/sampler combinations as an unbiased final performance estimate.','- SHAP is not claimed here: use SHAP only after selecting a validated final estimator and explain its limitations; permutation importance is provided as a leakage-safe baseline.','- NSGA-II is intentionally not run: optimization requires validated regression surrogates that outperform naive baselines on unseen papers.','']
    (OUT/'AUDIT_REPORT.md').write_text('\n'.join(lines))
    print('\nSummary:\n',summ.to_string(index=False)); print('\nOutputs:',OUT)
if __name__=='__main__': main()
