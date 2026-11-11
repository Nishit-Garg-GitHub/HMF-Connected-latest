# Head-to-head of decision tree, random forest, SVM and neural network
# under the nested protocol of 10_f1_improved.py (tuned decision rules,
# optional SMOTE, anode-text features), both split types.
from pathlib import Path
ROOT = Path(__file__).resolve().parent
import importlib.util, sys, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
spec=importlib.util.spec_from_file_location("m", ROOT / "10_f1_improved.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SEED=42
FAM={
 "Decision tree":{f"DT_l{l}":(DecisionTreeClassifier(min_samples_leaf=l,class_weight="balanced",random_state=SEED),False,"clf") for l in (1,3,5)},
 "Random forest":{f"RF_l{l}":(RandomForestClassifier(n_estimators=400,max_features="sqrt",min_samples_leaf=l,class_weight="balanced_subsample",random_state=SEED,n_jobs=1),False,"clf") for l in (1,3)},
 "SVM":{f"SVM_C{c}":(SVC(C=c,gamma="scale",class_weight="balanced",probability=True,random_state=SEED),True,"clf") for c in (1,10)},
 "Neural network":{"NN_32_a0.1":(MLPClassifier(hidden_layer_sizes=(32,),alpha=0.1,max_iter=1500,random_state=SEED),True,"clf"),
                   "NN_64x32_a0.001":(MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=1500,random_state=SEED),True,"clf")},
}
def cands():
    return [(n,s,k) for n in m.ZOO for s in ("none","smote") for k in (0,20)]
m.candidate_list=cands
df,feats=m.load_features(["text"])
rows=[]
for split in ("group","record"):
    m.SPLIT=split
    for scheme in ("K2_high","K3_paper"):
        for fam,z in FAM.items():
            m.ZOO.clear(); m.ZOO.update(z)
            r=pd.DataFrame(m.nested(df,feats,scheme))
            rows.append(dict(split=split,scheme=scheme,family=fam,macro=r.macro_f1.mean(),sd=r.macro_f1.std(ddof=0),weighted=r.weighted_f1.mean()))
            print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "family_comparison.csv",index=False)
