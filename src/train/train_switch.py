import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from src.tests.accuracy_test import test

pre = "data/processed/switch/"
X_tr = np.load(pre + "X_tr.npy")
y_tr = np.load(pre + "y_tr.npy").astype(np.int32)
X_va = np.load(pre + "X_va.npy")
y_va = np.load(pre + "y_va.npy").astype(np.int32)

le = joblib.load("models/switch/util/label_encoder.pkl")

n_classes = len(le.classes_)
"""
sw_clf = LGBMClassifier(
    objective="multiclass",
    boosting_type="gbdt",
    num_class=n_classes,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=127,
    min_child_samples=10,
    max_bin=255,
    subsample=0.9,
    colsample_bytree=0.9,
    verbosity=-1,
    n_estimators=10000,
    min_split_gain = 0.0,
    reg_alpha = 0.1,
    reg_lambda= 0.5, 
)

sw_clf.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="multi_logloss",
    callbacks=[
        early_stopping(stopping_rounds=20, verbose=True),
        log_evaluation(period=10)
    ]
)
joblib.dump(sw_clf, "models/switch/final/switch_clf_1.1.pkl")
test(sw_clf, X_va, y_va)
"""

sw_clf = XGBClassifier(
    objective="multi:softprob", 
    num_class=n_classes,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    random_state=42,
    n_jobs=6,
    early_stopping_rounds = 10
)
sw_clf.fit(
    X_tr, 
    y_tr,
    eval_set=[(X_va, y_va)],
    verbose=True
)
joblib.dump(sw_clf, "models/switch/final/switch_clf_2.0.pkl")

#best_iter = sw_clf.get_booster().best_iteration
#print("Best iteration:", best_iter)

print("Stage2b sw train acc:", sw_clf.score(X_tr, y_tr))
print("Stage2b sw val acc:", sw_clf.score(X_va,  y_va))

