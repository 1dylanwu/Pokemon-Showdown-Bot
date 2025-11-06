import random
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from pathlib import Path
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, f1_score, log_loss, make_scorer, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids
import xgboost as xgb
from src.tests.accuracy_test import test

pre = "data/processed/type/"
X_train, y_tr_type = np.load(pre + "X_train_clean.npy", mmap_mode="r").astype(np.float32), np.load(pre + "y_train_clean.npy", allow_pickle=True)
X_val, y_va_type = np.load(pre+"X_val_clean.npy", mmap_mode="r").astype(np.float32), np.load(pre+"y_val_clean.npy", allow_pickle=True)
X_test, y_te_type = np.load(pre+"X_test_clean.npy", mmap_mode="r"), np.load(pre+"y_test_clean.npy", allow_pickle=True)
"""
rus = RandomUnderSampler(
    sampling_strategy='auto',
    random_state=42
)


X_train, y_tr_type = rus.fit_resample(X_train, y_tr_type)
"""

def find_threshold_for_switch(y_true, proba_move, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        # predict 0=switch when proba_move < t
        y_pred = (proba_move >= t).astype(int)
        # compute F1 for switch (label=0)
        f1_s = f1_score(y_true, y_pred, pos_label=0)
        if f1_s > best_f1:
            best_f1, best_t = f1_s, t
    return best_t, best_f1



type_clf = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    class_weight="balanced",
    n_estimators=50000,
    n_jobs=5,
    verbosity = -1,
    learning_rate= 0.12, 
    num_leaves= 63, 
    max_depth= -1, 
    min_child_samples= 7, 
    max_bin= 64, 
    subsample= 1.0, 
    colsample_bytree= 1.0, 
    reg_alpha= 0.5, 
    reg_lambda= 2.0, 
    min_split_gain= 0.0
)



type_clf.fit(
    X_train, y_tr_type,
    eval_set=[(X_val, y_va_type)],
    eval_metric="binary_logloss",
    callbacks=[
        early_stopping(stopping_rounds=100, verbose=True),
        log_evaluation(period=1000)
    ]
)

joblib.dump(type_clf, "models/type/type_2.2.pkl")

#type_clf = joblib.load("models/type/type_2.2.pkl")
print("Stage1 train acc:", type_clf.score(X_train, y_tr_type))

test(type_clf, X_val, y_va_type, 0.36, True)

"""
dtrain = xgb.DMatrix(X_train, label=y_tr_type)
dval = xgb.DMatrix(X_val, label=y_va_type)
n_pos = np.sum(y_tr_type == 1)
n_neg = np.sum(y_tr_type == 0)
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "scale_pos_weight" : n_neg / n_pos
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=3500,
    evals=[(dval, "val")],
    #early_stopping_rounds=50,
    verbose_eval=200,
    maximize = True
)

joblib.dump(model, "models/type/type_3.1.pkl")

#model = joblib.load("models/stage1_type/type_clf_3.1.pkl")
dval = xgb.DMatrix(X_val)
y_val_proba = model.predict(dval)
y_val_pred = (y_val_proba > 0.38).astype(int)

acc = accuracy_score(y_va_type, y_val_pred)
auc = roc_auc_score(y_va_type, y_val_proba)
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation ROC AUC : {auc:.4f}")
num_move_preds = np.sum(y_val_pred)
total_preds = len(y_val_pred)
move_ratio = num_move_preds / total_preds

print(f"Predicted 'move' actions: {num_move_preds} out of {total_preds}")
print(f"Fraction predicted as 'switch': {1 - move_ratio:.4f}")
print(classification_report(y_va_type, y_val_pred, target_names=["switch", "move"]))
"""
"""
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=7,
    class_weight = "balanced_subsample",
    random_state=42,
    n_jobs= 5
)
params = {"n_estimators": [200, 500, 800],
    "max_depth": [8, 12, 16, 24, None],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 3, 5, 7, 15],
    "max_features": [0.25, 0.4, 0.6, "sqrt"],
    "class_weight": ["balanced", "balanced_subsample", None]
}


rf_clf.fit(X_train, y_tr_type)
joblib.dump(rf_clf, "models/type/type_clf_4.0.pkl")

#rf_clf = joblib.load("models/type/type_clf_4.0.pkl")
proba_move_val = rf_clf.predict_proba(X_val)[:, 1]

best_t, best_f1 = find_threshold_for_switch(y_va_type, proba_move_val)

test(rf_clf, X_val, y_va_type, best_t, True)
print("Best threshold for switch (val):", best_t, " with F1 for switch:", best_f1)
"""