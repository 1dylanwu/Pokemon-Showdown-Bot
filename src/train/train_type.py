import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from pathlib import Path
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report
import xgboost as xgb
from src.utils.utils import split_action_type
from src.tests.accuracy_test import test

pre = "data/processed/type/"
X_train, y_tr_type = np.load(pre + "X_train_clean.npy").astype(np.float32), np.load(pre + "y_train_clean.npy", allow_pickle=True)
X_val, y_va_type = np.load(pre+"X_val_clean.npy").astype(np.float32), np.load(pre+"y_val_clean.npy", allow_pickle=True)
X_test, y_te_type = np.load(pre+"X_test_clean.npy"), np.load(pre+"y_test_clean.npy", allow_pickle=True)

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

"""
type_clf = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=4500,
    learning_rate=0.1,
    max_depth=7,
    min_child_samples = 50,
    n_jobs=5,
    verbosity = -1,
    num_leaves= 63,
    max_bin = 252,
    is_unbalance = True,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5
)

type_clf.fit(
    X_train, y_tr_type,
    eval_set=[(X_val, y_va_type)],
    eval_metric="binary_logloss",
    callbacks=[
        #early_stopping(stopping_rounds=500, verbose=True),
        log_evaluation(period=100)
    ]
)

joblib.dump(type_clf, "models/type/type_2.0.pkl")
"""
type_clf = joblib.load("models/type/type_2.0.pkl")
print("Stage1 train acc:", type_clf.score(X_train, y_tr_type))
test(type_clf, X_val, y_va_type, 0.38, True)

"""
type_clf = HistGradientBoostingClassifier(min_samples_leaf = 10, max_iter = 1000, max_depth = 4, learning_rate = 0.03, class_weight = 'balanced', verbose = 1)

type_clf.fit(X_train, y_tr_type)
y_train_pred = type_clf.predict(X_train)
y_val_pred = type_clf.predict(X_val)

print(f"Train accuracy: {accuracy_score(y_tr_type, y_train_pred):.4f}")
print(f" Val accuracy: {accuracy_score(y_va_type, y_val_pred):.4f}")

#if(type_clf.score(X_val, y_va_type) > .7919):
    #print("New best model! Saving...")
joblib.dump(type_clf, "models/stage1_type/type_clf_1.0.pkl")
"""
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
    num_boost_round=750,
    evals=[(dval, "val")],
    #early_stopping_rounds=500,
    verbose_eval=50,
    maximize = True
)

joblib.dump(model, "models/stage1_type/type_clf_3.2.pkl")

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
    max_depth=None,
    min_samples_leaf=7,
    class_weight="balanced",
    random_state=42,
    n_jobs= 5
)

rf_clf.fit(X_train, y_tr_type)
joblib.dump(rf_clf, "models/stage1_type/type_clf_4.1.pkl")
proba_move_val = rf_clf.predict_proba(X_val)[:, 1]

best_t, best_f1 = find_threshold_for_switch(y_va_type, proba_move_val)
print(f"Best threshold for switch-vs-move: {best_t:.2f} (Switch F1 = {best_f1:.3f})")

y_pred_val = (proba_move_val >= best_t).astype(int)

print(classification_report(
    y_va_type, 
    y_pred_val, 
    target_names=["switch", "move"]
))

proba_move_test = rf_clf.predict_proba(X_test)[:, 1]
y_pred_test = (proba_move_test >= best_t).astype(int)
print("Test set performance:")
print(classification_report(y_te_type, y_pred_test, target_names=["switch", "move"]))
"""