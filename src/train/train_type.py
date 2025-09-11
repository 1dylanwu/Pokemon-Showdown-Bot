import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from src.utils.utils import split_action_type

pre = "data/processed/general/"
X_train, y_train = np.load(pre + "X_train.npy").astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
X_val, y_val = np.load(pre+"X_val.npy").astype(np.float32), np.load(pre+"y_val.npy", allow_pickle=True)
X_test, y_test = np.load(pre+"X_test.npy"), np.load(pre+"y_test.npy", allow_pickle=True)

# separate out the action types

y_tr_type = split_action_type(y_train)
y_va_type = split_action_type(y_val)
y_te_type = split_action_type(y_test)

y_tr_type = (y_tr_type == "switch").astype(int)
y_va_type = (y_va_type == "switch").astype(int)

# binary classifier to check for either move or switch

type_clf = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=-1,
    class_weight="balanced",
    random_state=42,
    n_jobs=2,
    verbosity = -1,
    num_leaves= 255,
    lambda_l2 = 1.0,
    lambda_l1 = 1.0,
    feature_fraction = 1.0,
    bagging_freq = 10,
    bagging_fraction = 1.0
)

type_clf.fit(
    X_train, y_tr_type,
    eval_set=[(X_val, y_va_type)],
    eval_metric="binary_logloss",
    callbacks=[
        early_stopping(stopping_rounds=100, verbose=True),
        log_evaluation(period=100)
    ]
)

print("Stage1 train acc:", type_clf.score(X_train, y_tr_type))
print("Stage1 val acc:", type_clf.score(X_val, y_va_type))

if(type_clf.score(X_val, y_va_type) > 0.7922905602192908):
    print("New best model! Saving...")
    joblib.dump(type_clf, "models/stage1_type/final/type_clf_2.1.pkl")
"""
type_clf = HistGradientBoostingClassifier(min_samples_leaf = 10, max_iter = 1000, max_depth = 4, learning_rate = 0.03, class_weight = 'balanced', verbose = 1)

type_clf.fit(X_train, y_tr_type)
y_train_pred = type_clf.predict(X_train)
y_val_pred = type_clf.predict(X_val)

print(f"Train accuracy: {accuracy_score(y_tr_type, y_train_pred):.4f}")
print(f" Val accuracy: {accuracy_score(y_va_type, y_val_pred):.4f}")

if(type_clf.score(X_val, y_va_type) > .7919):
    print("New best model! Saving...")
    joblib.dump(type_clf, "models/stage1_type/final/type_clf_1.0.pkl")

result = permutation_importance(
    type_clf,
    X_val,
    y_va_type,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

# Display sorted importances
importances = result.importances_mean
indices = np.argsort(importances)[::-1]

for i in indices:
    print(f"Feature {i}: Importance = {importances[i]:.4f}")
"""