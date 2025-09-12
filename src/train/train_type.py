import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score
from src.utils.utils import split_action_type

pre = "data/processed/type/"
X_train, y_train = np.load(pre + "X_train_clean.npy").astype(np.float32), np.load(pre + "y_train_clean.npy", allow_pickle=True)
X_val, y_val = np.load(pre+"X_val_clean.npy").astype(np.float32), np.load(pre+"y_val_clean.npy", allow_pickle=True)
X_test, y_test = np.load(pre+"X_test_clean.npy"), np.load(pre+"y_test_clean.npy", allow_pickle=True)

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
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=7,
    class_weight="balanced",
    random_state=42,
    min_child_samples = 5,
    n_jobs=2,
    verbosity = -1,
    num_leaves= 63,
    max_bin = 252
)

type_clf.fit(
    X_train, y_tr_type,
    eval_set=[(X_val, y_va_type)],
    eval_metric="multi_error",
    callbacks=[
        early_stopping(stopping_rounds=100, verbose=True),
        log_evaluation(period=100)
    ]
)

print("Stage1 train acc:", type_clf.score(X_train, y_tr_type))
print("Stage1 val acc:", type_clf.score(X_val, y_va_type))

#if(type_clf.score(X_val, y_va_type) > 0.7922905602192908):
    #print("New best model! Saving...")
joblib.dump(type_clf, "models/stage1_type/type_clf_2.0.pkl")
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