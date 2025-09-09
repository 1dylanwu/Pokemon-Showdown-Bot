import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path
import pandas as pd

def split_action_type(y):
    types = np.array(["move" if act.startswith("move_") else "switch"
                      for act in y])
    return types

pre = "data/processed/"
X_train, y_train = np.load(pre + "X_train.npy", ), np.load(pre + "y_train.npy", allow_pickle=True)
X_val,   y_val   = np.load(pre+"X_val.npy"),   np.load(pre+"y_val.npy", allow_pickle=True)
X_test,  y_test  = np.load(pre+"X_test.npy"),  np.load(pre+"y_test.npy", allow_pickle=True)

# separate out the action types


y_tr_type = split_action_type(y_train)
y_va_type = split_action_type(y_val)
y_te_type = split_action_type(y_test)

# change to float32 to save memory
X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)

# binary classifier to check for either move or switch

type_clf = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=500,
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
"""
type_clf = HistGradientBoostingClassifier(min_samples_leaf = 10, max_iter = 100, max_depth = 3, learning_rate = 0.05, class_weight = 'balanced')
"""

type_clf.fit(X_train, y_tr_type)

print("Stage1 train acc:", type_clf.score(X_train, y_tr_type))
print("Stage1 val acc:", type_clf.score(X_val, y_va_type))
joblib.dump(type_clf, "models/stage1_type/final/type_clf_2.0.pkl")
"""
if(type_clf.score(X_val, y_va_type) > 0.7947):
    print("New best model! Saving...")
    joblib.dump(type_clf, "models/stage1_type/final/type_clf_2.1.pkl")
    """