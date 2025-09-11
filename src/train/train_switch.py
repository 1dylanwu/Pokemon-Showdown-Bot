import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

pre = "data/processed/switch/"
X_tr = np.load(pre + "X_tr_sw.npy")
y_tr = np.load(pre + "y_tr_sw.npy").astype(np.int32)
X_va = np.load(pre + "X_va_sw.npy")
y_va = np.load(pre + "y_va_sw.npy").astype(np.int32)

le = joblib.load("models/stage2_switch/util/label_encoder.pkl")

n_classes = len(le.classes_)

params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": ["multi_logloss", "multi_error"],
    "learning_rate": 0.02,
    "max_depth": 8,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1
}
sw_clf = LGBMClassifier(
    objective="multiclass",
    num_class=n_classes,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=63,
    min_child_samples=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.0, 
    verbosity=-1,
    n_estimators=3000

)

sw_clf.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="multi_error",
    callbacks=[
        early_stopping(stopping_rounds=150, verbose=True),
        log_evaluation(period=10)
    ]
)
#best_iter = sw_clf.get_booster().best_iteration
#print("Best iteration:", best_iter)


joblib.dump(sw_clf, "models/stage2_switch/final/switch_clf_1.0.pkl")

print("Stage2b sw train acc:", sw_clf.score(X_tr, y_tr))
print("Stage2b sw val acc:", sw_clf.score(X_va,  y_va))