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
"""
sw_clf = LGBMClassifier(
    objective="multiclass",
    num_class=n_classes,
    learning_rate=0.01,
    max_depth=8,
    num_leaves=63,
    min_child_samples=150,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbosity=-1,
    n_estimators=10,
    min_split_gain = 0.1,
    reg_alpha = 0.5,
    reg_lambda= 1.5, 
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
joblib.dump(sw_clf, "models/stage2_switch/final/switch_clf_TEST.pkl")
probs = sw_clf.predict_proba(X_va)
print(probs.shape)

"""
sw_clf = XGBClassifier(
    objective="multi:softprob", 
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    random_state=42,
    n_jobs=5,
    early_stopping_rounds = 10
)
sw_clf.fit(
    X_tr, 
    y_tr,
    eval_set=[(X_va, y_va)],
    verbose=True
)
joblib.dump(sw_clf, "models/stage2_switch/final/switch_clf_2.0.pkl")

#best_iter = sw_clf.get_booster().best_iteration
#print("Best iteration:", best_iter)

print("Stage2b sw train acc:", sw_clf.score(X_tr, y_tr))
print("Stage2b sw val acc:", sw_clf.score(X_va,  y_va))

