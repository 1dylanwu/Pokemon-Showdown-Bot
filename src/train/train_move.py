import numpy as np
import joblib
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier

from src.tests.accuracy_test import test

X_tr_moves = np.load("data/processed/move/X_tr.npy")
y_tr_moves_enc = np.load("data/processed/move/y_tr.npy")
X_va_moves = np.load("data/processed/move/X_va.npy")
y_va_moves_enc = np.load("data/processed/move/y_va.npy")
X_te_moves = np.load("data/processed/move/X_te.npy")
y_te_moves_enc = np.load("data/processed/move/y_te.npy")
le = joblib.load("models/move/util/label_encoder.pkl")



move_clf = LGBMClassifier(
    objective="multiclass",
    num_class=len(np.unique(y_tr_moves_enc)),
    boosting_type="gbdt",
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=63,
    class_weight="balanced",
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_jobs=5,
    verbosity=-1
)

move_clf.fit(
    X_tr_moves, y_tr_moves_enc,
    eval_set=[(X_va_moves, y_va_moves_enc)],
    eval_metric="multi_error",
    callbacks=[
        early_stopping(stopping_rounds=10, verbose = True),
        log_evaluation(period=100)
    ]
)
joblib.dump(move_clf, "models/move/move_clf_1.0.pkl")
test(move_clf, X_te_moves, y_te_moves_enc)

"""
move_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=5,
    random_state=42
)
move_clf.fit(X_tr_moves, y_tr_moves_enc)
tr_acc = move_clf.score(X_tr_moves, y_tr_moves_enc)
va_acc = move_clf.score(X_va_moves,  y_va_moves_enc)
print("Stage2a move train acc:", tr_acc)
print("Stage2a move val acc:", va_acc)

#if(tr_acc > 0.49704120035654153 or va_acc > 0.29993106165058103):
joblib.dump((move_clf), "models/stage2_move/final/move_clf_2.0.pkl")
    """
"""
print(len(le.classes_))
move_clf = XGBClassifier(
    objective="multi:softprob", 
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=5,
    n_estimators=150,
    random_state=42,
    n_jobs=5,
    early_stopping_rounds = 15
)
move_clf.fit(
    X_tr_moves, 
    y_tr_moves_enc,
    eval_set=[(X_va_moves, y_va_moves_enc)],
    verbose=True
)
joblib.dump(move_clf, "models/move/final/move_clf_3.0.pkl")
test(move_clf, X_te_moves, y_te_moves_enc)
"""