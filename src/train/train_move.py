import numpy as np
import joblib
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

X_tr_moves = np.load("data/processed/move/X_tr_moves.npy")
y_tr_moves_enc = np.load("data/processed/move/y_tr_moves.npy")
X_va_moves = np.load("data/processed/move/X_va_moves.npy")
y_va_moves_enc = np.load("data/processed/move/y_va_moves.npy")

move_clf = LGBMClassifier(
    objective="multiclass",
    num_class=len(np.unique(y_tr_moves_enc)),
    boosting_type="gbdt",
    n_estimators=6000,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=63,
    min_data_in_leaf=200,
    class_weight="balanced",
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_jobs=4,
    verbosity=-1
)

print("Training Stage-2 Move model with LightGBM…")

move_clf.fit(
    X_tr_moves, y_tr_moves_enc,
    eval_set=[(X_va_moves, y_va_moves_enc)],
    eval_metric="multi_error",
    callbacks=[
        early_stopping(stopping_rounds=200, verbose=True),
        log_evaluation(period=10)
    ]
)
"""
move_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
move_clf.fit(X_tr_moves, y_tr_moves_enc)
"""
tr_acc = move_clf.score(X_tr_moves, y_tr_moves_enc)
va_acc = move_clf.score(X_va_moves,  y_va_moves_enc)
print("Stage2a move train acc:", tr_acc)
print("Stage2a move val acc:", va_acc)

if(tr_acc > 0.49704120035654153 or va_acc > 0.29993106165058103):
    joblib.dump((move_clf), "models/stage2_move/final/move_clf_1.2.pkl")