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

pre = "data/processed/type/"
X_train, y_train = np.load(pre + "X_train_clean.npy").astype(np.float32), np.load(pre + "y_train_clean.npy", allow_pickle=True)
X_val, y_val = np.load(pre+"X_val_clean.npy").astype(np.float32), np.load(pre+"y_val_clean.npy", allow_pickle=True)
X_test, y_test = np.load(pre+"X_test_clean.npy"), np.load(pre+"y_test_clean.npy", allow_pickle=True)

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


# separate out the action types

y_tr_type = split_action_type(y_train)
y_va_type = split_action_type(y_val)
y_te_type = split_action_type(y_test)

y_tr_type = np.array([1 if act.startswith("move_") else 0 for act in y_train])
y_va_type = np.array([1 if act.startswith("move_") else 0 for act in y_val])

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

print("Stage1 train acc:", type_clf.score(X_train, y_tr_type))
print("Stage1 val acc:", type_clf.score(X_val, y_va_type))

joblib.dump(type_clf, "models/stage1_type/type_clf_2.3.pkl")

#type_clf = joblib.load("models/stage1_type/type_clf_2.2.pkl")
y_pred = type_clf.predict(X_val)
frac_switches_pred = np.mean(y_pred == 0)
print(f"Predicted switch fraction: {frac_switches_pred:.3f}")

proba_move_val = type_clf.predict_proba(X_val)[:, 1]
best_t, best_re = find_threshold_for_switch(y_va_type, proba_move_val)
print(f"Best switch‐recall {best_re:.3f} at threshold {best_t:.3f}")
y_pred_switch = (proba_move_val >= .38).astype(int)

    # 2. Frequency of predicting switch vs. move
frac_switch_pred = np.mean(y_pred_switch == 0)
frac_switch_true = np.mean(y_va_type == 0)

    # 3. Confusion matrix & classification report
cm = confusion_matrix(y_va_type, y_pred_switch)
report = classification_report(
    y_va_type, y_pred_switch,
    target_names=["switch", "move"],
    digits=4
)
auc = roc_auc_score(y_va_type, proba_move_val)

print(f"Actual switch fraction : {frac_switch_true:.3f}")
print(f"Predicted switch fraction: {frac_switch_pred:.3f}")
print(f"ROC AUC               : {auc:.4f}")
print("\nConfusion Matrix:")
print(pd.DataFrame(cm,
                       index=["true_switch", "true_move"],
                       columns=["pred_switch", "pred_move"]))
print("\nClassification Report:")
print(report)

"""
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