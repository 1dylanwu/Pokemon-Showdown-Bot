import random
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, log_loss
from src.utils.utils import split_action_type
import joblib
import json
import os

pre = "data/processed/type/"
X_train, y_tr_type = np.load(pre + "X_train_clean.npy").astype(np.float32), np.load(pre + "y_train_clean.npy", allow_pickle=True)
X_val, y_va_type = np.load(pre+"X_val_clean.npy").astype(np.float32), np.load(pre+"y_val_clean.npy", allow_pickle=True)
X_test, y_te_type = np.load(pre+"X_test_clean.npy"), np.load(pre+"y_test_clean.npy", allow_pickle=True)

base_clf = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    class_weight="balanced",
    random_state=42,
    n_jobs=5,
    verbosity=-1
)
tried_path = "models/type/utils/type_params.json"
if os.path.exists(tried_path):
    with open(tried_path, "r") as f:
        tried_list = json.load(f) or []
else:
    tried_list = []
tried_params = set(tuple(sorted(d.items())) for d in tried_list)
"""
param_grid = {"learning_rate": [0.1, 0.12, 0.15],
        "num_leaves": [31, 63],
        "max_depth": [8, 12, -1],
        "min_child_samples": [1, 5, 7],
        "max_bin": [64, 128],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [ 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
        "min_split_gain": [0.0, 0.01, 0.05],
}
"""
param_grid = {"learning_rate": [0.11, 0.12, 0.13],
        "num_leaves": [31, 63],
        "max_depth": [12, -1],
        "min_child_samples": [1, 7],
        "max_bin": [64, 128],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [ 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
        "min_split_gain": [0.0],
}
def sample_random(param_distributions, exclude, n_iter=15, seed=2007):
    rng = random.Random(seed)
    keys = list(param_distributions.keys())
    samples = []
    seen = set(exclude)

    while len(samples) < n_iter:
        candidate = {k: rng.choice(param_distributions[k]) for k in keys}
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            samples.append(candidate)
            seen.add(key)
    return samples

best_score = 0.5109
best_params = None
best_model = None
new_trials = []

random_trials = sample_random(param_grid, exclude=tried_params, n_iter=15, seed=2025)

for i, params in enumerate(random_trials, 1):
    print(f"\nTrial {i} with params: {params}")
    
    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=50000,
        n_jobs=5,
        random_state=2007,
        class_weight="balanced",
        verbosity=-1,
        **params
    )
    
    model.fit(
        X_train, y_tr_type,
        eval_set=[(X_val, y_va_type)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(stopping_rounds=100)],
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    score = log_loss(y_va_type, val_pred)
    print(f"LogLoss: {score:.4f}")
    
    if score < best_score and score < 0.52:
        best_score = score
        best_params = params
        best_model = model
        joblib.dump(best_model, "best_lgbm_model.pkl")
        print("Saved new best model!")
    
    tried_params.add(tuple(sorted(params.items())))
    new_trials.append(params)

tried_dict = { tuple(sorted(p.items())): p for p in tried_list }
for p in new_trials:
    tried_dict[tuple(sorted(p.items()))] = p
all_tried_list = list(tried_dict.values())

with open(tried_path, "w") as f:
    json.dump(all_tried_list, f, indent=2)
