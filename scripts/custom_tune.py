import numpy as np
import joblib
import time
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import ParameterSampler, cross_val_score
from sklearn.preprocessing import LabelEncoder
# custom parameter tuning script for stage 1 classifier with autosaving feature since my computer kept crashing
# config
SEED          = 42
CHECKPOINT_F = Path("models/stage1_type/random_search_checkpoint.pkl")
BEST_MODEL_F  = Path("models/stage1_type/type_clf_best.pkl")
N_ITER        = 30
CV_FOLDS      = 3
SAVE_INTERVAL = 600   # seconds between auto-saves (10 minutes)
# load the data
pre       = "data/processed/"
X_train   = np.load(pre + "X_train.npy").astype(np.float32)
y_train   = np.load(pre + "y_train.npy", allow_pickle=True)

def split_action_type(y):
    return np.array(["move" if act.startswith("move_") else "switch" for act in y])

y_tr_type = split_action_type(y_train)

# hyperparameters to search over
param_distributions = {
    "learning_rate":     [0.01, 0.05, 0.1],
    "max_iter":          [100, 200, 300],
    "max_depth":         [3, 5, None],
    "min_samples_leaf":  [5, 10, 20],
    "class_weight":      [None, "balanced"]
}

# create or resume search from checkpoiint
if CHECKPOINT_F.exists():
    checkpoint = joblib.load(CHECKPOINT_F)
    tried_params = checkpoint["results"]
    best = checkpoint["best"]
    param_list = checkpoint["param_list"]
    start_idx = len(tried_params)
    last_save = checkpoint.get("timestamp", time.time())
    print(f"Resuming from iteration {start_idx}/{len(param_list)}")
else:
    param_list = list(ParameterSampler(param_distributions,
                                       n_iter=N_ITER,
                                       random_state=SEED))
    tried_params = []   # list of dicts: {"params":…, "mean_score":…, "scores":…}
    best = {"params": None, "score": -np.inf, "model": None}
    start_idx = 0
    last_save = time.time()

# search loop while saving periodically
for idx in range(start_idx, len(param_list)):
    params = param_list[idx]
    clf = HistGradientBoostingClassifier(random_state=SEED, **params)

    # cross validation
    scores = cross_val_score(clf, X_train, y_tr_type,
                             cv=CV_FOLDS, n_jobs=2,
                             scoring="accuracy")
    mean_score = scores.mean()
    tried_params.append({
        "params":      params,
        "scores":      scores,
        "mean_score":  mean_score
    })

    # if its the best so far, fit on the full data and save
    if mean_score > best["score"]:
        best["score"] = mean_score
        best["params"] = params
        best["model"] = clf.fit(X_train, y_tr_type)
        joblib.dump(best["model"], BEST_MODEL_F)
        print(f"[{idx+1}] New best: {mean_score:.4f} with {params}")

    else:
        print(f"[{idx+1}] Tried {params} → mean CV acc {mean_score:.4f}")

    # periodic checkpoint save (every iteration or every SAVE_INTERVAL secs)
    now = time.time()
    if now - last_save >= SAVE_INTERVAL or idx == len(param_list)-1:
        checkpoint = {
            "param_list": param_list,
            "results":    tried_params,
            "best":       {"params": best["params"], "score": best["score"]},
            "timestamp":  now
        }
        joblib.dump(checkpoint, CHECKPOINT_F)
        print(f"Checkpoint saved at iteration {idx+1}")
        last_save = now

print("Search complete.")
print("Best params:", best["params"])
print("Best CV accuracy:", best["score"])