import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, log_loss
from src.utils.utils import split_action_type
import joblib

pre = "data/processed/general/"
X_train, y_train = np.load(pre + "X_train.npy").astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
X_val, y_val = np.load(pre+"X_val.npy").astype(np.float32), np.load(pre+"y_val.npy", allow_pickle=True)

y_tr_type = split_action_type(y_train)
y_va_type = split_action_type(y_val)
y_tr_type = (y_tr_type == "switch").astype(int)

base_clf = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

param_dist = {
    "n_estimators": [500, 750, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 5, 7, 9, -1],
    "num_leaves": [31, 63, 127, 255],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "bagging_freq": [0, 5, 10],
    "lambda_l1": [0.0, 0.1, 1.0],
    "lambda_l2": [0.0, 0.1, 1.0]
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(log_loss, greater_is_better=False)

search = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_dist,
    n_iter=50,
    scoring=scorer,
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs= 2
)

search.fit(X_train, y_tr_type)
print("Best log-loss (negative):", search.best_score_)
print("Best parameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

joblib.dump(search.best_estimator_, "models/stage1_type/final/type_clf_2.1.pkl")