import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV


def predict_action(X_row, active_species, revealed_self,
                   type_clf, move_clf, le_moves, sw_clf, le_sw,
                   full_movepools):
    # Stage 1
    action_type = type_clf.predict(X_row)[0]  # "move" or "switch"

    if action_type == "move":
        probs = move_clf.predict_proba(X_row)[0]
        classes = le_moves.classes_

        # mask using full_movepools[active_species]
        legal = full_movepools.get(active_species, set())
        mask = np.array([cls.split("_",1)[1] in legal for cls in classes])

    else:  # "switch"
        probs = sw_clf.predict_proba(X_row)[0]
        classes = le_sw.classes_

        # mask using revealed_self
        mask = np.array([cls.split("_",1)[1] in revealed_self for cls in classes])

    # zero‐out illegal, renormalize
    probs = probs * mask
    if probs.sum() <= 0:
        probs = mask.astype(float) / mask.sum()
    else:
        probs /= probs.sum()

    return classes[np.argmax(probs)]

type_clf = joblib.load("models/stage1_type/type_clf_2.0.pkl")
move_clf = joblib.load("models/stage2_move/move_clf_1.0.pkl")
sw_clf = joblib.load("models/stage2_switch/switch_clf_1.0.pkl")
full_movepools = joblib.load("data/processed/full_movepools.pkl")


i = 42
X_row = X_val[i].reshape(1, -1)
act_true = y_val[i]
active_sp = df_val["p1a_active"].iloc[i]
revealed = set(df_val["p1_team_species"].iloc[i])

pred = predict_action(
    X_row, active_sp, revealed,
    type_clf, move_clf, le_moves, sw_clf, le_sw, full_movepools
)
print("true:", act_true, "pred:", pred)
