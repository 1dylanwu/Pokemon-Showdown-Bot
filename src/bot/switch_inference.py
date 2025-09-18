import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import ast
from src.utils.utils import split_action_type

pre = "data/processed/switch/"
model = joblib.load("models/stage2_switch/final/switch_clf_1.0.pkl")
le = joblib.load("models/stage2_switch/util/label_encoder.pkl")
X_te = np.load(pre + "X_va_sw.npy")
y_te = np.load(pre + "y_va_sw.npy").astype(int)
# base switch rows for split
df = pd.read_csv("data/parsed/val.csv")
df = df[(df["action_type"] == "switch") & (df["turn"] != 0)].reset_index(drop=True)
# team availability info for mask
team_info_path = "data/parsed/team_info/val_team_info.csv"
team_df = pd.read_csv(
    team_info_path,
    converters={
        "p1_available": lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else ([] if pd.isna(x) else [x]),
        "p2_available": lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else ([] if pd.isna(x) else [x]),
    },
)

# merge availability onto switch rows
df = df.merge(
    team_df[["replay_id", "turn", "p1_available", "p2_available"]],
    on=["replay_id", "turn"],
    how="left",
)

def switch_mask(
    df: pd.DataFrame,
    le_switch: LabelEncoder,
) -> np.ndarray:
    classes = le_switch.classes_
    n_rows, n_cols = len(df), len(classes)
    mask = np.zeros((n_rows, n_cols), dtype = bool)

    for i, row in df.iterrows():
        side = row["side"][:2] # only p1 instead of p1a
        active = row[f"state_{side}a_active"]
        # Use available mons from team-info CSV (revealed minus fainted) for the correct side
        available = row[f"{side}_available"] if f"{side}_available" in row and isinstance(row[f"{side}_available"], (list, tuple)) else []
        # Exclude the currently active mon
        bench = [s for s in available if s != active]

        for mon in bench:
            if mon in classes:
                idx = le_switch.transform([mon])[0]
                mask[i][idx] = True

    return mask

mask_te = switch_mask(df, le)
proba_raw = model.predict_proba(X_te)
y_pred_raw = np.argmax(proba_raw, axis=1)
acc_raw = accuracy_score(y_te, y_pred_raw)
print(f"Raw test accuracy:    {acc_raw:.4f}")

proba_mask = proba_raw.copy()
proba_mask[~mask_te] = 0
row_sums = proba_mask.sum(axis=1, keepdims=True)
valid_rows = row_sums.squeeze() > 0
proba_mask[valid_rows] /= row_sums[valid_rows]

y_pred_mask = np.argmax(proba_mask, axis=1)
acc_mask = accuracy_score(y_te, y_pred_mask)
print(f"Masked test accuracy: {acc_mask:.4f}")

true_in_mask = mask_te[np.arange(len(y_te)), y_te]
coverage = true_in_mask.mean()
acc_cond = accuracy_score(
    y_te[true_in_mask],
    y_pred_mask[true_in_mask]
) if coverage > 0 else np.nan

labels = np.arange(len(le.classes_))
print(f"True‐label coverage:  {coverage:.4%}")
print(f"Cond’l masked acc:    {acc_cond:.4f}  " 
      "(only where true switch was legal)")
top3_acc = top_k_accuracy_score(y_te, proba_mask, k=3, labels=labels)
print(f"Top-3 masked accuracy: {top3_acc:.4f}")