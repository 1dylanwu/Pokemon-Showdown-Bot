import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import random

MODEL_PATH    = Path("models/stage2_move/final/move_clf_1.1.pkl")
LABEL_ENCODER    = Path("models/stage2_move/util/label_encoder.pkl")
FULL_MOVEPOOLS   = Path("data/processed/move/full_movepools.pkl")
X_VAL_PATH       = Path("data/processed/move/X_va_moves.npy")
Y_VAL_PATH       = Path("data/processed/move/y_va_moves.npy")
VAL_CSV_PATH     = Path("data/parsed/val.csv") 

# load everything
move_clf = joblib.load(MODEL_PATH)
le_moves = joblib.load(LABEL_ENCODER)
movepools = joblib.load(FULL_MOVEPOOLS)

X_val = np.load(X_VAL_PATH)
y_val_enc = np.load(Y_VAL_PATH, allow_pickle=True)

# process csv (filter for only moves and remove state prefix)
df = pd.read_csv(VAL_CSV_PATH, dtype=str)
df = df[df["action_type"] == "move"].copy()
df.rename(columns=lambda c: c[6:] if c.startswith("state_") else c, inplace=True)

# create list of the active species per row
species_val = np.where(
    df["side"] == "p1a",
    df["p1a_active"],
    df["p2a_active"]
).tolist()
opponent_species = np.where(
    df["side"] == "p1a",
    df["p2a_active"],
    df["p1a_active"]
)


# create list of total available moves per row, according to possible movesets json
legal_moves_batch = [
    movepools.get(species, [])
    for species in species_val
]

def apply_legal_move_mask(probs: np.ndarray,
                          legal_moves_batch: list[list[str]],
                          le_moves
                         ) -> np.ndarray:
    # zero out illegal move probabilities and renormalize
    masked = np.zeros_like(probs)
    class_map = {}
    for idx, mv_label in enumerate(le_moves.classes_):
        raw = mv_label.split("_", 1)[1] if "_" in mv_label else mv_label
        class_map[raw] = idx

    for i in range(len(probs)):
        allowed = legal_moves_batch[i]
        idxs = [class_map[mv] for mv in allowed if mv in class_map]
        if idxs:
            row = probs[i]
            m = np.zeros_like(row)
            m[idxs] = row[idxs]
            total = m.sum()
            masked[i] = (m / total) if total > 0 else row
        else:
            masked[i] = probs[i]
    return masked

# ─── PREDICT + MASK + EVALUATE ───────────────────────────────────────────────
labels = np.arange(len(le_moves.classes_))
probs = move_clf.predict_proba(X_val)
probs_masked = apply_legal_move_mask(probs, legal_moves_batch, le_moves)
y_pred_enc = np.argmax(probs_masked, axis=1)
missed = 0
for i, allowed in enumerate(legal_moves_batch):
    true_move = le_moves.inverse_transform([y_val_enc[i]])[0]
    raw_true = true_move.split("_", 1)[1] if "_" in true_move else true_move
    if raw_true not in allowed:
        missed += 1
print(f"{missed} / {len(y_val_enc)} true moves missing from legal move masks")

# ─── Log a Few Failures ──────────────────────────────────────────────────────
sample_indices = random.sample(range(len(y_val_enc)), 10)

for i in sample_indices:
    true_move = le_moves.inverse_transform([y_val_enc[i]])[0]
    pred_move = le_moves.inverse_transform([y_pred_enc[i]])[0]
    allowed = legal_moves_batch[i]
    species = species_val[i]
    opponent = opponent_species[i]
    top3 = np.argsort(probs_masked[i])[::-1][:3]
    top3_moves = le_moves.inverse_transform(top3)

    print(f"[{i}] Species: {species}")
    print(f"Opponent: {opponent}")
    print(f"     True: {true_move}")
    print(f"     Pred: {pred_move}")
    print(f"     Legal: {allowed}")
    print(f"     Top-3: {top3_moves}")
    if true_move.split("_", 1)[1] not in allowed:
        print("True move was masked out!")
    print()

print("\nSamples where true move is NOT in top-10 (up to 10):")
count = 0
for i in range(len(y_val_enc)):
    true_idx = y_val_enc[i]
    top10 = np.argsort(probs_masked[i])[::-1][:10]

    if true_idx not in top10:
        true_move = le_moves.inverse_transform([true_idx])[0]
        pred_move = le_moves.inverse_transform([top10[0]])[0]
        top10_moves = le_moves.inverse_transform(top10)
        species = species_val[i]
        allowed = legal_moves_batch[i]

        print(f"[{i}] Species: {species}")
        print(f"     True: {true_move}")
        print(f"     Pred: {pred_move}")
        print(f"     Top-10: {top10_moves}")
        print(f"     Legal: {allowed}")
        print()
        count += 1
        if count >= 10:
            break
"""
acc_top1 = accuracy_score(y_val_enc, y_pred_enc)
acc_top3 = top_k_accuracy_score(y_val_enc, probs_masked, k=3, labels = labels)
acc_top5 = top_k_accuracy_score(y_val_enc, probs_masked, k=5, labels = labels)
acc_top10 = top_k_accuracy_score(y_val_enc, probs_masked, k=10, labels = labels)

print("Stage-2 Move Accuracy with Legal-Move Masking:")
print(f" → Top-1 Accuracy: {acc_top1:.4f}")
print(f" → Top-3 Accuracy: {acc_top3:.4f}")
print(f" → Top-5 Accuracy: {acc_top5:.4f}")
print(f" → Top-10 Accuracy: {acc_top10:.4f}")
"""