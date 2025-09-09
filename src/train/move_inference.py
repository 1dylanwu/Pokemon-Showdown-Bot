import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import random

MODEL_PATH = Path("models/stage2_move/final/move_clf_1.0.pkl")
LABEL_ENCODER = Path("models/stage2_move/util/label_encoder.pkl")
FULL_MOVEPOOLS = Path("data/processed/move/full_movepools_from_data.pkl")
X_VAL_PATH = Path("data/processed/move/X_va_moves.npy")
Y_VAL_PATH = Path("data/processed/move/y_va_moves.npy")
VAL_NAME_PATH = Path("data/processed/move/species_va.npy")
VAL_MOVE_PATH = Path("data/processed/move/species_va_moves.npy")

# load everything
move_clf = joblib.load(MODEL_PATH)
le_moves = joblib.load(LABEL_ENCODER)
movepools = joblib.load(FULL_MOVEPOOLS)

X_val = np.load(X_VAL_PATH)
y_val_enc = np.load(Y_VAL_PATH, allow_pickle=True)

# create list of the active species per row and the chosen moves
species_val = np.load(VAL_NAME_PATH, allow_pickle = True)
species_val_move = np.load(VAL_MOVE_PATH, allow_pickle=True)

# create list of total available moves per row, according to possible movesets
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

# predict, mask, and evaluate
labels = np.arange(len(le_moves.classes_))
probs = move_clf.predict_proba(X_val)
probs_masked = apply_legal_move_mask(probs, legal_moves_batch, le_moves)
y_pred_enc = np.argmax(probs_masked, axis=1)

acc_top1 = accuracy_score(y_val_enc, y_pred_enc)
acc_top3 = top_k_accuracy_score(y_val_enc, probs_masked, k=3, labels = labels)
acc_top5 = top_k_accuracy_score(y_val_enc, probs_masked, k=5, labels = labels)
acc_top10 = top_k_accuracy_score(y_val_enc, probs_masked, k=10, labels = labels)

print("Stage-2 Move Accuracy with Legal-Move Masking:")
print(f" → Top-1 Accuracy: {acc_top1:.4f}")
print(f" → Top-3 Accuracy: {acc_top3:.4f}")
print(f" → Top-5 Accuracy: {acc_top5:.4f}")
print(f" → Top-10 Accuracy: {acc_top10:.4f}")
