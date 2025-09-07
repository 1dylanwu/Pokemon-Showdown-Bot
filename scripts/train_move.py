import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from train_type import split_action_type

pre = "data/processed/"
X_train, y_train = np.load(pre + "X_train.npy", ).astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
X_val,   y_val   = np.load(pre+"X_val.npy").astype(np.float32),   np.load(pre+"y_val.npy", allow_pickle=True)
X_test,  y_test  = np.load(pre+"X_test.npy").astype(np.float32),  np.load(pre+"y_test.npy", allow_pickle=True)

y_tr_type = split_action_type(y_train)
y_va_type = split_action_type(y_val)
y_te_type = split_action_type(y_test)

move_idx_tr = np.where(y_tr_type == "move")[0]
move_idx_va = np.where(y_va_type == "move")[0]

X_tr_moves = X_train[move_idx_tr]
y_tr_moves = y_train[move_idx_tr]

X_va_moves = X_val[move_idx_va]
y_va_moves = y_val[move_idx_va]

# encode move labels into ints
le_moves = LabelEncoder().fit(y_tr_moves)
y_tr_moves_enc = le_moves.transform(y_tr_moves)
y_va_moves_enc = le_moves.transform(y_va_moves)

move_clf = RandomForestClassifier(
    class_weight="balanced_subsample", n_estimators=200, n_jobs=-1
)
move_clf.fit(X_tr_moves, y_tr_moves_enc)

print("Stage2a move train acc:", move_clf.score(X_tr_moves, y_tr_moves_enc))
print("Stage2a move val   acc:", move_clf.score(X_va_moves,  y_va_moves_enc))

joblib.dump((move_clf, le_moves), "models/stage2_move/move_clf_1.0.pkl")