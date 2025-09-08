import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

pre = "data/processed/"
X_train, y_train = np.load(pre + "X_train.npy", ).astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
X_val,   y_val   = np.load(pre+"X_val.npy").astype(np.float32),   np.load(pre+"y_val.npy", allow_pickle=True)
X_test,  y_test  = np.load(pre+"X_test.npy").astype(np.float32),  np.load(pre+"y_test.npy", allow_pickle=True)

def split_action_type(y):
    types = np.array(["move" if act.startswith("move_") else "switch"
                      for act in y])
    return types

y_tr_type = split_action_type(y_train)
y_va_type = split_action_type(y_val)
y_te_type = split_action_type(y_test)

move_idx_tr = np.where(y_tr_type == "move")[0]
move_idx_va = np.where(y_va_type == "move")[0]

X_tr_moves = X_train[move_idx_tr]
y_tr_moves = y_train[move_idx_tr]

X_va_moves = X_val[move_idx_va]
y_va_moves = y_val[move_idx_va]

imp = SimpleImputer(strategy="mean", fill_value=0)
X_tr_moves = imp.fit_transform(X_tr_moves)
X_va_moves = imp.transform(X_va_moves)

# encode move labels into ints
le_moves = LabelEncoder().fit(y_tr_moves)
y_tr_moves_enc = le_moves.transform(y_tr_moves)
y_va_moves_enc = le_moves.transform(y_va_moves)

joblib.dump(imp, "models/stage2_move/util/imputer.pkl")
joblib.dump(le_moves, "models/stage2_move/util/label_encoder.pkl")
np.save("data/processed/move/X_tr_moves.npy", X_tr_moves)
np.save("data/processed/move/y_tr_moves.npy", y_tr_moves_enc)
np.save("data/processed/move/X_va_moves.npy", X_va_moves)
np.save("data/processed/move/y_va_moves.npy", y_va_moves_enc)