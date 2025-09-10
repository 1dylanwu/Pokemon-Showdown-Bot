import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

pre = "data/processed/general"
X_train, y_train = np.load(pre + "X_train.npy", ).astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
X_val, y_val = np.load(pre+"X_val.npy").astype(np.float32), np.load(pre+"y_val.npy", allow_pickle=True)
X_test, y_test = np.load(pre+"X_test.npy").astype(np.float32), np.load(pre+"y_test.npy", allow_pickle=True)

# y_tr_type[i] = "move" or "switch", length is full y_train array
# move_idx_tr = indices of only the "move" actions in y_train,length is number of move actions
y_tr_type = np.array(["move" if act.startswith("move_") else "switch"
                      for act in y_train])
move_idx_tr = np.where(y_tr_type == "move")[0]

y_va_type = np.array(["move" if act.startswith("move_") else "switch" 
                      for act in y_val])
move_idx_va = np.where(y_va_type == "move")[0]

train_df = pd.read_csv("data/parsed/train.csv", dtype=str)
val_df = pd.read_csv("data/parsed/val.csv",   dtype=str)

def clean_df(df):
    df.drop(columns=["turn"], errors="ignore", inplace=True) 
    df.rename(columns=lambda c: c[6:] if c.startswith("state_") else c, inplace=True)
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
    df = df[df["turn"] > 0]
    df = df[df["action_type"] == "move"]
    return df

train_df = clean_df(train_df)
val_df   = clean_df(val_df)

# return the active species for each row based off which side is active (p1a or p2a)
species_train_full = np.where(train_df["side"] == "p1a",
                              train_df["p1a_active"],
                              train_df["p2a_active"])
species_val_full = np.where(val_df["side"] == "p1a",
                              val_df["p1a_active"],
                              val_df["p2a_active"])

species_tr_moves = train_df["action"]
species_va_moves = val_df["action"]

np.save("data/processed/move/species_tr_moves.npy", species_tr_moves)
np.save("data/processed/move/species_va_moves.npy", species_va_moves)
np.save("data/processed/move/species_tr.npy", species_train_full)
np.save("data/processed/move/species_va.npy", species_val_full)


move_idx_tr = np.where(y_tr_type == "move")[0]
move_idx_va = np.where(y_va_type == "move")[0]

X_tr_moves = X_train[move_idx_tr]
y_tr_moves = y_train[move_idx_tr]

X_va_moves = X_val[move_idx_va]
y_va_moves = y_val[move_idx_va]

if np.isnan(X_tr_moves).any() or np.isnan(X_va_moves).any():
    print("NaNs detected — fitting SimpleImputer")
    imp = SimpleImputer(strategy="mean")
    X_tr_moves = imp.fit_transform(X_tr_moves)
    X_va_moves = imp.transform(X_va_moves)
    joblib.dump(imp, "models/stage2_move/util/imputer.pkl")
else:
    print("No NaNs detected — skipping imputer")
    imp = None  # or leave as-is


# encode move labels into ints
le_moves = LabelEncoder().fit(y_tr_moves)
y_tr_moves_enc = le_moves.transform(y_tr_moves)
y_va_moves_enc = le_moves.transform(y_va_moves)


joblib.dump(le_moves, "models/stage2_move/util/label_encoder.pkl")
np.save("data/processed/move/X_tr_moves.npy", X_tr_moves)
np.save("data/processed/move/y_tr_moves.npy", y_tr_moves_enc)
np.save("data/processed/move/X_va_moves.npy", X_va_moves)
np.save("data/processed/move/y_va_moves.npy", y_va_moves_enc)
