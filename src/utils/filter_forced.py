# builds dataset without forced switches so that the type model can be correctly trained
import numpy as np

def is_forced(label: str) -> bool:
    return str(label).startswith("forced_switch_")

def is_move(label: str) -> bool:
    return str(label).startswith("move_")

def filter_forced(X, y):
    mask = np.array([not str(label).startswith("forced_switch_") for label in y])
    X_clean = X[mask]
    y_clean = np.array([1 if is_move(label) else 0 for label in y[mask]], dtype=np.int32)
    return X_clean, y_clean


pre = "data/processed/general/"
X_train = np.load(pre + "X_train.npy").astype(np.float32)
y_train = np.load(pre + "y_train.npy", allow_pickle=True)

X_val = np.load(pre + "X_val.npy").astype(np.float32)
y_val = np.load(pre + "y_val.npy", allow_pickle=True)

X_test = np.load(pre + "X_test.npy").astype(np.float32)
y_test = np.load(pre + "y_test.npy", allow_pickle=True)

X_train_clean, y_train_clean = filter_forced(X_train, y_train)
X_val_clean, y_val_clean = filter_forced(X_val, y_val)
X_test_clean, y_test_clean = filter_forced(X_test, y_test)

np.save("data/processed/type/X_train_clean.npy", X_train_clean)
np.save("data/processed/type/y_train_clean.npy", y_train_clean)
np.save("data/processed/type/X_val_clean.npy", X_val_clean)
np.save("data/processed/type/y_val_clean.npy", y_val_clean)
np.save("data/processed/type/X_test_clean.npy", X_test_clean)
np.save("data/processed/type/y_test_clean.npy", y_test_clean)
