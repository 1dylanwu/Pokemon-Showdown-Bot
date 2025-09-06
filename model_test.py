import numpy as np


X_train = np.load('data/processed/X_train.npy', allow_pickle=False)
y_train = np.load('data/processed/y_train.npy', allow_pickle=False)


print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")