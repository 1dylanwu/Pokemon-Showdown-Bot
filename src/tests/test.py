import numpy as np
X_new = np.load("data/processed/X_train.npy")
print("New X_train shape:", X_new.shape)
print("First row turn value:", X_new[0, 1])
