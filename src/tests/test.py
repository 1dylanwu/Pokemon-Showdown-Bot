import numpy as np
import pandas as pd

# Load the .npy file
data = np.load("data/processed/move/X_va_moves.npy", allow_pickle=True)
print(data.shape)
