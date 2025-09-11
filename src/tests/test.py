import numpy as np
import pandas as pd

# Load the .npy file
npy_path = "data/processed/general/X_val_SWITCH.npy"  # change this to your actual path
csv_path = "X_val_SWITCH.csv"

# Load the array
arr = np.load(npy_path)

# Convert to DataFrame
df = pd.DataFrame(arr)

# Save to CSV
df.to_csv(csv_path, index=False)

print(f"Saved {df.shape[0]} rows × {df.shape[1]} columns to {csv_path}")