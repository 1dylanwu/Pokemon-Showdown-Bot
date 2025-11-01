import pandas as pd

# Path to your validation CSV
val_path = "data/parsed/val.csv"

# Load the CSV
df_val = pd.read_csv(val_path)

# Confirm the target row
target_index = 1266
target_label = df_val.loc[target_index, "action"]  # adjust column name if needed

if target_label == "Aerial Ace":
    print(f"Removing row {target_index} with label: '{target_label}'")
    df_val = df_val.drop(index=target_index)
    df_val.to_csv(val_path, index=False)
    print("CSV updated successfully.")
else:
    print(f"Row {target_index} does not contain 'aerial ace'. Found: '{target_label}'")