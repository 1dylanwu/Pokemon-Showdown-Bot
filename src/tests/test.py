import pandas as pd

# Path to your validation CSV
val_path = "data/parsed/val.csv"

# Load the CSV
df = pd.read_csv(val_path)

# Filter out rows where action == "Aerial Ace"
df_cleaned = df[df["action"] != "Aerial Ace"]

# Save the cleaned CSV
df_cleaned.to_csv("data/parsed/val_cleaned.csv", index=False)

print(f"Removed {len(df) - len(df_cleaned)} row(s) with action 'Aerial Ace'.")