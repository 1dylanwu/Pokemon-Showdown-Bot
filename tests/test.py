import pandas as pd

# 1) Load the entire file in one go so pandas can see all values before inferring types
df = pd.read_csv("data/parsed/train.csv", low_memory=False)

# 2) Grab the column‐indices from the warning
mixed_idxs = [14,15,24,25,28,30,34,37,38,40]
for i in mixed_idxs:
    col = df.columns[i]
    types = df[col].apply(type).value_counts().to_dict()
    print(f"Column #{i} → '{col}':", types)
