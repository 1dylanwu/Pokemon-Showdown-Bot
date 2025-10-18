import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib
pipeline       = joblib.load("data/processed/general/pipeline.pkl")


ct = pipeline.named_steps["trans"]
required_cols = ct.feature_names_in_
print(required_cols)