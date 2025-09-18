import joblib
from pathlib import Path
booster = joblib.load("models/stage1_type/type_clf_3.2.pkl")
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
pipeline = joblib.load("data/processed/general/pipeline.pkl")
feature_names = pipeline.named_steps["trans"].get_feature_names_out()
booster.feature_names = list(feature_names)

xgb.plot_importance(booster, max_num_features=30)
plt.tight_layout()
plt.show()