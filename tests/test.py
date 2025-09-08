from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

pipeline = joblib.load("data/processed/pipeline.pkl")
ct = pipeline.named_steps["trans"]

feature_names = ct.get_feature_names_out()
print(feature_names[:10])