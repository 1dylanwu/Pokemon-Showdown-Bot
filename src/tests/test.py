import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

paths = [
    "data/processed/general/y_train.npy",
    "data/processed/general/y_val.npy",
    "data/processed/general/y_test.npy"
]

all_pokemon = []
for path in paths:
    labels = np.load(path, allow_pickle=True)
    for label in labels:
        if label.startswith("switch_") or label.startswith("forced_switch_"):
            name = label.replace("forced_switch_", "switch_")
            pokemon = name[len("switch_"):]
            all_pokemon.append(pokemon)


le = LabelEncoder()
le.fit(all_pokemon)


joblib.dump(le, "models/stage2_switch/util/label_encoder.pkl")
print(f"Saved encoder with {len(le.classes_)} Pokémon species.")
