import pandas as pd
import numpy as np
import joblib
from src.ingest.add_type_features import get_pokemon_types
from src.ingest.preprocess import build_feature_matrix, flatten_sets, flatten_haz

class StateProcessor:
    def __init__(
        self,
        pipeline_path:  str = "data/processed/general/pipeline.pkl",
        mlb1_path:      str = "data/processed/general/mlb_p1_team.pkl",
        mlb2_path:      str = "data/processed/general/mlb_p2_team.pkl",
        mlb_types_path: str = "data/processed/general/mlb_types.pkl",
        hz1_path:       str = "data/processed/general/hazard_keys_p1.pkl",
        hz2_path:       str = "data/processed/general/hazard_keys_p2.pkl",
    ):
        # Load the fitted preprocessing pipeline
        self.pipeline       = joblib.load(pipeline_path)

        # Load your multilabel binarizers and hazard‐key lists
        self.mlb_p1_team    = joblib.load(mlb1_path)
        self.mlb_p2_team    = joblib.load(mlb2_path)
        self.mlb_types      = joblib.load(mlb_types_path)
        self.hazard_keys_p1 = joblib.load(hz1_path)
        self.hazard_keys_p2 = joblib.load(hz2_path)

        # Capture the columns that the ColumnTransformer expects
        ct = self.pipeline.named_steps["trans"]
        self.required_cols = ct.feature_names_in_

    def process(self, state: dict) -> np.ndarray:
        """
        Turn a single battle-state dict into a feature vector.
        1) Auto-inject hp_frac from current/max HP
        2) Ensure active-mon types
        3) Build one-row DataFrame and expand teams, hazards, types
        4) Align to pipeline inputs and transform
        """

        if "p1a_active" in state and "p1a_types" not in state:
            state["p1a_types"] = get_pokemon_types(state["p1a_active"])
        if "p2a_active" in state and "p2a_types" not in state:
            state["p2a_types"] = get_pokemon_types(state["p2a_active"])

        # 2) One-row DataFrame
        df = pd.DataFrame([state])

        for col in ["p1a_types", "p2a_types"]:
            df[col] = df[col].apply(
                lambda types: [t.lower() for t in types] if isinstance(types, list) else types
            )

        # 3) Multi-hot encode each team
        p1_ts, self.mlb_p1_team = flatten_sets(
            df, "p1_team_species", "p1_team_", mlb=self.mlb_p1_team
        )
        p2_ts, self.mlb_p2_team = flatten_sets(
            df, "p2_team_species", "p2_team_", mlb=self.mlb_p2_team
        )

        # 4) Flatten hazards
        p1_haz = flatten_haz(df, "hazards_p1", "p1_haz_", keys=self.hazard_keys_p1)
        p2_haz = flatten_haz(df, "hazards_p2", "p2_haz_", keys=self.hazard_keys_p2)

        # 5) One-hot encode active-mon types
        p1_ty, self.mlb_types = flatten_sets(
            df, "p1a_types", "p1_type_", mlb=self.mlb_types
        )
        p2_ty, _ = flatten_sets(
            df, "p2a_types", "p2_type_", mlb=self.mlb_types
        )

        # 6) Build the raw feature matrix just like in training
        raw = pd.concat([
            df.select_dtypes("number"),
            df.select_dtypes("object"),
            p1_ts, p2_ts,
            p1_haz, p2_haz,
            p1_ty, p2_ty
        ], axis=1)

        result = build_feature_matrix(
            raw,
            mlb1=self.mlb_p1_team,
            mlb2=self.mlb_p2_team,
            mlb_types=self.mlb_types
        )
        X_proc           = result[0]
        self.mlb_p1_team = result[2]
        self.mlb_p2_team = result[3]
        self.mlb_types   = result[4]

        # 7) Align to the transformer’s expected columns
        X_aligned = X_proc.reindex(columns=self.required_cols, fill_value=0)

        # 8) Apply the pipeline to get the final numpy array
        X_final = self.pipeline.transform(X_aligned)

        return X_final.astype(np.float32)