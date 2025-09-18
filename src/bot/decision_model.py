import ast
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Union
from src.bot.state_processor import StateProcessor
from src.utils.utils import split_action_type
from src.ingest.add_type_features import get_pokemon_types
from src.ingest.preprocess import load_and_clean, build_feature_matrix, flatten_haz


class DecisionModel:
    """
    Hierarchical model that combines type classification, move prediction, 
    switch prediction, and forced switch prediction for Pokemon battles.

    1. Checks if either active Pokemon has "faint" status -> use forced switch model
    2. Otherwise, use type classifier with threshold to decide move vs switch
    3. If move: use move model with legal move masking
    4. If switch: use switch model with legal switch masking
    """
    
    def __init__(self, model_dir: str = "models", move_threshold: float = 0.62):
        self.move_threshold = move_threshold
        self.model_dir = Path(model_dir)
        self._load_models()
        self._load_utilities()
        self.class_map = {
            mv_label.split("_", 1)[1] if "_" in mv_label else mv_label: idx
            for idx, mv_label in enumerate(self.le_moves.classes_)
        }
        
    def _load_models(self):
        """Load all required models"""
        # Type classifier (stage 1)
        self.type_clf = joblib.load(self.model_dir / "stage1_type/type_clf_3.1.pkl")
        
        # Move classifier (stage 2)
        self.move_clf = joblib.load(self.model_dir / "stage2_move/final/move_clf_3.1.pkl")
        
        # Switch classifier (stage 2) 
        self.switch_clf = joblib.load(self.model_dir / "stage2_switch/final/switch_clf_2.0.pkl")
        
        # Forced switch classifier (stage 3)
        self.forced_clf = joblib.load(self.model_dir / "stage3_forced/final/switch_clf_2.0.pkl")
        
    def _load_utilities(self):
        """Load label encoders and other utilities"""
        # Label encoders
        self.le_moves = joblib.load(self.model_dir / "stage2_move/util/label_encoder.pkl")
        self.le_switch = joblib.load(self.model_dir / "stage2_switch/util/label_encoder.pkl")
        self.le_forced = joblib.load(self.model_dir / "stage3_forced/util/label_encoder.pkl")

        self.processor = StateProcessor(
            pipeline_path="data/processed/general/pipeline.pkl",
            mlb1_path="data/processed/general/mlb_p1_team.pkl",
            mlb2_path="data/processed/general/mlb_p2_team.pkl",
            mlb_types_path="data/processed/general/mlb_types.pkl",
            hz1_path="data/processed/general/hazard_keys_p1.pkl",
            hz2_path="data/processed/general/hazard_keys_p2.pkl"
        )
    
    
    def _check_fainted_status(self, state: Dict) -> Tuple[bool, Optional[str]]:
        """
        Check if either active Pokemon has fainted status.
        
        Args:
            state: Battle state dictionary with p1a_status and p2a_status
            
        Returns:
            (is_fainted, fainted_side) where fainted_side is 'p1a' or 'p2a' or None
        """
        p1_status = state.get("p1a_status", "").lower()
        p2_status = state.get("p2a_status", "").lower()
        
        if p1_status == "faint":
            return True, "p1a"
        elif p2_status == "faint":
            return True, "p2a"
        else:
            return False, None
    
    def _apply_legal_move_mask(self, probs: np.ndarray, legal_moves: List[str]) -> np.ndarray:
        """
        Apply legal move masking to move probabilities.
        
        Args:
            probs: Raw move probabilities from model
            legal_moves: List of legal moves for the active Pokemon
            
        Returns:
            Masked and renormalized probabilities
        """
        masked = np.zeros_like(probs)
        
        # Apply mask
        allowed_indices = [self.class_map[mv] for mv in legal_moves if mv in self.class_map]
        
        if allowed_indices:
            row = probs[0]  # Single prediction
            m = np.zeros_like(row)
            m[allowed_indices] = row[allowed_indices]
            total = m.sum()
            masked[0] = (m / total) if total > 0 else row
        else:
            masked[0] = probs[0]
            
        return masked

    
    def _apply_legal_switch_mask(self, probs: np.ndarray, legal_switches: List[str], forced: bool = False) -> np.ndarray:
        """
        Apply legal switch masking to switch probabilities.
        
        Args:
            probs: Raw switch probabilities from model
            legal_switches: List of legal switch options
            
        Returns:
            Masked and renormalized probabilities
        """
        encoder = self.le_forced if forced else self.le_switch
        masked = np.zeros_like(probs)
        
        # Get indices of legal switches
        legal_indices = []
        for switch in legal_switches:
            if switch in encoder.classes_:
                idx = encoder.transform([switch])[0]
                legal_indices.append(idx)
        if legal_indices:
            row = probs[0]  # Single prediction
            m = np.zeros_like(row)
            m[legal_indices] = row[legal_indices]
            total = m.sum()
            masked[0] = (m / total) if total > 0 else row
        else:
            masked[0] = probs[0]
            
        return masked
    
    def predict(self, state: Dict, legal_moves: List, legal_switches: List) -> Dict:
        """
        Make a hierarchical prediction for a given battle state and side.
        
        Args:
            state: Battle state dictionary containing all relevant battle information
            side: Side to make prediction for ('p1' or 'p2')
            
        Returns:
            Dictionary containing:
            - action_type: 'move', 'switch', or 'forced_switch'
            - action: The predicted move or switch target
            - confidence: Prediction confidence score
            - probabilities: Raw probabilities for debugging
        """
        # Get side
        side = state["side"][:2]
        # Check for fainted status first
        is_fainted, fainted_side = self._check_fainted_status(state)
        
        if is_fainted and fainted_side == f"{side}a":
            # Use forced switch model
            X_features = self.processor.process(state)
            forced_probs = self.forced_clf.predict_proba(X_features)
            
            # Apply legal switch masking for forced switches
            masked_probs = self._apply_legal_switch_mask(forced_probs, legal_switches)
            
            predicted_idx = np.argmax(masked_probs[0])
            predicted_switch = self.le_switch.inverse_transform([predicted_idx])[0]
            
            return {
                "action_type": "forced_switch",
                "action": predicted_switch,
                "confidence": float(masked_probs[0][predicted_idx]),
                "probabilities": masked_probs[0].tolist()
            }
        
        # Use type classifier to decide move vs switch
        X_features = self.processor.process(state)
        X_df = pd.DataFrame(X_features, columns=self.processor.pipeline.named_steps["trans"].get_feature_names_out())

        dX = xgb.DMatrix(X_features)
        type_probs = self.type_clf.predict(dX)
        
        move_prob = type_probs[0]  # Probability of move (class 1)
        print(f"prob of move: {move_prob}")
        # Apply threshold
        if move_prob >= self.move_threshold:
            # Predict move
            move_probs = self.move_clf.predict_proba(X_features)
            
            # Apply legal move masking
            active_species = state.get(f"{side}a_active", "")
            masked_probs = self._apply_legal_move_mask(move_probs, legal_moves)
            
            predicted_idx = np.argmax(masked_probs[0])
            predicted_move = self.le_moves.inverse_transform([predicted_idx])[0]
            return {
                "action_type": "move",
                "action": predicted_move,
                "confidence": float(masked_probs[0][predicted_idx]),
                "probabilities": masked_probs[0].tolist(),
                "type_confidence": float(move_prob)
            }
        else:
            # Predict switch
            switch_probs = self.switch_clf.predict_proba(X_features)
            # Apply legal switch masking
            masked_probs = self._apply_legal_switch_mask(switch_probs, legal_switches)
            
            predicted_idx = np.argmax(masked_probs[0])
            predicted_switch = self.le_switch.inverse_transform([predicted_idx])[0]
            
            return {
                "action_type": "switch", 
                "action": predicted_switch,
                "confidence": float(masked_probs[0][predicted_idx]),
                "probabilities": masked_probs[0].tolist(),
                "type_confidence": float(1 - move_prob)
            }
    
    def predict_forced(self, state: Dict, legal_switches: List) -> Dict:
        side = state["side"][:2]
        X_features = self.processor.process(state)
        forced_probs = self.forced_clf.predict_proba(X_features)
        # Apply legal switch masking for forced switches
        masked_probs = self._apply_legal_switch_mask(forced_probs, legal_switches)
            
        predicted_idx = np.argmax(masked_probs[0])
        predicted_switch = self.le_switch.inverse_transform([predicted_idx])[0]
            
        return {
            "action_type": "forced_switch",
            "action": predicted_switch,
            "confidence": float(masked_probs[0][predicted_idx]),
            "probabilities": masked_probs[0].tolist()
        }