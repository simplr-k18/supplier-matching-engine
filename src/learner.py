import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

class MatchLearner:
    """
    Machine Learning component that learns from user corrections.
    Uses features from matching algorithms to predict if pairs should match.
    """
    
    def __init__(self, model_path='models/match_model.pkl'):
        self.model_path = Path(model_path)
        self.model = None
        self.is_trained = False
        
        # Try to load existing model
        if self.model_path.exists():
            self.load_model()
    
    def extract_features(self, scores_dict):
        """
        Convert similarity scores dict to feature vector.
        
        Args:
            scores_dict: Dict with keys like 'levenshtein', 'fuzz_ratio', etc.
        
        Returns:
            numpy array of features
        """
        features = [
            scores_dict.get('levenshtein', 0),
            scores_dict.get('fuzz_ratio', 0),
            scores_dict.get('fuzz_partial', 0),
            scores_dict.get('fuzz_token_sort', 0),
            scores_dict.get('fuzz_token_set', 0),
            scores_dict.get('jaro_winkler', 0),
            scores_dict.get('overall', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def add_training_example(self, scores_dict, is_match):
        """
        Add a user-labeled example to training data.
        
        Args:
            scores_dict: Dictionary of similarity scores
            is_match: True if user accepted, False if rejected
        """
        # Create training data file if doesn't exist
        training_file = Path('data/training_data.csv')
        
        # Prepare row
        row = {
            'levenshtein': scores_dict.get('levenshtein', 0),
            'fuzz_ratio': scores_dict.get('fuzz_ratio', 0),
            'fuzz_partial': scores_dict.get('fuzz_partial', 0),
            'fuzz_token_sort': scores_dict.get('fuzz_token_sort', 0),
            'fuzz_token_set': scores_dict.get('fuzz_token_set', 0),
            'jaro_winkler': scores_dict.get('jaro_winkler', 0),
            'overall': scores_dict.get('overall', 0),
            'is_match': 1 if is_match else 0
        }
        
        # Append to CSV
        df_new = pd.DataFrame([row])
        
        if training_file.exists():
            df_existing = pd.read_csv(training_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(training_file, index=False)
        else:
            df_new.to_csv(training_file, index=False)
    
    def train(self, min_samples=10):
        """
        Train the model on accumulated user corrections.
        
        Args:
            min_samples: Minimum number of samples needed to train
        
        Returns:
            True if training succeeded, False otherwise
        """
        training_file = Path('data/training_data.csv')
        
        if not training_file.exists():
            return False
        
        df = pd.read_csv(training_file)
        
        if len(df) < min_samples:
            return False
        
        # Prepare features and labels
        X = df.drop('is_match', axis=1).values
        y = df['is_match'].values
        
        # Check if we have both classes
        if len(np.unique(y)) < 2:
            return False
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return True
    
    def predict(self, scores_dict):
        """
        Predict if a pair should match using trained model.
        
        Args:
            scores_dict: Dictionary of similarity scores
        
        Returns:
            Tuple of (prediction, confidence)
            prediction: 1 if match, 0 if not
            confidence: probability score (0-1)
        """
        if not self.is_trained:
            # Fallback to rule-based if not trained
            return (1 if scores_dict.get('overall', 0) >= 80 else 0, 0.5)
        
        features = self.extract_features(scores_dict)
        prediction = self.model.predict(features)[0]
        confidence = self.model.predict_proba(features)[0][prediction]
        
        return (prediction, confidence)
    
    def get_feature_importance(self):
        """
        Get which features are most important for predictions.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained or self.model is None:
            return None
        
        feature_names = [
            'levenshtein', 'fuzz_ratio', 'fuzz_partial',
            'fuzz_token_sort', 'fuzz_token_set', 'jaro_winkler', 'overall'
        ]
        
        importances = self.model.feature_importances_
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def save_model(self):
        """Save trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Could not load model: {e}")
            return False
    
    def get_training_stats(self):
        """Get statistics about training data."""
        training_file = Path('data/training_data.csv')
        
        if not training_file.exists():
            return {
                'total_samples': 0,
                'accepted': 0,
                'rejected': 0,
                'model_trained': False
            }
        
        df = pd.read_csv(training_file)
        
        return {
            'total_samples': len(df),
            'accepted': len(df[df['is_match'] == 1]),
            'rejected': len(df[df['is_match'] == 0]),
            'model_trained': self.is_trained
        }