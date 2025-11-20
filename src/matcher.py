import pandas as pd
from fuzzywuzzy import fuzz
from Levenshtein import distance as levenshtein_distance
import jellyfish
from src.learner import MatchLearner
import re

class SupplierMatcher:
    """
    Core supplier matching engine using multiple algorithms.
    """
    
    def __init__(self, threshold=80, use_ml=False):
        """
        Args:
            threshold (int): Minimum similarity score (0-100) to consider a match
            use_ml (bool): Whether to use ML predictions alongside rule-based matching
        """
        self.threshold = threshold
        self.match_history = []
        self.use_ml = use_ml
        self.learner = MatchLearner() if use_ml else None

    def normalize_name(self, name):
        """Clean and standardize company name."""
        if pd.isna(name):
            return ""
        
        # Convert to string and lowercase
        name = str(name).lower().strip()
        
        # Remove common suffixes
        suffixes = ['inc', 'inc.', 'corp', 'corp.', 'corporation', 'company', 
                   'co', 'co.', 'ltd', 'ltd.', 'llc', 'l.l.c.', 'limited']
        for suffix in suffixes:
            name = re.sub(r'\b' + suffix + r'\b', '', name)
        
        # Remove special characters
        name = re.sub(r'[^\w\s]', ' ', name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def calculate_similarity(self, name1, name2):
        """
        Calculate similarity using multiple algorithms.
        Returns dict with scores from each method.
        """
        # Normalize both names
        n1 = self.normalize_name(name1)
        n2 = self.normalize_name(name2)
        
        if not n1 or not n2:
            return {'overall': 0}
        
        scores = {
            'levenshtein': 100 - (levenshtein_distance(n1, n2) / max(len(n1), len(n2)) * 100),
            'fuzz_ratio': fuzz.ratio(n1, n2),
            'fuzz_partial': fuzz.partial_ratio(n1, n2),
            'fuzz_token_sort': fuzz.token_sort_ratio(n1, n2),
            'fuzz_token_set': fuzz.token_set_ratio(n1, n2),
            'jaro_winkler': jellyfish.jaro_winkler_similarity(n1, n2) * 100,
        }
        
        # Calculate weighted average (you can tune these weights)
        scores['overall'] = (
            scores['levenshtein'] * 0.15 +
            scores['fuzz_ratio'] * 0.20 +
            scores['fuzz_token_sort'] * 0.25 +
            scores['fuzz_token_set'] * 0.25 +
            scores['jaro_winkler'] * 0.15
        )
        
        return scores
    
    def find_matches(self, df, name_column='vendor_name'):
        """
        Find potential matches in a DataFrame.
        Returns DataFrame with match groups.
        """
        matches = []
        processed = set()
        
        for idx1, row1 in df.iterrows():
            if idx1 in processed:
                continue
                
            name1 = row1[name_column]
            match_group = [idx1]
            
            for idx2, row2 in df.iterrows():
                if idx1 >= idx2 or idx2 in processed:
                    continue
                
                name2 = row2[name_column]
                scores = self.calculate_similarity(name1, name2)
                
                if scores['overall'] >= self.threshold:
                    match_group.append(idx2)
                    processed.add(idx2)
                    
                    matches.append({
                        'name_1': name1,
                        'name_2': name2,
                        'similarity': round(scores['overall'], 2),
                        'levenshtein': round(scores['levenshtein'], 2),
                        'token_sort': round(scores['fuzz_token_sort'], 2),
                        'jaro_winkler': round(scores['jaro_winkler'], 2)
                    })
            
            processed.add(idx1)
        
        return pd.DataFrame(matches)

    def find_matches_with_ml(self, df, name_column='vendor_name'):
        """
        Find matches using both rule-based and ML approaches.
        """
        matches = []
        processed = set()
        
        for idx1, row1 in df.iterrows():
            if idx1 in processed:
                continue
                
            name1 = row1[name_column]
            match_group = [idx1]
            
            for idx2, row2 in df.iterrows():
                if idx1 >= idx2 or idx2 in processed:
                    continue
                
                name2 = row2[name_column]
                scores = self.calculate_similarity(name1, name2)
                
                # Rule-based decision
                rule_based_match = scores['overall'] >= self.threshold
                
                # ML-based decision (if available)
                ml_match = False
                ml_confidence = 0.0
                
                if self.use_ml and self.learner and self.learner.is_trained:
                    ml_prediction, ml_confidence = self.learner.predict(scores)
                    ml_match = ml_prediction == 1
                
                # Combine decisions (ML overrides if confident)
                is_match = rule_based_match
                if self.use_ml and self.learner.is_trained:
                    if ml_confidence > 0.7:  # High confidence threshold
                        is_match = ml_match
                
                if is_match:
                    match_group.append(idx2)
                    processed.add(idx2)
                    
                    matches.append({
                        'name_1': name1,
                        'name_2': name2,
                        'similarity': round(scores['overall'], 2),
                        'levenshtein': round(scores['levenshtein'], 2),
                        'token_sort': round(scores['fuzz_token_sort'], 2),
                        'jaro_winkler': round(scores['jaro_winkler'], 2),
                        'ml_confidence': round(ml_confidence * 100, 2) if self.use_ml else None,
                        'ml_prediction': 'Match' if ml_match else 'No Match' if self.use_ml else None
                    })
            
            processed.add(idx1)
        
        return pd.DataFrame(matches)