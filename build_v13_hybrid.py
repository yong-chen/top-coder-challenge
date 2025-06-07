#!/usr/bin/env python3
"""
V13 - Hybrid Model (V2 + Rule-Based Adjustments)
Combines the best of ML and business rules
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

class HybridReimbursementModel:
    def __init__(self, v2_model_path):
        """Initialize with the trained V2 model"""
        model_data = joblib.load(v2_model_path)
        self.v2_model = model_data['model']
        self.feature_cols = model_data['feature_cols']
    
    def _prepare_features(self, X):
        """Create all features used by V2 model"""
        X_processed = X[['days', 'miles', 'receipts']].copy()
        
        # Add all features used in V2
        X_processed['days_squared'] = X_processed['days'] ** 2
        X_processed['miles_squared'] = X_processed['miles'] ** 2
        X_processed['receipts_squared'] = X_processed['receipts'] ** 2
        X_processed['days_miles'] = X_processed['days'] * X_processed['miles']
        X_processed['days_receipts'] = X_processed['days'] * X_processed['receipts']
        X_processed['miles_receipts'] = X_processed['miles'] * X_processed['receipts']
        
        # Add categorical features
        X_processed['high_expense'] = (X_processed['receipts'] > 1000).astype(int)
        X_processed['long_distance'] = (X_processed['miles'] > 500).astype(int)
        X_processed['very_high_expense'] = (X_processed['receipts'] > 2000).astype(int)
        X_processed['short_trip'] = (X_processed['days'] <= 2).astype(int)
        X_processed['medium_trip'] = ((X_processed['days'] > 2) & (X_processed['days'] <= 5)).astype(int)
        X_processed['long_trip'] = (X_processed['days'] > 5).astype(int)
        X_processed['short_distance'] = (X_processed['miles'] <= 200).astype(int)
        
        # Ensure all required features are present
        for col in self.feature_cols:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        return X_processed[self.feature_cols]
    
    def predict(self, X):
        """
        Make predictions with V2 + rule-based adjustments
        
        Adjustment Rules:
        1. Cap high-receipt predictions
        2. Reduce very short, high-mileage trips
        3. Ensure minimum reimbursement
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['days', 'miles', 'receipts'])
        
        # Prepare features for V2
        X_processed = self._prepare_features(X)
        
        # Get V2 predictions
        v2_preds = self.v2_model.predict(X_processed)
        
        # Apply adjustments
        adjusted_preds = []
        for idx, (_, row) in enumerate(X.iterrows()):
            pred = v2_preds[idx]
            days, miles, receipts = row['days'], row['miles'], row['receipts']
            
            # 1. Cap high-receipt predictions
            if receipts > 1500:
                cap = (days * 150) + (miles * 0.6)
                pred = min(pred, cap)
            
            # 2. Reduce very short, high-mileage trips
            if days <= 2 and miles > 500:
                pred *= 0.7  # 30% reduction
            
            # 3. Ensure minimum reimbursement
            min_reimbursement = max(50 * days, 100)
            pred = max(pred, min_reimbursement)
            
            adjusted_preds.append(pred)
        
        return np.array(adjusted_preds)

def load_data():
    """Load and prepare the data"""
    with open('public_cases.json', 'r') as f:
        raw_data = json.load(f)
    
    data = []
    for case in raw_data:
        data.append({
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        })
    return pd.DataFrame(data)

def evaluate_model(model, X, y):
    """Evaluate model and print metrics"""
    # Get predictions for all test samples
    y_pred = model.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate accuracy within 10%
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = np.abs((y - y_pred) / y)
        accuracy = np.mean(rel_errors < 0.10) * 100
    
    # Calculate improvement over baseline
    baseline_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
    improvement = (1 - (mae / baseline_mae)) * 100
    
    return mae, accuracy, improvement, y_pred

def main():
    print("🚀 V13 - Hybrid Model (V2 + Rule-Based Adjustments)")
    print("-" * 60)
    
    # Load data
    df = load_data()
    X = df[['days', 'miles', 'receipts']]
    y = df['reimbursement']
    
    # Train/test split (80/20) with shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize V13 model
    print("\n🔧 Initializing V13 model...")
    model = HybridReimbursementModel('reimbursement_model.pkl')
    
    # Evaluate V13 on test set
    print("📊 Evaluating V13...")
    mae, accuracy, improvement, y_pred = evaluate_model(model, X_test, y_test)
    
    # Get V2 performance for comparison on same test set
    print("\n🔍 Evaluating V2 baseline...")
    v2_model_data = joblib.load('reimbursement_model.pkl')
    v2_model = v2_model_data['model']
    
    # Prepare features for V2 using the same preprocessing
    X_processed = model._prepare_features(X_test)
    v2_pred = v2_model.predict(X_processed)
    v2_mae = mean_absolute_error(y_test, v2_pred)
    
    print(f"\n📈 Performance Metrics:")
    print(f"- V13 MAE: ${mae:.2f}")
    print(f"- V2 MAE:  ${v2_mae:.2f}")
    print(f"- Improvement: {improvement:.1f}% over baseline")
    print(f"- Accuracy within 10%: {accuracy:.1f}%")
    
    # Show worst predictions
    test_df = X_test.copy()
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    test_df['error'] = np.abs(y_test - y_pred)
    
    print("\n❌ Top 5 Worst Predictions:")
    print(test_df.nlargest(5, 'error')[['days', 'miles', 'receipts', 'actual', 'predicted', 'error']].to_string())
    
    # Save model
    model_data = {
        'model': model,
        'mae': mae,
        'accuracy': accuracy
    }
    joblib.dump(model_data, 'reimbursement_model_v13.pkl')
    print("\n💾 Saved V13 model to 'reimbursement_model_v13.pkl'")

if __name__ == "__main__":
    main()
