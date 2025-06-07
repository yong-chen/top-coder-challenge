#!/usr/bin/env python3
"""
Build V6 Model - Simplified RandomForest with Key Features
Focuses on the most reliable patterns from our analysis
"""

import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def engineer_v6_features(df):
    """Create minimal, robust features"""
    print("📊 ENGINEERING V6 FEATURES")
    print("-" * 35)
    
    # Base features
    X = pd.DataFrame()
    X['days'] = df['days']
    X['miles'] = df['miles']
    X['receipts'] = df['receipts']
    
    # Simple interactions (from V2)
    X['days_sq'] = X['days'] ** 2
    X['miles_sq'] = X['miles'] ** 2
    X['receipts_sq'] = X['receipts'] ** 2
    X['days_miles'] = X['days'] * X['miles']
    X['days_receipts'] = X['days'] * X['receipts']
    X['miles_receipts'] = X['miles'] * X['receipts']
    
    # Simple business logic (no data leakage)
    X['is_long_trip'] = (X['days'] > 5).astype(int)
    X['is_long_distance'] = (X['miles'] > 500).astype(int)
    X['is_high_expense'] = (X['receipts'] > X['receipts'].quantile(0.8)).astype(int)
    
    print(f"✅ Created {X.shape[1]} features")
    return X

def train_v6_model(X, y):
    """Train and evaluate V6 model"""
    print("\n🎓 TRAINING V6 MODEL")
    print("-" * 35)
    
    # Simple RandomForest with good defaults
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,  # Shallower to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X, y, 
        cv=5, 
        scoring='neg_mean_absolute_error'
    )
    
    avg_mae = -np.mean(cv_scores)
    print(f"📊 CV MAE (V6): ${avg_mae:.2f}")
    
    # Train final model
    model.fit(X, y)
    print("✅ Model trained")
    
    # Feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 TOP FEATURES:")
    print(importances.head(10).to_string(index=False))
    
    return model, avg_mae

def main():
    # Load data
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
    df = pd.DataFrame(data)
    
    # Engineer features
    X = engineer_v6_features(df)
    y = df['reimbursement']
    
    # Train model
    model, mae_v6 = train_v6_model(X, y)
    
    # Save model
    model_data = {
        'model': model,
        'feature_cols': list(X.columns),
        'mae': mae_v6
    }
    joblib.dump(model_data, 'reimbursement_model_v6.pkl')
    
    print(f"\n🚀 V6 MODEL SAVED (MAE: ${mae_v6:.2f})")
    print(f"   Improvement over V2 ($28.52): ${28.52 - mae_v6:.2f}")
    print(f"   Improvement over V5 ($65.97): ${65.97 - mae_v6:.2f}")

if __name__ == "__main__":
    main()
