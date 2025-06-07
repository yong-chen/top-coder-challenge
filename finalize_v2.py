#!/usr/bin/env python3
"""
Finalize V2 Model for Submission
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def load_v2_model():
    """Load the V2 model and its feature processor"""
    print("🚀 Loading V2 model...")
    model_data = joblib.load('reimbursement_model.pkl')
    return model_data['model'], model_data['feature_cols']

def prepare_v2_features(df, feature_cols):
    """Prepare features using the same logic as V2"""
    # Create a copy to avoid modifying the original
    X = df[['days', 'miles', 'receipts']].copy()
    
    # Add all features used in V2
    X['days_squared'] = X['days'] ** 2
    X['miles_squared'] = X['miles'] ** 2
    X['receipts_squared'] = X['receipts'] ** 2
    X['days_miles'] = X['days'] * X['miles']
    X['days_receipts'] = X['days'] * X['receipts']
    X['miles_receipts'] = X['miles'] * X['receipts']
    
    # Add categorical features
    X['low_expense'] = (X['receipts'] <= 50).astype(int)
    X['medium_expense'] = ((X['receipts'] > 50) & (X['receipts'] <= 500)).astype(int)
    X['high_expense'] = ((X['receipts'] > 500) & (X['receipts'] <= 1000)).astype(int)
    X['very_high_expense'] = (X['receipts'] > 1000).astype(int)
    
    # Add per-day metrics
    X['miles_per_day'] = X['miles'] / (X['days'] + 1e-6)
    X['receipts_per_day'] = X['receipts'] / (X['days'] + 1e-6)
    
    # Add trip duration categories
    X['short_trip'] = (X['days'] <= 2).astype(int)
    X['medium_trip'] = ((X['days'] > 2) & (X['days'] <= 5)).astype(int)
    X['long_trip'] = (X['days'] > 5).astype(int)
    
    # Add distance categories
    X['short_distance'] = (X['miles'] <= 200).astype(int)
    X['long_distance'] = (X['miles'] > 500).astype(int)
    
    # Ensure all features are in the correct order
    return X[feature_cols]

def main():
    print("🔍 FINALIZING V2 MODEL")
    print("-" * 50)
    
    # Load the V2 model
    model, feature_cols = load_v2_model()
    print(f"✅ Loaded model with {len(feature_cols)} features")
    
    # Load test data
    print("\n📊 Loading test data...")
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Prepare test data
    data = []
    for case in test_cases:
        data.append({
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'actual': case['expected_output']
        })
    
    df = pd.DataFrame(data)
    
    # Prepare features
    X = prepare_v2_features(df, feature_cols)
    y = df['actual']
    
    # Make predictions
    print("\n📈 Making predictions...")
    y_pred = model.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    accuracy_10pct = np.mean(np.abs((y - y_pred) / y) < 0.10) * 100
    
    print(f"\n📊 Performance Metrics:")
    print(f"- Mean Absolute Error: ${mae:.2f}")
    print(f"- Accuracy within 10%: {accuracy_10pct:.1f}%")
    
    # Show some examples
    print("\n📋 Example Predictions:")
    print("Days | Miles | Receipts | Actual  | Predicted | Error")
    print("-" * 60)
    
    sample = df.sample(5, random_state=42)
    for _, row in sample.iterrows():
        X_sample = prepare_v2_features(pd.DataFrame([row]), feature_cols)
        pred = model.predict(X_sample)[0]
        error = abs(pred - row['actual'])
        print(f"{int(row['days']):4d} | {int(row['miles']):5d} | ${row['receipts']:7.2f} | ${row['actual']:7.2f} | ${pred:7.2f} | ${error:5.2f}")
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
