#!/usr/bin/env python3
"""
Final Evaluation of V2 Model
"""

import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error

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

def prepare_features(df):
    """Prepare features for V2 model"""
    X = df[['days', 'miles', 'receipts']].copy()
    
    # Base features
    X['days_squared'] = X['days'] ** 2
    X['miles_squared'] = X['miles'] ** 2
    X['receipts_squared'] = X['receipts'] ** 2
    
    # Interaction terms
    X['days_miles'] = X['days'] * X['miles']
    X['days_receipts'] = X['days'] * X['receipts']
    X['miles_receipts'] = X['miles'] * X['receipts']
    
    # Expense categories
    X['low_expense'] = (X['receipts'] <= 50).astype(int)
    X['medium_expense'] = ((X['receipts'] > 50) & (X['receipts'] <= 500)).astype(int)
    X['high_expense'] = ((X['receipts'] > 500) & (X['receipts'] <= 1000)).astype(int)
    X['very_high_expense'] = (X['receipts'] > 1000).astype(int)
    
    # Per-day metrics
    X['miles_per_day'] = X['miles'] / (X['days'] + 1e-6)
    X['receipts_per_day'] = X['receipts'] / (X['days'] + 1e-6)
    
    # Trip duration categories
    X['short_trip'] = (X['days'] <= 2).astype(int)
    X['medium_trip'] = ((X['days'] > 2) & (X['days'] <= 5)).astype(int)
    X['long_trip'] = (X['days'] > 5).astype(int)
    
    # Distance categories
    X['short_distance'] = (X['miles'] <= 200).astype(int)
    X['long_distance'] = (X['miles'] > 500).astype(int)
    
    # Ensure all required features are present in the correct order
    expected_features = [
        'days', 'days_miles', 'days_receipts', 'days_squared', 'high_expense',
        'long_distance', 'long_trip', 'low_expense', 'medium_expense',
        'medium_trip', 'miles', 'miles_per_day', 'miles_receipts',
        'miles_squared', 'receipts', 'receipts_per_day', 'receipts_squared',
        'short_distance', 'short_trip', 'very_high_expense'
    ]
    
    # Add any missing features with zeros
    for feat in expected_features:
        if feat not in X.columns:
            X[feat] = 0
    
    # Return only the expected features in the correct order
    return X[expected_features]
    
    return X

def main():
    print("🔍 FINAL V2 EVALUATION")
    print("-" * 50)
    
    # Load data
    print("\n📊 Loading data...")
    df = load_data()
    X = prepare_features(df)
    y = df['reimbursement']
    
    # Load V2 model
    print("🚀 Loading V2 model...")
    model_data = joblib.load('reimbursement_model.pkl')
    model = model_data['model']
    
    # Make predictions
    print("📈 Making predictions...")
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
        pred = model.predict(prepare_features(pd.DataFrame([row])))[0]
        error = abs(pred - row['reimbursement'])
        print(f"{int(row['days']):4d} | {int(row['miles']):5d} | ${row['receipts']:7.2f} | ${row['reimbursement']:7.2f} | ${pred:7.2f} | ${error:5.2f}")
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
