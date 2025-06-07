#!/usr/bin/env python3
"""
Generate Submission Files for TopCoder Challenge
"""

import json
import joblib
import numpy as np
import pandas as pd

def load_model():
    """Load the trained V2 model"""
    print("🚀 Loading V2 model...")
    model_data = joblib.load('reimbursement_model.pkl')
    return model_data['model'], model_data['feature_cols']

def prepare_features(df, feature_cols):
    """Prepare features for prediction"""
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

def generate_private_results():
    """Generate private_results.txt for submission"""
    # Load model
    model, feature_cols = load_model()
    
    # Load private test cases
    print("📂 Loading private test cases...")
    with open('private_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Prepare test data
    data = []
    for case in test_cases:
        data.append({
            'days': case['trip_duration_days'],
            'miles': case['miles_traveled'],
            'receipts': case['total_receipts_amount']
        })
    
    df = pd.DataFrame(data)
    
    # Prepare features and predict
    print("🔮 Making predictions...")
    X = prepare_features(df, feature_cols)
    predictions = model.predict(X)
    
    # Save to file (one prediction per line, rounded to 2 decimal places)
    with open('private_results.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred:.2f}\n")
    
    print(f"✅ Saved {len(predictions)} predictions to private_results.txt")

def main():
    print("🚀 TOPCODER CHALLENGE SUBMISSION")
    print("-" * 50)
    
    # Generate the required submission file
    generate_private_results()
    
    print("\n📋 NEXT STEPS:")
    print("1. Verify private_results.txt looks correct")
    print("2. Add arjun-krishna1 as collaborator to your GitHub repo")
    print("3. Submit via the TopCoder challenge portal")
    print("\n🎉 GOOD LUCK! 🍀")

if __name__ == "__main__":
    main()
