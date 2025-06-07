#!/usr/bin/env python3
"""
Fast Evaluation Script - V4 with refined 5-pattern expense caps
"""

import json
import pandas as pd
import joblib
import numpy as np
from calculate_reimbursement_v4 import create_features, apply_refined_caps

def main():
    print("🧾 Black Box Challenge - Fast Evaluation V4")
    print("=" * 50)
    
    # Load model once
    print("📁 Loading model...")
    try:
        model_data = joblib.load('reimbursement_model.pkl')
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test cases
    print("📁 Loading test cases...")
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    print(f"✅ Loaded {len(test_cases)} test cases")
    
    # Batch process all cases
    print("🧮 Processing all cases with V4 (ML + Refined 5-Pattern Caps)...")
    
    exact_matches = 0
    close_matches = 0
    total_error = 0.0
    successful_runs = 0
    errors = []
    
    for i, case in enumerate(test_cases):
        if i % 100 == 0:
            print(f"Progress: {i}/1000")
        
        try:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            # Create features and get ML prediction
            features_df = create_features(days, miles, receipts)
            X = features_df[feature_cols]
            ml_prediction = model.predict(X)[0]
            
            # Apply refined expense caps (V4 improvement)
            predicted = apply_refined_caps(ml_prediction, receipts, days, miles)
            predicted = round(predicted, 2)
            
            error = abs(predicted - expected)
            total_error += error
            successful_runs += 1
            
            # Track accuracy
            if error <= 0.01:
                exact_matches += 1
            elif error <= 1.00:
                close_matches += 1
            
            # Store large errors for analysis
            if error > 20:
                errors.append({
                    'days': days,
                    'miles': miles, 
                    'receipts': receipts,
                    'expected': expected,
                    'predicted': predicted,
                    'error': error
                })
                
        except Exception as e:
            print(f"   ❌ Error on case {i+1}: {e}")
    
    # Results
    print(f"\nProgress: {len(test_cases)}/1000")
    print("\n📊 EVALUATION RESULTS (V4)")
    print("=" * 35)
    
    if successful_runs > 0:
        avg_error = total_error / successful_runs
        exact_percent = (exact_matches / successful_runs) * 100
        close_percent = (close_matches / successful_runs) * 100
        score = avg_error + (1000 - exact_matches) * 0.1
        
        print(f"Successful runs:    {successful_runs} / {len(test_cases)}")
        print(f"Exact matches:      {exact_matches} ({exact_percent:.1f}%) [within ±$0.01]")
        print(f"Close matches:      {close_matches} ({close_percent:.1f}%) [within ±$1.00]")
        print(f"Average error:      ${avg_error:.2f}")
        print(f"Score:              {score:.2f} (lower is better)")
        
        # Show performance assessment
        if exact_percent > 95:
            print("\n🎉 Excellent! Your implementation is highly accurate!")
        elif exact_percent > 80:
            print("\n👍 Good work! Your implementation is quite accurate.")
        elif exact_percent > 50:
            print("\n🔧 Getting there! Consider refining your algorithm.")
        else:
            print("\n🚀 Keep working! There's room for improvement.")
            
        # Show worst errors
        if len(errors) > 0:
            print(f"\n🔍 Largest errors ({len(errors)} cases > $20):")
            sorted_errors = sorted(errors, key=lambda x: x['error'], reverse=True)
            for err in sorted_errors[:5]:
                print(f"   {err['days']:2d}d, {err['miles']:3d}mi, ${err['receipts']:6.2f} → "
                      f"Expected: ${err['expected']:7.2f}, Got: ${err['predicted']:7.2f}, "
                      f"Error: ${err['error']:6.2f}")
        else:
            print("\n🎉 No errors > $20!")
    else:
        print("❌ No successful runs! Please check your implementation.")

if __name__ == "__main__":
    main()
