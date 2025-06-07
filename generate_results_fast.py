#!/usr/bin/env python3
"""
Fast Private Results Generation - Load model once, process all cases in batch
"""

import json
import pandas as pd
import joblib
import numpy as np
from calculate_reimbursement_v2 import create_features

def main():
    print("Black Box Challenge - Fast Private Results Generation")
    print("=" * 55)
    
    # Load model once
    print("Loading ML model...")
    try:
        model_data = joblib.load('reimbursement_model.pkl')
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load private test cases
    print("Loading private test cases...")
    try:
        with open('private_cases.json', 'r') as f:
            test_cases = json.load(f)
        print(f"✅ Loaded {len(test_cases)} private test cases")
    except Exception as e:
        print(f"❌ Error loading private_cases.json: {e}")
        return
    
    # Batch process all cases
    print("Processing all cases in batch...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, case in enumerate(test_cases):
        if i % 500 == 0:
            print(f"Progress: {i}/{len(test_cases)}")
        
        try:
            # Private cases have direct structure (no 'input' wrapper)
            days = case['trip_duration_days']
            miles = case['miles_traveled']
            receipts = case['total_receipts_amount']
            
            # Create features and predict
            features_df = create_features(days, miles, receipts)
            X = features_df[feature_cols]
            predicted = model.predict(X)[0]
            predicted = round(predicted, 2)
            
            results.append(str(predicted))
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Error on case {i+1}: {e}")
            results.append("0.00")
            failed += 1
    
    print(f"Progress: {len(test_cases)}/{len(test_cases)}")
    
    # Write results to file
    print("\nWriting results to private_results.txt...")
    with open('private_results.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    print("✅ Results written to private_results.txt")
    print("\nSummary:")
    print(f"   Total cases: {len(test_cases)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All test cases processed successfully!")
        print("📋 Next steps:")
        print("   1. Add arjun-krishna1 as collaborator to your GitHub repo")
        print("   2. Submit private_results.txt via the submission form")
    else:
        print(f"\n⚠️  {failed} test cases failed. Review implementation.")

if __name__ == "__main__":
    main()
