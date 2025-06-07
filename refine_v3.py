#!/usr/bin/env python3
"""
Refine V3 Algorithm - Analyze remaining errors and improve expense caps
"""

import json
import pandas as pd
import numpy as np
import joblib
from calculate_reimbursement_v3 import create_features

def analyze_v3_errors():
    """Analyze where V3 is still failing"""
    print("🔍 ANALYZING V3 REMAINING ERRORS")
    print("=" * 50)
    
    # Load model and data
    model_data = joblib.load('reimbursement_model.pkl')
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Analyze the worst V3 errors
    worst_cases = [
        (8, 795, 1645.99, 644.69),
        (4, 69, 2321.49, 322.00),
        (11, 740, 1171.99, 902.09),
        (8, 482, 1411.49, 631.81),
        (5, 516, 1878.49, 669.85)
    ]
    
    print("🎯 WORST V3 ERROR ANALYSIS:")
    print("Days | Miles | Receipts | Expected | Ratio | Pattern")
    print("-" * 60)
    
    error_patterns = []
    
    for days, miles, receipts, expected in worst_cases:
        ratio = expected / receipts
        
        # Find similar cases
        similar_cases = []
        for case in test_cases:
            c_days = case['input']['trip_duration_days']
            c_miles = case['input']['miles_traveled']
            c_receipts = case['input']['total_receipts_amount']
            c_expected = case['expected_output']
            
            # Look for similar patterns (±1 day, ±100 miles, ±200 receipts)
            if (abs(c_days - days) <= 1 and 
                abs(c_miles - miles) <= 100 and 
                abs(c_receipts - receipts) <= 200):
                similar_cases.append((c_days, c_miles, c_receipts, c_expected, c_expected/c_receipts))
        
        if similar_cases:
            avg_ratio = np.mean([s[4] for s in similar_cases])
            pattern = f"Similar n={len(similar_cases)}, avg_ratio={avg_ratio:.3f}"
        else:
            pattern = "No similar cases"
        
        print(f"{days:2d}d  | {miles:4d}mi | ${receipts:7.2f} | ${expected:8.2f} | {ratio:.3f} | {pattern}")
        
        error_patterns.append({
            'days': days,
            'miles': miles, 
            'receipts': receipts,
            'expected': expected,
            'ratio': ratio,
            'similar_count': len(similar_cases) if similar_cases else 0
        })
    
    print()
    
    # Look for systematic patterns
    print("🔍 SYSTEMATIC ERROR PATTERNS:")
    
    # Low reimbursement despite high expenses
    low_ratios = [p for p in error_patterns if p['ratio'] < 0.4]
    if low_ratios:
        print(f"⚠️  Very low ratio cases ({len(low_ratios)}): Receipts >$1400 but reimb <40%")
        print("   → Need more aggressive caps for these specific patterns")
    
    # High mileage + high expenses
    high_both = [p for p in error_patterns if p['miles'] > 400 and p['receipts'] > 1400]
    if high_both:
        print(f"⚠️  High mileage + expenses ({len(high_both)}): Complex trips with severe caps")
        print("   → Need special handling for high-complexity, high-expense cases")
    
    # Short trips with very high expenses
    short_expensive = [p for p in error_patterns if p['days'] <= 5 and p['receipts'] > 2000]
    if short_expensive:
        print(f"⚠️  Short + expensive trips ({len(short_expensive)}): Likely anomalous")
        print("   → Need different cap logic for short high-expense trips")
    
    return error_patterns

def create_refined_caps():
    """Create more nuanced expense cap logic"""
    print("\n🔨 CREATING REFINED CAP LOGIC")
    print("=" * 50)
    
    cap_logic = """
def apply_refined_caps(ml_prediction, receipts, days, miles):
    '''
    Refined expense cap logic based on V3 error analysis
    '''
    
    # Calculate trip complexity score
    complexity_score = (miles / days) if days > 0 else 0
    expense_per_day = receipts / days if days > 0 else receipts
    
    # Pattern 1: Short trips with very high expenses - severe caps
    if days <= 5 and receipts > 2000:
        # These are likely anomalous - cap very aggressively
        base_allowance = days * 60 + miles * 0.25
        excess_allowance = (receipts - 2000) * 0.1  # Only 10% of excess
        return min(ml_prediction, base_allowance + excess_allowance)
    
    # Pattern 2: High complexity + high expense trips  
    elif complexity_score > 100 and receipts > 1400:
        # Complex routes with high expenses - moderate but firm caps
        base_trip_cost = days * 70 + miles * 0.35
        receipt_portion = min(receipts * 0.6, 800)  # Cap receipt portion at $800
        return min(ml_prediction, base_trip_cost + receipt_portion)
    
    # Pattern 3: Very high expenses (>$2500)
    elif receipts > 2500:
        base_trip_cost = days * 80 + miles * 0.4
        receipt_portion = receipts * 0.4  # More aggressive than before
        return min(ml_prediction, base_trip_cost + receipt_portion)
    
    # Pattern 4: High expenses ($2000-2500)
    elif receipts > 2000:
        base_trip_cost = days * 80 + miles * 0.4
        receipt_portion = receipts * 0.65  # Slightly more aggressive
        return min(ml_prediction, base_trip_cost + receipt_portion)
    
    # Pattern 5: Medium-high expenses ($1500-2000)
    elif receipts > 1500:
        # Light smoothing
        return ml_prediction * 0.95
    
    else:
        # Low/medium expenses - no caps
        return ml_prediction
    """
    
    print("Created refined cap logic with 5 patterns:")
    print("1. Short + very expensive trips: Severe caps")  
    print("2. High complexity + expensive: Moderate caps")
    print("3. Very high expenses (>$2500): Aggressive caps")
    print("4. High expenses ($2000-2500): Moderate caps") 
    print("5. Medium expenses ($1500-2000): Light smoothing")
    
    return cap_logic

def main():
    error_patterns = analyze_v3_errors()
    refined_logic = create_refined_caps()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Implement refined cap logic in calculate_reimbursement_v4.py")
    print("2. Test on worst error cases")
    print("3. Run full evaluation")
    print("4. Compare with V2/V3 performance")

if __name__ == "__main__":
    main()
