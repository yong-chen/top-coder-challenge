#!/usr/bin/env python3
"""
Error Analysis Script - Analyze patterns in our algorithm's errors
"""

import json
import pandas as pd
import numpy as np
from calculate_reimbursement import calculate_reimbursement

def load_and_predict():
    """Load data and generate predictions"""
    print("🔍 Analyzing Algorithm Errors...")
    
    with open('public_cases.json', 'r') as f:
        raw_data = json.load(f)
    
    data = []
    for case in raw_data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculate_reimbursement(days, miles, receipts)
        error = abs(predicted - expected)
        
        data.append({
            'days': days,
            'miles': miles, 
            'receipts': receipts,
            'expected': expected,
            'predicted': predicted,
            'error': error,
            'error_pct': (error / expected) * 100 if expected > 0 else 0
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Analyzed {len(df)} cases")
    return df

def analyze_error_patterns(df):
    """Analyze patterns in errors"""
    print("\n📊 ERROR PATTERN ANALYSIS")
    print("=" * 50)
    
    # Overall performance
    exact_matches = len(df[df['error'] <= 0.01])
    close_matches = len(df[df['error'] <= 1.00])
    avg_error = df['error'].mean()
    
    print(f"Exact matches (±$0.01): {exact_matches}/1000 ({exact_matches/10:.1f}%)")
    print(f"Close matches (±$1.00): {close_matches}/1000 ({close_matches/10:.1f}%)")
    print(f"Average error: ${avg_error:.2f}")
    print()
    
    # Error by trip duration
    print("🗓️ Errors by Trip Duration:")
    day_errors = df.groupby('days')['error'].agg(['count', 'mean', 'std']).reset_index()
    for _, row in day_errors.iterrows():
        if row['count'] >= 5:
            print(f"   {int(row['days']):2d} days: ${row['mean']:6.2f} avg error (n={int(row['count']):2d})")
    
    # Find systematic bias by duration
    print("\n🎯 Systematic Bias Analysis:")
    bias_analysis = df.groupby('days').agg({
        'predicted': 'mean',
        'expected': 'mean', 
        'error': 'mean',
        'receipts': 'count'  # Use receipts for count instead of days
    }).reset_index()
    bias_analysis['bias'] = bias_analysis['predicted'] - bias_analysis['expected']
    
    print("   Duration | Our Avg | Expected Avg | Bias | Count")
    print("   ---------|---------|------------- |------|------")
    for _, row in bias_analysis.iterrows():
        if row['receipts'] >= 5:  # Only show rows with good sample size
            print(f"   {int(row['days']):2d} days  | ${row['predicted']:7.2f} | ${row['expected']:9.2f} | {row['bias']:+5.2f} | {int(row['receipts']):3d}")
    
    print()
    
    # Worst cases analysis
    print("🔴 Worst Error Cases:")
    worst_cases = df.nlargest(10, 'error')
    for _, case in worst_cases.head(5).iterrows():
        print(f"   {int(case['days']):2d}d, {int(case['miles']):3d}mi, ${case['receipts']:6.2f} → "
              f"Expected: ${case['expected']:7.2f}, Got: ${case['predicted']:7.2f}, Error: ${case['error']:6.2f}")
    
    print()
    
    # Error patterns by expense level
    print("💰 Errors by Expense Level:")
    df['expense_band'] = pd.cut(df['receipts'], 
                               bins=[0, 50, 200, 500, 1000, 5000],
                               labels=['$0-50', '$50-200', '$200-500', '$500-1000', '$1000+'])
    
    expense_errors = df.groupby('expense_band')['error'].agg(['count', 'mean']).reset_index()
    for _, row in expense_errors.iterrows():
        if pd.notna(row['expense_band']):
            print(f"   {row['expense_band']:>10}: ${row['mean']:6.2f} avg error (n={int(row['count']):2d})")
    
    return df

def identify_fixes(df):
    """Identify specific fixes needed"""
    print("\n🔧 RECOMMENDED FIXES")
    print("=" * 50)
    
    # Check 1-day trips specifically (saw errors in evaluation)
    one_day = df[df['days'] == 1]
    if len(one_day) > 0:
        avg_bias = (one_day['predicted'] - one_day['expected']).mean()
        print(f"1. One-day trips: Average bias = ${avg_bias:+.2f}")
        if abs(avg_bias) > 5:
            print(f"   🎯 FIX: Adjust 1-day base rate by ${-avg_bias:.2f}")
    
    # Check low-expense cases  
    low_expense = df[df['receipts'] < 50]
    if len(low_expense) > 0:
        avg_bias = (low_expense['predicted'] - low_expense['expected']).mean()
        print(f"2. Low-expense cases (<$50): Average bias = ${avg_bias:+.2f}")
        if abs(avg_bias) > 5:
            print(f"   🎯 FIX: Adjust low-expense rate")
    
    # Check medium trips (3-5 days) 
    medium_trips = df[(df['days'] >= 3) & (df['days'] <= 5)]
    if len(medium_trips) > 0:
        avg_bias = (medium_trips['predicted'] - medium_trips['expected']).mean()
        print(f"3. Medium trips (3-5 days): Average bias = ${avg_bias:+.2f}")
    
    # Check high-mileage cases
    high_miles = df[df['miles'] > 800]
    if len(high_miles) > 0:
        avg_bias = (high_miles['predicted'] - high_miles['expected']).mean()
        print(f"4. High-mileage trips (>800mi): Average bias = ${avg_bias:+.2f}")
        if abs(avg_bias) > 10:
            print(f"   🎯 FIX: Adjust high-mileage rate")

def main():
    df = load_and_predict()
    df = analyze_error_patterns(df)
    identify_fixes(df)
    
    print("\n🎯 Next steps: Apply fixes and re-evaluate")

if __name__ == "__main__":
    main()
