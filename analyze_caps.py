#!/usr/bin/env python3
"""
Analyze the actual expense cap patterns from the data
"""

import json
import pandas as pd
import numpy as np

def analyze_expense_patterns():
    """Analyze how expenses affect reimbursement"""
    
    print("🔍 ANALYZING EXPENSE CAP PATTERNS")
    print("=" * 50)
    
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
    
    # Focus on high-expense cases
    print("📊 High Expense Cases Analysis:")
    
    high_expense = df[df['receipts'] > 2000].sort_values('receipts')
    print(f"Cases with >$2000 receipts: {len(high_expense)}")
    
    if len(high_expense) > 0:
        print("\nHigh expense cases:")
        print("Days | Miles | Receipts | Reimbursement | Ratio")
        print("-" * 50)
        for _, row in high_expense.head(10).iterrows():
            ratio = row['reimbursement'] / row['receipts']
            print(f"{int(row['days']):2d}d  | {int(row['miles']):4d}mi | ${row['receipts']:7.2f} | ${row['reimbursement']:9.2f} | {ratio:.3f}")
    
    # Analyze expense ratio patterns
    print(f"\n💰 Expense Ratio Analysis:")
    df['expense_ratio'] = df['reimbursement'] / df['receipts']
    
    expense_bands = [
        (0, 500, "$0-500"),
        (500, 1000, "$500-1000"), 
        (1000, 1500, "$1000-1500"),
        (1500, 2000, "$1500-2000"),
        (2000, 2500, "$2000-2500"),
        (2500, 5000, "$2500+")
    ]
    
    print("Receipt Range | Count | Avg Ratio | Min Ratio | Max Ratio")
    print("-" * 55)
    
    for min_r, max_r, label in expense_bands:
        band_data = df[(df['receipts'] >= min_r) & (df['receipts'] < max_r)]
        if len(band_data) > 0:
            avg_ratio = band_data['expense_ratio'].mean()
            min_ratio = band_data['expense_ratio'].min()
            max_ratio = band_data['expense_ratio'].max()
            print(f"{label:12} | {len(band_data):5d} | {avg_ratio:8.3f} | {min_ratio:8.3f} | {max_ratio:8.3f}")
    
    # Look at specific problematic cases from V3
    print(f"\n🎯 V3 ERROR ANALYSIS:")
    problem_cases = [
        (1, 697, 2148.50, 1421.07),
        (1, 793, 2171.07, 1421.36),
        (1, 872, 2420.07, 1456.34),
        (5, 644, 2383.17, 1785.53),
        (1, 673, 2026.16, 1372.83)
    ]
    
    print("Days | Miles | Receipts | Expected | Actual Pattern")
    print("-" * 50)
    
    for days, miles, receipts, expected in problem_cases:
        # Find similar cases in data
        similar = df[
            (abs(df['days'] - days) <= 1) & 
            (abs(df['miles'] - miles) <= 100) & 
            (abs(df['receipts'] - receipts) <= 200)
        ]
        
        if len(similar) > 0:
            avg_reimb = similar['reimbursement'].mean()
            avg_ratio = (similar['reimbursement'] / similar['receipts']).mean()
            print(f"{days:2d}d  | {miles:4d}mi | ${receipts:7.2f} | ${expected:8.2f} | Avg: ${avg_reimb:.2f} (ratio: {avg_ratio:.3f})")
        else:
            ratio = expected / receipts
            print(f"{days:2d}d  | {miles:4d}mi | ${receipts:7.2f} | ${expected:8.2f} | No similar (ratio: {ratio:.3f})")

def main():
    analyze_expense_patterns()

if __name__ == "__main__":
    main()
