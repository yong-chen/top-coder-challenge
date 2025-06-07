#!/usr/bin/env python3
"""
V9 - Rule-Based Reimbursement Calculator
A transparent, business-rule driven approach
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def calculate_reimbursement_v9(days, miles, receipts):
    """
    Calculate reimbursement using business rules
    
    Rules:
    1. Base rate: $100/day + $0.50/mile
    2. Receipt multipliers:
       - < $50: 1.0x
       - $50-$200: 1.2x
       - $200-$500: 1.5x
       - > $500: 2.0x
    3. Trip length adjustments:
       - 1 day: +10%
       - 7+ days: -10%
    4. Distance surcharge:
       - 500-1000 miles: +15%
       - >1000 miles: +25%
    """
    # Base components
    daily_rate = 100
    mileage_rate = 0.50
    
    # Base calculation
    base = (days * daily_rate) + (miles * mileage_rate)
    
    # Receipt multiplier
    if receipts < 50:
        receipt_multiplier = 1.0
    elif receipts < 200:
        receipt_multiplier = 1.2
    elif receipts < 500:
        receipt_multiplier = 1.5
    else:
        receipt_multiplier = 2.0
    
    # Apply receipt multiplier
    total = base * receipt_multiplier
    
    # Trip length adjustments
    if days == 1:
        total *= 1.10  # Day trips are more expensive
    elif days >= 7:
        total *= 0.90  # Weekly discount
    
    # Distance surcharge
    if miles > 1000:
        total *= 1.25
    elif miles > 500:
        total *= 1.15
    
    return round(total, 2)

def evaluate_model(predict_func, X, y):
    """Evaluate model using MAE"""
    y_pred = [predict_func(row['days'], row['miles'], row['receipts']) 
              for _, row in X.iterrows()]
    mae = mean_absolute_error(y, y_pred)
    return mae, y_pred

def main():
    # Load data
    with open('public_cases.json', 'r') as f:
        raw_data = json.load(f)
    
    # Prepare data
    data = []
    for case in raw_data:
        data.append({
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        })
    df = pd.DataFrame(data)
    
    # Evaluate V9
    X = df[['days', 'miles', 'receipts']]
    y = df['reimbursement']
    
    mae, y_pred = evaluate_model(calculate_reimbursement_v9, X, y)
    
    print("🚀 V9 RULE-BASED MODEL")
    print("-" * 35)
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Comparison to V2 ($28.52): ${28.52 - mae:+.2f}")
    
    # Show some examples
    print("\n📋 EXAMPLE PREDICTIONS:")
    print("Days | Miles | Receipts | Actual | Predicted | Error")
    print("-" * 60)
    
    for i in range(5):
        row = df.sample(1).iloc[0]
        pred = calculate_reimbursement_v9(row['days'], row['miles'], row['receipts'])
        error = abs(pred - row['reimbursement'])
        print(f"{int(row['days']):4d} | {int(row['miles']):5d} | ${row['receipts']:7.2f} | ${row['reimbursement']:7.2f} | ${pred:7.2f} | ${error:5.2f}")

if __name__ == "__main__":
    main()
