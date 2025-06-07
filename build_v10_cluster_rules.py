#!/usr/bin/env python3
"""
V10 - Cluster-Based Rules with Receipt Tiers
Combines clustering insights with tiered receipt multipliers
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

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

def assign_clusters(df):
    """Assign trips to clusters based on features"""
    # Prepare features for clustering
    X = df[['days', 'miles', 'receipts']].copy()
    X['log_receipts'] = np.log1p(X['receipts'])
    X['log_miles'] = np.log1p(X['miles'])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[['days', 'log_miles', 'log_receipts']])
    
    # Fit KMeans (5 clusters from elbow analysis)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Map clusters to business meanings
    cluster_map = {
        0: 'long_high_mileage',
        1: 'medium_low_receipts',
        2: 'short_high_spend',
        3: 'long_local',
        4: 'quick_local'
    }
    
    return [cluster_map[c] for c in clusters]

def calculate_reimbursement_v10(days, miles, receipts):
    """
    Calculate reimbursement using cluster-based rules and receipt tiers
    
    Cluster Rules:
    1. long_high_mileage: Base + 20% premium
    2. medium_low_receipts: Base + receipt multiplier only
    3. short_high_spend: Base + 15% premium
    4. long_local: Base - 10% (extended local stay discount)
    5. quick_local: Flat $50/day + receipts (capped at 2x)
    
    Receipt Tiers:
    - < $50: 1.0x
    - $50-$200: 1.2x
    - $200-$500: 1.5x
    - > $500: 2.0x
    """
    # Base calculation
    daily_rate = 100
    mileage_rate = 0.50
    base = (days * daily_rate) + (miles * mileage_rate)
    
    # Determine receipt multiplier
    if receipts < 50:
        receipt_multiplier = 1.0
    elif receipts < 200:
        receipt_multiplier = 1.2
    elif receipts < 500:
        receipt_multiplier = 1.5
    else:
        receipt_multiplier = 2.0
    
    # Determine cluster
    # For prediction, we'll use a simplified rule-based cluster assignment
    if days > 7 and miles > 500:
        cluster = 'long_high_mileage'
    elif days <= 3 and receipts > 1000:
        cluster = 'short_high_spend'
    elif days > 7 and miles < 100:
        cluster = 'long_local'
    elif days <= 2 and miles < 200:
        cluster = 'quick_local'
    else:
        cluster = 'medium_low_receipts'
    
    # Apply cluster-specific rules
    if cluster == 'long_high_mileage':
        total = base * 1.20 * receipt_multiplier
    elif cluster == 'medium_low_receipts':
        total = base * receipt_multiplier
    elif cluster == 'short_high_spend':
        total = base * 1.15 * receipt_multiplier
    elif cluster == 'long_local':
        total = base * 0.90 * receipt_multiplier
    else:  # quick_local
        total = min((50 * days) + (receipts * 2), base * 2)
    
    # Ensure minimum reimbursement
    min_reimbursement = max(50 * days, 100)
    total = max(total, min_reimbursement)
    
    return round(total, 2)

def evaluate_model(predict_func, X, y):
    """Evaluate model using MAE and print examples"""
    y_pred = [predict_func(row['days'], row['miles'], row['receipts']) 
             for _, row in X.iterrows()]
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate accuracy within 10%
    accuracy = np.mean(np.abs((y - y_pred) / y) < 0.10) * 100
    
    return mae, accuracy, y_pred

def main():
    # Load data
    df = load_data()
    
    # Assign clusters for analysis (not used in prediction)
    df['cluster'] = assign_clusters(df)
    
    # Evaluate V10
    X = df[['days', 'miles', 'receipts']]
    y = df['reimbursement']
    
    mae, accuracy, y_pred = evaluate_model(calculate_reimbursement_v10, X, y)
    
    print("🚀 V10 CLUSTER-BASED RULES")
    print("-" * 35)
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Accuracy within 10%: {accuracy:.1f}%")
    print(f"Comparison to V2 ($28.52): ${28.52 - mae:+.2f}")
    
    # Show cluster distribution
    print("\n📊 Cluster Distribution:")
    print(df['cluster'].value_counts())
    
    # Show some examples
    print("\n📋 EXAMPLE PREDICTIONS:")
    print("Days | Miles | Receipts | Cluster           | Actual  | Predicted | Error")
    print("-" * 80)
    
    sample = df.sample(5, random_state=42)
    for _, row in sample.iterrows():
        pred = calculate_reimbursement_v10(row['days'], row['miles'], row['receipts'])
        error = abs(pred - row['reimbursement'])
        print(f"{int(row['days']):4d} | {int(row['miles']):5d} | ${row['receipts']:7.2f} | {row['cluster'][:15]:15s} | ${row['reimbursement']:7.2f} | ${pred:7.2f} | ${error:5.2f}")
    
    # Save model
    model_data = {
        'model_type': 'rule_based_v10',
        'mae': mae,
        'accuracy': accuracy
    }
    joblib.dump(model_data, 'reimbursement_model_v10.pkl')
    print("\n💾 Saved V10 model to 'reimbursement_model_v10.pkl'")

if __name__ == "__main__":
    main()
