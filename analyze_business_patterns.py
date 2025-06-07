#!/usr/bin/env python3
"""
Business Pattern Analysis - Travel Expense Clustering & Temporal Trends
Looking for real corporate travel patterns that would exist in a 60-year system
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load data and create business-relevant features"""
    print("🏢 BUSINESS PATTERN ANALYSIS")
    print("=" * 50)
    
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
    
    # Create business-relevant features
    df['daily_expense'] = df['receipts'] / df['days']
    df['miles_per_day'] = df['miles'] / df['days']
    df['expense_per_mile'] = df['receipts'] / (df['miles'] + 1)  # Avoid div by 0
    df['reimbursement_rate'] = df['reimbursement'] / df['receipts']
    df['trip_efficiency'] = df['reimbursement'] / (df['days'] * df['miles'] + 1)
    
    # Travel type indicators (business logic)
    df['is_local'] = (df['miles'] < 100) & (df['days'] <= 2)
    df['is_long_haul'] = df['miles'] > 500
    df['is_extended'] = df['days'] > 7
    df['is_high_expense'] = df['receipts'] > df['receipts'].quantile(0.8)
    
    return df

def analyze_travel_patterns(df):
    """Identify natural travel patterns using clustering"""
    print("\n📊 TRAVEL PATTERN CLUSTERING")
    print("-" * 30)
    
    # Features for clustering (business-relevant dimensions)
    cluster_features = [
        'days', 'miles', 'daily_expense', 'miles_per_day', 
        'expense_per_mile', 'reimbursement_rate'
    ]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features])
    
    # Find optimal number of clusters
    inertias = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Use elbow method to suggest optimal k
    diffs = np.diff(inertias)
    diff_ratios = diffs[:-1] / diffs[1:]
    optimal_k = np.argmax(diff_ratios) + 3  # +3 because of indexing
    
    print(f"Optimal number of travel patterns: {optimal_k}")
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['travel_pattern'] = kmeans.fit_predict(X_scaled)
    
    # Analyze each pattern
    print("\n🎯 TRAVEL PATTERN PROFILES:")
    print("Pattern | Count | Avg Days | Avg Miles | Daily $ | Reimb Rate | Description")
    print("-" * 80)
    
    patterns = []
    for pattern in range(optimal_k):
        group = df[df['travel_pattern'] == pattern]
        
        avg_days = group['days'].mean()
        avg_miles = group['miles'].mean()
        avg_daily_exp = group['daily_expense'].mean()
        avg_reimb_rate = group['reimbursement_rate'].mean()
        
        # Business interpretation
        if avg_miles < 200 and avg_days <= 3:
            desc = "Local/Regional"
        elif avg_miles > 800 and avg_days >= 5:
            desc = "Long-distance"
        elif avg_daily_exp > 200:
            desc = "High-expense"
        elif avg_reimb_rate > 2.0:
            desc = "High-reimbursement"
        else:
            desc = "Standard"
        
        print(f"   {pattern}    | {len(group):5d} | {avg_days:7.1f} | {avg_miles:8.0f} | {avg_daily_exp:6.0f} | {avg_reimb_rate:9.2f} | {desc}")
        
        patterns.append({
            'pattern': pattern,
            'count': len(group),
            'avg_days': avg_days,
            'avg_miles': avg_miles,
            'avg_daily_expense': avg_daily_exp,
            'avg_reimb_rate': avg_reimb_rate,
            'description': desc
        })
    
    return df, patterns

def analyze_temporal_trends(df):
    """Look for time-based patterns (inflation, policy changes)"""
    print(f"\n⏰ TEMPORAL TREND ANALYSIS")
    print("-" * 30)
    
    # We don't have explicit dates, but we can look for systematic trends
    # that might indicate chronological ordering
    
    # Sort by reimbursement rate (proxy for policy era)
    df_sorted = df.sort_values('reimbursement_rate')
    df_sorted['case_order'] = range(len(df_sorted))
    
    # Check for trends across the sorted cases
    window_size = 100  # Rolling window
    
    trends = []
    for i in range(0, len(df_sorted) - window_size, window_size):
        window = df_sorted.iloc[i:i+window_size]
        
        trends.append({
            'window': i // window_size,
            'avg_reimb_rate': window['reimbursement_rate'].mean(),
            'avg_daily_expense': window['daily_expense'].mean(),
            'avg_miles_per_day': window['miles_per_day'].mean(),
            'case_range': f"{i}-{i+window_size}"
        })
    
    trend_df = pd.DataFrame(trends)
    
    print("Window | Cases | Avg Reimb Rate | Avg Daily $ | Avg Miles/Day")
    print("-" * 60)
    for _, row in trend_df.iterrows():
        print(f"  {row['window']:2d}   | {row['case_range']:8s} | {row['avg_reimb_rate']:11.3f} | {row['avg_daily_expense']:8.0f} | {row['avg_miles_per_day']:10.1f}")
    
    # Look for significant trends
    reimb_trend = stats.linregress(trend_df.index, trend_df['avg_reimb_rate'])
    expense_trend = stats.linregress(trend_df.index, trend_df['avg_daily_expense'])
    
    print(f"\n📈 TREND ANALYSIS:")
    print(f"Reimbursement rate trend: slope={reimb_trend.slope:.4f}, p-value={reimb_trend.pvalue:.4f}")
    print(f"Daily expense trend: slope={expense_trend.slope:.4f}, p-value={expense_trend.pvalue:.4f}")
    
    if reimb_trend.pvalue < 0.05:
        print(f"✅ Significant reimbursement rate trend detected!")
    if expense_trend.pvalue < 0.05:
        print(f"✅ Significant expense inflation trend detected!")
    
    return trend_df

def analyze_expense_regimes(df):
    """Identify different expense policy regimes"""
    print(f"\n💼 EXPENSE POLICY REGIME ANALYSIS")
    print("-" * 35)
    
    # Group by reimbursement rate ranges (likely policy tiers)
    df['reimb_tier'] = pd.qcut(df['reimbursement_rate'], 
                               q=5, 
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    regime_analysis = df.groupby('reimb_tier').agg({
        'days': ['count', 'mean'],
        'miles': 'mean',
        'receipts': 'mean',
        'reimbursement': 'mean',
        'daily_expense': 'mean'
    }).round(2)
    
    print("Reimbursement Regime Analysis:")
    print(regime_analysis)
    
    # Look for policy breakpoints
    print(f"\n🎯 POLICY INSIGHTS:")
    for tier in df['reimb_tier'].unique():
        if pd.isna(tier):
            continue
            
        group = df[df['reimb_tier'] == tier]
        avg_rate = group['reimbursement_rate'].mean()
        
        if tier == 'Very Low':
            print(f"• {tier} reimbursement ({avg_rate:.2f}x): Likely strict policy or specific trip types")
        elif tier == 'Very High':
            print(f"• {tier} reimbursement ({avg_rate:.2f}x): Likely special circumstances or executive travel")
        else:
            print(f"• {tier} reimbursement ({avg_rate:.2f}x): Standard business travel")

def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Analyze travel patterns
    df, patterns = analyze_travel_patterns(df)
    
    # Look for temporal trends
    trend_df = analyze_temporal_trends(df)
    
    # Analyze expense regimes
    analyze_expense_regimes(df)
    
    print(f"\n🚀 RECOMMENDATIONS FOR MODEL IMPROVEMENT:")
    print("1. Use travel pattern clusters as features")
    print("2. Include reimbursement tier as a categorical feature")
    print("3. Consider separate models for different patterns")
    print("4. Add business logic features (local/long-haul indicators)")

if __name__ == "__main__":
    main()
