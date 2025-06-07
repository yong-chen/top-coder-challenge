#!/usr/bin/env python3
"""
Black Box Challenge - Data Analysis Script
Deep statistical analysis of 1,000 historical reimbursement cases
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the test cases from JSON file"""
    print("🔍 Loading historical reimbursement data...")
    
    with open('public_cases.json', 'r') as f:
        raw_data = json.load(f)
    
    # Convert to pandas DataFrame for analysis
    data = []
    for case in raw_data:
        data.append({
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'], 
            'receipts': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} cases")
    print(f"   Days range: {int(df['days'].min())}-{int(df['days'].max())}")
    print(f"   Miles range: {int(df['miles'].min())}-{int(df['miles'].max())}")
    print(f"   Receipts range: ${df['receipts'].min():.2f}-${df['receipts'].max():.2f}")
    print(f"   Reimbursement range: ${df['reimbursement'].min():.2f}-${df['reimbursement'].max():.2f}")
    print()
    
    return df

def basic_statistics(df):
    """Calculate basic statistics and correlations"""
    print("📊 BASIC STATISTICAL ANALYSIS")
    print("=" * 50)
    
    # Basic stats
    print("Summary Statistics:")
    print(df.describe())
    print()
    
    # Correlation analysis
    print("🔗 Correlation Analysis:")
    corr_matrix = df.corr()
    print(corr_matrix)
    print()
    
    # Individual correlations with reimbursement
    for col in ['days', 'miles', 'receipts']:
        r, p = pearsonr(df[col], df['reimbursement'])
        print(f"   {col.title()} correlation: r={r:.4f}, p={p:.2e}")
    
    print()
    return corr_matrix

def analyze_patterns(df):
    """Look for non-linear patterns and thresholds"""
    print("🔍 PATTERN ANALYSIS")
    print("=" * 50)
    
    # Days analysis - looking for "sweet spots"
    print("📅 Days Analysis:")
    days_stats = df.groupby('days')['reimbursement'].agg(['count', 'mean', 'std']).reset_index()
    days_stats = days_stats[days_stats['count'] >= 5]  # Only days with sufficient data
    
    print("Reimbursement by Trip Duration:")
    for _, row in days_stats.head(10).iterrows():
        print(f"   {int(row['days']):2d} days: ${row['mean']:7.2f} avg (n={int(row['count']):2d})")
    
    # Look for the Marcus "5-6 day sweet spot"
    if len(days_stats) > 0:
        best_days = int(days_stats.loc[days_stats['mean'].idxmax(), 'days'])
        print(f"   🎯 Best performing duration: {best_days} days")
    
    print()
    
    # Miles analysis - looking for distance thresholds
    print("🚗 Miles Analysis:")
    # Create mileage bands
    df['mile_band'] = pd.cut(df['miles'], bins=[0, 100, 200, 300, 500, 800, 2000], 
                            labels=['0-100', '100-200', '200-300', '300-500', '500-800', '800+'])
    
    mile_stats = df.groupby('mile_band')['reimbursement'].agg(['count', 'mean', 'std']).reset_index()
    print("Reimbursement by Distance Band:")
    for _, row in mile_stats.iterrows():
        if pd.notna(row['mile_band']):
            print(f"   {row['mile_band']:>8} miles: ${row['mean']:7.2f} avg (n={int(row['count']):2d})")
    
    print()
    
    # Receipts analysis - looking for caps/penalties
    print("🧾 Receipts Analysis:")
    # Create receipt bands
    df['receipt_band'] = pd.cut(df['receipts'], 
                               bins=[0, 50, 100, 200, 500, 1000, 5000], 
                               labels=['$0-50', '$50-100', '$100-200', '$200-500', '$500-1000', '$1000+'])
    
    receipt_stats = df.groupby('receipt_band')['reimbursement'].agg(['count', 'mean', 'std']).reset_index()
    print("Reimbursement by Receipt Amount:")
    for _, row in receipt_stats.iterrows():
        if pd.notna(row['receipt_band']):
            print(f"   {row['receipt_band']:>10}: ${row['mean']:7.2f} avg (n={int(row['count']):2d})")
    
    print()

def test_employee_theories(df):
    """Test specific theories from employee interviews"""
    print("🕵️ TESTING EMPLOYEE THEORIES")
    print("=" * 50)
    
    # Marcus's theory: "efficiency bonus" for high miles/day ratio
    print("Theory 1: Efficiency Bonus (Marcus)")
    df['efficiency'] = df['miles'] / df['days']
    
    # Look at high efficiency trips
    high_eff = df[df['efficiency'] > df['efficiency'].quantile(0.8)]
    low_eff = df[df['efficiency'] < df['efficiency'].quantile(0.2)]
    
    print(f"   High efficiency (top 20%): ${high_eff['reimbursement'].mean():.2f} avg")
    print(f"   Low efficiency (bottom 20%): ${low_eff['reimbursement'].mean():.2f} avg")
    
    # Statistical test
    t_stat, p_val = stats.ttest_ind(high_eff['reimbursement'], low_eff['reimbursement'])
    print(f"   T-test p-value: {p_val:.4f} {'✅ Significant!' if p_val < 0.05 else '❌ Not significant'}")
    print()
    
    # Lisa's theory: Complex interactions
    print("Theory 2: Complex Interactions (Lisa)")
    # Try to find cases where simple linear combination doesn't work
    
    # Simple linear model baseline
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X_simple = df[['days', 'miles', 'receipts']]
    y = df['reimbursement']
    
    model_simple = LinearRegression().fit(X_simple, y)
    pred_simple = model_simple.predict(X_simple)
    r2_simple = r2_score(y, pred_simple)
    
    print(f"   Simple linear model R²: {r2_simple:.4f}")
    
    # Add interaction terms
    df['days_miles'] = df['days'] * df['miles']
    df['days_receipts'] = df['days'] * df['receipts']  
    df['miles_receipts'] = df['miles'] * df['receipts']
    
    X_interact = df[['days', 'miles', 'receipts', 'days_miles', 'days_receipts', 'miles_receipts']]
    model_interact = LinearRegression().fit(X_interact, y)
    pred_interact = model_interact.predict(X_interact)
    r2_interact = r2_score(y, pred_interact)
    
    print(f"   With interactions R²: {r2_interact:.4f}")
    print(f"   Improvement: {r2_interact - r2_simple:.4f} {'✅ Better!' if r2_interact > r2_simple + 0.01 else '❌ Minimal'}")
    print()
    
    # Dave's theory: Route complexity (high miles for duration)
    print("Theory 3: Route Complexity Bonus (Dave)")
    
    # For each duration, find cases with unusually high mileage
    complex_cases = []
    for days in df['days'].unique():
        day_data = df[df['days'] == days]
        if len(day_data) >= 5:  # Need enough data
            high_mile_threshold = day_data['miles'].quantile(0.8)
            complex_trips = day_data[day_data['miles'] > high_mile_threshold]
            normal_trips = day_data[day_data['miles'] <= day_data['miles'].quantile(0.5)]
            
            if len(complex_trips) > 0 and len(normal_trips) > 0:
                complex_avg = complex_trips['reimbursement'].mean()
                normal_avg = normal_trips['reimbursement'].mean()
                print(f"   {days:2d} days - Complex routes: ${complex_avg:6.2f}, Normal routes: ${normal_avg:6.2f}")
    
    print()

def find_formula_components(df):
    """Try to reverse engineer the formula components"""
    print("🧮 REVERSE ENGINEERING FORMULA COMPONENTS")
    print("=" * 50)
    
    # Try to find base per-day rate
    print("🗓️ Per-Day Component Analysis:")
    
    # Look at trips with minimal receipts and miles to isolate per-day rate
    minimal_cases = df[(df['receipts'] < 10) & (df['miles'] < 50)]
    if len(minimal_cases) > 0:
        print(f"   Minimal expense cases (n={len(minimal_cases)}):")
        for _, case in minimal_cases.head(5).iterrows():
            per_day = case['reimbursement'] / case['days']
            print(f"     {int(case['days']):2d}d, {int(case['miles']):2d}mi, ${case['receipts']:.2f} → ${case['reimbursement']:.2f} (${per_day:.2f}/day)")
        
        avg_per_day = (minimal_cases['reimbursement'] / minimal_cases['days']).mean()
        print(f"   🎯 Estimated base per-day rate: ${avg_per_day:.2f}")
    else:
        print("   No minimal expense cases found")
    
    print()
    
    # Try to find mileage rate  
    print("🚗 Mileage Component Analysis:")
    
    # Look at cases with similar days/receipts, varying miles
    for days in [1, 2, 3]:
        day_cases = df[(df['days'] == days) & (df['receipts'] < 100)]
        if len(day_cases) >= 5:
            # Simple correlation between miles and reimbursement for this duration
            r, p = pearsonr(day_cases['miles'], day_cases['reimbursement'])
            print(f"   {days} day trips: miles correlation r={r:.3f}, p={p:.3f}")
            
            # Try to estimate rate per mile
            if len(day_cases) >= 3:
                high_mile = day_cases[day_cases['miles'] > day_cases['miles'].quantile(0.8)]
                low_mile = day_cases[day_cases['miles'] < day_cases['miles'].quantile(0.2)]
                
                if len(high_mile) > 0 and len(low_mile) > 0:
                    mile_diff = high_mile['miles'].mean() - low_mile['miles'].mean()
                    reimb_diff = high_mile['reimbursement'].mean() - low_mile['reimbursement'].mean()
                    if mile_diff > 0:
                        est_rate = reimb_diff / mile_diff
                        print(f"     Estimated rate: ${est_rate:.3f} per mile")
    
    print()
    
    # Receipt analysis
    print("🧾 Receipt Component Analysis:")
    
    # Look for receipt reimbursement patterns
    for receipt_range in [(0, 50), (50, 150), (150, 300)]:
        range_cases = df[(df['receipts'] >= receipt_range[0]) & 
                        (df['receipts'] < receipt_range[1]) &
                        (df['days'] <= 3) & (df['miles'] < 200)]
        
        if len(range_cases) >= 3:
            r, p = pearsonr(range_cases['receipts'], range_cases['reimbursement'])
            print(f"   ${receipt_range[0]}-{receipt_range[1]} range: correlation r={r:.3f}")

def analyze_outliers(df):
    """Find and analyze outlier cases"""
    print("🎯 OUTLIER ANALYSIS")
    print("=" * 50)
    
    # Calculate expected vs actual based on simple assumptions
    # Rough estimate: $100/day + $0.50/mile + receipts
    df['simple_estimate'] = df['days'] * 100 + df['miles'] * 0.50 + df['receipts']
    df['residual'] = df['reimbursement'] - df['simple_estimate']
    
    # Find biggest positive outliers (system pays more than expected)
    print("🔺 Biggest Positive Outliers (system pays MORE than simple estimate):")
    positive_outliers = df.nlargest(5, 'residual')
    for _, case in positive_outliers.iterrows():
        print(f"   {int(case['days']):2d}d, {int(case['miles']):2d}mi, ${case['receipts']:.2f} → "
              f"${case['reimbursement']:.2f} vs ${case['simple_estimate']:.2f} "
              f"(+${case['residual']:.2f})")
    
    print()
    
    # Find biggest negative outliers (system pays less than expected)  
    print("🔻 Biggest Negative Outliers (system pays LESS than simple estimate):")
    negative_outliers = df.nsmallest(5, 'residual')
    for _, case in negative_outliers.iterrows():
        print(f"   {int(case['days']):2d}d, {int(case['miles']):2d}mi, ${case['receipts']:.2f} → "
              f"${case['reimbursement']:.2f} vs ${case['simple_estimate']:.2f} "
              f"({case['residual']:.2f})")
    
    print()

def main():
    """Main analysis function"""
    print("🧾 BLACK BOX REIMBURSEMENT SYSTEM - DATA ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    df = load_data()
    
    # Basic statistics
    corr_matrix = basic_statistics(df)
    
    # Pattern analysis
    analyze_patterns(df)
    
    # Test employee theories
    test_employee_theories(df)
    
    # Try to find formula components
    find_formula_components(df)
    
    # Analyze outliers
    analyze_outliers(df)
    
    print("🎯 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Key findings will be used to build the reimbursement algorithm.")
    print()

if __name__ == "__main__":
    main()
