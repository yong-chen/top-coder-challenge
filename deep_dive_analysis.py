#!/usr/bin/env python3
"""
Deep Dive Analysis - Uncovering Hidden Patterns
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set up visualization without requiring display
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Configure matplotlib to not try to show plots
plt.ioff()  # Turn off interactive mode

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

def analyze_distributions(df):
    """Analyze distributions of key variables"""
    print("📊 DISTRIBUTION ANALYSIS")
    print("-" * 35)
    
    # Basic stats
    print("\n📈 Basic Statistics:")
    print(df.describe().round(2))
    
    # Check for multimodality using skewness and kurtosis
    print("\n🔍 Distribution Shape Analysis:")
    print(f"{'Variable':15s} {'Skewness':>10s} {'Kurtosis':>10s}  Interpretation")
    print("-" * 60)
    
    for col in ['days', 'miles', 'receipts', 'reimbursement']:
        skewness = stats.skew(df[col])
        kurt = stats.kurtosis(df[col])
        
        # Simple interpretation
        if abs(skewness) > 1:
            shape = "Highly skewed"
        elif abs(skewness) > 0.5:
            shape = "Moderately skewed"
        else:
            shape = "Roughly symmetric"
            
        if kurt > 3:
            shape += " (Leptokurtic - heavy-tailed)"
        elif kurt < 0:
            shape += " (Platykurtic - light-tailed)"
            
        print(f"{col:15s} {skewness:10.2f} {kurt:10.2f}  {shape}")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of Key Variables', fontsize=16)
    
    # Ensure we have a 2D array of axes
    if len(axes.shape) == 1:
        axes = axes.reshape(-1, 2)
    
    for i, col in enumerate(['days', 'miles', 'receipts', 'reimbursement']):
        ax = axes[i//2, i%2]
        sns.histplot(df[col], kde=True, ax=ax, bins=30)
        ax.set_title(f'Distribution of {col}')
        
        # Add vertical lines for percentiles
        for p in [25, 50, 75, 95]:
            val = np.percentile(df[col], p)
            ax.axvline(val, color='r', linestyle='--', alpha=0.7)
            ax.text(val, ax.get_ylim()[1]*0.9, f'{p}%', color='r')
    
    plt.tight_layout()
    plt.savefig('distributions.png')
    print("\n✅ Saved distribution plots to 'distributions.png'")

def analyze_relationships(df):
    """Analyze relationships between variables"""
    print("\n🔗 RELATIONSHIP ANALYSIS")
    print("-" * 35)
    
    # Calculate derived metrics
    df['daily_expense'] = df['receipts'] / df['days']
    df['miles_per_day'] = df['miles'] / df['days']
    df['reimbursement_rate'] = df['reimbursement'] / df['receipts']
    
    # Correlation matrix
    corr = df.corr()
    print("\n📈 Correlation Matrix:")
    print(corr.round(2))
    
    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Relationship Analysis', fontsize=16)
    
    # Ensure we have a 2D array of axes
    if len(axes.shape) == 1:
        axes = axes.reshape(-1, 2)
    
    # Days vs Reimbursement
    sns.scatterplot(x='days', y='reimbursement', data=df, ax=axes[0,0], alpha=0.6)
    axes[0,0].set_title('Days vs Reimbursement')
    
    # Miles vs Reimbursement
    sns.scatterplot(x='miles', y='reimbursement', data=df, ax=axes[0,1], alpha=0.6)
    axes[0,1].set_title('Miles vs Reimbursement')
    
    # Receipts vs Reimbursement
    sns.scatterplot(x='receipts', y='reimbursement', data=df, ax=axes[1,0], alpha=0.6)
    axes[1,0].set_title('Receipts vs Reimbursement')
    
    # Daily Expense vs Reimbursement Rate
    sns.scatterplot(x='daily_expense', y='reimbursement_rate', data=df, ax=axes[1,1], alpha=0.6)
    axes[1,1].set_title('Daily Expense vs Reimbursement Rate')
    
    plt.tight_layout()
    plt.savefig('relationships.png')
    print("✅ Saved relationship plots to 'relationships.png'")

def analyze_clusters(df):
    """Cluster similar trips to find patterns"""
    print("\n🔍 CLUSTER ANALYSIS")
    print("-" * 35)
    
    # Prepare features for clustering
    X = df[['days', 'miles', 'receipts']].copy()
    X['log_receipts'] = np.log1p(X['receipts'])
    X['log_miles'] = np.log1p(X['miles'])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[['days', 'log_miles', 'log_receipts']])
    
    # Find optimal number of clusters using elbow method
    inertias = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('elbow_curve.png')
    print("✅ Saved elbow curve to 'elbow_curve.png'")
    
    # Fit with optimal k (visually determined)
    optimal_k = 5  # Adjust based on elbow plot
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_summary = df.groupby('cluster').agg({
        'days': ['mean', 'count'],
        'miles': 'mean',
        'receipts': 'mean',
        'reimbursement': 'mean'
    }).round(2)
    
    print("\n📊 Cluster Summary:")
    print(cluster_summary)
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        np.log1p(df['miles']), 
        df['reimbursement'], 
        c=df['cluster'], 
        cmap='viridis',
        alpha=0.6
    )
    plt.xlabel('Log(Miles + 1)')
    plt.ylabel('Reimbursement ($)')
    plt.title('Trip Clusters by Miles and Reimbursement')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('clusters.png')
    print("✅ Saved cluster visualization to 'clusters.png'")

def analyze_outliers(df):
    """Identify and analyze outliers"""
    print("\n🔍 OUTLIER ANALYSIS")
    print("-" * 35)
    
    # Calculate z-scores
    z_scores = np.abs(stats.zscore(df[['days', 'miles', 'receipts', 'reimbursement']]))
    
    # Find outliers (z-score > 3)
    outliers = (z_scores > 3).any(axis=1)
    print(f"\nNumber of outliers (z > 3): {outliers.sum()} / {len(df)} ({(outliers.sum()/len(df)*100):.1f}%)")
    
    # Analyze outlier characteristics
    if outliers.any():
        print("\nOutlier Statistics:")
        print(df[outliers].describe().round(2))
    
    # Plot outliers
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x=np.log1p(df['receipts']), 
        y=df['reimbursement'],
        hue=outliers,
        palette={True: 'red', False: 'blue'},
        alpha=0.6
    )
    plt.title('Outliers in Receipts vs Reimbursement')
    plt.savefig('outliers.png')
    print("✅ Saved outlier visualization to 'outliers.png'")

def main():
    # Load data
    df = load_data()
    
    # Run analyses
    analyze_distributions(df)
    analyze_relationships(df)
    analyze_clusters(df)
    analyze_outliers(df)
    
    print("\n🎉 Analysis complete! Check the generated plots for insights.")

if __name__ == "__main__":
    main()
