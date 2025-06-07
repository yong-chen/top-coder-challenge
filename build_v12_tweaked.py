#!/usr/bin/env python3
"""
V12 - V2 with Minimal, Targeted Tweaks
Incremental improvements to V2 based on error analysis
"""

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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

def create_v12_features(df):
    """Create features for V12 (V2 features + small tweaks)"""
    X = pd.DataFrame()
    
    # 1. Base V2 features
    X['days'] = df['days']
    X['miles'] = df['miles']
    X['receipts'] = df['receipts']
    
    # 2. V2 interactions (keep what worked)
    X['days_miles'] = X['days'] * X['miles']
    X['days_receipts'] = X['days'] * X['receipts']
    X['miles_receipts'] = X['miles'] * X['receipts']
    
    # 3. Small tweaks based on error analysis
    # - Log transform receipts to reduce impact of high values
    X['log_receipts'] = np.log1p(X['receipts'])
    
    # - Days squared (capture non-linearity in trip length)
    X['days_sq'] = X['days'] ** 2
    
    # - Receipt tiers (simplified from V10)
    X['receipt_tier'] = pd.cut(
        X['receipts'],
        bins=[0, 50, 200, 500, float('inf')],
        labels=[1, 2, 3, 4],
        include_lowest=True
    )
    
    # - Is local trip (days <= 2 and miles < 100)
    X['is_local'] = ((X['days'] <= 2) & (X['miles'] < 100)).astype(int)
    
    return X

def train_v12_model(X, y):
    """Train V12 model with similar structure to V2"""
    # Use same hyperparameters as V2 for fair comparison
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Simple train-test split (no time series)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, mae, importances

def analyze_errors(model, X, y):
    """Analyze where the model makes mistakes"""
    y_pred = model.predict(X)
    errors = np.abs(y - y_pred)
    
    # Add to dataframe for analysis
    df = X.copy()
    df['actual'] = y
    df['predicted'] = y_pred
    df['error'] = errors
    
    # Find worst predictions
    worst = df.nlargest(5, 'error')
    
    return worst

def main():
    print("🚀 V12 - V2 with Minimal, Targeted Tweaks")
    print("-" * 50)
    
    # Load data
    df = load_data()
    
    # Create features
    X = create_v12_features(df)
    y = df['reimbursement']
    
    # Train model
    print("\n🔧 Training V12 model...")
    model, mae, importances = train_v12_model(X, y)
    
    print(f"\n📊 V12 Mean Absolute Error: ${mae:.2f}")
    print(f"   Compared to V2 ($28.52): ${28.52 - mae:+.2f}")
    
    # Feature importance
    print("\n🔍 Top 5 Features:")
    print(importances.head().to_string(index=False))
    
    # Analyze errors
    print("\n🔎 Analyzing errors...")
    worst_predictions = analyze_errors(model, X, y)
    print("\n❌ Worst Predictions:")
    print(worst_predictions[['days', 'miles', 'receipts', 'actual', 'predicted', 'error']].to_string())
    
    # Save model
    model_data = {
        'model': model,
        'feature_cols': list(X.columns),
        'mae': mae
    }
    joblib.dump(model_data, 'reimbursement_model_v12.pkl')
    print("\n💾 Saved V12 model to 'reimbursement_model_v12.pkl'")

if __name__ == "__main__":
    main()
