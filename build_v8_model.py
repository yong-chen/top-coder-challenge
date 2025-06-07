#!/usr/bin/env python3
"""
Build V8 Model - Optimized Feature Engineering
Uses feature selection and model tuning
"""

import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

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

def create_v8_features(df):
    """Create optimized features based on V6 analysis"""
    print("🔧 CREATING V8 FEATURES")
    print("-" * 35)
    
    X = pd.DataFrame()
    
    # 1. Base features (from V2)
    X['days'] = df['days']
    X['miles'] = df['miles']
    X['receipts'] = df['receipts']
    
    # 2. Top interactions (from V6 importance)
    X['days_miles'] = X['days'] * X['miles']
    X['days_receipts'] = X['days'] * X['receipts']
    X['miles_receipts'] = X['miles'] * X['receipts']
    
    # 3. Polynomial features (better scaling)
    X['sqrt_receipts'] = np.sqrt(X['receipts'] + 1)
    X['log_receipts'] = np.log1p(X['receipts'])
    
    # 4. Business logic features
    X['daily_expense'] = X['receipts'] / (X['days'] + 1)
    X['miles_per_day'] = X['miles'] / (X['days'] + 1)
    
    # 5. Categorical features (one-hot encoded later)
    X['expense_tier'] = pd.qcut(X['receipts'], q=5, labels=False, duplicates='drop')
    X['trip_type'] = 0  # Default
    X.loc[(X['days'] <= 2) & (X['miles'] < 100), 'trip_type'] = 1  # Local
    X.loc[(X['days'] > 7) | (X['miles'] > 500), 'trip_type'] = 2  # Long
    
    print(f"✅ Created {X.shape[1]} features")
    return X

def train_v8_model(X, y):
    """Train and evaluate V8 model with feature selection"""
    print("\n🎓 TRAINING V8 MODEL")
    print("-" * 35)
    
    # Define feature types
    numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_features = ['expense_tier', 'trip_type']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', 'passthrough', categorical_features)
        ])
    
    # Base model for feature selection
    selector = SelectFromModel(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        threshold='median'  # Keep top 50% features
    )
    
    # Final model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', model)
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    avg_mae = -np.mean(cv_scores)
    print(f"📊 CV MAE (V8): ${avg_mae:.2f}")
    
    # Train final model
    pipeline.fit(X, y)
    
    # Get selected features
    selected_features = np.array(X.columns)[selector.get_support()]
    print(f"\n🔍 SELECTED FEATURES ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
    
    return pipeline, avg_mae

def main():
    # Load and prepare data
    df = load_data()
    X = create_v8_features(df)
    y = df['reimbursement']
    
    # Train model
    model, mae = train_v8_model(X, y)
    
    # Save model
    model_data = {
        'model': model,
        'feature_cols': list(X.columns),
        'mae': mae
    }
    joblib.dump(model_data, 'reimbursement_model_v8.pkl')
    
    print(f"\n🚀 V8 MODEL SAVED (MAE: ${mae:.2f})")
    print(f"   Improvement over V2 ($28.52): ${28.52 - mae:.2f}")
    print(f"   Improvement over V6 ($80.62): ${80.62 - mae:.2f}")

if __name__ == "__main__":
    main()
