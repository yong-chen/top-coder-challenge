#!/usr/bin/env python3
"""
Build Regression Model - Create a data-driven algorithm using machine learning
"""

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib

def load_data():
    """Load and prepare data"""
    print("🔍 Loading data for regression model...")
    
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
    print(f"✅ Loaded {len(df)} cases")
    return df

def create_features(df):
    """Create engineered features based on our analysis"""
    df = df.copy()
    
    # Basic features
    df['days_squared'] = df['days'] ** 2
    df['miles_squared'] = df['miles'] ** 2
    df['receipts_squared'] = df['receipts'] ** 2
    
    # Interaction features (Lisa's theory)
    df['days_miles'] = df['days'] * df['miles']
    df['days_receipts'] = df['days'] * df['receipts'] 
    df['miles_receipts'] = df['miles'] * df['receipts']
    
    # Route complexity (Dave's theory)
    df['miles_per_day'] = df['miles'] / df['days']
    df['receipts_per_day'] = df['receipts'] / df['days']
    
    # Expense bands (from our analysis)
    df['low_expense'] = (df['receipts'] <= 50).astype(int)
    df['medium_expense'] = ((df['receipts'] > 50) & (df['receipts'] <= 500)).astype(int)
    df['high_expense'] = (df['receipts'] > 1000).astype(int)
    df['very_high_expense'] = (df['receipts'] > 2000).astype(int)
    
    # Trip duration bands
    df['short_trip'] = (df['days'] <= 2).astype(int)
    df['medium_trip'] = ((df['days'] >= 3) & (df['days'] <= 5)).astype(int)
    df['long_trip'] = (df['days'] >= 7).astype(int)
    
    # Distance bands
    df['short_distance'] = (df['miles'] <= 200).astype(int)
    df['long_distance'] = (df['miles'] >= 800).astype(int)
    
    print(f"✅ Created features: {list(df.columns)}")
    return df

def test_models(df):
    """Test different regression models"""
    print("\n🧪 TESTING REGRESSION MODELS")
    print("=" * 50)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'reimbursement']
    X = df[feature_cols]
    y = df['reimbursement']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Polynomial (degree 2)': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        if name == 'Polynomial (degree 2)':
            # Create polynomial features
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
        else:
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        results[name] = {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'model': model
        }
        
        print(f"{name:20}: R² = {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['mean_r2'])
    print(f"\n🎯 Best model: {best_model_name} (R² = {results[best_model_name]['mean_r2']:.4f})")
    
    return results, feature_cols, X, y

def build_final_model(df, model_type='RandomForest'):
    """Build the final model"""
    print(f"\n🔨 BUILDING FINAL {model_type.upper()} MODEL")
    print("=" * 50)
    
    # Prepare data
    feature_cols = [col for col in df.columns if col != 'reimbursement']
    X = df[feature_cols]
    y = df['reimbursement']
    
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        model.fit(X, y)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🔍 Feature Importance (Top 10):")
        for _, row in importance.head(10).iterrows():
            print(f"   {row['feature']:20}: {row['importance']:.4f}")
        
    else:  # Linear
        model = LinearRegression()
        model.fit(X, y)
        
        # Coefficients
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("📊 Coefficients (Top 10 by magnitude):")
        for _, row in coef_df.head(10).iterrows():
            print(f"   {row['feature']:20}: {row['coefficient']:+8.2f}")
    
    # Model performance
    train_score = model.score(X, y)
    predictions = model.predict(X)
    mae = np.mean(np.abs(predictions - y))
    
    print(f"\n📈 Model Performance:")
    print(f"   Training R²: {train_score:.4f}")
    print(f"   Mean Absolute Error: ${mae:.2f}")
    
    # Save model
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'model_type': model_type
    }
    joblib.dump(model_data, 'reimbursement_model.pkl')
    print(f"✅ Model saved as 'reimbursement_model.pkl'")
    
    return model, feature_cols

def main():
    df = load_data()
    df_features = create_features(df)
    
    # Test different models
    results, feature_cols, X, y = test_models(df_features)
    
    # Build final model (choose best performing)
    best_r2 = max(results.values(), key=lambda x: x['mean_r2'])['mean_r2']
    
    if best_r2 > 0.9:
        print(f"\n🎉 Excellent model performance! R² = {best_r2:.4f}")
        model_type = 'RandomForest'  # Use RF for high accuracy
    else:
        print(f"\n⚠️  Model performance moderate: R² = {best_r2:.4f}")
        model_type = 'Linear'  # Use simpler model
    
    model, feature_cols = build_final_model(df_features, model_type)
    
    print("\n🎯 Next: Create calculate_reimbursement_v2.py using this model")

if __name__ == "__main__":
    main()
