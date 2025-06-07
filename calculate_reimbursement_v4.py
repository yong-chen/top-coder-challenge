#!/usr/bin/env python3
"""
Black Box Challenge - Reimbursement Calculation V4
Refined ML + Rule-based approach with 5-pattern expense caps
"""

import sys
import pandas as pd
import joblib
import numpy as np

def create_features(days, miles, receipts):
    """Create the same features used in model training"""
    data = {
        'days': days,
        'miles': miles,
        'receipts': receipts,
        
        # Squared features
        'days_squared': days ** 2,
        'miles_squared': miles ** 2,
        'receipts_squared': receipts ** 2,
        
        # Interaction features (confirmed important!)
        'days_miles': days * miles,
        'days_receipts': days * receipts,
        'miles_receipts': miles * receipts,
        
        # Per-day features
        'miles_per_day': miles / days if days > 0 else 0,
        'receipts_per_day': receipts / days if days > 0 else 0,
        
        # Expense band features
        'low_expense': 1 if receipts <= 50 else 0,
        'medium_expense': 1 if 50 < receipts <= 500 else 0,
        'high_expense': 1 if receipts > 1000 else 0,
        'very_high_expense': 1 if receipts > 2000 else 0,
        
        # Trip duration features
        'short_trip': 1 if days <= 2 else 0,
        'medium_trip': 1 if 3 <= days <= 5 else 0,
        'long_trip': 1 if days >= 7 else 0,
        
        # Distance features
        'short_distance': 1 if miles <= 200 else 0,
        'long_distance': 1 if miles >= 800 else 0
    }
    
    return pd.DataFrame([data])

def apply_refined_caps(ml_prediction, receipts, days, miles):
    """
    Refined expense cap logic based on V3 error analysis
    Implements 5-pattern approach for different trip types
    """
    
    # Calculate trip complexity score
    complexity_score = (miles / days) if days > 0 else 0
    expense_per_day = receipts / days if days > 0 else receipts
    
    # Pattern 1: Short trips with very high expenses - severe caps
    if days <= 5 and receipts > 2000:
        # These are likely anomalous - cap very aggressively
        base_allowance = days * 60 + miles * 0.25
        excess_allowance = (receipts - 2000) * 0.1  # Only 10% of excess
        capped = base_allowance + excess_allowance
        return min(ml_prediction, capped)
    
    # Pattern 2: High complexity + high expense trips  
    elif complexity_score > 100 and receipts > 1400:
        # Complex routes with high expenses - moderate but firm caps
        base_trip_cost = days * 70 + miles * 0.35
        receipt_portion = min(receipts * 0.6, 800)  # Cap receipt portion at $800
        capped = base_trip_cost + receipt_portion
        return min(ml_prediction, capped)
    
    # Pattern 3: Very high expenses (>$2500)
    elif receipts > 2500:
        base_trip_cost = days * 80 + miles * 0.4
        receipt_portion = receipts * 0.4  # More aggressive than V3
        capped = base_trip_cost + receipt_portion
        return min(ml_prediction, capped)
    
    # Pattern 4: High expenses ($2000-2500)
    elif receipts > 2000:
        base_trip_cost = days * 80 + miles * 0.4
        receipt_portion = receipts * 0.65  # Slightly more aggressive than V3
        capped = base_trip_cost + receipt_portion
        return min(ml_prediction, capped)
    
    # Pattern 5: Medium-high expenses ($1500-2000)
    elif receipts > 1500:
        # Light smoothing
        return ml_prediction * 0.95
    
    else:
        # Low/medium expenses - no caps needed
        return ml_prediction

def calculate_reimbursement(days, miles, receipts):
    """
    Calculate reimbursement using refined ML + 5-pattern cap approach
    """
    try:
        # Load the trained model
        model_data = joblib.load('reimbursement_model.pkl')
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        
        # Create features
        features_df = create_features(days, miles, receipts)
        
        # Ensure we have all required features in correct order
        X = features_df[feature_cols]
        
        # Get ML prediction
        ml_prediction = model.predict(X)[0]
        
        # Apply refined expense caps (V4 improvement)
        final_prediction = apply_refined_caps(ml_prediction, receipts, days, miles)
        
        # Round to 2 decimal places
        return round(final_prediction, 2)
        
    except FileNotFoundError:
        # Fallback to simple estimation if model not found
        print("Warning: Model file not found, using fallback calculation", file=sys.stderr)
        return round(days * 100 + miles * 0.5 + receipts * 0.8, 2)
    
    except Exception as e:
        print(f"Error in calculation: {e}", file=sys.stderr)
        return 0.0

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement_v4.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = int(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = calculate_reimbursement(days, miles, receipts)
        print(result)
        
    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
