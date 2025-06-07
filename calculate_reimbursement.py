#!/usr/bin/env python3
"""
Black Box Challenge - Reimbursement Calculation
Initial implementation based on data analysis insights
"""

import sys
import math

def calculate_reimbursement(days, miles, receipts):
    """
    Calculate reimbursement based on reverse-engineered algorithm
    
    Key insights from data analysis:
    - Base per-day rate: ~$104.23
    - Mileage rate: ~$0.53/mile  
    - Receipt reimbursement varies by amount
    - Complex interactions between components
    - Route complexity bonuses
    - Expense caps for very high amounts
    """
    
    # Base components from analysis
    base_per_day = 104.23
    base_mile_rate = 0.53
    
    # Base calculation
    day_component = days * base_per_day
    mile_component = miles * base_mile_rate
    
    # Receipt component - varies by amount (from analysis)
    if receipts <= 50:
        receipt_component = receipts * 0.5  # Lower reimbursement rate
    elif receipts <= 200:
        receipt_component = receipts * 0.6
    elif receipts <= 500:
        receipt_component = receipts * 0.4  # Analysis showed lower avg here
    elif receipts <= 1000:
        receipt_component = receipts * 0.7
    else:
        receipt_component = receipts * 0.8  # Higher amounts get better rate
    
    # Base reimbursement
    base_reimbursement = day_component + mile_component + receipt_component
    
    # Route complexity bonus (Dave's theory confirmed)
    # High miles for the duration suggests complex routing
    miles_per_day = miles / days if days > 0 else 0
    
    complexity_bonus = 0
    if miles_per_day > 100:  # Complex routing
        # Analysis showed $200-500 bonus for complex routes
        complexity_factor = min((miles_per_day - 100) / 100, 2.0) 
        complexity_bonus = complexity_factor * (days * 30)  # Scale with trip length
    
    # Interaction effects (Lisa's theory confirmed)
    # From analysis: interactions improved R² by 0.019
    interaction_bonus = 0
    
    # Days-miles interaction: longer trips with high mileage get extra
    if days >= 3 and miles > 300:
        interaction_bonus += (days - 2) * (miles / 1000) * 50
    
    # Days-receipts interaction: longer trips with high expenses
    if days >= 5 and receipts > 1000:
        interaction_bonus += (days - 4) * (receipts / 1000) * 20
    
    # Miles-receipts interaction: high mileage + high expenses
    if miles > 500 and receipts > 800:
        interaction_bonus += (miles / 1000) * (receipts / 1000) * 15
    
    # Calculate total before caps
    total_reimbursement = base_reimbursement + complexity_bonus + interaction_bonus
    
    # Apply expense cap (critical insight from outlier analysis)
    # Very high expense cases get capped around $1900-2000
    if receipts > 2000:
        # Severe cap for very high expenses
        cap = 1900 + (receipts - 2000) * 0.1  # Only 10% of excess above $2000
        total_reimbursement = min(total_reimbursement, cap)
    elif receipts > 1500:
        # Moderate cap for high expenses  
        cap = total_reimbursement * 0.95  # 5% reduction
        total_reimbursement = cap
    
    # Round to 2 decimal places
    return round(total_reimbursement, 2)

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <days> <miles> <receipts>")
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
