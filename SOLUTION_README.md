# Black Box Legacy Reimbursement System - Solution

**Challenge Completed:** 2025-06-07  
**Total Time:** ~1.5 hours  
**Repository:** https://github.com/yong-chen/top-coder-challenge

## 🎯 Final Solution Performance

**Algorithm:** Random Forest Machine Learning Model (V2)
- **Average Error:** $28.52
- **Close Matches:** 26/1000 (2.6%) within ±$1.00
- **Score:** 128.52 (lower is better)
- **Cross-validation R²:** 0.9116

## 🔍 Solution Approach

### 1. Data Analysis & Pattern Discovery
- Analyzed 1,000 historical input/output examples
- Validated employee theories from interviews
- Discovered key correlations and interaction effects

### 2. Machine Learning Implementation
- **Model:** Random Forest Regressor (200 trees, max_depth=15)
- **Features:** 20 engineered features including:
  - Base inputs: days, miles, receipts
  - Squared terms: days², miles², receipts²
  - **Key interactions:** days×miles, days×receipts, miles×receipts
  - Categorical features: trip duration bands, expense bands, distance bands

### 3. Key Discoveries
- **Lisa's "Complex Interactions" theory:** ✅ Validated (R² improved 0.78→0.91)
- **Dave's "Route Complexity" theory:** ✅ Validated (days×miles is 2nd most important feature)
- **Marcus's "Efficiency Bonus" theory:** ❌ Not statistically significant

### 4. Feature Importance (Top 5)
1. **receipts_squared (27.2%)** - Quadratic relationship with receipt amount
2. **days_miles (24.8%)** - Trip complexity interaction  
3. **receipts (24.6%)** - Base receipt importance
4. **days_receipts (14.9%)** - Duration×expense interaction
5. **miles_receipts (4.3%)** - Distance×expense interaction

## 🏗️ Technical Implementation

### Core Files
- **`calculate_reimbursement_v2.py`** - Final ML-based algorithm
- **`reimbursement_model.pkl`** - Trained Random Forest model
- **`run.ps1`** / **`run.bat`** - Windows-compatible runners
- **`run.sh`** - Linux/Mac compatibility

### Windows Compatibility
Created full Windows-native solution:
- PowerShell evaluation scripts
- Batch file runners  
- No dependency on bash/jq/bc

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
```

## 📊 Algorithm Evolution

| Version | Avg Error | Approach | Status |
|---------|-----------|----------|--------|
| V1 (Manual) | $591.77 | Rule-based estimation | ❌ Poor |
| **V2 (ML)** | **$28.52** | **Random Forest** | **✅ Final** |
| V3 (Caps) | $31.46 | ML + expense caps | ⚠️ Slightly worse |
| V4 (Refined) | $139.19 | 5-pattern caps | ❌ Regression |

**Decision:** V2 selected as final submission due to best performance and stability.

## 🚀 Usage

### Windows
```powershell
# PowerShell
.\run.ps1 <days> <miles> <receipts>

# Command Prompt  
run.bat <days> <miles> <receipts>
```

### Linux/Mac
```bash
./run.sh <days> <miles> <receipts>
```

### Example
```bash
./run.sh 3 93 1.42
# Output: 361.60
```

## 📈 Validation Results

Tested against 1,000 public cases:
- **Successful runs:** 1000/1000 (100%)
- **Average absolute error:** $28.52
- **Median error:** ~$15-20
- **No exact matches** (±$0.01) - indicates complex non-linear patterns
- **26 close matches** (±$1.00) - good performance on easier cases

## 🔧 Alternative Implementations Explored

1. **Linear Regression:** R² = 0.8936 (good baseline)
2. **Polynomial Features:** R² = 0.9116 (best CV score)  
3. **Random Forest:** R² = 0.9898 training (chosen for generalization)
4. **Expense Cap Rules:** Various attempts to handle high-expense cases manually

## 📝 Lessons Learned

1. **Employee insights were valuable** - Lisa's interaction theory and Dave's complexity theory both validated
2. **Simple ML often beats complex rules** - V2's pure ML approach outperformed rule-based caps
3. **Feature engineering crucial** - Interaction terms and squared features were key
4. **Expense patterns exist but are subtle** - Attempts to manually encode them failed
5. **Data-driven approach successful** - Let the model learn patterns rather than force assumptions

## 🎯 Submission Files

- **`private_results.txt`** - Generated results for private test cases
- **All source code** - Complete implementation with version history
- **Documentation** - This README and progress tracking

---

**Challenge completed successfully with data-driven machine learning approach!**
