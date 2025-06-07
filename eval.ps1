# Black Box Challenge Evaluation Script - PowerShell Version
# Tests your reimbursement calculation implementation against 1,000 historical cases

Write-Host " Black Box Challenge - Reimbursement System Evaluation" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host " Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host " Error: Python is required but not found!" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Check if run script exists
if (-not (Test-Path "run.ps1") -and -not (Test-Path "run.bat")) {
    Write-Host " Error: Neither run.ps1 nor run.bat found!" -ForegroundColor Red
    Write-Host "Please create your implementation script." -ForegroundColor Yellow
    exit 1
}

# Determine which run script to use
$runScript = if (Test-Path "run.ps1") { "powershell -File run.ps1" } else { "run.bat" }

# Load test cases
Write-Host " Loading test cases from public_cases.json..." -ForegroundColor Yellow

if (-not (Test-Path "public_cases.json")) {
    Write-Host " Error: public_cases.json not found!" -ForegroundColor Red
    exit 1
}

try {
    $testCases = Get-Content "public_cases.json" | ConvertFrom-Json
    Write-Host " Loaded $($testCases.Count) test cases" -ForegroundColor Green
} catch {
    Write-Host " Error: Failed to parse public_cases.json" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host " Running evaluation..." -ForegroundColor Yellow
Write-Host "Progress: " -NoNewline

$exactMatches = 0
$closeMatches = 0
$totalError = 0.0
$successfulRuns = 0
$maxDisplayErrors = 10
$displayedErrors = 0

for ($i = 0; $i -lt $testCases.Count; $i++) {
    $testCase = $testCases[$i]
    $inputData = $testCase.input
    $expected = [double]$testCase.expected_output
    
    # Show progress
    if ($i % 100 -eq 0) {
        Write-Host "$i " -NoNewline -ForegroundColor Green
    }
    
    try {
        # Run the implementation
        $result = if ($runScript -like "*run.ps1*") {
            powershell -File run.ps1 $inputData.trip_duration_days $inputData.miles_traveled $inputData.total_receipts_amount
        } else {
            & cmd /c "run.bat $($inputData.trip_duration_days) $($inputData.miles_traveled) $($inputData.total_receipts_amount)"
        }
        
        $actual = [double]$result
        $errorAmount = [Math]::Abs($actual - $expected)
        
        $totalError += $errorAmount
        $successfulRuns++
        
        # Check accuracy
        if ($errorAmount -le 0.01) {
            $exactMatches++
        } elseif ($errorAmount -le 1.00) {
            $closeMatches++
        } else {
            # Display first few large errors for debugging
            if ($displayedErrors -lt $maxDisplayErrors) {
                Write-Host ""
                Write-Host "   Large error - Input: $($inputData.trip_duration_days)d, $($inputData.miles_traveled)mi, `$$($inputData.total_receipts_amount) | Expected: `$$expected | Got: `$$actual | Error: `$$($errorAmount.ToString('F2'))" -ForegroundColor Red
                $displayedErrors++
            }
        }
        
    } catch {
        Write-Host ""
        Write-Host "   Failed on case $($i+1): Input: $($inputData.trip_duration_days)d, $($inputData.miles_traveled)mi, `$$($inputData.total_receipts_amount)" -ForegroundColor Red
        Write-Host "      Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host ""
Write-Host " EVALUATION RESULTS" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan

if ($successfulRuns -gt 0) {
    $avgError = $totalError / $successfulRuns
    $exactPercent = ($exactMatches / $successfulRuns) * 100
    $closePercent = ($closeMatches / $successfulRuns) * 100
    $score = $avgError + (1000 - $exactMatches) * 0.1
    
    Write-Host "Successful runs:    $successfulRuns / $($testCases.Count)" -ForegroundColor Green
    Write-Host "Exact matches:      $exactMatches ($($exactPercent.ToString('F1'))%) [within ±`$0.01]" -ForegroundColor $(if ($exactPercent -gt 80) { "Green" } elseif ($exactPercent -gt 50) { "Yellow" } else { "Red" })
    Write-Host "Close matches:      $closeMatches ($($closePercent.ToString('F1'))%) [within ±`$1.00]" -ForegroundColor $(if ($closePercent -gt 90) { "Green" } elseif ($closePercent -gt 70) { "Yellow" } else { "Red" })
    Write-Host "Average error:      `$$($avgError.ToString('F2'))" -ForegroundColor $(if ($avgError -lt 1) { "Green" } elseif ($avgError -lt 5) { "Yellow" } else { "Red" })
    Write-Host "Score:              $($score.ToString('F2')) (lower is better)" -ForegroundColor $(if ($score -lt 50) { "Green" } elseif ($score -lt 200) { "Yellow" } else { "Red" })
    
    Write-Host ""
    if ($exactPercent -gt 95) {
        Write-Host " Excellent! Your implementation is highly accurate!" -ForegroundColor Green
    } elseif ($exactPercent -gt 80) {
        Write-Host " Good work! Your implementation is quite accurate." -ForegroundColor Yellow
    } elseif ($exactPercent -gt 50) {
        Write-Host " Getting there! Consider refining your algorithm." -ForegroundColor Yellow
    } else {
        Write-Host " Keep working! There's room for improvement." -ForegroundColor Red
    }
} else {
    Write-Host " No successful runs! Please check your implementation." -ForegroundColor Red
}

Write-Host ""
