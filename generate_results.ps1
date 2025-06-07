# Black Box Challenge - Results Generation Script (PowerShell)
# This script runs your implementation against private test cases and outputs results

Write-Host "Black Box Challenge - Generating Private Results" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is required but not found!" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Check if our V2 implementation exists
if (-not (Test-Path "calculate_reimbursement_v2.py")) {
    Write-Host "Error: calculate_reimbursement_v2.py not found!" -ForegroundColor Red
    exit 1
}

# Check if private test cases exist
if (-not (Test-Path "private_cases.json")) {
    Write-Host "Error: private_cases.json not found!" -ForegroundColor Red
    exit 1
}

# Load private test cases
Write-Host "Loading private test cases..." -ForegroundColor Yellow

try {
    $testCases = Get-Content "private_cases.json" | ConvertFrom-Json
    Write-Host "Loaded $($testCases.Count) private test cases" -ForegroundColor Green
} catch {
    Write-Host "Error: Failed to parse private_cases.json" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Generating results for private test cases..." -ForegroundColor Yellow
Write-Host "Progress: " -NoNewline

$results = @()
$successful = 0
$failed = 0

for ($i = 0; $i -lt $testCases.Count; $i++) {
    $testCase = $testCases[$i]
    $inputData = $testCase.input
    
    # Show progress
    if ($i % 100 -eq 0) {
        Write-Host "$i " -NoNewline -ForegroundColor Green
    }
    
    try {
        # Run our V2 implementation
        $result = python calculate_reimbursement_v2.py $inputData.trip_duration_days $inputData.miles_traveled $inputData.total_receipts_amount
        
        if ($result) {
            $results += $result
            $successful++
        } else {
            $results += "0.00"
            $failed++
        }
        
    } catch {
        Write-Host ""
        Write-Host "   Failed on case $($i+1): $($_.Exception.Message)" -ForegroundColor Red
        $results += "0.00"
        $failed++
    }
}

Write-Host ""
Write-Host ""
Write-Host "Writing results to private_results.txt..." -ForegroundColor Yellow

# Write results to file
$results | Out-File -FilePath "private_results.txt" -Encoding ASCII

Write-Host "Results written to private_results.txt" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "   Total cases: $($testCases.Count)" -ForegroundColor White
Write-Host "   Successful: $successful" -ForegroundColor Green
Write-Host "   Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })

if ($failed -eq 0) {
    Write-Host ""
    Write-Host "All test cases processed successfully!" -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "   1. Add arjun-krishna1 as collaborator to your GitHub repo" -ForegroundColor White
    Write-Host "   2. Submit private_results.txt via the submission form" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Some test cases failed. Review your implementation." -ForegroundColor Yellow
}

Write-Host ""
