# Black Box Challenge - PowerShell Implementation
# Usage: .\run.ps1 <trip_duration_days> <miles_traveled> <total_receipts_amount>

param(
    [Parameter(Mandatory=$true)][int]$TripDurationDays,
    [Parameter(Mandatory=$true)][int]$MilesTraveled,
    [Parameter(Mandatory=$true)][double]$TotalReceiptsAmount
)

# Call Python implementation V3 (ML + Expense Caps)
python calculate_reimbursement_v3.py $TripDurationDays $MilesTraveled $TotalReceiptsAmount
