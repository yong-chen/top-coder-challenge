@echo off
REM Black Box Challenge - Batch Implementation
REM Usage: run.bat <trip_duration_days> <miles_traveled> <total_receipts_amount>

python calculate_reimbursement.py %1 %2 %3
