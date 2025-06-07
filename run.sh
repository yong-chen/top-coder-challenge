#!/bin/bash

# Black Box Challenge - Final Submission Implementation
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Call Python implementation V2 (ML-based - BEST PERFORMANCE: $28.52 avg error)
python calculate_reimbursement_v2.py "$1" "$2" "$3"
