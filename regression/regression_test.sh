#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
VALIDATION_DIR="validation_set"
NEW_RESULTS_DIR="new_results"

echo "Starting regression test on the validation set..."

# Step 1: Run the pipeline on the validation set
echo "Running the pipeline on the validation set..."
cd ../src
python3 ./pipeline.py ../pipeline_config_validation.yaml -v 2>&1 | tee ../log_new.out
cd ..

# Step 2: Copy the new results
mkdir -p "$NEW_RESULTS_DIR"
cp output_validation/results/iapc_results.csv "$NEW_RESULTS_DIR/"

# Step 3: Compare Results
echo "Comparing baseline results with new results..."

# Paths to results
ORIGINAL_RESULTS="baseline_results/iapc_results.csv"
NEW_RESULTS="$NEW_RESULTS_DIR/iapc_results.csv"

# Check if the new results file exists
if [ ! -f "$NEW_RESULTS" ]; then
  echo "Regression test failed: New iapc_results.csv not found."
  exit 1
fi

# Use the comparison script
python regression/compare_results.py "$ORIGINAL_RESULTS" "$NEW_RESULTS"

echo "Regression test completed."
