#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
NUM_IMAGES=100  # Fixed number of images for the validation set
SOURCE_IMAGES_DIR="/home/swagh/bae-systems-gqp-24/xview_dataset_raw/train_images"
ANNOTATIONS_FILE="/home/swagh/bae-systems-gqp-24/xview_dataset_raw/xView_train.geojson"
CLASS_LABELS_FILE="/home/swagh/bae-systems-gqp-24/xview_dataset_raw/xview_class_labels.json"
VALIDATION_DIR="validation_set"
BASELINE_RESULTS_DIR="baseline_results"
SEED=42  # Fixed seed for reproducibility

# --------------------------------------
# Data Preparation
# --------------------------------------

# Step 1: Create validation directories
mkdir -p "$VALIDATION_DIR/images"
mkdir -p "$VALIDATION_DIR/annotations"

# Step 2: Select fixed random images and copy them to the validation directory
echo "Selecting $NUM_IMAGES fixed random images..."
if [ ! -f "$VALIDATION_DIR/images_selected.txt" ]; then
    find "$SOURCE_IMAGES_DIR" -type f | shuf --random-source=<(yes $SEED) -n "$NUM_IMAGES" > "$VALIDATION_DIR/images_selected.txt"
fi

# Copy images
echo "Copying images to validation set..."
while read img_path; do
    cp "$img_path" "$VALIDATION_DIR/images/"
done < "$VALIDATION_DIR/images_selected.txt"

# Step 3: Copy the annotations files
echo "Copying annotations files..."
cp "$ANNOTATIONS_FILE" "$VALIDATION_DIR/annotations/"
cp "$CLASS_LABELS_FILE" "$VALIDATION_DIR/annotations/"

# Step 4: Filter the annotations file
echo "Filtering annotations..."
python3 <<EOF
import json
import os

# Directory containing the validation images
image_dir = '$VALIDATION_DIR/images'
image_ids = set(os.listdir(image_dir))

# Load the annotations file
with open('$VALIDATION_DIR/annotations/xView_train.geojson', 'r') as f:
    data = json.load(f)

# Filter the features
filtered_features = []
for feature in data['features']:
    image_id_in_annotation = os.path.basename(feature['properties']['image_id'])
    if image_id_in_annotation in image_ids:
        filtered_features.append(feature)

# Create new data with filtered features
filtered_data = {
    'type': data['type'],
    'features': filtered_features
}

# Save the filtered annotations
with open('$VALIDATION_DIR/annotations/xView_train_filtered.geojson', 'w') as f:
    json.dump(filtered_data, f)
EOF

echo "Data preparation completed. Validation set is ready."

# --------------------------------------
# Baseline Results Generation
# --------------------------------------

# Step 1: Run the pipeline to generate baseline results
echo "Running the pipeline on the validation set to generate baseline results..."
cd ../src
python3 ./pipeline.py ../pipeline_config_validation.yaml -v 2>&1 | tee ../log_baseline.out
cd ..

# Step 2: Copy the baseline results
mkdir -p "$BASELINE_RESULTS_DIR"
cp output_validation/results/iapc_results.csv "$BASELINE_RESULTS_DIR/"

echo "Baseline results generated and stored in $BASELINE_RESULTS_DIR/iapc_results.csv."