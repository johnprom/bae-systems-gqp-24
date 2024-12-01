import sys
import pandas as pd

def main(original_file, new_file):
    # Load the CSV files
    original_df = pd.read_csv(original_file)
    new_df = pd.read_csv(new_file)

    # Display the columns for debugging purposes
    print("Baseline Results Columns:", original_df.columns.tolist())
    print("New Results Columns:", new_df.columns.tolist())

    # Define the metrics to compare
    metrics = ['object_name', 'mAP']

    # Check if the required metrics are present
    for metric in metrics:
        if metric not in original_df.columns or metric not in new_df.columns:
            print(f"Metric '{metric}' not found in one of the result files. Cannot compare.")
            sys.exit(1)

    # Merge the dataframes on 'object_name'
    merged_df = pd.merge(
        original_df[metrics],
        new_df[metrics],
        on='object_name',
        suffixes=('_baseline', '_new')
    )

    # Define tolerance thresholds
    PER_CLASS_TOLERANCE = 0.10  # 10% tolerance for per-class mAP
    OVERALL_TOLERANCE = 0.05    # 5% tolerance for overall mAP

    per_class_success = True

    # Compare mAP for each object_name
    for _, row in merged_df.iterrows():
        object_name = row['object_name']
        mAP_baseline = row['mAP_baseline']
        mAP_new = row['mAP_new']

        print(f"\nObject: {object_name}")
        print(f"  Baseline mAP: {mAP_baseline}")
        print(f"  New mAP: {mAP_new}")

        if mAP_new < mAP_baseline * (1 - PER_CLASS_TOLERANCE):
            print(f"  Warning: New mAP is worse than baseline beyond acceptable per-class tolerance.")
            per_class_success = False
        else:
            print(f"  New mAP is within acceptable per-class tolerance.")

    # Calculate overall mAP
    overall_mAP_baseline = merged_df['mAP_baseline'].mean()
    overall_mAP_new = merged_df['mAP_new'].mean()

    print(f"\nOverall Baseline mAP: {overall_mAP_baseline}")
    print(f"Overall New mAP: {overall_mAP_new}")

    success = True
    if overall_mAP_new < overall_mAP_baseline * (1 - OVERALL_TOLERANCE):
        print("Warning: Overall new mAP is worse than baseline beyond acceptable overall tolerance.")
        success = False
    else:
        print("Overall new mAP is within acceptable overall tolerance.")

    if success :
        print("\nRegression test succeeded: New mAP is within acceptable tolerance.")
        sys.exit(0)
    else:
        print("\nRegression test failed: Some metrics are worse than baseline beyond acceptable tolerance.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <original_csv> <new_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
