#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:21:55 2024

@author: dfox
"""
import math
import os

def get_model(config):
    if 'model' not in config or 'models' not in config:
        raise ValueError("model not configured")
        
    model_name = config['model']
    if model_name not in config['models']:
        raise ValueError(f"model {model_name} not configured")
        
    model_dict = config['models'][model_name]
    
    return model_dict

# TODO: KENDALL: Write this method. No worries about changing anything. Just make it work.
# This method calculates ONE data point on the IAPC
def run_eval(ctxt, baseline_image_size, degraded_image_size, val_degraded_dir_path):
    """
    Run the model evaluation at a specific resolution and calculate mAP.
    
    Args:
    - baseline_image_size: Tuple of (width, height) of the baseline images.
    - degraded_image_size: Tuple of (width, height) of the degraded images.
    - val_degraded_dir_path: Path to the validation images for the current resolution.
    
    Returns:
    - mAP: Mean Average Precision of the model for this resolution.
    """
    
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()
    top_dir = ctxt.get_top_dir()
    
    # Get the trained model and settings
    model_dict = get_model(config)
    input_image_size = model_dict['input_image_size']
    input_width = input_image_size
    input_height = input_image_size
    num_epochs = model_dict['epochs']
    
    trained_model_filename_template = model_dict['trained_model_filename']
    trained_model_filename = os.path.join(top_dir, config['trained_models_subdir'],
                                          trained_model_filename_template.format(
                                              width=baseline_image_size[0], height=baseline_image_size[1], epochs=num_epochs))
    
    # Load the trained model for evaluation
    model = YOLO(trained_model_filename)  # Load the model using the path to the fine-tuned model
    
    # Run evaluation
    results = model.val(data=data_config_path, imgsz=degraded_image_size)  # Evaluate on degraded images
    
    # Retrieve mAP from the evaluation results
    mAP = results['map']  # Assuming 'map' key contains the mean Average Precision
    
    # Log results to a structured file (for future analysis)
    update_results(ctxt, baseline_image_size, degraded_image_size, mAP)
    
    return mAP  # Return the mAP value for this resolution

def update_results(ctxt, orig_image_size, degraded_image_size, mAP):
    """
    Log the evaluation results (mAP and image sizes) for knee discovery.
    
    Args:
    - orig_image_size: Tuple of (width, height) of the original images.
    - degraded_image_size: Tuple of (width, height) of the degraded images.
    - mAP: Mean Average Precision of the model for this resolution.
    """
    
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    os.makedirs(results_path, exist_ok=True)

    # Log results to the results file (CSV or text)
    iapc_results_filename = os.path.join(results_path, "iapc_results.csv")
    
    # Append results (resolution and mAP) to CSV file
    with open(iapc_results_filename, 'a') as file:
        file.write(f"{orig_image_size[0]}x{orig_image_size[1]}, {degraded_image_size[0]}x{degraded_image_size[1]}, {mAP}\n")
    
    print(f"Logged IAPC results: Original {orig_image_size}, Degraded {degraded_image_size}, mAP {mAP}")

# This function is used to calculate the knee after multiple runs
def calculate_knee(ctxt, iapc_results_filename):
    """
    Read the IAPC results and calculate the knee using the Kneedle algorithm.
    
    Args:
    - ctxt: The pipeline context object.
    - iapc_results_filename: Path to the file containing IAPC results (resolution vs. mAP).
    
    Returns:
    - knee_point: The resolution at which the knee is detected.
    """
    from kneed import KneeLocator
    import pandas as pd
    
    # Read the results into a pandas DataFrame
    results_df = pd.read_csv(iapc_results_filename, header=None, names=['orig_resolution', 'degraded_resolution', 'mAP'])
    
    # Extract the degraded resolutions and mAP values
    degraded_resolutions = results_df['degraded_resolution'].apply(lambda x: int(x.split('x')[0]))  # Take the width as resolution
    iapc_values = 1 / results_df['mAP']  # Inverse of mAP for IAPC
    
    # Use Kneedle algorithm to find the knee
    kneedle = KneeLocator(degraded_resolutions, iapc_values, curve='convex', direction='increasing')
    
    # Get the knee point
    knee_point = kneedle.knee
    print(f"Knee detected at resolution {knee_point}")
    
    # Optionally log the knee point to a file 
    with open(os.path.join(ctxt.get_output_dir_path(), "knee_results.txt"), 'a') as knee_file:
        knee_file.write(f"Knee detected at resolution {knee_point}\n")
    
    return knee_point

    # TODO: KENDALL: Here run the evaluation code, calculating ONE data point on the IAPC
    # method = config['eval_method'] # detection or classification, only detection supported right now, so maybe leave this
    # comment in here or take this comment out entirely
    # results = run model here, stored in traine_model_filename, config['target_labels'], put result into trained_modelfilename
    # eval_detection
    # write results to some structured file, coordinate with DAN
    # ALSO, could cache the results in ctxt (class Pipeline in src/pipeline.py). That would be very useful.
    


