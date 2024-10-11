#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:21:55 2024

@author: dfox
"""
import math
import os
import pandas as pd
from ultralytics import YOLO

from util.util import load_pipeline_config
# from knee_discovery.knee_discovery import calc_degradation_factor

def get_model(config):
    if 'model' not in config or 'models' not in config:
        raise ValueError("model not configured")
        
    model_name = config['model']
    if model_name not in config['models']:
        raise ValueError(f"model {model_name} not configured")
        
    model_dict = config['models'][model_name]
    
    return model_dict

def update_results(ctxt, num_names, name_list, orig_image_size, degraded_image_size, mAP_list, is_knee):
    """
    Log the evaluation results (mAP and image sizes) for knee discovery.
    
    Args:
    - orig_image_size: Tuple of (width, height) of the original images.
    - degraded_image_size: Tuple of (width, height) of the degraded images.
    - mAP: Mean Average Precision of the model for this resolution.
    """
    
    # IAPC results file columns
    # self.iapc_columns = ['object_name', 'original_resolution_width', 'original_resolution_height', 'effective_resolution_width',
    #                      'effective_resolution_height', 'mAP', 'degradation_factor', 'knee']
    
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    os.makedirs(results_path, exist_ok=True)

    # Log results to the results file (CSV or text)
    eval_results_filename = os.path.join(results_path, config['knee_discovery']['eval_results_filename'])

    if ctxt.cache_results:
        if ctxt.results_cache_df is None:
            ctxt.results_cache_df = pd.DataFrame(columns=ctxt.iapc_columns)
            rcdf = ctxt.results_cache_df.copy()
        else:
            rcdf = ctxt.results_cache_df.copy()
    elif os.path.exists(eval_results_filename):
        rcdf = pd.read_csv(eval_results_filename, index_col=False)
    else:
        rcdf = pd.DataFrame(columns=ctxt.iapc_columns)
        
    for idx in range(num_names):
    # for idx, name in enumerate(name_list):
        degradation_factor = calc_degradation_factor(orig_image_size[0], orig_image_size[1],
                                                     degraded_image_size[0], degraded_image_size[1])
        # degradation_factor = calc_degradation_factor(orig_image_size, orig_image_size, degraded_image_size, degraded_image_size)
        rcdf_list = [name_list[idx], orig_image_size[0], orig_image_size[1], degraded_image_size[0], 
                                   degraded_image_size[1], mAP_list[idx], degradation_factor, is_knee]
        print(f"rcdf list length {len(rcdf_list)}, iapc columns length {len(ctxt.iapc_columns)}, rcdf length {rcdf.shape[1]}")
        rcdf.loc[rcdf.shape[0]] = [name_list[idx], orig_image_size[0], orig_image_size[1], degraded_image_size[0], 
                                   degraded_image_size[1], mAP_list[idx], degradation_factor, is_knee]
        print(f"Logged IAPC results: Object class {name_list[idx]}, Original {orig_image_size}, Degraded {degraded_image_size}, "
              + f"mAP {mAP_list[idx]}, knee {is_knee}")
    
    
    rcdf.to_csv(eval_results_filename, index=False)
    ctxt.results_cache_df = rcdf.copy()
    
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
    results_df = pd.read_csv(iapc_results_filename, header=None, names=['orig_resolution', 'degraded_resolution', 'mAP'], 
                             index_col=False)
    
    # Extract the degraded resolutions and mAP values
    degraded_resolutions = results_df['degraded_resolution'].apply(lambda x: int(x.split('x')[0]))  # Take the width as resolution
    iapc_values = 1 / results_df['mAP']  # Inverse of mAP for IAPC
    
    # Use Kneedle algorithm to find the knee
    kneedle = KneeLocator(degraded_resolutions, iapc_values, curve='convex', direction='increasing')
    
    # Get the knee point
    knee_point = kneedle.knee
    print(f"Knee detected at resolution {knee_point}")
    
    # Log the knee point to a file 
    with open(os.path.join(ctxt.get_output_dir_path(), "knee_results.txt"), 'a') as knee_file:
        knee_file.write(f"Knee detected at resolution {knee_point}\n")
    
    return knee_point

def calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h):
    # orig_res_w = IAPC_df['original_resolution_width'].astype(float)
    # orig_res_h = IAPC_df['original_resolution_height'].astype(float)
    # eff_res_w = IAPC_df['effective_resolution_width'].astype(float)
    # eff_res_h = IAPC_df['effective_resolution_height'].astype(float)
    
    degradation_factor_w = eff_res_w / orig_res_w
    degradation_factor_h = eff_res_h / orig_res_h
    degradation_factor = math.sqrt(degradation_factor_w * degradation_factor_h)
    # IAPC_df['degradation_factor'] = degradation_factor
    return degradation_factor # pd.Series

def run_eval(ctxt, baseline_image_size, degraded_image_size, val_degraded_dir_path):
    """
    Run the model evaluation at a specific resolution and calculate mAP.
    
    Args:
    - ctxt: The pipeline context object.
    - baseline_image_size: Tuple of (width, height) of the baseline images.
    - degraded_image_size: Tuple of (width, height) of the degraded images.
    - val_degraded_dir_path: Path to the validation images for the current resolution.
    
    Returns:
    - mAP: Mean Average Precision of the model for this resolution.
    """
    
    data_config_eval_path = ctxt.get_data_config_eval_dir_path()
    data_config_eval = load_pipeline_config(data_config_eval_path)
    
    if ctxt.final_weights_path is None or ctxt.final_weights_path == "" or not os.path.exists(ctxt.final_weights_path):
        model_to_use = ctxt.get_model_name()
    else:
        model_to_use = ctxt.final_weights_path
    model = YOLO(model_to_use)
    
    # Run evaluation
    results = model.val(data=data_config_eval_path, imgsz=list(baseline_image_size), cache=ctxt.use_eval_cache())
    
    # Retrieve mAP from the evaluation results
    print(f"type of maps return is {type(results.box.maps)}")
    print(f"{results.box.maps}")
    mAP_list = list(results.box.maps)  # Access mAP for object detection
    
    # log results to a structured file 
    update_results(ctxt, data_config_eval['nc'], model.names, baseline_image_size, degraded_image_size, mAP_list, "unknown")
    
    return mAP_list  # return the mAP value for resolution

