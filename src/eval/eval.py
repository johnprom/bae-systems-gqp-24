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

from util.util import load_pipeline_config, update_data_config_val_path
# from knee_discovery.knee_discovery import calc_degradation_factor

def get_model(config):
    """
    Retrieves the configuration dictionary for a specified model.

    Args:
        config (dict): The configuration dictionary that includes model settings.
    
    Raises:
        ValueError: If 'model' or 'models' is not present in the configuration,
                    or if the specified model name is not found within 'models'.
    
    Returns:
        dict: A dictionary of parameters and settings for the specified model.
    """

    if 'model' not in config or 'models' not in config:
        raise ValueError("model not configured")
        
    model_name = config['model']
    if model_name not in config['models']:
        raise ValueError(f"model {model_name} not configured")
        
    model_dict = config['models'][model_name]
    
    return model_dict

def drop_dup_results(ctxt, rcdf):
    """
    Removes duplicate entries in the results DataFrame based on a generated key.

    Args:
        ctxt: The pipeline context object.
        rcdf (pd.DataFrame): DataFrame containing evaluation results.
    
    Returns:
        pd.DataFrame: A DataFrame with duplicate rows removed based on the unique key.
    """
    temp_rcdf = rcdf.copy()
    # temp_rcdf['mAP_rounded'] = temp_rcdf['mAP'].round(3)  # Adjust the decimal places as needed
    temp_rcdf['key'] = (temp_rcdf['object_name'] + '_' + temp_rcdf['original_resolution_width'].astype(str) + '_'
                        + temp_rcdf['original_resolution_height'].astype(str) + '_'
                        + temp_rcdf['degradation_factor'].round(6).astype(str))
    temp_rcdf = temp_rcdf.drop_duplicates(subset='key', keep='last')
    temp_rcdf = temp_rcdf.drop('key', axis=1)
    return temp_rcdf

def update_results(ctxt, num_names, name_list, orig_image_size, degraded_image_size, mAP_list, is_knee):
    """
    Logs evaluation results (mAP and image sizes) for knee discovery.

    Args:
        ctxt: The pipeline context object.
        num_names (int): Number of object classes.
        name_list (list): List of object class names.
        orig_image_size (tuple): Tuple of (width, height) of the original images.
        degraded_image_size (tuple): Tuple of (width, height) of the degraded images.
        mAP_list (list): List of Mean Average Precision values for each object class.
        is_knee (bool): Flag indicating if the current evaluation is a knee point in knee discovery.
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
            if os.path.exists(eval_results_filename):
                rcdf = pd.read_csv(eval_results_filename, index_col=False)
            else:
                rcdf = pd.DataFrame(columns=ctxt.iapc_columns)
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
        original_gsd = config['pixel_size']
        gsd_per_pixel = original_gsd / degradation_factor
        class_pixel_area = ctxt.object_sizes.get(name_list[idx], 1)
        pixels_on_target = math.ceil((original_gsd ** 2) * class_pixel_area / (gsd_per_pixel ** 2))

        # degradation_factor = calc_degradation_factor(orig_image_size, orig_image_size, degraded_image_size, degraded_image_size)
        # rcdf_list = [name_list[idx], orig_image_size[0], orig_image_size[1], degraded_image_size[0], 
        #                            degraded_image_size[1], mAP_list[idx], degradation_factor, is_knee]
        rcdf.loc[rcdf.shape[0]] = [name_list[idx], orig_image_size[0], orig_image_size[1], degraded_image_size[0], 
                                   degraded_image_size[1], mAP_list[idx], degradation_factor, gsd_per_pixel, pixels_on_target, is_knee]
        if ctxt.verbose:
            print(f"Logged IAPC results: Object class {name_list[idx]}, Original {orig_image_size}, Degraded {degraded_image_size}, "
                  + f"mAP {mAP_list[idx]}, knee {is_knee}")
    
    
    rcdf = drop_dup_results(ctxt, rcdf)
    rcdf.to_csv(eval_results_filename, index=False)
    if ctxt.cache_results:
        ctxt.results_cache_df = rcdf.copy()
    
def update_knee_results(ctxt, name, orig_image_size, degradation_factor, mAP):
    """
    Logs specific evaluation results (mAP and image sizes) for a knee discovery point.

    Args:
        ctxt: The pipeline context object.
        name (str): Object class name.
        orig_image_size (tuple): Tuple of (width, height) of the original images.
        degradation_factor (float): Factor by which the image is degraded.
        mAP (float): Mean Average Precision of the model for the given resolution.
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
            if os.path.exists(eval_results_filename):
                rcdf = pd.read_csv(eval_results_filename, index_col=False)
            else:
                rcdf = pd.DataFrame(columns=ctxt.iapc_columns)
        else:
            rcdf = ctxt.results_cache_df.copy()
    elif os.path.exists(eval_results_filename):
        rcdf = pd.read_csv(eval_results_filename, index_col=False)
    else:
        rcdf = pd.DataFrame(columns=ctxt.iapc_columns)
        
    degraded_width = math.ceil(orig_image_size[0] * degradation_factor)
    degraded_height = math.ceil(orig_image_size[1] * degradation_factor)

    original_gsd = config['pixel_size']
    gsd_per_pixel = original_gsd / degradation_factor

    class_pixel_area = ctxt.object_sizes.get(name, 1)
    pixels_on_target = math.ceil((original_gsd ** 2) * class_pixel_area / (gsd_per_pixel ** 2))

    rcdf.loc[rcdf.shape[0]] = [name, orig_image_size[0], orig_image_size[1], degraded_width, 
                               degraded_height, mAP, degradation_factor, gsd_per_pixel, pixels_on_target, True]
    
    # rcdf = rcdf.sort_values(['object_name', 'degradation_factor'])
    
    if ctxt.verbose:
        print(f"Logged IAPC results: Object class {name}, Original {orig_image_size}, "
              + f"Degraded ({degraded_width}, {degraded_height}), mAP {mAP}, knee True")
    
    
    rcdf = drop_dup_results(ctxt, rcdf)
    rcdf.to_csv(eval_results_filename, index=False)
    if ctxt.cache_results:
        ctxt.results_cache_df = rcdf.copy()
    
def calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h):
    
    """
    Calculates the degradation factor based on original and effective resolutions.

    Args:
        orig_res_w (float): Original resolution width.
        orig_res_h (float): Original resolution height.
        eff_res_w (float): Effective resolution width.
        eff_res_h (float): Effective resolution height.
    
    Returns:
        float: The degradation factor calculated from the resolutions.
    """
    
    degradation_factor_w = eff_res_w / orig_res_w
    degradation_factor_h = eff_res_h / orig_res_h
    degradation_factor = math.sqrt(degradation_factor_w * degradation_factor_h)
    # IAPC_df['degradation_factor'] = degradation_factor
    return degradation_factor # pd.Series

def run_eval(ctxt, baseline_image_size, degraded_image_size, val_degraded_dir_path, knee):
    """
    Runs the model evaluation at a specific resolution and calculates mAP.

    Args:
        ctxt: The pipeline context object.
        baseline_image_size (tuple): Tuple of (width, height) of the baseline images.
        degraded_image_size (tuple): Tuple of (width, height) of the degraded images.
        val_degraded_dir_path (str): Path to the validation images for the current resolution.
        knee (bool): Indicator for knee point in knee discovery.
    
    Returns:
        None
    """
    
    update_data_config_val_path(ctxt, val_degraded_dir_path)
    
    data_config_path = ctxt.get_data_config_dir_path()
    data_config = load_pipeline_config(data_config_path)
    
    if ctxt.final_weights_path is None or ctxt.final_weights_path == "" or not os.path.exists(ctxt.final_weights_path):
        model_to_use = ctxt.get_model_name()
    else:
        model_to_use = ctxt.final_weights_path
    model = YOLO(model_to_use)
    if ctxt.use_cuda:
        model.to('cuda')
        if ctxt.verbose:
            print("Added cuda to the model")
            
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    os.makedirs(results_path, exist_ok=True)
    
    run_name = f"val_{degraded_image_size[0]}_{degraded_image_size[1]}"
    if isinstance(knee, bool) and knee:
        run_name += "_knee"

    # Run evaluation
    results = model.val(data=data_config_path, imgsz=list(baseline_image_size), cache=ctxt.use_eval_cache(),
                        save_json=True, project=results_path, name=run_name)
    
    # Retrieve mAP from the evaluation results
    mAP_list = list(results.box.maps)  # Access mAP for object detection
    if ctxt.verbose:
        print(mAP_list)
    
    # if all mAP are zero, return any value from degradation factor list, let's pick the minimum
    if len(mAP_list) == 0:
        mAP_list = [0.0] * len(model.names)

    # log results to a structured file 
    update_results(ctxt, data_config['nc'], model.names, baseline_image_size, degraded_image_size, mAP_list, knee)
    
    # return mAP_list  # no need to return this
