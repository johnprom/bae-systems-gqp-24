#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:47:32 2024

@author: dfox
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil
import sys

from kneed import KneeLocator
from PIL import Image

# from util.util import get_preprocessed_images_dir_path, calc_degradation_factor
from eval.eval import run_eval, update_results

exts = ('.tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')  # Common image file extensions
label_ext = '.txt'

def replace_image_ext_with_label(filepath):
    path = Path(filepath)

    if path.suffix.lower() in exts:
        return path.with_suffix(label_ext)

    return filepath

def degrade_images(ctxt, orig_image_size, degraded_image_size, degraded_dir, corrupted_counter):
    """
    Degrade images by resizing them and store them in the degraded_dir.
    For testing, this method is currently only copying images from baseline_dir to degraded_dir.
    """
    # orig_image_size and degraded_image_size are tuples of (width, height)
    # config = ctxt.get_pipeline_config()
    # data_config_path = ctxt.get_data_config_dir_path()

    print(f"degrade images {orig_image_size} -> {degraded_image_size} in degraded directory {degraded_dir}", flush=True)
    config = ctxt.config

    method = config['preprocess_method']
    # maxwidth, maxheight = find_maxres(ctxt.input_images_dir, method) # TODO: use
    preprocess_top_dir = ctxt.get_preprocessing_dir_path()
    val_template = config['preprocess_methods'][method]['val_baseline_subdir']
    
    # maxwidth = config['preprocess_methods'][method]['image_size']
    # maxheight = maxwidth
    maxwidth = orig_image_size[0]
    maxheight = orig_image_size[1]

    if method == 'padding':
        # ctxt.train_baseline_dir = os.path.join(preprocess_top_dir, train_template.format(maxwidth=maxwidth, maxheight=maxheight))
        val_baseline_dir = os.path.join(preprocess_top_dir, val_template.format(maxwidth=maxwidth, maxheight=maxheight))
    elif method == 'tiling':
        stride = config['preprocess_methods'][method]['stride']
        val_baseline_dir = os.path.join(preprocess_top_dir, 
                                        val_template.format(maxwidth=maxwidth, maxheight=maxheight, stride=stride))
    else:
        raise ValueError("Unknown preprocessing method: " + method)

    # baseline_dir = ctxt.val_baseline_dir
    os.makedirs(degraded_dir, exist_ok=True)
    
    if ctxt.val_image_filename_set is None:
        val_image_filenames = [os.path.join(val_baseline_dir, x) for x in os.listdir(val_baseline_dir) if x.endswith(exts)]
        ctxt.val_image_filename_set = set(val_image_filenames)
    else:
        val_image_filenames = list(ctxt.val_image_filename_set)
    num_images = len(val_image_filenames)
    val_label_filenames = [replace_image_ext_with_label(fn) for fn in val_image_filenames]
    print(f"val_image_filenames {val_image_filenames}")
    print(f"num_images {num_images}", flush=True)
    for idx, val_image_filename in enumerate(val_image_filenames):
        val_label_filename = val_label_filenames[idx]
        if os.path.exists(val_image_filename) and os.path.exists(val_label_filename):
            val_degraded_image_filename = os.path.join(degraded_dir, os.path.basename(val_image_filename))
            val_degraded_label_filename = os.path.join(degraded_dir, os.path.basename(val_label_filename))
            print(f"degraded image name {val_degraded_image_filename}")
            if not os.path.exists(val_degraded_image_filename):
                if orig_image_size == degraded_image_size:
                    # resolutions and hyperparameters are equal, simple copy
                    shutil.copyfile(val_image_filename, val_degraded_image_filename)
                else:
                    try:
                        print(degraded_image_size, val_image_filename, flush=True)
                        image = Image.open(val_image_filename)
                        val_shrunk_image = image.resize(degraded_image_size)
                        val_degraded_image = val_shrunk_image.resize(orig_image_size)
                        val_degraded_image.save(val_degraded_image_filename) 
                        print(val_degraded_image_filename, flush=True)
                        image.close()
                        val_shrunk_image.close()
                        val_degraded_image.close()
                    except OSError:
                        ctxt.val_image_filename_set.discard(val_image_filename)
                        corrupted_counter += 1
                        continue
                if not os.path.exists(val_degraded_label_filename):
                    shutil.copyfile(val_label_filename, val_degraded_label_filename)
        else:
            ctxt.val_image_filename_set.discard(val_image_filename)
        
    return num_images, corrupted_counter

def run_eval_on_initial_resolutions(ctxt):
    """
    Evaluate the model on the baseline resolution and log the results.
    """
    config = ctxt.get_pipeline_config()
    preprocess_method = config['preprocess_method']
    pp_params = config['preprocess_methods'][preprocess_method]
    width = pp_params['image_size']
    height = pp_params['image_size']
   
    # Run evaluation on the baseline resolution
    baseline_dir = ctxt.val_baseline_dir
    mAP = run_eval(ctxt, (width, height), (width, height), baseline_dir)
    # update_results(ctxt, (width, height), (width, height), mAP)
    # Note: No need to call update_results() here since run_eval() already calls it.

def run_eval_on_degraded_images(ctxt):
    if ctxt.verbose:
        print("Start run_eval_on_degraded_images")
    config = ctxt.get_pipeline_config()
    preprocess_method = config['preprocess_method']
    pp_params = config['preprocess_methods'][preprocess_method]
    kd_params = config['knee_discovery']
    val_template = pp_params['val_degraded_subdir']

    width = pp_params['image_size']
    height = pp_params['image_size']
    if preprocess_method == 'tiling':
        stride = pp_params['stride']
    else:
        stride = None
    
    
    longer = max(width, height)
    shorter = min(width, height)
    shorter_mult = shorter / longer

    long_low_range = math.ceil(kd_params['search_resolution_range'][0] * longer)
    long_high_range = math.ceil(kd_params['search_resolution_range'][1] * longer) + 1
    step = math.floor(kd_params['search_resolution_step'] * longer)

    # results = []

    if ctxt.verbose:
        print(f"degrading images from {long_low_range} to {long_high_range} step {step}")
    corrupted_counter = 0
    max_images = 0
    for degraded_long_res in range(long_low_range, long_high_range, step):
        degraded_short_res = math.ceil(shorter_mult * degraded_long_res)
        degraded_width = degraded_long_res if width == longer else degraded_short_res
        degraded_height = degraded_short_res if height == longer else degraded_long_res

        # Create directory path for degraded images
        val_degraded_dir = os.path.join(ctxt.get_preprocessing_dir_path(), val_template.format(
            maxwidth=width, maxheight=height, effective_width=degraded_width, effective_height=degraded_height, stride=stride))

        # Degrade images and run evaluation
        num_images, corrupted_counter = degrade_images(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir, 
                                                       corrupted_counter)
        if max_images == 0:
            max_images = num_images
        # TODO: SHUBHAM: I think you will implement your knee discovery algorithm around here
        # Note that we are optimizing for detection of ONE class, which means you would have to
        mAP_list = run_eval(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir)

    if corrupted_counter > 0:
        print(f"{corrupted_counter} out of {max_images} images are corrupted!")
    

def calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h):
    
    degradation_factor_w = eff_res_w / orig_res_w
    degradation_factor_h = eff_res_h / orig_res_h
    degradation_factor_area = degradation_factor_w * degradation_factor_h
    degradation_factor = degradation_factor_area.apply(math.sqrt)
    return degradation_factor # pd.Series

def calculate_knee(class_name, results_class_df):
    """
    Calculate the knee in the IAPC curve using mAP values.
    Args:
    - results: A list of dictionaries with 'degraded_width' and 'mAP' keys.

    Returns:
    - knee_resolution: The resolution at which the knee occurs.
    """
    orig_res_w = results_class_df['original_resolution_width'].astype(float)
    orig_res_h = results_class_df['original_resolution_height'].astype(float)
    eff_res_w = results_class_df['effective_resolution_width'].astype(float)
    eff_res_h = results_class_df['effective_resolution_height'].astype(float)
    mAP_values_series = results_class_df['mAP']
    degradation_factor_series = calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h)
    degradation_factor_list = degradation_factor_series.to_list()
    
    mAP_values = mAP_values_series.to_list()
    # if math.isclose(a, b, abs_tol=1e-7):

    # iapc_values = [1/mAP for mAP in mAP_values]

    # if all mAP are zero, return any value from degradation factor list, let's pick the minimum
    if all(mAP == 0.0 for mAP in mAP_values):
        return None
    
    print(class_name)
    print(degradation_factor_list)
    print(mAP_values)

    kneedle = KneeLocator(degradation_factor_list, mAP_values, curve='concave', direction='increasing')
    knee_degradation_factor = kneedle.knee
    
    return knee_degradation_factor


def plot_iapc_curve(results_class_df, class_name):
    """
    Plot the IAPC curve for the given results and highlight the knee point.
    """
    orig_res_w = results_class_df['original_resolution_width'].astype(float)
    orig_res_h = results_class_df['original_resolution_height'].astype(float)
    eff_res_w = results_class_df['effective_resolution_width'].astype(float)
    eff_res_h = results_class_df['effective_resolution_height'].astype(float)
    degradation_factor_series = calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h)

    mAP_values = results_class_df['mAP'].values
    degradation_factors = degradation_factor_series.values

    # Find the knee
    kneedle = KneeLocator(degradation_factors, mAP_values, curve='concave', direction='decreasing')
    knee_degradation_factor = kneedle.knee

    # Plotting the IAPC curve
    plt.figure(figsize=(10, 6))
    plt.plot(degradation_factors, mAP_values, marker='o', label=f'{class_name} IAPC')
    if knee_degradation_factor:
        plt.axvline(x=knee_degradation_factor, color='r', linestyle='--', label=f'Knee at {knee_degradation_factor}')
    plt.xlabel('Degradation Factor')
    plt.ylabel('Mean Average Precision (mAP)')
    plt.title(f'IAPC Curve for {class_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def run_knee_discovery(ctxt):
    print("Start with run_knee_discovery")
    
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    eval_results_filename = os.path.join(results_path, config['knee_discovery']['eval_results_filename'])
    
    if 'clean_subdir' in config['knee_discovery'] and config['knee_discovery']['clean_subdir']:
        if os.path.exists(results_path):
            shutil.rmtree(results_path)

    os.makedirs(results_path, exist_ok=True)
    
    # Run evaluation on initial (baseline) resolutions
    run_eval_on_degraded_images(ctxt)

    # Load the results from CSV (or from the cached results in memory)
    if ctxt.results_cache_df is not None:
        results_df = ctxt.results_cache_df
    else:
        if os.path.exists(eval_results_filename):
            results_df = pd.read_csv(eval_results_filename, index_col=False)
        else: 
            print(f"Knee discovery: {eval_results_filename} not found!")
            return
        # results = results_df.to_dict('records')

    ## Ensure degradation_factor is calculated
    if 'degradation_factor' not in results_df.columns:
        orig_res_w = results_df['original_resolution_width'].astype(float)
        orig_res_h = results_df['original_resolution_height'].astype(float)
        eff_res_w = results_df['effective_resolution_width'].astype(float)
        eff_res_h = results_df['effective_resolution_height'].astype(float)
        results_df['degradation_factor'] = calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h)

    # Ensure 'knee' column exists
    if 'knee' not in results_df.columns:
        results_df['knee'] = False

    class_names = results_df['object_name'].unique()

    # Set the desired convergence tolerance and maximum iterations
    desired_tolerance = 1e-2  # Adjust this value based on your needs
    mAP_tolerance = 1e-3      # Tolerance for mAP change convergence
    max_iterations = 10       # Maximum number of iterations to prevent excessive computations

    results_df['knee'] = False
    for class_name in class_names:
        results_class_df = results_df[results_df['object_name'] == class_name].copy()
        knee_degradation_factor = calculate_knee(class_name, results_class_df)
        print(f"knee_degradation_factor {knee_degradation_factor}")
        
        # # Initialize variables for convergence checking
        # new_knee_degradation_factor = knee_degradation_factor
        # old_knee_degradation_factor = None
        # new_knee_mAP = None
        # old_knee_mAP = None
        # iteration = 0

        # while (iteration < max_iterations and new_knee_degradation_factor is not None and
        #        (old_knee_degradation_factor is None or not np.isclose(new_knee_degradation_factor, old_knee_degradation_factor, atol=desired_tolerance))):
        #     iteration += 1
        #     old_knee_degradation_factor = new_knee_degradation_factor

        #     # Implement the knee discovery algorithm here
        #     # Find neighboring degradation factors
        #     degradation_factors = results_class_df['degradation_factor'].values
        #     mAP_values = results_class_df['mAP'].values

        #     sorted_indices = np.argsort(degradation_factors)
        #     degradation_factors_sorted = degradation_factors[sorted_indices]
        #     mAP_values_sorted = mAP_values[sorted_indices]

        #     # Find index of current knee degradation factor
        #     knee_index = np.where(np.isclose(degradation_factors_sorted, new_knee_degradation_factor, atol=1e-5))[0]

        #     if knee_index.size == 0:
        #         print(f"Knee degradation factor {new_knee_degradation_factor} not found in degradation factors.")
        #         break

        #     knee_index = knee_index[0]

        #     # Get mAP at the knee degradation factor
        #     new_knee_mAP = mAP_values_sorted[knee_index]

        #     # If old_knee_mAP is set, check mAP change for convergence
        #     if old_knee_mAP is not None:
        #         mAP_change = abs(new_knee_mAP - old_knee_mAP)
        #         if mAP_change < mAP_tolerance:
        #             print(f"Convergence achieved based on mAP change for class {class_name}.")
        #             break

        #     old_knee_mAP = new_knee_mAP

        #     # Find left and right indices
        #     if knee_index > 0:
        #         left_index = knee_index - 1
        #         left_degradation = degradation_factors_sorted[left_index]
        #         left_mAP = mAP_values_sorted[left_index]
        #         delta_mAP_left = abs(new_knee_mAP - left_mAP)
        #     else:
        #         left_degradation = None
        #         delta_mAP_left = 0

        #     if knee_index < len(degradation_factors_sorted) - 1:
        #         right_index = knee_index + 1
        #         right_degradation = degradation_factors_sorted[right_index]
        #         right_mAP = mAP_values_sorted[right_index]
        #         delta_mAP_right = abs(new_knee_mAP - right_mAP)
        #     else:
        #         right_degradation = None
        #         delta_mAP_right = 0

        #     # Decide which side has higher mAP difference
        #     if delta_mAP_left > delta_mAP_right and left_degradation is not None:
        #         # Choose left side
        #         degradation1 = left_degradation
        #         degradation2 = new_knee_degradation_factor
        #     elif right_degradation is not None:
        #         # Choose right side
        #         degradation1 = new_knee_degradation_factor
        #         degradation2 = right_degradation
        #     else:
        #         # Cannot refine further
        #         print("No further refinement possible.")
        #         break

        #     # Calculate new degradation factor between selected points
        #     new_degradation = (degradation1 + degradation2) / 2

        #     # Check if this degradation factor is already in our data
        #     if np.any(np.isclose(degradation_factors, new_degradation, atol=1e-4)):
        #         # This degradation factor is already in the data, cannot proceed
        #         print(f"Degradation factor {new_degradation} already evaluated.")
        #         break

        #     # Convert degradation factor to degraded width and height
        #     width = results_class_df['original_resolution_width'].iloc[0]
        #     height = results_class_df['original_resolution_height'].iloc[0]
        #     degraded_width = int(new_degradation * width)
        #     degraded_height = int(new_degradation * height)

        #     # Ensure degraded dimensions are within valid ranges
        #     degraded_width = max(1, min(int(width), degraded_width))
        #     degraded_height = max(1, min(int(height), degraded_height))

        #     # Degrade images at this new resolution and run evaluation
        #     preprocess_method = config['preprocess_method']
        #     pp_params = config['preprocess_methods'][preprocess_method]
        #     val_template = pp_params['val_degraded_subdir']
        #     stride = pp_params.get('stride', None)
        #     val_degraded_dir = os.path.join(ctxt.get_preprocessing_dir_path(), val_template.format(
        #         maxwidth=width, maxheight=height, effective_width=degraded_width, effective_height=degraded_height, stride=stride))

        #     num_images, corrupted_counter = degrade_images(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir, 0)
        #     mAP_list = run_eval(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir)
        #     # Note: No need to call update_results() here since run_eval() already calls it.

        #     # After updating results, refresh results_df
        #     if ctxt.results_cache_df is not None:
        #         results_df = ctxt.results_cache_df
        #     else:
        #         if os.path.exists(eval_results_filename):
        #             results_df = pd.read_csv(eval_results_filename, index_col=False)
        #         else:
        #             print(f"Knee discovery: {eval_results_filename} not found!")
        #             break

        #     results_class_df = results_df[results_df['object_name'] == class_name]

        #     # Recalculate the knee
        #     new_knee_degradation_factor = calculate_knee(class_name, results_class_df)

        # Mark the knee point
        # results_df.loc[results_df['object_name'] == class_name, 'knee'] = False
        # if new_knee_degradation_factor is not None:
        #     df = results_class_df[np.isclose(results_class_df['degradation_factor'], new_knee_degradation_factor, atol=1e-5)]
        #     results_df.loc[df.index, 'knee'] = True

        #     print(f"Knee discovered at degradation factor {new_knee_degradation_factor} for {class_name}")

        #     # Plot the IAPC curve for the class
        #    # plot_iapc_curve(results_class_df, class_name)
        # else:
        #     print(f"No knee found for class {class_name}")

        # Mark the knee point
        if knee_degradation_factor is not None:
            df = results_class_df[np.isclose(results_class_df['degradation_factor'], knee_degradation_factor, atol=1e-5)]
            results_df.loc[df.index, 'knee'] = True

            print(f"Knee discovered at degradation factor {knee_degradation_factor} for {class_name}")

            # Plot the IAPC curve for the class
            # plot_iapc_curve(results_class_df, class_name)
        else:
            print(f"No knee found for class {class_name}")

    results_df.to_csv(eval_results_filename, index=False)
    if ctxt.cache_results:
        ctxt.results_cache_df = results_df.copy()

    print("End knee discovery")
