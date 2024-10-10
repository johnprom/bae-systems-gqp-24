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
import shutil
import sys

from kneed import KneeLocator
from PIL import Image

# from util.util import get_preprocessed_images_dir_path, calc_degradation_factor
from eval.eval import run_eval, update_results

exts = ('.tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')  # Common image file extensions

# def find_maxres(baseline_dir, method):
#     # result_directories = []
#     maxwidth = 0
#     maxheight = 0

#     for dirpath, _, filenames in os.walk(baseline_dir):
#         if any(file.lower().endswith(exts) for file in filenames):
#             path_parts = os.path.abspath(dirpath).split(os.sep)
#             if method == 'tiling':
#                 if len(path_parts) >= 2:
#                     print(path_parts)
#                     maxwidth = path_parts[-2].split('_')[0]
#                     maxheight = path_parts[-2].split('_')[1]
#             elif method == 'padding':
#                 if len(path_parts) >= 1:
#                     maxwidth = path_parts[-1].split('_')[0]
#                     maxheight = path_parts[-1].split('_')[1]
#             break 

#     return maxwidth, maxheight


def degrade_images(ctxt, orig_image_size, degraded_image_size, degraded_dir, corrupted_counter):
    """
    Degrade images by resizing them and store them in the degraded_dir.
    For testing, this method is currently only copying images from baseline_dir to degraded_dir.
    """
    # orig_image_size and degraded_image_size are tuples of (width, height)
    # config = ctxt.get_pipeline_config()
    # data_config_path = ctxt.get_data_config_dir_path()

    print(f"degrade images {orig_image_size} -> {degraded_image_size} in degraded directory {degraded_dir}")
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
    
    if ctxt.val_image_set is None:
        val_images = [os.path.join(val_baseline_dir, x) for x in os.listdir(val_baseline_dir) if x.endswith(exts)]
        ctxt.val_image_set = set()
    else:
        val_images = list(ctxt.val_image_set)
    num_images = len(val_images)
    for i, val_image in enumerate(val_images):
        try:
            # print(i, degraded_image_size, val_image)
            image = Image.open(val_image)
            val_shrunk_image = image.resize(degraded_image_size)
            val_degraded_image = val_shrunk_image.resize(orig_image_size)
            if i == 0:
                val_degraded_image.show()
            val_degraded_image.save(os.path.join(degraded_dir, val_image))
            image.close()
            val_shrunk_image.close()
            val_degraded_image.close()
        except OSError:
            ctxt.val_image_set.discard(val_image)
            corrupted_counter += 1
            continue
        ctxt.val_image_set.add(val_image)
        
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
    update_results(ctxt, (width, height), (width, height), mAP)


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
        # TODO: SHUBHAM: I think you will implement your knee discovery algorithm here
        mAP_list = run_eval(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir)

    #     # Store results in list
    #     results.append({
    #         'orig_width': width,
    #         'orig_height': height,
    #         'degraded_width': degraded_width,
    #         'degraded_height': degraded_height,
    #         'mAP': mAP
    #     })

    # # Store results in CSV
    # results_df = pd.DataFrame(results)
    # results_file = os.path.join(ctxt.get_output_dir_path(), 'iapc_results.csv')
    # results_df.to_csv(results_file, index=False)

    # return results

    if corrupted_counter > 0:
        print(f"{corrupted_counter} out of {max_images} images are corrupted!")
    

def mark_as_knee(ctxt, name, orig_image_size, degraded_image_size):
    pass

def calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h):
    # arguments are all pd.Series, as is return
    # orig_res_w = IAPC_df['original_resolution_width'].astype(float)
    # orig_res_h = IAPC_df['original_resolution_height'].astype(float)
    # eff_res_w = IAPC_df['effective_resolution_width'].astype(float)
    # eff_res_h = IAPC_df['effective_resolution_height'].astype(float)
    
    degradation_factor_w = eff_res_w / orig_res_w
    degradation_factor_h = eff_res_h / orig_res_h
    degradation_factor_area = degradation_factor_w * degradation_factor_h
    degradation_factor = degradation_factor_area.apply(math.sqrt)
    # IAPC_df['degradation_factor'] = degradation_factor
    return degradation_factor # pd.Series

def calculate_knee(class_name, results_class_df):
    """
    Calculate the knee in the IAPC curve using mAP values.
    Args:
    - results: A list of dictionaries with 'degraded_width' and 'mAP' keys.

    Returns:
    - knee_resolution: The resolution at which the knee occurs.
    """
    # resolutions = [r['degraded_width'] for r in results]
    # mAP_values = [r['mAP'] for r in results]
    
    # # Convert to IAPC (1/mAP)
    # iapc_values = [1/mAP if mAP != 0 else float('inf') for mAP in mAP_values]

    # resolutions = list(results_class_df['degraded_width'])
    orig_res_w = results_class_df['original_resolution_width'].astype(float)
    orig_res_h = results_class_df['original_resolution_height'].astype(float)
    eff_res_w = results_class_df['effective_resolution_width'].astype(float)
    eff_res_h = results_class_df['effective_resolution_height'].astype(float)
    degradation_factor_series = calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h)
    print(f"type(degradation_factor_series) is {type(degradation_factor_series)}, shape {degradation_factor_series.shape}")
    mAP_values_series = results_class_df['mAP']
    print(f"type(mAP_values_series) is {type(mAP_values_series)}, shape {mAP_values_series.shape}")
    degradation_factor_list = degradation_factor_series.to_list()
    print(f"type(degradation_factor_list) is {type(degradation_factor_list)}, length {len(degradation_factor_list)}")
    
    mAP_values = results_class_df['mAP'].to_list()
    print(f"type(mAP_values) is {type(mAP_values)}, length {len(mAP_values)}")
    # if math.isclose(a, b, abs_tol=1e-7):

    # iapc_values = [1/mAP for mAP in mAP_values]

    # Use KneeLocator to find the knee
    kneedle = KneeLocator(degradation_factor_list, mAP_values, curve='convex', direction='increasing')
    knee_degradation_factor = kneedle.knee
    
    return knee_degradation_factor


def plot_iapc_curve(results_class_df, class_name):
    """
    Plot the IAPC curve for the given results and highlight the knee point.
    """
    # resolutions = [r['degraded_width'] for r in results]
    # mAP_values = [r['mAP'] for r in results]
    # iapc_values = [1/mAP if mAP != 0 else float('inf') for mAP in mAP_values]

    # resolutions = list(results_class_df['degraded_width'])
    # degradation_factor = calc_degradation_factor(results_class_df)
    orig_res_w = results_class_df['original_resolution_width'].astype(float)
    orig_res_h = results_class_df['original_resolution_height'].astype(float)
    eff_res_w = results_class_df['effective_resolution_width'].astype(float)
    eff_res_h = results_class_df['effective_resolution_height'].astype(float)
    degradation_factor = calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h)

    mAP_values = list(results_class_df['mAP'])
    # iapc_values = [1/mAP for mAP in mAP_values]

    # Find the knee
    kneedle = KneeLocator(degradation_factor, mAP_values, curve='convex', direction='increasing')
    knee_resolution = kneedle.knee

    # Plotting the IAPC curve
    plt.figure(figsize=(10, 6))
    plt.plot(degradation_factor, mAP_values, marker='o', label=f'{class_name} IAPC')
    if knee_resolution:
        plt.axvline(x=knee_resolution, color='r', linestyle='--', label=f'Knee at {knee_resolution}')
    plt.xlabel('Resolution')
    plt.ylabel('Inverse Average Precision (1/mAP)')
    plt.title(f'IAPC Curve for {class_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def run_knee_discovery(ctxt):
    print("Start with run_knee_discovery")
    
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    
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
        eval_results_filename = os.path.join(results_path, config['knee_discovery']['eval_results_filename'])
        if os.path.exists(eval_results_filename):
            results_df = pd.read_csv(eval_results_filename, index_col=False)
        else: 
            print(f"Knee discovery: {eval_results_filename} not found!")
            return
        # results = results_df.to_dict('records')

    # Calculate the knee for each class and plot the IAPC curve
    class_names = list(config['target_labels'].values())
    for class_name in class_names:
        results_class_df = results_df[results_df['object_name'] == class_name]
        knee_degradation_factor = calculate_knee(class_name, results_class_df)

        df = results_class_df[np.isclose(results_class_df['degradation_factor'], knee_degradation_factor, atol=1e-5)]
        results_class_df['knee'] = False
        results_class_df.at[df.index, 'knee'] = True

        print(f"Knee discovered at {knee_degradation_factor} for {class_name}")

        # Optionally, plot the IAPC curve for the class
        # plot_iapc_curve(results_class_df, class_name)  # class name pulled from class_filtering.py

    print("End knee discovery")
