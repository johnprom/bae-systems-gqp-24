#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:47:32 2024

@author: dfox
"""

import math
import os
import shutil

# from util.util import get_preprocessed_images_dir_path
from eval.eval import run_eval, update_results, calculate_knee

exts = ('.tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')  # Common image file extensions

def find_maxres(root_dir, method):
    # result_directories = []
    maxres = 0

    for dirpath, _, filenames in os.walk(root_dir):
        if any(file.lower().endswith(exts) for file in filenames):
            path_parts = os.path.abspath(dirpath).split(os.sep)
            if method == 'tiling':
                if len(path_parts) >= 2:
                    maxres = path_parts[-2].split('_')[0]
            elif method == 'padding':
                if len(path_parts) >= 1:
                    maxres = path_parts[-1].split('_')[0]
            break 

    return maxres


def degrade_images(ctxt, orig_image_size, degraded_image_size, degraded_dir):
    """
    Degrade images by resizing them and store them in the degraded_dir.
    For testing, this method is currently only copying images from baseline_dir to degraded_dir.
    """
    # orig_image_size and degraded_image_size are tuples of (width, height)
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()

    baseline_dir = ctxt.val_baseline_dir
    os.makedirs(degraded_dir, exist_ok=True)
    
    val_images = [os.path.join(baseline_dir, x) for x in os.listdir(baseline_dir) if x.endswith(exts)]
    for val_image in val_images:
        shutil.copy2(val_image, degraded_dir)


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
    config = ctxt.get_pipeline_config()
    preprocess_method = config['preprocess_method']
    pp_params = config['preprocess_methods'][preprocess_method]
    kd_params = config['knee_discovery']
    val_template = pp_params['val_degraded_subdir']

    width = pp_params['image_size']
    height = pp_params['image_size']
    longer = max(width, height)
    shorter = min(width, height)
    shorter_mult = shorter / longer

    long_low_range = math.ceil(kd_params['search_resolution_range'][0] * longer)
    long_high_range = math.ceil(kd_params['search_resolution_range'][1] * longer) + 1
    step = math.floor(kd_params['search_resolution_step'] * longer)

    results = []

    for degraded_long_res in range(long_low_range, long_high_range, step):
        degraded_short_res = math.ceil(shorter_mult * degraded_long_res)
        degraded_width = degraded_long_res if width == longer else degraded_short_res
        degraded_height = degraded_short_res if height == longer else degraded_long_res

        # Create directory path for degraded images
        val_degraded_dir = os.path.join(ctxt.get_preprocessing_dir_path(), val_template.format(
            maxwidth=ctxt.maxwidth, maxheight=ctxt.maxheight,
            effective_width=degraded_width, effective_height=degraded_height))

        # Degrade images and run evaluation
        degrade_images(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir)
        mAP = run_eval(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir)

        # Store results in list
        results.append({
            'orig_width': width,
            'orig_height': height,
            'degraded_width': degraded_width,
            'degraded_height': degraded_height,
            'mAP': mAP
        })

    # Store results in CSV
    results_df = pd.DataFrame(results)
    results_file = os.path.join(ctxt.get_output_dir_path(), 'iapc_results.csv')
    results_df.to_csv(results_file, index=False)

    return results
    

def calculate_knee(results):
    """
    Calculate the knee in the IAPC curve using mAP values.
    Args:
    - results: A list of dictionaries with 'degraded_width' and 'mAP' keys.

    Returns:
    - knee_resolution: The resolution at which the knee occurs.
    """
    resolutions = [r['degraded_width'] for r in results]
    mAP_values = [r['mAP'] for r in results]

    # Convert to IAPC (1/mAP)
    iapc_values = [1/mAP if mAP != 0 else float('inf') for mAP in mAP_values]

    # Use KneeLocator to find the knee
    kneedle = KneeLocator(resolutions, iapc_values, curve='convex', direction='increasing')
    knee_resolution = kneedle.knee

    return knee_resolution


def plot_iapc_curve(results, class_name):
    """
    Plot the IAPC curve for the given results and highlight the knee point.
    """
    resolutions = [r['degraded_width'] for r in results]
    mAP_values = [r['mAP'] for r in results]
    iapc_values = [1/mAP if mAP != 0 else float('inf') for mAP in mAP_values]

    # Find the knee
    kneedle = KneeLocator(resolutions, iapc_values, curve='convex', direction='increasing')
    knee_resolution = kneedle.knee

    # Plotting the IAPC curve
    plt.figure(figsize=(10, 6))
    plt.plot(resolutions, iapc_values, marker='o', label=f'{class_name} IAPC')
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

    # Run evaluation on initial (baseline) resolutions
    run_eval_on_degraded_images(ctxt)

    # Load the results from CSV (or from the cached results in memory)
    results_file = os.path.join(ctxt.get_output_dir_path(), 'iapc_results.csv')
    results_df = pd.read_csv(results_file)
    results = results_df.to_dict('records')

    # Calculate the knee for each class and plot the IAPC curve
    for class_name in ctxt.class_names:
        knee_resolution = calculate_knee(results)
        print(f"Knee discovered at {knee_resolution} for {class_name}")

        # Optionally, plot the IAPC curve for the class
        plot_iapc_curve(results, class_name)  # class name pulled from class_filtering.py

    print("End knee discovery")
