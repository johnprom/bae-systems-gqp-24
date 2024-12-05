#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:47:32 2024

@author: dfox
"""

import bisect
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil
from pwlf import PiecewiseLinFit
from scipy.interpolate import make_interp_spline

# Removed import of KneeLocator since we're replacing it
# from kneed import KneeLocator
from PIL import Image

# from util.util import get_preprocessed_images_dir_path, calc_degradation_factor
from eval.eval import run_eval, update_knee_results

exts = ('.tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')  # Common image file extensions
label_ext = '.txt'

def replace_image_ext_with_label(filepath):
    """
    Replaces the file extension of an image file with a label extension if it matches specified extensions.

    Args:
        filepath (str or Path): The file path of the image whose extension needs to be replaced.

    Returns:
        Path: A Path object with the file extension replaced by `label_ext` if it matches specified extensions.
              Otherwise, returns the original filepath.
    """
    path = Path(filepath)

    if path.suffix.lower() in exts:
        return path.with_suffix(label_ext)

    return filepath

def degrade_images(ctxt, orig_image_size, degraded_image_size, degraded_dir, corrupted_counter):
    """
    Degrades images by resizing them and stores them in the specified degraded directory.
    If `orig_image_size` and `degraded_image_size` are the same, images are simply copied.
    Otherwise, images are resized to the degraded size and then resized back to the original size.

    Args:
        ctxt: The pipeline context object containing configurations and settings.
        orig_image_size (tuple): A tuple (width, height) of the original image size.
        degraded_image_size (tuple): A tuple (width, height) of the degraded image size.
        degraded_dir (str): Directory where the degraded images will be stored.
        corrupted_counter (int): Counter tracking corrupted or unreadable images.

    Returns:
        tuple: A tuple containing:
            - num_images (int): The number of images processed.
            - corrupted_counter (int): Updated counter for corrupted images encountered.
    """
    # orig_image_size and degraded_image_size are tuples of (width, height)

    if ctxt.verbose:
        print(f"degrade images {orig_image_size} -> {degraded_image_size} in degraded directory {degraded_dir}", flush=True)
    config = ctxt.config

    method = config['preprocess_method']
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
    for idx, val_image_filename in enumerate(val_image_filenames):
        val_label_filename = val_label_filenames[idx]
        if os.path.exists(val_image_filename) and os.path.exists(val_label_filename):
            val_degraded_image_filename = os.path.join(degraded_dir, os.path.basename(val_image_filename))
            val_degraded_label_filename = os.path.join(degraded_dir, os.path.basename(val_label_filename))
            if not os.path.exists(val_degraded_image_filename):
                if orig_image_size == degraded_image_size:
                    # resolutions and hyperparameters are equal, simple copy
                    shutil.copyfile(val_image_filename, val_degraded_image_filename)
                else:
                    try:
                        image = Image.open(val_image_filename)
                        this_orig_image_size = image.size
                        this_degraded_width = int(math.ceil((degraded_image_size[0] / orig_image_size[0])
                                                            * this_orig_image_size[0]))
                        this_degraded_height = int(math.ceil((degraded_image_size[1] / orig_image_size[1])
                                                             * this_orig_image_size[1]))

                        val_shrunk_image = image.resize((this_degraded_width, this_degraded_height))
                        val_degraded_image = val_shrunk_image.resize(this_orig_image_size)
                        val_degraded_image.save(val_degraded_image_filename)
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

def run_eval_on_degraded_images(ctxt):
    """
    Evaluates the model on a range of degraded image resolutions, specified in the configuration,
    and logs the results. This function systematically reduces image resolution within the range
    defined by the knee discovery parameters, creating degraded copies and evaluating the model on each.

    Args:
        ctxt: The pipeline context object containing configuration settings, paths, and options.

    Behavior:
        - Retrieves baseline and knee discovery parameters from the pipeline configuration.
        - Iteratively reduces resolution of images in specified steps and evaluates the model on each degraded set.
        - Logs any corrupted images that could not be processed during degradation.

    Log Output:
        - Prints status messages if verbosity is enabled, detailing degradation ranges, steps, and corrupted images.
        - Final report of corrupted images if any were encountered during processing.

    """

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
        run_eval(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir)

    if corrupted_counter > 0:
        print(f"{corrupted_counter} out of {max_images} images are corrupted!")


def calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h):
    """
    Calculates the degradation factor based on the original and effective image resolutions.
    The degradation factor is computed as the square root of the area ratio between the effective
    and original resolutions.

    Args:
        orig_res_w (float or pd.Series): Original resolution width.
        orig_res_h (float or pd.Series): Original resolution height.
        eff_res_w (float or pd.Series): Effective resolution width.
        eff_res_h (float or pd.Series): Effective resolution height.

    Returns:
        pd.Series: The degradation factor calculated from the resolutions, representing the ratio
                   of effective resolution to original resolution.
    """

    degradation_factor_w = eff_res_w / orig_res_w
    degradation_factor_h = eff_res_h / orig_res_h
    degradation_factor_area = degradation_factor_w * degradation_factor_h
    degradation_factor = degradation_factor_area.apply(math.sqrt)
    return degradation_factor # pd.Series

def find_interp_range_indicies(x_array, x, granularity):
    # low_index = None
    # high_index = None

    diffs = [(val - x) for val in x_array]
    # absdiffs = [abs(d) for d in diffs]

    # closest_index = absdiffs.index(min(absdiffs)) # if equal, returns the lower one
    x_array = np.array(x_array)
    closest_index = np.argmin(np.abs(x_array - x))
    if abs(x_array[closest_index] - x) < (granularity / 2):
        low_index = closest_index
        high_index = closest_index
    elif diffs[closest_index] < 0.0:
        low_index = closest_index
        high_index = closest_index + 1
    else:
        low_index = closest_index - 1
        high_index = closest_index

    if low_index < 0:
        low_index = 0

    if high_index >= len(x_array):
        high_index = len(x_array) - 1

    return low_index, high_index

    # for idx, value in enumerate(x_array):
    #     if abs(value - x) <= granularity:
    #         return idx, idx
    # idx = bisect.bisect_right(x_array, x)
    # if idx == 0:
    #     return idx, idx
    # elif idx >= len(x_array):
    #     raise ValueError("find_interp_range(): x > any value in x_array")
    # else:
    #     return idx, (idx - 1)

def calculate_knee(ctxt, class_name, results_class_df):
    """
    Calculates the "knee" point in the IAPC curve using the piecewise linear fit method
    with spline interpolation to smooth the data.

    Args:
        ctxt: The pipeline context object containing configuration and verbosity settings.
        class_name (str): Name of the class for which the knee is being calculated.
        results_class_df (pd.DataFrame): DataFrame with columns 'original_resolution_width',
                                         'original_resolution_height', 'effective_resolution_width',
                                         'effective_resolution_height', and 'mAP' for the given class.

    Returns:
        tuple: A tuple containing two lists:
            - x_out_list (list): A list containing the degradation factor where the knee point is identified.
            - y_out_list (list): A list containing the corresponding mAP value at the knee point.

    Behavior:
        - Converts resolution columns to floating-point values and calculates degradation factors.
        - Filters out mAP values below a threshold to focus on meaningful data points.
        - Applies spline interpolation to smooth the data.
        - Uses piecewise linear fitting to identify the knee point.
        - Logs knee point if verbosity is enabled in the context.
        - Calls `update_knee_results` to store the knee point in the context.

    """

    # Get degradation factors and mAP values
    orig_res_w = results_class_df['original_resolution_width'].astype(float)
    orig_res_h = results_class_df['original_resolution_height'].astype(float)
    eff_res_w = results_class_df['effective_resolution_width'].astype(float)
    eff_res_h = results_class_df['effective_resolution_height'].astype(float)
    mAP_values_series = results_class_df['mAP']
    degradation_factor_series = calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h)

    degradation_factor_list = degradation_factor_series.to_list()
    mAP_values = mAP_values_series.to_list()
    width_list = orig_res_w.to_list()
    height_list = orig_res_h.to_list()

    # If all mAP are zero, return empty lists
    threshold = 0.01
    if all(mAP <= threshold for mAP in mAP_values):
        return [], []

    # Filter out mAP values <= threshold
    filtered_data = [(d, m, w, h) for d, m, w, h in zip(degradation_factor_list, mAP_values, width_list, height_list) if m > threshold]
    # filtered_data = [(d, m, w, h) for d, m, w, h in zip(degradation_factor_list, mAP_values, width_list, height_list)]
    if len(filtered_data) < 2:
        return [], []
    x_list, y_list, w_list, h_list = zip(*filtered_data)

    # Sort data based on x (degradation factors)
    sorted_data = sorted(zip(x_list, y_list, w_list, h_list), key=lambda pair: pair[0])
    x_sorted, y_sorted, w_sorted, h_sorted = zip(*sorted_data)

    # Convert lists to numpy arrays for interpolation
    x_array = np.array(x_sorted)
    y_array = np.array(y_sorted)

    # Apply spline interpolation to smooth the data
    config = ctxt.get_pipeline_config()
    knee_resolution_divisor, knee_resolution_granularity, knee_step = ctxt.get_knee_resolution_divisor()
    if ctxt.verbose:
        print(f"knee_resolution_divisor {knee_resolution_divisor}")
        print(f"knee_step {knee_step}")


    try:
        start = x_array.min()
        stop = x_array.max()
        num_data_points = len(x_array)
        # num_interpolation_points = int((math.ceil((stop - start) / interp_granularity)) + 1)
        num_interpolation_points = (num_data_points - 1) * knee_resolution_divisor + 1
        if ctxt.verbose:
            print(f"num_data_points {num_data_points}")
            print(f"num_interpolation_points {num_interpolation_points}")
        if num_interpolation_points > num_data_points:
            x_interp = np.linspace(start, stop, num=num_interpolation_points)
            spline = make_interp_spline(x_array, y_array, k=3)  # Cubic spline
            y_interp = spline(x_interp)
        else:
            x_interp = x_array
            y_interp = y_array
    except Exception as e:
        if ctxt.verbose:
            print(f"Spline interpolation failed for class {class_name}: {e}")
        return [], []

    # Apply piecewise linear fit with two segments on interpolated data
    try:
        pwlf = PiecewiseLinFit(x_interp, y_interp)
        breaks = pwlf.fit(2)
        knee_x = breaks[1]
        knee_x = round(knee_x / knee_resolution_granularity, 0) * knee_resolution_granularity
        # interp_granularity = knee_step / knee_resolution_divisor
        # tolerance = interp_granularity
        x_low_idx, x_high_idx = find_interp_range_indicies(x_array, knee_x, knee_resolution_granularity)
        if ctxt.verbose:
            print(f"x_high_idx {x_high_idx}, x_low_idx {x_low_idx}")
            print(f"knee_x {knee_x}, tolerance {knee_resolution_granularity}")
            print("x_array:")
            print(f"  {x_array}")
        # idx_high, idx_low = find_interp_indicies
        # x_high = math.ceil(knee_x / knee_step) * knee_step
        # x_low = math.floor(knee_x / knee_step) * knee_step
        if x_high_idx == x_low_idx:
            knee_y = y_array[x_high_idx]
        else:
            y0 = y_array[x_low_idx]
            x0 = x_array[x_low_idx]
            y1 = y_array[x_high_idx]
            x1 = x_array[x_high_idx]
            knee_y = y0 + (knee_x - x0) * ((y1 - y0) / (x1 - x0))
            if ctxt.verbose:
                print(f"x0 {x0}, x1 {x1}, y0 {y0}, y1 {y1}, knee_y {knee_y}")
        if ctxt.verbose:
            print(f"knee_x {knee_x} knee_y {knee_y}")

        # Find the index of the knee point in the interpolated data
        # knee_index = np.argmin(np.abs(x_interp - knee_x))

        # Map back to the closest original resolution for logging and updating results
        original_index = np.argmin(np.abs(x_array - knee_x))
        w_knee = w_sorted[original_index]
        h_knee = h_sorted[original_index]

        if ctxt.verbose:
            print(f"Knee found at degradation factor {knee_x} with mAP {knee_y} for class {class_name}")
            # print(f"Knee found at degradation factor {knee_x} for class {class_name}")

        # Update the knee results
        update_knee_results(ctxt, class_name, (w_knee, h_knee), knee_x, knee_y)

        return [knee_x], [knee_y]

    except Exception as e:
        if ctxt.verbose:
            print(f"Piecewise linear fit failed for class {class_name}: {e}")
        return [], []


def run_knee_discovery(ctxt):
    """
    Runs the knee discovery process to identify optimal image resolutions at which model performance
    (mean Average Precision, mAP) experiences a significant change. This process iteratively degrades
    image resolution, evaluates the model, and identifies knee points in the performance curve.

    Args:
        ctxt: The pipeline context object containing configurations, paths, and verbosity settings.

    Behavior:
        - Initializes output paths and clears previous knee discovery results if configured to do so.
        - Runs evaluation on initial degraded resolutions to populate a baseline.
        - Loads or calculates degradation factors for each resolution.
        - Iterates over each class to identify the knee point in the degradation curve.
            - If a binary search algorithm is specified, it refines the knee point by iterating over nearby
              degradation factors until convergence is achieved within a specified tolerance or a maximum
              number of iterations.
            - Uses piecewise linear fitting to identify knee points and logs details if verbosity is enabled.
            - Calls `update_knee_results` to store the knee point in the context.
        - Updates and saves the final knee discovery results to the specified output file.

    Log Output:
        - Prints status updates for the knee discovery process, including convergence checks and detected knee points.
        - Prints a summary of discovered knee points or a notification if no knee point is found for a class.

    Returns:
        None
    """
    print("Start with run_knee_discovery")

    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    eval_results_filename = os.path.join(results_path, config['knee_discovery']['eval_results_filename'])

    knee_only = config['knee_discovery'].get('calculate_knee_only', False)

    # if not doing preprocessing and fine-tuning, but
    if not knee_only:
        if os.path.exists(eval_results_filename):
            os.remove(eval_results_filename)

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

    # Ensure degradation_factor is calculated
    if 'degradation_factor' not in results_df.columns:
        orig_res_w = results_df['original_resolution_width'].astype(float)
        orig_res_h = results_df['original_resolution_height'].astype(float)
        eff_res_w = results_df['effective_resolution_width'].astype(float)
        eff_res_h = results_df['effective_resolution_height'].astype(float)
        results_df['degradation_factor'] = calc_degradation_factor(orig_res_w, orig_res_h, eff_res_w, eff_res_h)

    # Ensure 'knee' column exists and initialize it to False
    results_df['knee'] = False

    class_names = results_df['object_name'].unique()

    # The 'knee' column is initialized outside the loop to avoid overwriting

    for class_name in class_names:
        results_class_df = results_df[results_df['object_name'] == class_name].copy()
        _, _ = calculate_knee(ctxt, class_name, results_class_df)

    print("End knee discovery")
