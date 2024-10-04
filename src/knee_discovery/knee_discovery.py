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
from eval.eval import run_eval

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

# def run_eval_on_initial_resolutions(config, data_config_path):
    
# TODO: KENDALL There's a lot to do here in this file, and I don't know how much of it has been done yet.
# Coordinate with DAN on how to store the IAPC and the knees, as he will be doing the report generation code.

# TODO: KENDALL: Write this method
def find_knees(ctxt):
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    os.makedirs(results_path, exist_ok=True)
    model_name = config['model']
    
    # TODO: KENDALL: This needs to be changed to the specific label we are looking for.
    # Maybe needs to be added to pipeline_config.yaml? Coordinate with DAN and GABE and AMIT.
    target_label = list(config['target_labels'].values())[0]
    
    # TODO: KENDALL: change this to a real file. Should be put into pipeline_config.yaml.
    # OR if it makes sense use the results file for the IAPC (used in update_results() below)
    # ALSO, could cache the results in ctxt. That would be very useful.
    knee_results_filename = os.path.join(results_path, "dummy_knees")
  
    # TODO: KENDALL: Find knees and update perhaps a knee .csv
    # ALSO, could cache the results in ctxt. That would be very useful.
    
    # TODO: KENDALL: DUMMY CODE FOR TESTING ONLY. DELETE.
    with open(knee_results_filename, "a") as fp:
        fp.write(f"Dummy results, model {model_name}, object_type {target_label}")
    
# TODO: Write this method
def update_results(ctxt, orig_image_size, degraded_image_size, mAP):
    # orig_image_size and degraded_image_size are tuples of (width, height)
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    os.makedirs(results_path, exist_ok=True)

    # TODO: KENDALL: change this to a real file. Should be put into pipeline_config.yaml.
    iapc_results_filename = os.path.join(results_path, "dummy_results")
    
    # TODO: Update the results file here
    # Perhaps read in a .csv with pandas, update with another data point, sort if necessary, and then write to the csv
    # ALSO, could cache the results in ctxt. That would be very useful.
    
    # DUMMY CODE FOR TESTING ONLY. REWRITE.
    with open(iapc_results_filename, "a") as fp:
        fp.write(f"Dummy results, orig {orig_image_size}, degraded {degraded_image_size}, mAP {mAP}")

# TODO: Write this method
def degrade_images(ctxt, orig_image_size, degraded_image_size, degraded_dir):
    # orig_image_size and degraded_image_size are tuples of (width, height)
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()

    baseline_dir = ctxt.val_baseline_dir
    os.makedirs(degraded_dir, exist_ok=True)
    
    # TODO: KENDALL: degrade images in baseline_dir, put in degraded_dir
    # FOR TESTING PURPOSES ONLY, SIMPLY COPY IMAGES
    # Delete this code block when coding the actual solution
    # print(baseline_dir, degraded_dir)
    val_images = [os.path.join(baseline_dir, x) for x in os.listdir(baseline_dir) if x.endswith(exts)]
    for val_image in val_images:
        shutil.copy2(val_image, degraded_dir)

def run_eval_on_initial_resolutions(ctxt):
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()
    preprocess_method = config['preprocess_method']
    pp_params = config['preprocess_methods'][preprocess_method]
    width = pp_params['image_size']
    height = pp_params['image_size']
   
    # run eval on baseline (path already set) and write to results file
    baseline_dir = ctxt.val_baseline_dir
    
    mAP = run_eval(ctxt, (width, height), (width, height), baseline_dir)
    update_results(ctxt, (width, height), (width, height), mAP) # TODO: Write this method

# TODO: KENDALL: Update code, currently only here for testing purposes
def run_eval_on_degraded_images(ctxt):
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()
    output_top_dir = ctxt.get_output_dir_path()
    preprocess_dir_path = ctxt.get_preprocessing_dir_path()
    
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

    # TODO: KENDALL: From here on is the code that degrades the images,
    # runs the evaluation on each group of degraded images, and finds the knee
    # DO NOT FEEL LIKE YOU CAN'T CHANGE THIS CODE. WHAT IS HERE IS ONLY AN EXAMPLE TO AID DEVELOPMENT.
    for degraded_long_res in range(long_low_range, long_high_range, step):
        degraded_short_res = math.ceil(shorter_mult * degraded_long_res)
        if width == longer:
            degraded_width = degraded_long_res
            degraded_height = degraded_short_res
        else:
            degraded_height = degraded_long_res
            degraded_width = degraded_short_res
            
        
            
        if preprocess_method == 'padding':
            val_degraded_dir = os.path.join(preprocess_dir_path, val_template.format(maxwidth=ctxt.maxwidth, 
                                                                                     maxheight=ctxt.maxheight,
                                                                                     effective_width=degraded_width,
                                                                                     effective_height=degraded_height))
        elif preprocess_method == 'tiling':
            stride = pp_params['stride']
            val_degraded_dir = os.path.join(preprocess_dir_path, val_template.format(width=ctxt.maxwidth, height=ctxt.maxheight, 
                                                                                     stride=stride,
                                                                                     effective_width=degraded_width,
                                                                                     effective_height=degraded_height))
 
        # degrade the images here, place in val_degraded_dir
        degrade_images(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir) # TODO: write this method

        # run eval  
        mAP = run_eval(ctxt, (width, height), (degraded_width, degraded_height), val_degraded_dir)
        update_results(ctxt, (width, height), (degraded_width, degraded_height), mAP)
        # write to results file, maybe pandas with a .csv? That would be easiest for report generation.
        
    # TODO: KENDALL:
    # based on the mAP values returned at each resolution, loop through smartly to find the knee
    # can determine knee here, or gather data and do it in run_knee_discovery
    # again, write to a results file, maybe pandas with a .csv? That would be easiest for report generation.
    # Coordinate with DAN as necessary, he knows pandas very well
    

# TODO: KENDALL: Update this method, currently only here for testing purposes
def run_knee_discovery(ctxt):
    print("Start with run_knee_discovery")

    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()
    output_top_dir = ctxt.get_output_dir_path()
    results_dir = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    
    if 'clean_subdir' in config['knee_discovery'] and config['knee_discovery']['clean_subdir']:
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
 
    preprocess_top_dir = ctxt.get_preprocessing_dir_path()
    method = config['preprocess_method']
    params = ctxt.config['preprocess_methods'][method]
    train_template = params['train_baseline_subdir']
    val_template = params['val_baseline_subdir']
    
    if ctxt.train_baseline_dir == "" or ctxt.maxwidth == 0:
        if method == 'padding':
            # need to parse directories to find train baseline directory
            baseline_top_dir = os.path.join(preprocess_top_dir, 'baseline')
            ctxt.maxwidth = find_maxres(baseline_top_dir, 'padding')
            ctxt.maxheight = ctxt.maxwidth
            ctxt.train_baseline_dir = os.path.join(preprocess_top_dir, train_template.format(maxwidth=ctxt.maxwidth, 
                                                                                             maxheight=ctxt.maxheight))
            ctxt.val_baseline_dir = os.path.join(preprocess_top_dir, val_template.format(maxwidth=ctxt.maxwidth, 
                                                                                         maxheight=ctxt.maxheight))
        elif method == 'tiling':
            image_size = params['image_size']
            ctxt.maxwidth = image_size
            ctxt.maxheight = image_size
            stride = params['stride']
            ctxt.train_baseline_dir = os.path.join(preprocess_top_dir, 
                                                   train_template.format(width=ctxt.maxwidth, height=ctxt.maxheight, stride=stride))
            ctxt.val_baseline_dir = os.path.join(preprocess_top_dir, 
                                                 val_template.format(width=ctxt.maxwidth, height=ctxt.maxheight, stride=stride))
        else:
            raise ValueError("Unknown preprocessing method: " + method)

    os.makedirs(ctxt.train_baseline_dir, exist_ok=True)
    os.makedirs(ctxt.val_baseline_dir, exist_ok=True)
    
    if ('clean_named_preprocess_subdir' in config['knee_discovery']
        and config['knee_discovery']['clean_named_preprocess_subdir'] != ""):
        degraded_dir = os.path.join(ctxt.get_preprocessing_dir_path(), config['knee_discovery']['clean_named_preprocess_subdir'])
        if os.path.exists(degraded_dir):
            shutil.rmtree(degraded_dir)

    # val_degraded_images_dir = get_preprocessed_images_dir_path(ctxt, 'degraded', 'val')

    run_eval_on_initial_resolutions(ctxt)
    run_eval_on_degraded_images(ctxt)
    
    # TODO: KENDALL: read results from results file OR get from a cache stored in ctxt OR something similar
    # File could be a .csv file and retrieves knee from its data, placing into perhaps another file?
    find_knees(ctxt)

    print("End knee discovery")
