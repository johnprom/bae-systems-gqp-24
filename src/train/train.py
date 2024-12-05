#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:01:23 2024

@author: dfox
"""

import copy
import os
import pandas as pd
import random
import shutil
import math

from ultralytics import YOLO
from itertools import product
from sklearn.model_selection import ParameterSampler
from util.util import update_data_config_train_path, update_data_config_val_path

def train_val_split(ctxt):
    """
    Splits the dataset into training and validation sets.
    """

    config = ctxt.get_pipeline_config()
    train_split = 0.80  # Ratio for train split
    image_ext = ['.tif', '.tiff', '.png', '.gif', '.jpg', '.jpeg']

    interim_images_list = [x for x in os.listdir(ctxt.train_baseline_dir) if x[-4:] in image_ext or x[-5:] in image_ext]

    random.seed(43)
    num_train = math.ceil(len(interim_images_list) * train_split)

    # Split this list into a random train list and a random val list
    ctxt.train_hyperparameter_images_list = random.sample(interim_images_list, num_train)
    ctxt.val_hyperparameter_images_list = [x for x in interim_images_list if x not in ctxt.train_hyperparameter_images_list]

    ctxt.train_hyperparameter_labels_list = [(x.rsplit('.', 1)[0] + '.txt') for x in ctxt.train_hyperparameter_images_list]
    ctxt.val_hyperparameter_labels_list = [(x.rsplit('.', 1)[0] + '.txt') for x in ctxt.val_hyperparameter_images_list]

    for filename in ctxt.train_hyperparameter_images_list:
        if ctxt.verbose:
            print(f"Copying file {filename} into directory {ctxt.train_hyperparameter_dir}")
        shutil.copy2(os.path.join(ctxt.train_baseline_dir, filename), ctxt.train_hyperparameter_dir)
    for filename in ctxt.val_hyperparameter_images_list:
        if ctxt.verbose:
            print(f"Copying file {filename} into directory {ctxt.val_hyperparameter_dir}")
        shutil.copy2(os.path.join(ctxt.train_baseline_dir, filename), ctxt.val_hyperparameter_dir)

    for filename in ctxt.train_hyperparameter_labels_list:
        full_path = os.path.join(ctxt.train_baseline_dir, filename)
        if os.path.exists(full_path):
            if ctxt.verbose:
                print(f"Copying file {filename} into directory {ctxt.train_hyperparameter_dir}")
            shutil.copy2(full_path, ctxt.train_hyperparameter_dir)
    for filename in ctxt.val_hyperparameter_labels_list:
        full_path = os.path.join(ctxt.train_baseline_dir, filename)
        if os.path.exists(full_path):
            if ctxt.verbose:
                print(f"Copying file {filename} into directory {ctxt.val_hyperparameter_dir}")
            shutil.copy2(full_path, ctxt.val_hyperparameter_dir)

def update_hyperparameters(ctxt, model_params):
    """
    Updates the hyperparameters into the configured filename

    Args:
        ctxt: Context object containing configuration, model details, and utility methods.
        model_params: Dictionary of parameters used for the YOLO model evaluation

    Returns:
        None
    """
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    hyper_path = os.path.join(output_top_dir, config['train']['output_subdir'])

    if 'train' not in config or 'hyperparams_filename' not in config['train']:
        hyperparams_filename = os.path.join(hyper_path, 'hyperparams.csv')
    else:
        hyperparams_filename = os.path.join(hyper_path, config['train']['hyperparams_filename'])

    os.makedirs(hyper_path, exist_ok=True)

    # Log hyperparameters
    df = pd.DataFrame(columns=['parameter', 'value'])
    for key, value in model_params.items():
        if key != 'data':
            df.loc[df.shape[0]] = [key, str(value)]
    df.to_csv(hyperparams_filename, index=False)

def run_hyperparameter_tuning(ctxt, fractional_factorial=False):
    """
    Runs the fine-tuning process for a YOLO model using the specified context and configuration.

    Args:
        ctxt: Context object containing configuration, model details, and utility methods.

    Returns:
        None: Trains the model and saves the fine-tuned weights to the specified path.
    """
    print("Start Running Fine-tuning.")

    config = ctxt.get_pipeline_config()

    preprocess_top_dir = ctxt.get_preprocessing_dir_path()
    method = ctxt.config['preprocess_method']  # Currently 'padding' or 'tiling'

    params = ctxt.config['preprocess_methods'][method]

    train_template = params['train_hyperparameter_subdir']
    val_template = params['val_hyperparameter_subdir']

    image_size = params['image_size']
    ctxt.maxwidth = image_size
    ctxt.maxheight = image_size
    stride = params['stride']
    ctxt.train_hyperparameter_dir = os.path.join(preprocess_top_dir, train_template.format(maxwidth=ctxt.maxwidth, maxheight=ctxt.maxheight, stride=stride))
    ctxt.val_hyperparameter_dir = os.path.join(preprocess_top_dir, val_template.format(maxwidth=ctxt.maxwidth, maxheight=ctxt.maxheight, stride=stride))

    os.makedirs(ctxt.train_hyperparameter_dir, exist_ok=True)
    os.makedirs(ctxt.val_hyperparameter_dir, exist_ok=True)

    train_val_split(ctxt)

    update_data_config_train_path(ctxt, ctxt.train_hyperparameter_dir)
    update_data_config_val_path(ctxt, ctxt.val_hyperparameter_dir)

    model_name = config['model']
    model_dict = config['models'][model_name]
    base_params = model_dict['params']
    hyperparameter_grid = model_dict.get('hyperparameters', {})

    best_mAP = 0
    best_params = {}

    # Generate all combinations of hyperparameters from YAML
    keys, values = zip(*hyperparameter_grid.items())
    if len(hyperparameter_grid) > 0:
        combinations = [dict(zip(keys, v)) for v in product(*values)]
    else:
        combinations = [base_params]

    # Check if fractional factorial flag is set
    if fractional_factorial and len(hyperparameter_grid) > 0:
        # Use ParameterSampler for a structured, fractional search
        n_iter = max(len(combinations) // 2, 1) # Use half of the combinations as an example
        param_distributions = {key: values for key, values in hyperparameter_grid.items()}
        combinations = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))
        print(f"Starting fractional factorial hyperparameter tuning with {len(combinations)} combinations...")
    else:
        print(f"Starting full factorial hyperparameter tuning with {len(combinations)} combinations...")

    if not ctxt.is_model_yolo():
        raise ValueError(f"unknown deep learning model {ctxt.get_model_name()} specified.")

    # Loop through all hyperparameter combinations
    for i, combination in enumerate(combinations):
        if ctxt.verbose:
            print(f"Testing combination {i+1}/{len(combinations)}: {combination}")

        # Merge base params with the current combination
        current_params = {**base_params, **combination}

        # Dynamically pass all parameters from YAML to the model's train method
        train_params = {
            key: value
            for key, value in current_params.items()
        }

        # Initialize YOLO model
        yolo_id = ctxt.get_yolo_id()
        model = YOLO(yolo_id)

        # Train the model with the dynamically read hyperparameters
        results = model.train(
            **train_params,
            data=ctxt.get_data_config_dir_path(),
            device='cuda' if ctxt.use_cuda else 'cpu'
        )

        # Extract the average mAP (mAP@50-95)
        avg_mAP = results.box.map  # Access the mAP score
        print(f"Extracted avg mAP: {avg_mAP}")

        # Update best model if current avg mAP is better
        if avg_mAP > best_mAP:
            best_mAP = avg_mAP
            best_params = current_params
            # ctxt.final_weights_path = os.path.join(
            #     ctxt.get_output_dir_path(),
            #     config['train']['output_subdir'],
            #     config['train'].get('trained_model_filename', 'best_model.pt')
            # )
            os.makedirs(os.path.dirname(ctxt.final_weights_path), exist_ok=True)
            model.save(ctxt.final_weights_path)
            if ctxt.verbose:
                print(f"New best model saved at: {ctxt.final_weights_path}")

    # Update hyperparameters log
    update_hyperparameters(ctxt, best_params)
    update_data_config_train_path(ctxt, ctxt.train_baseline_dir)
    update_data_config_val_path(ctxt, ctxt.val_baseline_dir)  # Reset to baseline data paths

    if ctxt.verbose:
        print(f"Best Hyperparameters: {best_params}")
        print(f"Best Average mAP: {best_mAP}")
        print(f"Final weights saved to: {ctxt.final_weights_path}")

    print("Finished Running Fine-tuning.")
