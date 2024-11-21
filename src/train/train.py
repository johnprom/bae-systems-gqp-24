#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:01:23 2024

@author: dfox
"""

import os
from ultralytics import YOLO
from itertools import product
import pandas as pd

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

def run_hyperparameter_tuning(ctxt):
    """
    Runs hyperparameter tuning by iterating over parameter combinations,
    selects the best model based on avg mAP, and saves the best model's weights.

    Args:
        ctxt: The pipeline context object containing configuration and paths.
    """
    config = ctxt.get_pipeline_config()
    model_name = config['model']
    model_dict = config['models'][model_name]
    base_params = model_dict['params']
    hyperparameter_grid = model_dict.get('hyperparameters', {})

    best_mAP = 0
    best_params = {}

    # Generate all combinations of hyperparameters from YAML
    keys, values = zip(*hyperparameter_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    print(f"Starting hyperparameter tuning with {len(combinations)} combinations...")

    for i, combination in enumerate(combinations):
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
            ctxt.final_weights_path = os.path.join(
                ctxt.get_output_dir_path(),
                config['train']['output_subdir'],
                config['train'].get('trained_model_filename', 'best_model.pt')
            )
            os.makedirs(os.path.dirname(ctxt.final_weights_path), exist_ok=True)
            model.save(ctxt.final_weights_path)
            print(f"New best model saved at: {ctxt.final_weights_path}")

        # Update hyperparameters log
        update_hyperparameters(ctxt, current_params)

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Average mAP: {best_mAP}")
    print(f"Final weights saved to: {ctxt.final_weights_path}")
