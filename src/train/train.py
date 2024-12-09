#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:01:23 2024

@author: dfox
"""

import os
import pandas as pd
import random
import shutil
import math
from itertools import product
from sklearn.model_selection import ParameterSampler
from ultralytics import YOLO
from util.util import update_data_config_train_path, update_data_config_val_path

def train_val_split(ctxt):
    """
    Splits the dataset into training and validation sets.
    """
    config = ctxt.get_pipeline_config()
    train_split = 0.80
    image_ext = ['.tif', '.tiff', '.png', '.gif', '.jpg', '.jpeg']

    interim_images_list = [x for x in os.listdir(ctxt.train_baseline_dir) if x.lower().endswith(tuple(image_ext))]

    random.seed(43)
    num_train = math.ceil(len(interim_images_list) * train_split)

    ctxt.train_hyperparameter_images_list = random.sample(interim_images_list, num_train)
    ctxt.val_hyperparameter_images_list = [x for x in interim_images_list if x not in ctxt.train_hyperparameter_images_list]

    ctxt.train_hyperparameter_labels_list = [(x.rsplit('.', 1)[0] + '.txt') for x in ctxt.train_hyperparameter_images_list]
    ctxt.val_hyperparameter_labels_list = [(x.rsplit('.', 1)[0] + '.txt') for x in ctxt.val_hyperparameter_images_list]

    for filename in ctxt.train_hyperparameter_images_list:
        shutil.copy2(os.path.join(ctxt.train_baseline_dir, filename), ctxt.train_hyperparameter_dir)
    for filename in ctxt.val_hyperparameter_images_list:
        shutil.copy2(os.path.join(ctxt.train_baseline_dir, filename), ctxt.val_hyperparameter_dir)
    for filename in ctxt.train_hyperparameter_labels_list:
        full_path = os.path.join(ctxt.train_baseline_dir, filename)
        if os.path.exists(full_path):
            shutil.copy2(full_path, ctxt.train_hyperparameter_dir)
    for filename in ctxt.val_hyperparameter_labels_list:
        full_path = os.path.join(ctxt.train_baseline_dir, filename)
        if os.path.exists(full_path):
            shutil.copy2(full_path, ctxt.val_hyperparameter_dir)

def update_hyperparameters(ctxt, model_params):
    """
    Updates the hyperparameters into the configured filename
    """
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    hyper_path = os.path.join(output_top_dir, config['train']['output_subdir'])

    hyperparams_filename = os.path.join(hyper_path, 'hyperparams.csv')
    os.makedirs(hyper_path, exist_ok=True)

    # Log hyperparameters
    df = pd.DataFrame(columns=['parameter', 'value'])
    for key, value in model_params.items():
        df.loc[df.shape[0]] = [key, str(value)]
    df.to_csv(hyperparams_filename, index=False)

def run_hyperparameter_tuning(ctxt, fractional_factorial=False):
    """
    Runs the fine-tuning process for a YOLO model using the specified context and configuration.
    """
    print("Start Running Fine-tuning.")

    config = ctxt.get_pipeline_config()

    preprocess_top_dir = ctxt.get_preprocessing_dir_path()
    method = ctxt.config['preprocess_method']
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
    hyperparameter_grid = model_dict.get('hyperparameters', {})

    # Define default values for hyperparameters if not provided
    default_hyperparameters = {
        "cls": [1.5],
        "imgsz": [640],
        "epochs": [100],
        "batch": [16],
        "freeze": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    }

    if not hyperparameter_grid:
        print("No hyperparameters defined. Using default settings.")
        hyperparameter_grid = default_hyperparameters

    # Generate hyperparameter combinations
    keys, values = zip(*hyperparameter_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Fractional factorial tuning if enabled
    if fractional_factorial and hyperparameter_grid:
        param_distributions = {key: values for key, values in hyperparameter_grid.items()}
        n_iter = max(len(hyperparameter_combinations) // 2, 1)
        hyperparameter_combinations = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))

    best_mAP = 0
    best_params = {}

    for i, combination in enumerate(hyperparameter_combinations):
        print(f"Testing combination {i + 1}/{len(hyperparameter_combinations)}: {combination}")

        # Initialize YOLO model
        yolo_id = ctxt.get_yolo_id()
        model = YOLO(yolo_id)

        # Train the model
        results = model.train(
            **combination,
            data=ctxt.get_data_config_dir_path(),
            device='cuda' if ctxt.use_cuda else 'cpu'
        )

        avg_mAP = results.box.map
        print(f"Extracted avg mAP: {avg_mAP}")

        # Update best model if better
        if avg_mAP > best_mAP:
            best_mAP = avg_mAP
            best_params = combination
            os.makedirs(os.path.dirname(ctxt.final_weights_path), exist_ok=True)
            model.save(ctxt.final_weights_path)

    update_hyperparameters(ctxt, best_params)
    update_data_config_train_path(ctxt, ctxt.train_baseline_dir)
    update_data_config_val_path(ctxt, ctxt.val_baseline_dir)

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Average mAP: {best_mAP}")
    print("Finished Running Fine-tuning.")
