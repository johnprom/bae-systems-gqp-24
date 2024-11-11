#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:01:23 2024

@author: dfox
"""
import copy
import os
import pandas as pd
from ultralytics import YOLO

def update_hyperparameters(ctxt, model_params):
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_output_dir_path()
    hyper_path = os.path.join(output_top_dir, config['train']['output_subdir'])
    if 'train' not in config or 'hyperparams_filename' not in config['train']:
        hyperparams_filename = os.path.join(hyper_path, 'hyperparams.csv')
    else:
        hyperparams_filename = os.path.join(hyper_path, config['train']['hyperparams_filename'])

    os.makedirs(hyper_path, exist_ok=True)

    # Log hyperparameters
    # df = pd.DataFrame.from_dict(model_params)
    df = pd.DataFrame(columns=['parameter', 'value'])
    for key, value in model_params.items():
        if key != 'data':
            df.loc[df.shape[0]] = [key, str(value)]
    df.to_csv(hyperparams_filename, index=False)

def run_finetuning(ctxt):
    """
    Runs the fine-tuning process for a YOLO model using the specified context and configuration.

    Args:
        ctxt: Context object containing configuration, model details, and utility methods.

    Returns:
        None: Trains the model and saves the fine-tuned weights to the specified path.
    """
    
    print("Start Running Fine-tuning.")
            
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()

    if not ctxt.is_model_yolo():
        raise ValueError(f"unkown deep learning model {ctxt.get_model_name()} specified.")

    yolo_id = ctxt.get_yolo_id()
    model_name = config['model']
    model_dict = config['models'][model_name]
    model_params = copy.deepcopy(model_dict['params'])
    if ctxt.verbose:
        print("About to instantiate the model")
    base_model = YOLO(yolo_id)
    if ctxt.use_cuda:
        base_model.to('cuda')
        if ctxt.verbose:
            print("Added cuda to the model")
    
    # trained_model_filename_template = model_dict['trained_model_filename']
    model_params['data'] = data_config_path
    if ctxt.verbose:
        print(f"Start training model, params {model_params}")
    ft_stats = base_model.train(**model_params)
    if ctxt.verbose:
        print("Stats from training:")
        print(ft_stats)
    os.makedirs(os.path.dirname(ctxt.final_weights_path), exist_ok=True)
    base_model.save(ctxt.final_weights_path)

    update_hyperparameters(ctxt, model_params)
    
    print("Finished Running Fine-tuning.")
