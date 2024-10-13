#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:01:23 2024

@author: dfox
"""
import copy
import os
from ultralytics import YOLO

def run_finetuning(ctxt):
    
    print("Start Running Fine-tuning.")
            
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()

    if not ctxt.is_model_yolo():
        raise ValueError(f"unkown deep learning model {ctxt.get_model_name()} specified.")

    yolo_id = ctxt.get_yolo_id()
    model_name = config['model']
    print(f"Fine-tuning using model {model_name}")
    model_dict = config['models'][model_name]
    model_params = copy.deepcopy(model_dict['params'])
    if ctxt.verbose:
        print("About to instantiate the model")
    base_model = YOLO(yolo_id)
    if False:
        if ctxt.verbose:
            print("Adding cuda to the model")
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
    
    print("Finished Running Fine-tuning.")
