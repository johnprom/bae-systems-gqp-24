#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:21:55 2024

@author: dfox
"""
import math
import os

def get_model(config):
    if 'model' not in config or 'models' not in config:
        raise ValueError("model not configured")
        
    model_name = config['model']
    if model_name not in config['models']:
        raise ValueError(f"model {model_name} not configured")
        
    model_dict = config['models'][model_name]
    
    return model_dict

# TODO: KENDALL: Write this method. No worries about changing anything. Just make it work.
# This method calculates ONE data point on the IAPC
def run_eval(ctxt, baseline_image_size, degraded_image_size, val_degraded_dir_path):
    
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()
    top_dir = ctxt.get_top_dir()
    output_dir = ctxt.get_output_dir_path()
    
    model_dict = get_model(config)
    
    input_image_size = model_dict['input_image_size']
    input_width = input_image_size
    input_height = input_image_size
    num_epochs = model_dict['epochs']
    
    trained_model_filename_template = model_dict['trained_model_filename']
    trained_model_filename = os.path.join(top_dir, config['trained_models_subdir'],
                                          trained_model_filename_template.format(
                                              width=baseline_image_size[0], height=baseline_image_size[1], epochs=num_epochs))

    # TODO: KENDALL: Here run the evaluation code, calculating ONE data point on the IAPC
    # method = config['eval_method'] # detection or classification, only detection supported right now, so maybe leave this
    # comment in here or take this comment out entirely
    # results = run model here, stored in traine_model_filename, config['target_labels'], put result into trained_modelfilename
    # eval_detection
    # write results to some structured file, coordinate with DAN
    # ALSO, could cache the results in ctxt (class Pipeline in src/pipeline.py). That would be very useful.
    
    # TODO: KENDALL: Dummy value, put in real one. Note this routine doesn't have to return anything.
    mAP = 0.0
    return mAP                                                      

