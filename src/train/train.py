#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:01:23 2024

@author: dfox
"""
import os
from ultralytics import YOLO

# def run_finetuning(config, data_config_path):
def run_finetuning(ctxt):
    
    print("Start Running Fine-tuning.")
            
    config = ctxt.get_pipeline_config()
    data_config_path = ctxt.get_data_config_dir_path()

    if config['model'].startswith('yolo'):
        model_dict = config['models'][config['model']]
        base_model = YOLO(config['model'])
        
        trained_model_filename_template = model_dict['trained_model_filename']
        input_width = model_dict['input_image_size']
        input_height = input_width
        # input_width, input_height = model_dict['input_image_size'][0], model_dict['input_image_size'][1]
        num_epochs = model_dict['epochs']
    else:
        raise ValueError(f"unkown deep learning model {config['model']} specified.")
    
    # TODO: KENDALL? GABE? DAN?
    # Train your model here. The following is just dummy code
    # ft_model = base_model.train(data=data_config_path, epochs=num_epochs, imgsz=input_width)
    
    final_weights_path = os.path.join(ctxt.get_top_dir(), config['trained_models_subdir'], 
                                      trained_model_filename_template.format(width=input_width, height=input_height, 
                                                                             epochs=num_epochs))
    
    # TODO: KENDALL? GABE? DAN?
    # tune the model here, use config.target_labels
    # Save the pretrained weights here. Again, the code below is just an example and not specifically how it should be done
    # Obviously, take out the first print statement
    print(f"final weights would be in {final_weights_path} but tuning code not run yet")
    # base_model.save(final_weights_path) # change to ft_model.save(final_weights_path) when data is there
    print("Finished Running Fine-tuning.")

