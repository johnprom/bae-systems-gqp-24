import geojson
import json
import os
import shutil
import yaml
from enum import Enum

def load_pipeline_config(config_path):
    # top_path = os.path.join(os.path.dirname(__file__), '..')
    # config_path = os.path.join(top_path, 'pipeline_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# update a field in the YOLO data config
def update_yaml_file(ctxt, update_field, new_value):
    data_config_path = ctxt.get_data_config_dir_path()
    with open(data_config_path, 'r') as file:
        data = yaml.safe_load(file)
    data[update_field] = new_value
    with open(data_config_path, 'w') as file:
        yaml.dump(data, file)

    data_config_path = ctxt.get_data_config_eval_dir_path()
    with open(data_config_path, 'r') as file:
        data = yaml.safe_load(file)
    data[update_field] = new_value
    with open(data_config_path, 'w') as file:
        yaml.dump(data, file)

def update_data_config_train_path(ctxt, path):
    update_yaml_file(ctxt, 'train', path)

def update_data_config_val_path(ctxt, path):
    update_yaml_file(ctxt, 'val', path)

def update_data_config_class_count(ctxt, class_count):
    update_yaml_file(ctxt, 'nc', class_count)

def update_data_config_class_names(ctxt, class_list):
    update_yaml_file(ctxt, 'names', class_list)
    ctxt.class_names = class_list

def load_class_mappings(ctxt):
    json_file_path = os.path.join(ctxt.get_input_images_dir_path(), ctxt.config['input_class_labels_filename'])
    with open(json_file_path, 'r') as file:
        mappings = json.load(file)
    return mappings

def get_class_name_from_id(ctxt, id_key):
    mappings = load_class_mappings(ctxt)
    return mappings.get(str(id_key), "Unknown ID")

def load_annotations_master(ctxt):
    file_path = os.path.join(ctxt.get_input_images_dir_path(), 'xView_train.geojson')
    with open(file_path) as f:
        geojson_data = geojson.load(f)
    return geojson_data

def write_to_annotations_filtered(ctxt, filtered_annotations):
    output_file = ctxt.get_filtered_labels_filename()
    with open(output_file, 'w') as f:
        json.dump(filtered_annotations, f, indent=4)

def get_preprocessed_images_dir_path(ctxt, transform_type, split, maxres):
    config = ctxt.get_pipeline_config()
    output_dir = os.path.join(ctxt.get_top_dir(), config['preprocess']['output_subdir'])
    method = config['preprocess_method']

    if method == 'tiling':
        params = config['preprocess_methods']['tiling']
        if transform_type == 'baseline':
            if split == 'train':
                template = params['train_baseline_subdir']
            elif split == 'val':
                template = params['val_baseline_subdir']
            else:
                raise ValueError("split not specified correctly")
            width = params['image_size']
            height = params['image_size']
            stride = params['stride']
            return os.path.join(output_dir, template.format(width=width, height=height, stride=stride))
        elif transform_type == 'degraded':
            if split == 'train':
                template = params['train_degraded_subdir']
            elif split == 'val':
                template = params['val_degraded_subdir']
            else:
                raise ValueError("split not specified correctly")
            width = params['image_size']
            height = params['image_size']
            stride = params['stride']
            return os.path.join(output_dir, template.format(width=width, height=height, stride=stride,
                                                            effective_width=width, effective_height=height))
        else:
            raise ValueError("transform_type not specified correctly")
    
    if method == 'padding':
        params = config['preprocess_methods']['padding']
        if transform_type == 'baseline':
            if split == 'train':
                template = params['train_baseline_subdir']
            elif split == 'val':
                template = params['val_baseline_subdir']
            else:
                raise ValueError("split not specified correctly")
            width = params['image_size']
            height = params['image_size']
            return os.path.join(output_dir, template.format(maxwidth=width, maxheight=height))
        elif transform_type == 'degraded':
            if split == 'train':
                template = params['train_degraded_subdir']
            elif split == 'val':
                template = params['val_degraded_subdir']
            else:
                raise ValueError("split not specified correctly")
            width = params['image_size']
            height = params['image_size']
            return os.path.join(output_dir, template.format(maxwidth=width, maxheight=height,
                                                            effective_width=width, effective_height=height))
        else:
            raise ValueError("transform_type not specified correctly")
    else:
        raise ValueError("Unknown preprocessing method: " + method)
