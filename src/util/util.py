import json
import os
import yaml
from enum import Enum

# enum for different preprocessing approaches
class ApproachType(Enum):
    TILING = "tiling"
    PADDING = "padding"

def load_pipeline_config():
    config_path = os.path.join(os.path.dirname(__file__), '../..', 'pipeline_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# update a field in the YOLO data config
def update_yaml_file(update_field, new_value):
    data_config_path = os.path.join(os.path.dirname(__file__), '../..', 'data_config.yaml')
    with open(data_config_path, 'r') as file:
        data = yaml.safe_load(file)
    data[update_field] = new_value
    with open(data_config_path, 'w') as file:
        yaml.dump(data, file)

def update_data_config_train_path(path):
    update_yaml_file('train', path)

def update_data_config_val_path(path):
    update_yaml_file('val', path)

def update_data_config_class_count(class_count):
    update_yaml_file('nc', class_count)

def update_data_config_class_names(class_list):
    update_yaml_file('names', class_list)

def load_class_mappings():
    json_file_path = os.path.join(os.path.dirname(__file__), '../../xview_dataset_raw/', 'xview_class_labels.json')
    with open(json_file_path, 'r') as file:
        mappings = json.load(file)
    return mappings

def get_class_name_from_id(id_key):
    mappings = load_class_mappings()
    return mappings.get(str(id_key), "Unknown ID")

def load_annotations_master():
    file_path = os.path.join(os.path.dirname(__file__), '../../xview_dataset_raw/', 'xView_train.geojson')
    with open(file_path) as f:
        geojson_data = json.load(f)
    return geojson_data

def write_to_annotations_filtered(filtered_annotations):
    output_file = os.path.join(os.path.dirname(__file__), '../../preprocessed_datasets/', 'filtered_labels.json')
    with open(output_file, 'w') as f:
        json.dump(filtered_annotations, f, indent=4)
