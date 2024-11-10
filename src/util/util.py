import geojson
import json
import os
import yaml

def load_pipeline_config(config_path):
    """
    Loads the pipeline configuration from a specified YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration settings.
    """

    # top_path = os.path.join(os.path.dirname(__file__), '..')
    # config_path = os.path.join(top_path, 'pipeline_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# update a field in the YOLO data config
def update_yaml_file(ctxt, update_field, new_value):
    """
    Updates a specific field in a YAML configuration file with a new value.

    Args:
        ctxt: Context object containing configuration and file path methods.
        update_field (str): The field in the YAML file to be updated.
        new_value: The new value to assign to the specified field.

    Returns:
        None: Modifies the YAML file in place.
    """
    data_config_path = ctxt.get_data_config_dir_path()
    with open(data_config_path, 'r') as file:
        data = yaml.safe_load(file)
    data[update_field] = new_value
    with open(data_config_path, 'w') as file:
        yaml.dump(data, file)

def update_data_config_train_path(ctxt, path):
    """
    Updates the 'train' field in the data configuration YAML file with the specified path.

    Args:
        ctxt: Context object containing configuration and file path methods.
        path (str): The new path for training data.

    Returns:
        None: Modifies the YAML file in place.
    """
    update_yaml_file(ctxt, 'train', path)

def update_data_config_val_path(ctxt, path):
    """
    Updates the 'val' field in the data configuration YAML file with the specified path.

    Args:
        ctxt: Context object containing configuration and file path methods.
        path (str): The new path for validation data.

    Returns:
        None: Modifies the YAML file in place.
    """
    update_yaml_file(ctxt, 'val', path)

def update_data_config_class_count(ctxt, class_count):
    """
    Updates the 'nc' field in the data configuration YAML file with the specified class count.

    Args:
        ctxt: Context object containing configuration and file path methods.
        class_count (int): The new class count to update in the configuration.

    Returns:
        None: Modifies the YAML file in place.
    """
    update_yaml_file(ctxt, 'nc', class_count)

def update_data_config_class_names(ctxt, class_list):
    """
    Updates the 'names' field in the data configuration YAML file with the specified list of class names
    and updates the context with the new class names.

    Args:
        ctxt: Context object containing configuration and file path methods.
        class_list (list): The new list of class names to update in the configuration.

    Returns:
        None: Modifies the YAML file in place and updates the context.
    """
    update_yaml_file(ctxt, 'names', class_list)
    ctxt.class_names = class_list

def load_class_mappings(ctxt):
    """
    Loads class label mappings from a JSON file.

    Args:
        ctxt: Context object containing configuration and file path methods.

    Returns:
        dict: A dictionary containing the class label mappings loaded from the JSON file.
    """
    json_file_path = ctxt.get_class_labels_filename()
    with open(json_file_path, 'r') as file:
        mappings = json.load(file)
    return mappings

def get_class_name_from_id(ctxt, id_key):
    """
    Retrieves the class name corresponding to a given ID from the class label mappings.

    Args:
        ctxt: Context object containing configuration and file path methods.
        id_key (int or str): The ID for which the class name is to be retrieved.

    Returns:
        str: The class name corresponding to the given ID. Returns "Unknown ID" if the ID is not found.
    """
    mappings = load_class_mappings(ctxt)
    return mappings.get(str(id_key), "Unknown ID")

def load_annotations_master(ctxt):
    """
    Loads annotation data from a GeoJSON file.

    Args:
        ctxt: Context object containing configuration and file path methods.

    Returns:
        dict: A dictionary containing the annotation data loaded from the GeoJSON file.
    """
    file_path = ctxt.get_train_geojson_filename()
    with open(file_path) as f:
        geojson_data = geojson.load(f)
    return geojson_data

def write_to_annotations_filtered(ctxt, filtered_annotations):
    """
    Writes filtered annotations to a JSON file.

    Args:
        ctxt: Context object containing configuration and file path methods.
        filtered_annotations (dict): The filtered annotations to be written to the file.

    Returns:
        None: Saves the filtered annotations to the specified file.
    """
    output_file = ctxt.get_filtered_labels_filename()
    with open(output_file, 'w') as f:
        json.dump(filtered_annotations, f, indent=4)

def get_preprocessed_images_dir_path(ctxt, transform_type, split, maxres):
    """
    Generates the directory path for preprocessed images based on the preprocessing method, transformation type, data split, and resolution.

    Args:
        ctxt: Context object containing configuration and file path methods.
        transform_type (str): The type of transformation ('baseline' or 'degraded').
        split (str): The data split ('train' or 'val').
        maxres (bool): Indicates if the maximum resolution is used (used for formatting paths).

    Returns:
        str: The directory path for the preprocessed images based on the provided parameters.

    Raises:
        ValueError: If the `transform_type`, `split`, or preprocessing method is not specified correctly.
    """
    config = ctxt.get_pipeline_config()
    output_dir = ctxt.get_preprocessing_dir_path()
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
