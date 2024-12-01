import math
import os
import random
import shutil
from util.util import update_data_config_train_path, update_data_config_val_path
from preprocessing.tiling import tiling
from preprocessing.padding import padding_and_annotation_adjustment
from preprocessing.class_filtering import filter_classes

# TODO: Move larger method to their own files

# def clean_preprocessing(config):
#     pp_dir = get_preprocessing_dir_path(config)
#     if os.path.exists(pp_dir):
#         shutil.rmtree(pp_dir)
#     print("Finished clearing preprocessed data folders.")

def train_test_split(ctxt):
    """
    Splits images and their annotations into training and validation sets based on a specified ratio
    and copies them into respective directories for model training and validation.

    Args:
        ctxt: Context object containing configuration, file paths, and verbosity settings.

    Workflow:
        - Retrieves the train-validation split ratio from the configuration.
        - Lists all images in the interim directory with supported extensions.
        - Calculates the number of images for the training set based on the split ratio.
        - Randomly selects training images and assigns the remaining to validation.
        - Creates lists of corresponding annotation files for each image set.
        - Copies images and annotations to the appropriate train/val directories.

    Returns:
        None: Files are directly copied into training and validation directories specified in `ctxt`.
    """
    config = ctxt.get_pipeline_config()
    train_split = config['preprocess']['train_split'] # 0.0 to 1.0
    # input_images_dir = os.path.join(config['top_dir'], config['input_images_subdir'])
    
    image_ext = ['.tif', '.tiff', '.png', '.gif', '.jpg', '.jpeg']
    
    interim_images_list = [x for x in os.listdir(ctxt.interim_images_dir) if x[-4:] in image_ext or x[-5:] in image_ext]
    
    random.seed(42)
    num_train = math.ceil(len(interim_images_list) * train_split)
    
    # Split this list into a random train list and a random val list 
    ctxt.train_images_list = random.sample(interim_images_list, num_train) # already initialized to []
    ctxt.val_images_list = [x for x in interim_images_list if x not in ctxt.train_images_list] # already initialized to []
    
    ctxt.train_labels_list = [(x.split('.')[-2] +'.txt') for x in ctxt.train_images_list]
    ctxt.val_labels_list = [(x.split('.')[-2] +'.txt') for x in ctxt.val_images_list]
    
    for filename in ctxt.train_images_list:
        if ctxt.verbose:
            print(f"copying file {filename} into directory {ctxt.train_baseline_dir}")
        shutil.copy2(os.path.join(ctxt.interim_images_dir, filename), ctxt.train_baseline_dir)
    for filename in ctxt.val_images_list:
        shutil.copy2(os.path.join(ctxt.interim_images_dir, filename), ctxt.val_baseline_dir)
    
    for filename in ctxt.train_labels_list:
        full_path = os.path.join(ctxt.interim_images_dir, filename)
        if os.path.exists(full_path):
            if ctxt.verbose:
                print(f"copying file {filename} into directory {ctxt.train_baseline_dir}")
            shutil.copy2(full_path, ctxt.train_baseline_dir)
    for filename in ctxt.val_labels_list:
        full_path = os.path.join(ctxt.interim_images_dir, filename)
        if os.path.exists(full_path):
            if ctxt.verbose:
                print(f"copying file {filename} into directory {ctxt.val_baseline_dir}")
            shutil.copy2(full_path, ctxt.val_baseline_dir)

    if ctxt.verbose:
        print(f"Finished splitting baseline data into train ({train_split}) and val ({1.0-train_split}) sets.")

# main entry point for preprocessing
def run_preprocessing(ctxt):
    """
    Runs the preprocessing pipeline, which includes cleaning directories, filtering classes, 
    and preparing images for training and validation.

    Args:
        ctxt: Context object containing configuration settings, file paths, and methods for preprocessing.

    Workflow:
        - **Clean Directories**: Clears output and interim directories if specified in the configuration.
        - **Filter Classes**: Filters annotations to retain only target classes specified in the configuration.
        - **Preprocessing Method**: Applies the specified preprocessing method:
            - `padding`: Adjusts image size with padding.
            - `tiling`: Divides images into tiles.
        - **Set Directories**: Creates train and validation directories based on the preprocessing method.
        - **Split Data**: Performs a train-test split on the preprocessed images.
        - **Update Config Paths**: Updates YOLO configuration with paths for training and validation datasets.

    Returns:
        None: This function directly modifies file directories and updates paths in YOLO data configuration.
    """
    print("Start preprocessing")
    config = ctxt.get_pipeline_config()
    output_top_dir = ctxt.get_preprocessing_dir_path()

    if 'clean_subdir' in config['preprocess'] and config['preprocess']['clean_subdir']:
        if os.path.exists(output_top_dir):
            shutil.rmtree(output_top_dir)

    os.makedirs(output_top_dir, exist_ok=True)
    
    # Temporary directory for placing preprocessed images in before train_test_split
    if os.path.exists(ctxt.interim_images_dir):
        shutil.rmtree(ctxt.interim_images_dir)

    os.makedirs(ctxt.interim_images_dir, exist_ok=True)
    
    filter_classes(ctxt, list(config['target_labels'].keys()))

    method = config['preprocess_method'] # Currently 'padding' or 'tiling'

    # Method-specific code    
    # use ctxt.interim_images_dir to temporarily place preprocessed files prior to train_test_split
    if method == 'tiling':
        tiling(ctxt)
    else:
        raise ValueError("Unknown preprocessing method: " + method)

    os.makedirs(ctxt.train_baseline_dir, exist_ok=True)
    os.makedirs(ctxt.val_baseline_dir, exist_ok=True)

    # The following randomly creates a train and val list of image filenames. It does NOT copy the images to anywhere.
    # The results are put into the ctxt, which is a singleton instantiated object of class Pipeline
    # They are put into ctxt.train_images_list and ctxt.val_images_list
    # And then they are placed in the correct directory for use with training and evaluation
    # These are ctxt.train_baseline_dir and ctxt.val_baseline_dir
    train_test_split(ctxt)
    
    # deletion of temporary directory for preprocessed images before train_test_split
    if os.path.exists(ctxt.interim_images_dir):
        shutil.rmtree(ctxt.interim_images_dir)
    
    # set path to the baseline preprocessed dataset in YOLO data config
    update_data_config_train_path(ctxt, ctxt.train_baseline_dir)
    update_data_config_val_path(ctxt, ctxt.val_baseline_dir)

    print("Exit Preprocessing")
