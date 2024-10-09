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

    preprocess_top_dir = ctxt.get_preprocessing_dir_path()
    method = ctxt.config['preprocess_method'] # Currently 'padding' or 'tiling'
    
    params = ctxt.config['preprocess_methods'][method]
    train_template = params['train_baseline_subdir']
    val_template = params['val_baseline_subdir']
    
    method = config['preprocess_method'] # Currently 'padding' or 'tiling'

    # Method-specific code    
    # use ctxt.interim_images_dir to temporarily place preprocessed files prior to train_test_split
    if method == 'padding':
        ctxt.maxwidth, ctxt.maxheight = padding_and_annotation_adjustment(ctxt) 
        ctxt.train_baseline_dir = os.path.join(preprocess_top_dir, train_template.format(maxwidth=ctxt.maxwidth, 
                                                                                         maxheight=ctxt.maxheight))
        ctxt.val_baseline_dir = os.path.join(preprocess_top_dir, val_template.format(maxwidth=ctxt.maxwidth, 
                                                                                     maxheight=ctxt.maxheight))
    elif method == 'tiling':
        tiling(ctxt)
        image_size = params['image_size']
        ctxt.maxwidth = image_size
        ctxt.maxheight = image_size
        stride = params['stride']
        ctxt.train_baseline_dir = os.path.join(preprocess_top_dir, 
                                               train_template.format(maxwidth=ctxt.maxwidth, maxheight=ctxt.maxheight, 
                                                                     stride=stride))
        ctxt.val_baseline_dir = os.path.join(preprocess_top_dir, 
                                             val_template.format(maxwidth=ctxt.maxwidth, maxheight=ctxt.maxheight, stride=stride))
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
