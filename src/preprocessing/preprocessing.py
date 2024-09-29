from util.util import ApproachType, update_data_config_train_path, update_data_config_val_path

from preprocessing.tiling import tiling
from preprocessing.padding import padding
from preprocessing.class_filtering import filter_classes

# TODO: Move larger method to their own files

def clear_preprocessed_files():
    print("Finished clearing preprocessed data folders.")

def train_test_split(train_split: int):
    print("Finished splitting baseline data into train and val sets.")

# main entry point for preprocessing
def run_preprocessing(params):
    if params['skip_step']:
        print('Skipping Preprocessing Module.')
    else:
        # clear contents of preprocessed data folders
        #  -> clear:
        #       '../preprocessed_datasets/*'
        if params['clear_data']:
            clear_preprocessed_files()

        # filter classes in main annotation file
        #  -> read from: 
        #       '../xview_dataset_raw/xView_train.geojson' : og master label file
        #  -> write to:
        #       '../preprocessed_datasets/filtered_labels.geojson' : master label file filtered for only target classes
        #       '../data_config.yaml  : reflect these changes in the YOLO data config
        filter_classes(params['target_classes'])

        # perform main preprocessing
        #  -> read from: 
        #       '../xview_dataset_raw/train_images' : folder containing og training images
        #       '../preprocessed_datasets/filtered_labels.geojson' : master label file filtered for only target classes
        #  -> write to:
        #       '../preprocessed_datasets/baseline_unsplit/images/      eg:123.tif
        #       '../preprocessed_datasets/baseline_unsplit/labels/      eg:123.txt
        approach = ApproachType[params['approach']]
        match approach:
            case ApproachType.TILING:
                tiling(params['imgsz'], params['stride'])
            case ApproachType.PADDING:
                padding(params['imgsz'])
            case _:
                raise ValueError("Unknown preprocessing approach:" + str(approach))

        # preform test/train split
        #  -> read from: 
        #       '../preprocessed_datasets/baseline_unsplit/images/
        #       '../preprocessed_datasets/baseline_unsplit/labels/
        #  -> write to:
        #       '../preprocessed_datasets/baseline-<approach>-<max_dim>-<imgsz>/train/images/
        #       '../preprocessed_datasets/baseline-<approach>-<max_dim>-<imgsz>/train/labels/
        #       '../preprocessed_datasets/baseline-<approach>-<max_dim>-<imgsz>/val/images/
        #       '../preprocessed_datasets/baseline-<approach>-<max_dim>-<imgsz>/val/labels/
        train_test_split(params['train_split'])

        # set path to the baseline preprocessed dataset in YOLO data config
        update_data_config_train_path('../preprocessed_datasets/baseline-<approach>-<max_dim>-<imgsz>/train')
        update_data_config_val_path('../preprocessed_datasets/baseline-<approach>-<max_dim>-<imgsz>/val')

        print("Exit Preprocessing with: " + str(params))
