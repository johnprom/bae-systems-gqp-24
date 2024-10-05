import geojson
import os
import shutil

from preprocessing.Tiling_images_to_yolo import XViewTiler

# image tiling scheme
def tiling(ctxt):
    print("Begin tiling approach")
    
    config = ctxt.get_pipeline_config() # dictionary of yaml configuration, in case you need it
    
    # TODO: AMIT: Remove this code block entirely
    # Dummy code for testing - for now just copies images and text files, does no processing on them
    # image_filelist = [x for x in os.listdir(ctxt.input_images_dir) if x.endswith('tif')]
    # label_filelist = [x for x in os.listdir(ctxt.input_images_dir) if x.endswith('txt')]
    # for filename in image_filelist:
    #     shutil.copy2(os.path.join(ctxt.input_images_dir, filename), ctxt.interim_images_dir)
    # for filename in label_filelist:
    #     shutil.copy2(os.path.join(ctxt.input_images_dir, filename), ctxt.interim_images_dir)
    # TODO: AMIT: End of code block to remove

    # Note: The following is Amit's code. It has been tested on the "happy path." He gets the credit for it.
    tile_size = config['preprocess_methods']['tiling']['image_size']
    overlap = config['preprocess_methods']['tiling']['stride']
    tiler = XViewTiler(tile_size=tile_size, overlap=overlap)
    
    geojson_path = ctxt.get_train_geojson_filename()
    with open(geojson_path, 'r') as f:
        gj = geojson.load (f)
    features = gj['features']
    
    # TODO: AMIT: This needs to be changed to the specific label we are looking for. 
    # Maybe needs to be added to pipeline_config.yaml? Coordinate with DAN and KENDALL and GABE.
    # For now just use the first label in target_labels in the pipeline_config.yaml file
    target_label = list(config['target_labels'].keys())[0] 

    result_dict = tiler.get_class_wise_data(features, ctxt.input_images_dir, target_label)
    # TODO: AMIT: The following code puts the .txt files in the same directory as the .png files (ctxt.interim_images_dir)
    # Is this okay? If not, add a configuration entry to pipeline_config.yaml file. Coordinate with DAN.
    tiler.tile_image_and_save(result_dict, ctxt.input_images_dir, ctxt.interim_images_dir, ctxt.interim_images_dir)

    print("Finished running tiling approach.")
