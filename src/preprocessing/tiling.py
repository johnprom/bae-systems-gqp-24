import geojson
import os
import shutil

from preprocessing.Tiling_images_to_yolo import XViewTiler

# image tiling scheme
def tiling(ctxt):
    print("Begin tiling approach")
    
    config = ctxt.get_pipeline_config() # dictionary of yaml configuration, in case you need it
    
    tile_size = config['preprocess_methods']['tiling']['image_size']
    overlap = config['preprocess_methods']['tiling']['stride']
    tiler = XViewTiler(tile_size=tile_size, overlap=overlap)
    
    geojson_path = ctxt.get_train_geojson_filename()
    with open(geojson_path, 'r') as f:
        gj = geojson.load (f)
    features = gj['features']
    
    result_dict = tiler.get_class_wise_data(features, ctxt.input_images_dir, ctxt.target_labels, ctxt.train_labels)
    
    tiler.tile_image_and_save(result_dict, ctxt.input_images_dir, ctxt.interim_images_dir, ctxt.interim_images_dir)

    print("Finished running tiling approach.")
