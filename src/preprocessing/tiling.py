import geojson
import os
import shutil

from preprocessing.Tiling_images_to_yolo import XViewTiler

# image tiling scheme
def tiling(ctxt):
    """
    Applies a tiling approach to preprocess images and annotations, dividing images into smaller tiles 
    with overlap and saving them for further training.

    Args:
        ctxt: Context object containing configuration, file paths, verbosity, and tiling parameters.

    Workflow:
        - **Initialize Tiler**: Retrieves tile size and overlap (stride) from the configuration and creates an instance of `XViewTiler`.
        - **Load GeoJSON Annotations**: Loads annotations from the GeoJSON file to retrieve bounding box information.
        - **Organize Annotations**: Groups features by class and image for tiling using the `get_class_wise_data` method.
        - **Tile and Save Images**: Splits each image into tiles and saves the tiles and corresponding annotations to the interim directory.
        - **Logging**: Outputs process details if verbose mode is enabled.

    Returns:
        None: This function performs tiling and saves the results directly to the interim images directory.
    """

    if ctxt.verbose:
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

    if ctxt.verbose:
        print("Finished running tiling approach.")
