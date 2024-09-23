# REFERENCE CODE 
# PLUMBING IS NOT DONE YET FOR THIS TO WORK

import os
import rasterio
import geopandas as gpd
import json

def pixel_to_relative_coords(pixel_coords, og_x, og_y, max_len):
    # [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = map(int, pixel_coords.split(','))
    # add offset to account for padding
    # this approach assumes "adding" the extra pixel to right/bottom instead of splitting a pixel for odd diffs
    x_diff = (max_len - og_x) // 2
    y_diff = (max_len - og_y) // 2
    xmin = xmin + x_diff
    xmax = xmax + x_diff
    ymin = ymin + y_diff
    ymax = ymax + y_diff
    # <x_center> <y_center> <width> <height>
    x_center = (xmin + ((xmax - xmin + 1)/2)) / max_len
    y_center = (ymin + ((ymax - ymin + 1)/2)) / max_len
    width = (xmax - xmin + 1) / max_len
    height = (ymax - ymin + 1) / max_len
    relative_coords = " ".join(map(str, [round(x_center,6), round(y_center,6), round(width,6), round(height,6)]))
    return relative_coords

directory = '../../xview_dataset_raw/train_images' # dir with og images
max_img_size = 4224 

# THIS IS TEMPORARY PATH
# we need to either temp store the preprocessed images first before binning them into train and test dirs
#  or.. do the split first
#  or.. abstract this method enough so we can pass in paths

target_folder = '../../preprocessed_datasets/dataset_1/labels' # dir for new annotation files
# TODO - Filter out annoations for unwanted object classes first (filter by type_id val)

# iterate through all files in the target directory
for filename in os.listdir(directory):
    if filename.endswith('.tif'):

        file_path = os.path.join(directory, filename)
        with rasterio.open(file_path) as src:
            pixel_x = src.width
            pixel_y = src.height
            target_img = filename
        print("target_img " + target_img)

        txt_filename = os.path.join(target_folder, target_img.replace('.tif', '.txt'))

        # load the og GeoJSON file
        with open('annotations_master/xView_train.geojson') as f:
            geojson_data = json.load(f)

        # iterate through features and filter by image_id
        filtered_features = [
            feature for feature in geojson_data['features']
            if feature['properties'].get('image_id') == target_img
        ]

        # write to annotation file (overwrite if it exists)
        with open(txt_filename, 'w') as txt_file:
            for feature in filtered_features:
                properties = feature['properties']
                coords = properties['bounds_imcoords']
                class_label = properties['type_id']
                image_id = properties['image_id']

                relative_coords = pixel_to_relative_coords(coords, pixel_x, pixel_y, max_img_size)

                # <class_id> <x_center> <y_center> <width> <height>
                feat_new_debug = "DEBUG - coords: " + coords + " relative: " + relative_coords + " label: " + str(class_label) + " img_id: " + image_id + " ogxy: " + str(pixel_x) + "," + str(pixel_y)
                feat_new = str(class_label) + " " + relative_coords

                txt_file.write(feat_new_debug + '\n' + feat_new + '\n') 

        print(f"Created or overwrote: '{txt_filename}'")
