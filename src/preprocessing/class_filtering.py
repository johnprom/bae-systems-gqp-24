from util.util import get_class_name_from_id, load_annotations_master, update_data_config_class_count, update_data_config_class_names, write_to_annotations_filtered
import math

def calculate_average_pixel_area(filtered_features):
    """
    Calculate the average pixel per object class in pixels.

    Args:
        filtered_features (list): List of filtered feature dictionaries containing 'bounds_imcoords' and 'type_id'.

    Returns:
        dict: Dictionary of average pixel area per class, rounded up to the nearest integer greater than or equal to 1.
    """

    # initialize an empty dict to store areas by class
    class_pixel_areas = {}

    # calculate pixel areas for each feature by class
    for feature in filtered_features:
        type_id = feature['properties']['type_id']
        bounds = feature['properties']['bounds_imcoords']
        x_min, y_min, x_max, y_max = map(int, bounds.split(','))
        
        # calculate area (in pixels of bounding box)
        area = (x_max - x_min + 1) * (y_max - y_min + 1)
        
        # append to list for this class ID
        if type_id not in class_pixel_areas:
            class_pixel_areas[type_id] = []
        class_pixel_areas[type_id].append(area)

    # calculate average pixel area per class
    # round up to nearest int with min val 1
    class_avg_areas = {}
    for cls, areas in class_pixel_areas.items():
        avg_area = math.ceil(sum(areas) / len(areas)) if areas else 0
        class_avg_areas[cls] = max(avg_area, 1)

    # log dictionary
    print("Average areas per class in pixels:", class_avg_areas)
    
    return class_avg_areas


def filter_classes(ctxt, target_class_ids: list[int]):
    """
    Filters the features in the original GeoJSON annotations based on specified class IDs
    and updates the YOLO configuration accordingly.

    Args:
        ctxt: Context object containing configuration, file paths, and other settings.
        target_class_ids (list[int]): List of class IDs to keep in the filtered annotations.
                                      These IDs will be remapped to start from 0.

    Workflow:
        - Loads the original GeoJSON annotations from the master file.
        - Filters the features to retain only those whose `type_id` matches one of `target_class_ids`.
        - For each retained feature, keeps only:
            - 'bounds_imcoords': Bounding box coordinates.
            - 'type_id': Remapped to start from 0 based on the position in `target_class_ids`.
            - 'image_id': The ID of the image the annotation belongs to.
        - Saves the filtered annotations to a new GeoJSON file.
        - Retrieves class names for each `target_class_id` and updates the YOLO configuration:
            - Sets the `class_names` in YOLO data configuration.
            - Sets the total `class_count` in YOLO data configuration.

    Returns:
        None: Writes output directly to a new GeoJSON file and updates YOLO configuration.

    """

    
    # load the original GeoJSON file
    geojson_data = load_annotations_master(ctxt)
    # if the feature is part of our target set keep and retain:
    #   type_id, bounds_imcoords, image_id
    # replace the value of 'type_id' with the index of the matching value from the target set
    # this starts the class ids from 0 in the order they are stated in config (eg: 21,34,55 -> 0,1,2)
    filtered_features = [
        {
            'properties': {
                'bounds_imcoords': feature['properties']['bounds_imcoords'],
                'type_id': target_class_ids.index(feature['properties'].get('type_id')),
                'image_id': feature['properties']['image_id']
            }
        }
        for feature in geojson_data['features']
        if feature['properties'].get('type_id') in target_class_ids
    ]
    write_to_annotations_filtered(ctxt, filtered_features)

    # store object class-id to avg pixel area dict in ctxt for use in report generation
    ctxt.object_sizes = calculate_average_pixel_area(filtered_features)

    # map target class ids to class names for YOLO data config
    target_class_names = [get_class_name_from_id(ctxt, id_key) for id_key in target_class_ids]

    # Store the class names in ctxt for use in knee_discovery.py
    ctxt.class_names = target_class_names
    
    # update YOLO data config
    update_data_config_class_names(ctxt, target_class_names)
    update_data_config_class_count(ctxt, len(target_class_names)) 

    print("Finished class filtering.")
    
