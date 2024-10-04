from util.util import get_class_name_from_id, load_annotations_master, update_data_config_class_count, update_data_config_class_names, write_to_annotations_filtered

def filter_classes(ctxt, target_class_ids: list[int]):
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
    # map target class ids to class names for YOLO data config
    target_class_names = [get_class_name_from_id(ctxt, id_key) for id_key in target_class_ids]
    # update YOLO data config
    update_data_config_class_names(ctxt, target_class_names)
    update_data_config_class_count(ctxt, len(target_class_names)) 

    print("Finished class filtering.")
    