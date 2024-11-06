import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import math

# Tile settings
TILE_SIZE = 640
OVERLAP = 100

# Updated xView class to YOLO class mapping
xview_class2index = {
    11: 'Fixed-wing Aircraft',
    12: 'Small Aircraft',
    13: 'Cargo Plane',
    15: 'Helicopter',
    17: 'Passenger Vehicle',
    18: 'Small Car',
    19: 'Bus',
    20: 'Pickup Truck',
    21: 'Utility Truck',
    23: 'Truck',
    24: 'Cargo Truck',
    25: 'Truck w/Box',
    26: 'Truck Tractor',
    27: 'Trailer',
    28: 'Truck w/Flatbed',
    29: 'Truck w/Liquid',
    32: 'Crane Truck',
    33: 'Railway Vehicle',
    34: 'Passenger Car',
    35: 'Cargo Car',
    36: 'Flat Car',
    37: 'Tank Car',
    38: 'Locomotive',
    40: 'Maritime Vessel',
    41: 'Motorboat',
    42: 'Sailboat',
    44: 'Tugboat',
    45: 'Barge',
    47: 'Fishing Vessel',
    49: 'Ferry',
    50: 'Yacht',
    51: 'Container Ship',
    52: 'Oil Tanker',
    53: 'Engineering Vehicle',
    54: 'Tower crane',
    55: 'Container Crane',
    56: 'Reach Stacker',
    57: 'Straddle Carrier',
    59: 'Mobile Crane',
    60: 'Dump Truck',
    61: 'Haul Truck',
    62: 'Scraper/Tractor',
    63: 'Front loader/Bulldozer',
    64: 'Excavator',
    65: 'Cement Mixer',
    66: 'Ground Grader',
    71: 'Hut/Tent',
    72: 'Shed',
    73: 'Building',
    74: 'Aircraft Hangar',
    76: 'Damaged Building',
    77: 'Facility',
    79: 'Construction Site',
    83: 'Vehicle Lot',
    84: 'Helipad',
    86: 'Storage Tank',
    89: 'Shipping container lot',
    91: 'Shipping Container',
    93: 'Pylon',
    94: 'Tower'
}

class XViewTiler:
    def __init__(self, tile_size=TILE_SIZE, overlap=OVERLAP):
        """
        Initializes the tiler with specified tile size and overlap.

        Args:
            tile_size (int): The size of each square tile (default 640).
            overlap (int): The number of overlapping pixels between tiles (default 100).
        """
        self.tile_size = tile_size
        self.overlap = overlap

    def convert_bbox_to_yolo_format(self, bbox, img_w, img_h):
        """
        Converts bounding box coordinates to YOLO format (center coordinates, width, height), normalized by image dimensions.

        Args:
            bbox (list[int]): Bounding box as [x1, y1, x2, y2].
            img_w (int): Width of the image or tile.
            img_h (int): Height of the image or tile.

        Returns:
            list[float]: Bounding box in YOLO format as [x_center, y_center, width, height].
        """
        x1, y1, x2, y2 = bbox
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        bbox_width = (x2 - x1) / img_w
        bbox_height = (y2 - y1) / img_h
        return [x_center, y_center, bbox_width, bbox_height]

    def save_yolo_labels(self, bboxes, classes, output_path):
        """
        Saves bounding boxes and classes in YOLO format to a .txt file.

        Args:
            bboxes (list[list[float]]): List of bounding boxes in YOLO format.
            classes (list[int]): List of class IDs corresponding to each bounding box.
            output_path (str): File path for saving the label file.
        """
        
        with open(output_path, 'w') as f:
            for bbox, cls in zip(bboxes, classes):
                yolo_bbox = ' '.join([str(x) for x in bbox])
                f.write(f"{cls} {yolo_bbox}\n")

    def get_class_wise_data(self, features, file_path, class_id_list, train_id_list):
        """
        Filters and organizes annotations by class and image, based on a specified list of class IDs.

        Args:
            features (list[dict]): List of features from the GeoJSON file with bounding box data.
            file_path (str): Directory path where images are stored.
            class_id_list (list[int]): List of target class IDs to filter.
            train_id_list (list[int]): List of new class IDs for remapping.

        Returns:
            dict: Dictionary with image filenames as keys and bounding box data for each class as values.
        """
        
        for class_id in class_id_list:
            if class_id not in xview_class2index:
                print(f"Class ID {class_id} does not exist in the label dictionary.")
                return {}

        data_dict = {}
        for filename in os.listdir(file_path):
            if filename.endswith('.tif'):
                bounding_boxes = []
                for new_class_id, class_id in enumerate(class_id_list):
                    for feature in features:
                        if feature['properties'].get('image_id') == filename and feature['properties'].get('type_id') == class_id:
                            bounds_str = feature['properties']['bounds_imcoords']
                            coordinates = list(map(int, bounds_str.split(',')))
                            bounding_boxes.append({
                                'class_id': train_id_list[new_class_id],
                                'coordinates': coordinates
                            })
                    if bounding_boxes:
                        data_dict[filename] = bounding_boxes
        return data_dict

    def tile_image_and_save(self, data_dict, file_path, output_img_dir, output_lbl_dir):
        """
        Tiles images from `data_dict`, adjusts bounding boxes for each tile, and saves the tiles and labels.

        Args:
            data_dict (dict): Dictionary where each key is an image filename, and the value is a list of bounding box annotations.
            file_path (str): Directory path where the original images are stored.
            output_img_dir (str): Directory path to save the tiled image files.
            output_lbl_dir (str): Directory path to save YOLO format label files.

        Workflow:
            - **Load Image**: Loads each image specified in `data_dict`.
            - **Calculate Tiling Steps**: Determines the number of tiles needed in both x and y directions, based on tile size and overlap.
            - **Generate Tiles**: Iterates over each tile position, crops the tile, and adjusts bounding boxes to fit within the tile.
            - **Filter and Save Annotations**: Filters bounding boxes within each tile and saves them in YOLO format along with the tile image.
        """
        for img_id, annotations in data_dict.items():
            img_path = os.path.join(file_path, img_id)
            if not os.path.exists(img_path):
                print(f"Image {img_id} not found.")
                continue

            img = Image.open(img_path)
            img_w, img_h = img.size
            x_steps = math.ceil((img_w - self.overlap) / (self.tile_size - self.overlap))
            y_steps = math.ceil((img_h - self.overlap) / (self.tile_size - self.overlap))

            for x in range(x_steps):
                for y in range(y_steps):
                    x_start = max(x * (self.tile_size - self.overlap), 0)
                    y_start = max(y_start := y * (self.tile_size - self.overlap), 0)
                    x_end = min(x_start + self.tile_size, img_w)
                    y_end = min(y_start + self.tile_size, img_h)
                    ################
                    tile_width = x_end - x_start  # New line to calculate tile width
                    tile_height = y_end - y_start
                    if tile_width != self.tile_size and tile_height != self.tile_size:
                        continue  # Skip this tile if neither dimension matches targetTileSize (640)
                    #################
                    tile = img.crop((x_start, y_start, x_end, y_end))

                    tile_bboxes = []
                    tile_classes = []

                    for annotation in annotations:
                        cls_id = annotation['class_id']
                        bbox = annotation['coordinates']

                        if (bbox[0] >= x_start and bbox[2] <= x_end) and (bbox[1] >= y_start and bbox[3] <= y_end):
                            adjusted_bbox = [
                                bbox[0] - x_start,
                                bbox[1] - y_start,
                                bbox[2] - x_start,
                                bbox[3] - y_start
                            ]
                            tile_bboxes.append(self.convert_bbox_to_yolo_format(adjusted_bbox, self.tile_size, self.tile_size))
                            tile_classes.append(cls_id)

                    if tile_bboxes:
                        tile_filename = f"{img_id.split('.')[0]}_{x}_{y}.png"
                        tile.save(os.path.join(output_img_dir, tile_filename))

                        label_filename = f"{img_id.split('.')[0]}_{x}_{y}.txt"
                        self.save_yolo_labels(tile_bboxes, tile_classes, os.path.join(output_lbl_dir, label_filename))

