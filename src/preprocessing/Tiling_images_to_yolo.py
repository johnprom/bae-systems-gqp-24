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
        self.tile_size = tile_size
        self.overlap = overlap

    def convert_bbox_to_yolo_format(self, bbox, img_w, img_h):
        """Convert (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height) normalized."""
        x1, y1, x2, y2 = bbox
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        bbox_width = (x2 - x1) / img_w
        bbox_height = (y2 - y1) / img_h
        return [x_center, y_center, bbox_width, bbox_height]

    def save_yolo_labels(self, bboxes, classes, output_path):
        """Save YOLO format labels to a .txt file."""
        with open(output_path, 'w') as f:
            for bbox, cls in zip(bboxes, classes):
                yolo_bbox = ' '.join([str(x) for x in bbox])
                f.write(f"{cls} {yolo_bbox}\n")

    def get_class_wise_data(self, features, file_path, class_id):
        """Filter class-wise data for a specific class."""
        if class_id not in xview_class2index:
            print(f"Class ID {class_id} does not exist in the label dictionary.")
            return {}

        data_dict = {}
        for filename in os.listdir(file_path):
            if filename.endswith('.tif'):
                image_path = os.path.join(file_path, filename)
                bounding_boxes = []
                for feature in features:
                    if feature['properties'].get('image_id') == filename and feature['properties'].get('type_id') == class_id:
                        bounds_str = feature['properties']['bounds_imcoords']
                        coordinates = list(map(int, bounds_str.split(',')))
                        bounding_boxes.append({
                            'class_id': feature['properties'].get('type_id'),
                            'coordinates': coordinates
                        })
                if bounding_boxes:
                    data_dict[filename] = bounding_boxes
        return data_dict

    def tile_image_and_save(self, data_dict, file_path, output_img_dir, output_lbl_dir):
        """
        Function to tile images from a data_dict, adjust bounding boxes for each tile, and save the tiles and labels.
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

                    tile = img.crop((x_start, y_start, x_end, y_end))

                    tile_bboxes = []
                    tile_classes = []

                    for annotation in annotations:
                        cls = annotation['class_id']
                        bbox = annotation['coordinates']

                        if (bbox[0] >= x_start and bbox[2] <= x_end) and (bbox[1] >= y_start and bbox[3] <= y_end):
                            adjusted_bbox = [
                                bbox[0] - x_start,
                                bbox[1] - y_start,
                                bbox[2] - x_start,
                                bbox[3] - y_start
                            ]
                            tile_bboxes.append(self.convert_bbox_to_yolo_format(adjusted_bbox, self.tile_size, self.tile_size))
                            tile_classes.append(cls)

                    if tile_bboxes:
                        tile_filename = f"{img_id.split('.')[0]}_{x}_{y}.png"
                        tile.save(os.path.join(output_img_dir, tile_filename))

                        label_filename = f"{img_id.split('.')[0]}_{x}_{y}.txt"
                        self.save_yolo_labels(tile_bboxes, tile_classes, os.path.join(output_lbl_dir, label_filename))

        print("Tiling and label generation complete!")

# To execute
# if __name__ == "__main__":
#     tiler = XViewTiler(tile_size=640, overlap=100)

#     # Load geojson
#     with open('xView_train.geojson', 'r') as f:
#         gj = json.load(f)
#     features = gj['features']

#     # Get data for a specific class
#     file_path = 'train_images'
#     output_img_dir = 'output_images'
#     output_lbl_dir = 'output_labels'
#     class_id = 21  # Utility Truck

#     result_dict = tiler.get_class_wise_data(features, file_path, class_id)

#     # Tile and save
#     tiler.tile_image_and_save(result_dict, file_path, output_img_dir, output_lbl_dir)
