import os
from PIL import Image
from util.util import write_to_annotations_filtered

# Function to pad the image
def pad_image(image, target_size):
    """
    Pads an image to the target size, ensuring the aspect ratio remains.
    Args:
    - image: PIL Image object to be padded.
    - target_size: The size to pad the image to (e.g., 5120x5120).
    """
    width, height = image.size
    left_padding = (target_size - width) // 2
    right_padding = target_size - width - left_padding
    top_padding = (target_size - height) // 2
    bottom_padding = target_size - height - top_padding

    # Create new padded image with a black background
    padded_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    padded_image.paste(image, (left_padding, top_padding))

    return padded_image, left_padding, top_padding

# Function to adjust bounding box annotations for padding and convert to YOLO format
def adjust_bboxes_and_convert_to_yolo(bboxes, left_padding, top_padding, original_width, original_height, target_size):
    """
    Adjust bounding boxes to account for padding and convert them to YOLO format.
    Args:
    - bboxes: List of bounding boxes in the format (xmin, ymin, xmax, ymax).
    - left_padding: Pixels added to the left side during padding.
    - top_padding: Pixels added to the top side during padding.
    - original_width: Original width of the image.
    - original_height: Original height of the image.
    - target_size: Final size of the padded image.

    Returns:
    - List of bounding boxes in YOLO format: <class_id> <x_center> <y_center> <width> <height>
    """
    adjusted_bboxes = []
    
    for bbox in bboxes:
        # Split the bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, bbox['properties']['bounds_imcoords'].split(','))
        class_id = bbox['properties']['type_id']

        # Adjust bounding box coordinates to account for padding
        xmin = xmin + left_padding
        xmax = xmax + left_padding
        ymin = ymin + top_padding
        ymax = ymax + top_padding

        # Convert to YOLO format (relative to the padded image size)
        x_center = (xmin + xmax) / 2 / target_size
        y_center = (ymin + ymax) / 2 / target_size
        width = (xmax - xmin) / target_size
        height = (ymax - ymin) / target_size

        # Append the bounding box in YOLO format
        adjusted_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return adjusted_bboxes

def find_max_image_size(image_folder):
    """ Find the maximum dimensions among all images in the folder """
    max_width = 0
    max_height = 0
    for filename in os.listdir(image_folder):
        if filename.endswith('.tif'):
            image_path = os.path.join(image_folder, filename)
            with Image.open(image_path) as img:
                if img.width > max_width:
                    max_width = img.width
                if img.height > max_height:
                    max_height = img.height
    max_side = max(max_width, max_height)
    return (max_side + 31) // 32 * 32 # round up to the nearest multiple of 32

# Padding function that incorporates annotations
def padding_and_annotation_adjustment(ctxt):
    input_folder = ctxt.input_images_dir
    labels_folder = ctxt.interim_images_dir  # Folder for filtered annotations
    output_folder = ctxt.interim_images_dir  # Folder for processed images and annotations
    os.makedirs(output_folder, exist_ok=True)
    
    target_size = find_max_image_size(input_folder)  # Dynamically calculate target size
    if ctxt.verbose:
        print(f"Max image size for padding: {target_size}x{target_size}")

    # Loop through all images in the input folder
    for img_file in os.listdir(input_folder):
        if img_file.endswith('.tif'):
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path)
            padded_img, left_padding, top_padding = pad_image(img, target_size)
            
            # Save the padded image (converting to .png)
            padded_img.save(os.path.join(output_folder, img_file.replace('.tif', '.png')))
            if ctxt.verbose:
                print(f"Padded and saved {img_file}.")
            
            # Adjust annotations for this image
            annotation_file = os.path.join(labels_folder, img_file.replace('.tif', '.txt'))
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    bboxes = [line.strip() for line in f.readlines()]

                # Convert bounding boxes and adjust for padding
                adjusted_bboxes = adjust_bboxes_and_convert_to_yolo(
                    bboxes, left_padding, top_padding, img.width, img.height, target_size
                )

                # Save adjusted annotations in YOLO format
                adjusted_annotation_file = os.path.join(output_folder, img_file.replace('.tif', '.txt'))
                with open(adjusted_annotation_file, 'w') as f:
                    f.write("\n".join(adjusted_bboxes))
                
                if ctxt.verbose:
                    print(f"Adjusted annotations and saved for {img_file}.")
    
    print("Finished padding and annotation adjustment.")
    return target_size, target_size  # Return final padded image dimensions (width, height)
