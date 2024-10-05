import os
import shutil
from PIL import Image

# Function to pad the image to the target size
def pad_image(image, target_size):
    """
    Pads an image to the target size, ensuring the aspect ratio remains.
    
    Args:
    - image: PIL Image object to be padded.
    - target_size: The size to pad the image to (e.g., 5120x5120).
    
    Returns:
    - Padded image with a black background.
    """
    width, height = image.size
    left_padding = (target_size - width) // 2
    right_padding = target_size - width - left_padding
    top_padding = (target_size - height) // 2
    bottom_padding = target_size - height - top_padding

    # Create new padded image with a black background
    padded_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    padded_image.paste(image, (left_padding, top_padding))

    return padded_image

# Function to find the maximum dimensions among all images in the folder
def find_max_image_size(image_folder):
    """
    Find the maximum dimensions among all images in the folder.
    
    Args:
    - image_folder: Folder path where the images are located.
    
    Returns:
    - max_side: The maximum image size, rounded to the nearest multiple of 32.
    """
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
    return (max_side + 31) // 32 * 32  # round up to the nearest multiple of 32

# Padding function that processes all images in the input folder
def padding(ctxt):
    """
    Pads all images in the input folder to the maximum size and saves the padded images.
    
    Args:
    - ctxt: Context object that holds pipeline configuration and paths.
    
    Returns:
    - target_size, target_size: The width and height of the padded images.
    """
    config = ctxt.get_pipeline_config()  # dictionary of yaml configuration, in case you need it
    
    input_folder = ctxt.input_images_dir
    output_folder = ctxt.interim_images_dir
    os.makedirs(output_folder, exist_ok=True)

    # Find the max image size in the folder
    target_size = find_max_image_size(input_folder)

    if ctxt.verbose:
        print(f"Max image size for padding: {target_size}x{target_size}")

    # Loop through all .tif images in the input folder and pad them
    for img_file in os.listdir(input_folder):
        if img_file.endswith('.tif'):
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path)
            padded_img = pad_image(img, target_size)
            padded_img.save(os.path.join(output_folder, img_file.replace('.tif', '.png')))  # converting .tif to .png
            
            if ctxt.verbose:
                print(f"Padded and saved {img_file} as {img_file.replace('.tif', '.png')}.")
    
    print("Finished running padding approach.")
    return target_size, target_size  # Return the actual padded size
