import os
import shutil

from PIL import Image

# image normalization with padding scheme
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

    return padded_image

def find_max_image_size(image_folder):
    """ Find the maximum dimensions among all images in the folder """"
    max_width = 0
    max_height = 0
    for filename in or.listdir(image_folder):
        if filename.endswith('.tif'):
            image_path = os.path.join(image_foder, filename)
            with Image.open(image_path) as img:
                if img.width > max_width:
                    max_width = img.width
                if img.hight > max_height:
                    max_heght = img.hight
    max_side = max(max_width, max_height)
    return (max_side + 31) // 32 * 32 # round up to the nearest multiple of 32

def padding(ctxt):
    config = ctxt.get_pipeline_config() # dictionary of yaml configuration, in case you need it
    
    input_folder = ctxt.input_images_dir
    output_folder = ctxt.interim_images_dir
    os.makedirs(output_folder, exist_ok = True)
    target_size = find_max_image_size(input_folder)

    if ctxt.verbose:
        print(f"Max image size for padding: {target_size}x{target_size}")

    # Loop through all .tif images in the input folder and pad them
    for img_file in os.listdir(input_folder):
        if img_file.endswith('.tif'):
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path)
            padded_img = pad_image(img, target_size)
            padded_img.save(os.path.join(output_folder, img_file.replace('.tif', '.png')) # converting .tif to .png to match tiling
            if ctxt.verbose:
                print(f"Padded and saved {img_file}.")
            
    print("Finished running padding approach.")
    return imgsz, imgsz # width, height
