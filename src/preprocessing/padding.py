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

def padding(ctxt):
    config = ctxt.get_pipeline_config() # dictionary of yaml configuration, in case you need it
    
    # TODO: GABE: Loop through each image (and corresponding .txt file) in ctxt.input_images_dir, and pad
    # Place image and .txt files in ctxt.iterim_images_dir
    # Note that tiling.py converts to .png format, so you should do that as well, though I *think* it's ok if you don't
    # Coordinate with AMIT
    
    # Dummy code for testing - for now just copies images, does no processing on them
    # TODO: GABE: Remove this code block
    # image_filelist = [x for x in os.listdir(ctxt.input_images_dir) if x.endswith('tif')]
    # label_filelist = [x for x in os.listdir(ctxt.input_images_dir) if x.endswith('txt')]
    # for filename in image_filelist:
    #     shutil.copy2(os.path.join(ctxt.input_images_dir, filename), ctxt.interim_images_dir)
    # for filename in label_filelist:
    #     shutil.copy2(os.path.join(ctxt.input_images_dir, filename), ctxt.interim_images_dir)
    # End Dummy code for testing
    
    input_folder = ctxt.input_images_dir
    output_folder = ctxt.interim_images_dir
    imgsz = 4224

    for img_file in os.listdir(input_folder):
        if img_file.endswith('.tif'):
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path)
            padded_img = pad_image(img, imgsz)
            padded_img.save(os.path.join(output_folder, img_file))
            if ctxt.verbose:
                print(f"Padded and saved {img_file}.")
            
    print("Finished running padding approach.")
    # TODO: GABE: return the post-padding but pre-shrinking resolution: (4224, 4224)
    # Replace what is below with a calculation of it
    return imgsz, imgsz # width, height

# from PIL import Image

# def pad_image(image, target_size):
#     """
#     Pads an image to the target size, ensuring the aspect ratio remains.
#     Args:
#     - image: PIL Image object to be padded.
#     - target_size: The size to pad the image to (e.g., 5120x5120).
#     """
#     width, height = image.size
#     left_padding = (target_size - width) // 2
#     right_padding = target_size - width - left_padding
#     top_padding = (target_size - height) // 2
#     bottom_padding = target_size - height - top_padding

#     # Create new padded image with a black background
#     padded_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
#     padded_image.paste(image, (left_padding, top_padding))

#     return padded_image

# def padding(imgsz: int):
#     """Pads the images based on the desired size."""
#     input_folder = '../xview_dataset_raw/train_images/'
#     output_folder = '../preprocessed_datasets/padded_images/'
#     os.makedirs(output_folder, exist_ok=True)

#     for img_file in os.listdir(input_folder):
#         if img_file.endswith('.tif'):
#             img_path = os.path.join(input_folder, img_file)
#             img = Image.open(img_path)
#             padded_img = pad_image(img, imgsz)
#             padded_img.save(os.path.join(output_folder, img_file))
#             print(f"Padded and saved {img_file}.")
