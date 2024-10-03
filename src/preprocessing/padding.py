from PIL import Image

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

def padding(imgsz: int):
    """Pads the images based on the desired size."""
    input_folder = '../xview_dataset_raw/train_images/'
    output_folder = '../preprocessed_datasets/padded_images/'
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(input_folder):
        if img_file.endswith('.tif'):
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path)
            padded_img = pad_image(img, imgsz)
            padded_img.save(os.path.join(output_folder, img_file))
            print(f"Padded and saved {img_file}.")
