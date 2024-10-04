import os
import shutil

# image normalization with padding scheme
def padding(ctxt):
    config = ctxt.get_pipeline_config() # dictionary of yaml configuration, in case you need it
    
    # TODO: GABE: Loop through each image (and corresponding .txt file) in ctxt.input_images_dir, and pad
    # Place image and .txt files in ctxt.iterim_images_dir
    # Note that tiling.py converts to .png format, so you should do that as well, though I *think* it's ok if you don't
    # Coordinate with AMIT
    
    # Dummy code for testing - for now just copies images, does no processing on them
    # TODO: GABE: Remove this code block
    image_filelist = [x for x in os.listdir(ctxt.input_images_dir) if x.endswith('tif')]
    label_filelist = [x for x in os.listdir(ctxt.input_images_dir) if x.endswith('txt')]
    for filename in image_filelist:
        shutil.copy2(os.path.join(ctxt.input_images_dir, filename), ctxt.interim_images_dir)
    for filename in label_filelist:
        shutil.copy2(os.path.join(ctxt.input_images_dir, filename), ctxt.interim_images_dir)
    # End Dummy code for testing

    print("Finished running padding approach.")
    # TODO: GABE: return the post-padding but pre-shrinking resolution: (4224, 4224)
    # Replace what is below with a calculation of it
    return 4224, 4224 # width, height