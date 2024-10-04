# from yolov8.ultralytics.ultralytics import YOLO (for if you have yolo src code locally)
import argparse
import os
import pprint
import shutil

from knee_discovery.knee_discovery import run_knee_discovery
from preprocessing.preprocessing import run_preprocessing
# from reports.reports import generate_report
from train.train import run_finetuning
from util.util import load_pipeline_config
# import yaml

# TODO: DAN Move generate_report out of this file

def generate_report(ctxt):
    print("Finished generating report (not supported yet): ")

class Pipeline:
    def __init__(self, args):
        self.config = load_pipeline_config(args.config_filename)
        self.verbose = args.verbose
  
        self.input_images_dir = self.get_input_images_dir_path()
        self.interim_images_dir = self.get_interim_images_dir_path()
        
        # Here is where to store state information. Do not put state information into self.config
        self.train_images_list = [] # list of image files in the training set
        self.val_images_list = [] # list of image files in the validation set

        # The following code applies to all methods. If a new method is created that is not applicable to some lines here, break
        # them out into the proceeding if statement
        self.train_baseline_dir = ""
        self.val_baseline_dir = ""
        
        self.maxwidth = 0
        self.maxheight = 0

    def get_pipeline_config(self):
        return self.config
    
    def get_top_dir(self):
        if 'top_dir' in self.config:
            return self.config['top_dir']
        return os.path.join(os.path.dirname(__file__), '..')

    def get_data_config_dir_path(self):
        return os.path.join(self.get_top_dir(), self.config['data_config_filename'])
        
    def get_output_dir_path(self):
        return os.path.join(self.get_top_dir(), self.config['output_subdir'])
    
    def get_preprocessing_dir_path(self):
        return os.path.join(self.get_top_dir(), self.config['preprocess']['output_subdir'])
    
    def get_input_images_dir_path(self):
        return os.path.join(self.get_top_dir(), self.config['input_images_subdir'])
    
    def get_interim_images_dir_path(self):
        return os.path.join(self.get_input_images_dir_path(), self.config['interim_images_subdir'])
    
    def get_filtered_labels_filename(self):
        return os.path.join(self.get_preprocessing_dir_path(), self.config['preprocess']['filtered_labels_filename'])

    def get_train_geojson_filename(self):
        return os.path.join(self.get_input_images_dir_path(), self.config['input_images_labels_filename'])
    
    def run_clean(self):
        # remove output directory if exists
        # This does NOT delete preprocessed images and not tuned models either
        # It just deletes results and reports
        if self.verbose:
            print("start running clean with the following configuration:")
            pprint.pprint(self.config)
        output_dir = self.get_output_dir_path()
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        if self.verbose:
            print("done running clean")
    
    def run_pipeline(self):
        
        pipeline_config = self.config
    
        if "run_clean" in pipeline_config and pipeline_config["run_clean"]:
            self.run_clean()
    
        if "run_preprocess" in pipeline_config and pipeline_config["run_preprocess"]:
            run_preprocessing(self)
    
        if "run_train" in pipeline_config and pipeline_config["run_train"]:
            run_finetuning(self)
    
        if "run_knee_discovery" in pipeline_config and pipeline_config["run_knee_discovery"]:
            run_knee_discovery(self)
    
        if "generate_report" in pipeline_config and pipeline_config["generate_report"]:
            generate_report(self)
    
        print("Pipeline Finished.")

# Define the main function
def main():
    parser = argparse.ArgumentParser(description="Pipeline for determining optimal satellite imagery resolution.")
    
    parser.add_argument("config_filename", type=str, help="The path to the configuration file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Specify verbosity")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Processing file: {args.config_filename} with verbosity on.")
    else:
        print(f"Processing file: {args.config_filename}")
        
    pipeline_context = Pipeline(args)

    config = pipeline_context.get_pipeline_config()
    
    pipeline_context.run_pipeline()
        
    print("Exited Main.")

# Ensure the script runs only when executed directly (not when imported as a module)
if __name__ == "__main__":
    main()
