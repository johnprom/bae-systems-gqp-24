# from yolov8.ultralytics.ultralytics import YOLO (for if you have yolo src code locally)
import argparse
import os
import pandas as pd
import pprint
import shutil
import time
import torch

from knee_discovery.knee_discovery import run_knee_discovery
from preprocessing.preprocessing import run_preprocessing
# from reports.reports import generate_report
from train.train import run_finetuning
from util.util import load_pipeline_config
from reports.report import generate_report
# import yaml

class Pipeline:
    def __init__(self, args):
        """
        Initializes the Pipeline class with the provided arguments.
        Args:
            args (argparse.Namespace): Command-line arguments, including config_filename and verbosity flag.
        """

        
        self.config_filename = args.config_filename
        self.config = load_pipeline_config(self.config_filename)
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
        
        self.set_final_weights_path()

        self.train_labels_to_xview = {}
        self.xview_to_train_labels = {}
        self.target_labels = list(self.config['target_labels'].keys())
        self.xview_names = list(self.config['target_labels'].values())
        for idx, label in enumerate(self.target_labels):
            self.train_labels_to_xview[idx] = {
                'xview_type_id': label,
                'xview_name': self.xview_names[idx]
                }
            self.xview_to_train_labels[label] = {
                'train_type_id': idx,
                'train_name': self.xview_names[idx]
                }
        self.train_labels = list(range(len(self.target_labels)))
        self.train_names = self.xview_names.copy()
        if 'target_names' in self.config['report']:
            self.report_names = self.config['report']['target_names']
        else:
            self.report_names = self.xview_names.copy()

        self.iapc_columns = ['object_name', 'original_resolution_width', 'original_resolution_height', 'effective_resolution_width',
                             'effective_resolution_height', 'mAP', 'degradation_factor', 'knee']
        # self.eval_results_filename = self.config['knee_discovery']['eval_results_filename']
        self.results_cache_df = None
        self.cache_results = False
        self.knee_discovery_search_algorithm = None
        if 'run_knee_discovery' in self.config and self.config['run_knee_discovery']:
            if 'cache_results' in self.config['knee_discovery'] and self.config['knee_discovery']['cache_results']:
                self.cache_results = True
            if 'search_algorithm' in self.config['knee_discovery']:
                self.knee_discovery_search_algorithm = self.config['knee_discovery']['search_algorithm']
                
        self.class_names = None
        self.val_image_filename_set = None
        
        if 'use_cuda' in self.config and self.config['use_cuda'] and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        
        if 'run_clean' in self.config and self.config['run_clean']:
            self.run_clean()
            
    def get_pipeline_config(self):
        """
        Returns the pipeline configuration dictionary.

        Returns:
            dict: The loaded configuration dictionary for the pipeline.
        """

        return self.config
    
    def get_top_dir(self):
        """
        Returns the top directory path for the pipeline.

        Returns:
            str: Path of the top directory.
        """
        if 'top_dir' in self.config:
            return self.config['top_dir']
        return os.path.join(os.path.dirname(__file__), '..')

    def get_input_dir(self):
        """
        Returns the input directory path for the pipeline.

        Returns:
            str: Path of the input directory.
        """
        if 'input_dir' in self.config:
            return self.config['input_dir']
        return self.get_top_dir()

    def get_data_config_dir_path(self):
        """
        Returns the path to the data configuration directory.

        Returns:
            str: Full path of the data configuration file.
        """
        return os.path.join(self.get_top_dir(), self.config['data_config_filename'])
        
    def get_output_dir_path(self):
        """
        Returns the output directory path for storing results.

        Returns:
            str: Full path to the output directory.
        """
        if 'output_dir' in self.config:
            return self.config['output_dir']
        return os.path.join(self.get_top_dir(), self.config['output_subdir'])
    
    def get_preprocessing_dir_path(self):
        """
        Returns the directory path for storing preprocessed data.

        Returns:
            str: Full path to the preprocessing output directory.
        """
        return os.path.join(self.get_top_dir(), self.config['preprocess']['output_subdir'])
    
    def get_input_images_dir_path(self):
        """
        Returns the directory path where input images are stored.

        Returns:
            str: Full path to the directory containing input images.
        """

        return os.path.join(self.get_input_dir(), self.config['input_images_subdir'])
    
    def get_interim_images_dir_path(self):
        """
        Returns the directory path for storing interim processed images.

        Returns:
            str: Full path to the directory containing interim images, used during processing.
        """

        if 'interim_subdir' in self.config['preprocess']:
            return os.path.join(self.get_top_dir(), self.config['preprocess']['interim_subdir'])
        return os.path.join(self.get_input_images_dir_path(), self.config['interim_images_subdir'])
    
    def get_filtered_labels_filename(self):
        """
        Returns the filename for storing filtered labels in the preprocessing directory.

        Returns:
            str: Full path to the file containing filtered labels.
        """
        return os.path.join(self.get_preprocessing_dir_path(), self.config['preprocess']['filtered_labels_filename'])

    def get_train_geojson_filename(self):
        """
        Returns the filename for storing training data labels in GeoJSON format.

        Returns:
            str: Full path to the GeoJSON file containing training labels.
        """
        return os.path.join(self.get_input_dir(), self.config['input_images_labels_filename'])

    def get_class_labels_filename(self):
        """
        Returns the filename for storing class labels for the input images.

        Returns:
            str: Full path to the file containing class labels for the input images.
        """

        return os.path.join(self.get_input_dir(), self.config['input_class_labels_filename'])
    
    def get_model_name(self):
        """
        Retrieves the name of the deep learning model specified in the configuration.

        Raises:
            ValueError: If the 'model' or 'models' section is missing in the configuration
                        or if the specified model name is unknown.

        Returns:
            str: The name of the model specified in the configuration.
        """
        if 'model' not in self.config or 'models' not in self.config:
            raise ValueError("deep learning model uspecified in configuration file.")
        model_name = self.config['model']
        if model_name not in self.config['models']:
            raise ValueError(f"unknown deep learning model {model_name} specified.")
        return model_name
    
    def get_yolo_id(self):
        """
        Retrieves the YOLO model ID based on model parameters from the configuration.

        Returns:
            str: The YOLO model ID, typically a name identifying the YOLO model variant.
        """
        params = self.get_model_params()
        return params['name']
    
    def get_model_params(self):
        """
        Retrieves the configuration parameters for the specified model.

        Raises:
            ValueError: If the specified model is not found in the 'models' section of the configuration.

        Returns:
            dict: A dictionary of parameters for the specified model.
        """
        model_name = self.get_model_name()
        if model_name not in self.config['models']:
            raise ValueError(f"unkown deep learning model {model_name} specified.")            
        model_dict = self.config['models'][model_name]
        return model_dict
    
    def get_train_labels_from_target_labels(self, target_labels):
        """
        Converts a list of target labels to their corresponding training labels.

        Args:
            target_labels (list): A list of target labels to be converted.

        Returns:
            list: A list of training labels corresponding to the input target labels.
        """

        return [self.get_train_label_from_xview_label(xvl) for xvl in target_labels]
    
    def get_train_label_from_xview_label(self, xview_label):
        """
        Converts an individual xView label to its corresponding training label ID.

        Args:
            xview_label (str): The xView label to convert.

        Returns:
            int: The training label ID that corresponds to the input xView label.
        """
        return self.xview_to_train_labels[xview_label]['train_type_id']
    
    # def get_model_id(self):
    #     model_dict = self.get_model_params()
    #     if self.get_model_name() not in model_dict:
    #         raise ValueError(f"deep learning model {self.get_model_name()} not in 'models' in yaml configuration file")
    #     if 'name' not in model_dict:
    #         raise ValueError(f"YOLO deep learning model id not specified")
    #     return model_dict['id']

    def set_final_weights_path(self):
        """
        Sets the final weights path for the trained model based on the configuration and model parameters.
        Raises ValueError if an unknown deep learning model is specified.
        """
        if not self.is_model_yolo():
            ve = f"unknown deep learning model {self.get_model_name()} specified."
            raise ValueError(ve)
    
        # yolo_id = ctxt.get_yolo_id()
        model_name = self.config['model']
        model_dict = self.config['models'][model_name]
        model_params = model_dict['params']
        # base_model = YOLO(yolo_id)
        
        trained_model_filename_template = None
        if 'trained_model_filename' in model_dict:
            if 'trained_model_filename' != "":
                trained_model_filename_template = model_dict['trained_model_filename']
        if trained_model_filename_template is None:
            self.final_weights_path = None
            return
        
        # input_width = model_params['imgsz']
        # input_height = input_width
        # # input_width, input_height = model_dict['input_image_size'][0], model_dict['input_image_size'][1]
        # num_epochs = model_params['epochs']
        # batch_size = model_params['batch']
        
        # TODO: KENDALL? GABE? DAN?
        # Train your model here. The following is just dummy code
        
        # model_params['data'] = self.config['data_config_path']
        # train_params = ctxt.config['models']
        # train_params = {
        # 'data': 'data_config_path
        # 'epochs': num_epochs,
        # 'imgsz': input_width,
        # 'batch': 16,              # Batch size
        # 'freeze': [0, 1, 2, 3, 4, 5, 6],  # Freeze first 7 layers (backbone)
        # }
        
        # ft_model = base_model.train(data=data_config_path, epochs=num_epochs, imgsz=input_width, batch=16, freeze=list(range(7)))
        # ft_model = base_model.train(**model_params)
        
        params_str = ""
        keys = sorted(list(model_params.keys()))
        for key in keys:
            params_str += "_"
            params_str += str(key)
            params_str += "="
            params_str += str(model_params[key])

        params_hash = hash(params_str)
        self.final_weights_path = os.path.join(
            self.get_top_dir(), self.config['trained_models_subdir'], 
            trained_model_filename_template.format(hashed_params=str(params_hash)))
    
        # if 'freeze' in model_params and len(model_params['freeze']) > 0:
        #     freeze_str = '_'.join(map(str, model_params['freeze']))
        # else:
        #     freeze_str = 'No_Freeze'
        # self.final_weights_path = os.path.join(
        #     self.get_top_dir(), self.config['trained_models_subdir'], 
        #     trained_model_filename_template.format(width=model_params['imgsz'], 
        #                                            height=model_params['imgsz'],
        #                                            epochs=model_params['epochs'],
        #                                            batch=model_params['batch'],
        #                                            freeze=freeze_str))
        
    def is_model_yolo(self):
        """
        Checks if the specified model is a YOLO model.

        Returns:
            bool: True if the model is a YOLO model, False otherwise.
        """

        yolo_id = self.get_model_name()
        if type(yolo_id) == str and yolo_id.startswith('yolo'):
            return True
        return False
    
    def use_eval_cache(self):
        """
        Determines if the evaluation cache should be used based on configuration flags.

        Returns:
            bool: True if the evaluation cache should be used, False otherwise.
        """

        if 'run_clean' in self.config and self.config['run_clean']:
            return False
        if "run_preprocess" in self.config and self.config["run_preprocess"]:
            return False
    
        if "run_train" in self.config and self.config["run_train"]:
            return False
        
        if "run_knee_discovery" in self.config and self.config["run_knee_discovery"]:
            if 'knee_discovery' in self.config and 'use_eval_cache' in self.config['knee_discovery']:
                return self.config['knee_discovery']['use_eval_cache']

        return False        
    
    def run_clean(self):
        """
        Cleans up the output directory by removing results and reports.
        This does NOT delete preprocessed images or fine-tuned models.
        """
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
        """
        Runs the complete pipeline including preprocessing, training, knee discovery, and reporting.
        """

        
        pipeline_config = self.config
    
        start_pp = time.time()
        if "run_preprocess" in pipeline_config and pipeline_config["run_preprocess"]:
            run_preprocessing(self)
    
        start_tr = time.time()
        print(f"preprocessing duration {start_tr - start_pp} seconds")
        if "run_train" in pipeline_config and pipeline_config["run_train"]:
            run_finetuning(self)
    
        start_kd = time.time()
        print(f"training duration {start_kd - start_tr} seconds")
        if "run_knee_discovery" in pipeline_config and pipeline_config["run_knee_discovery"]:
            run_knee_discovery(self)
    
        start_gr = time.time()
        print(f"knee discovery duration {start_gr - start_kd} seconds")
        if "generate_report" in pipeline_config and pipeline_config["generate_report"]:
            generate_report(self)
    
        finish = time.time()
        print(f"generate report duration {finish - start_gr} seconds")
        print("Pipeline Finished.")
        print(f"Total pipeline duration {finish - start_pp} seconds")

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
