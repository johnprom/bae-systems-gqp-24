# from yolov8.ultralytics.ultralytics import YOLO (for if you have yolo src code locally)
from ultralytics import YOLO
from preprocessing.preprocessing import run_preprocessing
import os

# TODO: Generalize and move these methods out of this file

def run_finetuning(params, data_config_path):
    if params['skip_step']:
        print('Skipping Fine-Tuning Module.')
    else:
        base_model = YOLO(params['model'])
        #ft_model = base_model.train(data=data_config_path, epochs=config['epochs'], imgsz=config['imgsz'])
        final_weights_path = 'finetuned_models/latest_model.pt'
        base_model.save(final_weights_path) # change to ft_model.save(final_weights_path) when data is there
        print("Finished Running Fine-tuning.")

def run_eval_on_initial_resolutions(config, data_config_path):
    # run eval on baseline (path already set) and write to results file 
    for res in config['search_grid']:
        # res_path = create_degraded set based on baseline and write to preprocessed datasets
        # set path in data config to res_path
        # run eval and write to results file 
        print("---")

def generate_report(params):
    print("Finished generating report with: " + str(params))

def run_eval(model, dataset_path):
    results = ft_model.val(data=data_config_path)
    # write results to some structured file

def run_pipeline(pipeline_config):
    # get path for YOLO data config file
    data_config_path = os.path.join(os.path.dirname(__file__), '..', 'data_config.yaml')

    # run_preprocessing(imgsz: int, approach: ApproachType, stride: int, train_split: int, clear_data: Boolean, target_classes: [int])
    run_preprocessing(pipeline_config['preprocessing_params']) 

    # run_finetuning(epochs: int, imgsz: int, model: str, data_config_path: str)
    run_finetuning(pipeline_config['train_params'], data_config_path)

    # run_eval_on_initial_resolutions(search_grid: [int], enable_eval: false, data_config_path: str)
    #run_eval_on_initial_resolutions(pipeline_config['eval_params'], data_config_path, ft_model)

    # TODO: run_optimizer(data_config_path, ft_model)
    # in a loop: 
    #  optimizer calls elbow fn and decides if another datapoint is needed (continue or return next_res)
    #  generate degraded set
    #  set path in data config to res_path
    #  run eval and write to results file 

    # generate_report()
    generate_report(pipeline_config['report_params'])

    print("Pipeline Finished.")
