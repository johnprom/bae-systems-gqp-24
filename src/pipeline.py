from yolov8.ultralytics.ultralytics import YOLO
from preprocessing.preprocessing import run_preprocessing

'''
TODO generalize and invoke the following steps:
    1 - Invoke preprocessing(pipeline_config.preprocessing_params)
    2 - Invoke init pretrained model + fine-tuning(pipeline_config.train_params) 
    3 - Invoke model evaluation(pipeline_config.eval_params)
    4 - Invoke curve generation + find elbow(pipeline_config.output_params) 
'''

'''
    For a given object_class:
        Preprocess
            Preprocess image
            Filter geojson for target object class
                Preprocess annotations to YOLO
                Filter object classes also in the yaml
        For each model Load pretrained model and perform fine-tuning 
            model = YOLO('yolov8n.pt')
            model.train(data='../data_config.yaml', epochs=50, imgsz=640)
        For each generated model variation
            get MAP scores
            results = model.val(data='../data_config.yaml)
        For each score, res pair
            plot curve
            find elbow
            return (object_class : elbow val in real-world distance)
'''

def run_pipeline(pipeline_config):

    params="some params" # we can parse these from pipeline config
    run_preprocessing(params)

    print("Pipeline Finished.")
