## Getting Started

### Running the Pipeline

To run the Image Resolution-Performance pipeline, do the following:

```
cd .../bae-systems-gqp-24/src

python3 ./pipeline.py ../pipeline_config.yaml

Use the -v command-line option for verbosity:

python3 ./pipeline.py ../pipeline_config.yaml -v
```


## How to Configure the Pipeline Config

To properly configure the pipeline for your dataset and model, you need to set up various parameters in the configuration file. Below is an explanation of how to configure each section.

### 1. **Global Parameters**
These are the essential parameters that define the general setup for your pipeline.

- **`top_dir`**: This is the path to the top-level directory containing all the directories for input, output, and data configuration files. By default, it's set to `../`.
  
- **`input_dir`**: The directory containing your raw input data. This should point to the directory where the dataset (e.g., `xview_dataset_raw`) resides.

- **`output_dir`**: The directory where all processed outputs will be saved. The default is set to `../output`.

- **`run_clean`**: Whether to delete the output directory before starting the pipeline. Default is `false`.

- **`data_config_filename`**: Path to the data configuration file (e.g., `data_config.yaml`), which contains detailed information about the dataset and classes.

- **`input_images_subdir`**: The subdirectory within `input_dir` where the images are located. Default is the root directory `./`.

- **`input_images_labels_filename`**: Path to the input labels file in `.geojson` format, detailing the annotations for the dataset (e.g., `xView_train.geojson`).

- **`input_class_labels_filename`**: The file containing the class labels, typically in JSON format (e.g., `xview_class_labels.json`).

- **`model`**: The name of the model to use for training. In this case, `yolov8m` is used.

- **`use_cuda`**: Set to `true` to use CUDA if available, enabling GPU acceleration.

- **`trained_models_subdir`**: Subdirectory where the fine-tuned models will be saved. The default is `finetuned_models`.

- **`pixel_size`**: The size of each pixel in meters. This must be provided (e.g., `0.3`).

- **`preprocess_method`**: The method for preprocessing the dataset. Currently, only `tiling` is supported.

- **`target_labels`**: A list of target labels (such as 'Fixed-wing Aircraft', 'Small Aircraft', etc.) that the model will detect.

### 2. **Preprocessing Parameters**
The preprocessing module is responsible for preparing the dataset for training.

- **`output_subdir`**: Directory where the final preprocessed dataset will be stored. Set relative to `top_dir`.

- **`interim_subdir`**: The subdirectory for interim files during preprocessing.

- **`filtered_labels_filename`**: The file name for storing filtered labels after preprocessing.

- **`clean_subdir`**: Whether to delete the `output_subdir` before preprocessing. Default is `true`.

- **`train_split`**: The proportion of the dataset to be used for training. Default is `0.8` (80% training, 20% validation).

### 3. **Training Parameters**
This section handles the model training configurations.

- **`output_subdir`**: Directory where training results will be stored, relative to `output_dir`.

- **`hyperparameters_filename`**: The filename for storing hyperparameters used for training (default is `hyperparams.csv`).

- **`fractional_factorial`**: Set to `true` for using fractional factorial grid search for hyperparameter tuning.

### 4. **Knee Discovery Parameters**
The knee discovery module helps optimize the resolution for training and model evaluation.

- **`output_subdir`**: Directory where results of knee discovery will be saved.

- **`clean_subdir`**: Whether to delete the `output_subdir` before running knee discovery. Default is `true`.

- **`cache_results`**: Set to `true` to cache results for improved performance.

- **`eval_results_filename`**: The file where the cumulative evaluation results will be stored.

- **`search_resolution_range`**: The range of resolution fractions to search for optimal performance. Default is `[0.05, 1.0]`.

- **`search_resolution_step`**: The step size for resolution search.

- **`knee_resolution_interpolation_divisor`**: Granularity for fitting a spline for knee discovery. Default is `5`.

### 5. **Report Parameters**
The report module generates PDF reports of the model's performance.

- **`output_subdir`**: Directory where the report will be stored.

- **`clean_subdir`**: Whether to delete the `output_subdir` before generating the report. Default is `false`.

- **`report_filename`**: The filename for the generated PDF report. Default is `generated_report.pdf`.

- **`curves_per_graph`**: The number of curves to display on each graph in the report. Default is `5`.
  
- **`display_labels`**: A list of object classes to include in the generated report. If specified, only these classes will be included, even if the pipeline processed additional classes. If not specified, all target labels processed by the pipeline will be included by default.

### 6. **Preprocessing Methods**
Currently, the only supported preprocessing method is tiling.

- **`image_size`**: The size of the image tiles, recommended to be `640` for best performance.

- **`stride`**: The overlap between tiles, which ensures that bounding boxes are captured across tile boundaries.

- **`train_baseline_subdir`**: The directory for storing the tiled images for training.

- **`val_baseline_subdir`**: The directory for storing the tiled images for validation.

- **`train_degraded_subdir`**: The directory for storing degraded training images, if applicable.

- **`val_degraded_subdir`**: The directory for storing degraded validation images.

- **`output_basename_append`**: A string appended to the filename of each tiled image for uniqueness.

### 7. **ML Model Parameters**
This section defines the details of the YOLOv8 model used for detection.

- **`name`**: The model name, in this case, `yolov8m`.

- **`pretrained_id`**: The identifier for the pretrained model to use if no pretrained weights are found.

- **`trained_model_filename`**: The filename where the fine-tuned model will be saved.

- **`use_eval_cache`**: Set to `true` to enable caching for evaluation.

- **`params`**: Parameters for fine-tuning, such as the number of epochs, batch size, and layers to freeze.

- **`hyperparameters`**: A list of hyperparameters used for training, such as image size, batch size, and the number of epochs.

---




## How to run the Pipeline on your own dataset

### **Premise**  

**If the starting input data is in the xView GeoJSON format:**  

- **Hook up the directory path** in the `config.yaml` file.  
- **Set the class mappings** in the `config.yaml` file.  
- **...and you’re good to go!** 🎯  

**If the starting input data is NOT the xView GeoJSON format:**  

### **Data Conversion to GeoJSON Format**  

This guide explains how to convert your dataset into **GeoJSON format**, making it compatible with various mapping and object detection pipelines.


### **Step 1: Understand Your Input Data**  
Your input data could be in formats like CSV, JSON, or any custom format containing spatial information (coordinates).  

### **Example Input Data (CSV):**
```csv
id,latitude,longitude,class
1,34.0522,-118.2437,building
2,40.7128,-74.0060,vehicle
```


### **Step 2: Create a Script for Conversion**  
Write a custom script that reads your dataset and converts it to GeoJSON format.

### **What to Include in the Script:**  
- Read the input file (e.g., CSV or JSON).  
- Extract relevant properties like coordinates, IDs, and classes.  
- Format them into GeoJSON format using libraries like `geojson`, `json`, or `pandas`.  
- Save the output as a `.geojson` file.

### **Step 3: Expected Output Format**  

### **Example Output (GeoJSON):**  

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-118.2437, 34.0522]
      },
      "properties": {
        "id": 1,
        "class": "building"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-74.0060, 40.7128]
      },
      "properties": {
        "id": 2,
        "class": "vehicle"
      }
    }
  ]
}
```


### **Next Steps**  

- **Test Your Script:** Ensure the generated GeoJSON is valid.  
- **Use the GeoJSON File:** Plug it into your object detection or mapping pipeline.  

---

## High-level View
![image](https://github.com/user-attachments/assets/9e9dc4be-36b5-43d4-ae44-ec4b25f1216d)

At a high level, the main pipeline execution is composed of 4 independent modules. Each module can run in isolation. A diagram outlining the high-level design of the pipeline is shown above.

Note that this design makes several assumptions about input data. First, the datasets contain satellite imagery with wide dimensions. The dataset is also required to have a known GSD, which is constant across imagery in the dataset. Additionally, the dataset should contain an accurate ground truth with high-quality labels.

The pipeline contains four modules that are assumed to reside on the same file system. These four modules are: Preprocessing, Fine-tuning, Knee Discovery, and Report Generation. These modules can be run in order during the main pipeline execution, or they may be separate processes run in serial (they cannot be run in parallel). For example, this modularity should allow users to run the Knee Discovery module and Report Generation module independently without requiring preprocessing and fine-tuning to be run again. Further, the Pipeline is configured via a main configuration file. This configuration file determines the high-level behavior of the pipeline as well as parameters specific to each module during runtime. Note that in the diagram, the thick arrows are execution flow, while the thin arrows are filesystem I/O. The goal of each module is summarized below:
### Preprocessing Module
The Preprocessing module fetches raw image and label data from the file. Preprocessing is then done to target certain object classes and augment images and labels to meet the input requirements of the model architecture. The preprocessing module then splits into train and test sets and writes the categorized and preprocessed baseline dataset to file. 
### Fine Tuning Module
The Fine Tuning module fetches the preprocessed baseline dataset from the file. Next, pretrained model weights are loaded. The pretrained model is then “fine-tuned” by performing additional training on the pretrained model using the baseline preprocessed dataset. The Fine Tuning module then saves the fine-tuned model to file. 
### Knee Discovery Module 
The Knee Discovery module generates data points along a curve of data resolution versus detention performance (IRPC). This module invokes an evaluation subprocess that fetches the baseline dataset from the file, generates a degraded copy of the dataset with a lower effective resolution and runs evaluation on the fine-tuned model. The performance evaluation results are written to file.
### Report Generation Module
The Report Generation module fetches the results file (generated by the Knee Discovery module). The results are compiled and packaged into a Portable Document Format (PDF) report with relevant metrics and visualizations to effectively convey the results. This report is then saved to file.

## Documentation

### 1 -	Reusable Pipeline 
The Pipeline class and the main routine are located in src/pipeline.py.

This is the high-level main routine that manages the execution of the modules illustrated above. This high-level routine runs each module consecutively, as configured in the configuration yaml file (specified in the first argument at the command line). It instantiates a Pipeline class singleton which contains the context of the pipeline run. This context is passed around from module to module throughout the code. 

Each pipeline module can be run independently. There are restrictions, though, in how the pipeline is run: 

Even though not every module needs to be run for any execution, the pipeline always runs the modules in a specific order:
Data Preprocessing Module
Model Fine-Tuning Module
Detection and Knee Discovery Module
Report Generation Module
Modules can only be run if all of the previous modules have completed successfully.
The first three modules (Data Preprocessing, Model Fine-Tuning, and Detection and Knee Discovery module) must be run with the same configuration yaml file. If a change is made to the configuration file, then the pipeline needs to restart with the Data Processing Module.
The Report Generation Module can be run with configuration yaml file settings different from the ones used for the previous modules. However, if configuration options are added to the configuration yaml file (which would require pipeline code changes), then it is possible that the report generation module needs to be re-run. To be safe, re-run the entire pipeline if configuration options, and therefore code, are changed.

The main driver for the entire Pipeline is the Pipeline singleton object, the reference of which is passed into all routines in the pipeline as ctxt.

### 2 -	Data Preprocessing Module
src/preprocessing/annotation_preprocessing.py
src/preprocessing/class_filtering.py
src/preprocessing/preprocessing.py
src/preprocessing/tiling.py
src/preprocessing/Tiling_images_to_yolo.py

The main preprocessing source code is in src/preprocessing/preprocessing.py.

In order to ensure that raw data is converted into formats appropriate for training, evaluation, and model fine-tuning, data preprocessing is an essential step in the pipeline. The modular elements of the preprocessing pipeline are thoroughly explained in the ensuing subsections.

The code in preprocessing.py prepares the dataset for training and evaluation by performing essential preprocessing steps such as tiling, padding, filtering classes, and splitting the data into training and validation sets. It ensures the data is structured, optimized, and compatible with the YOLO model's requirements.

Implementation Details:
Train-Test Split:
The train_test_split(ctxt) function divides the dataset into training and validation sets based on a user-defined ratio in the configuration file.
Integration with Tiling and Padding:
Calls the tiling function to divide large images into smaller, manageable tiles, improving training efficiency.
Calls the padding_and_annotation_adjustment function to apply padding and adjust bounding box annotations to maintain consistency in image dimensions.
Class Filtering:
Utilizes the filter_classes function to remove irrelevant classes and retain only those specified in the configuration.
Utility Integration:
Updates YOLO configuration paths for training and validation datasets using:
update_data_config_train_path
Update_data_config_val_path

### 2.1 - Annotation Processing
The annotation processing source code is in src/preprocessing/annotation_preprocessing.py.

The primary objective of this module is to prepare annotation data for compatibility with the YOLO model by converting bounding box pixel coordinates into normalized relative values. This ensures that annotations adhere to YOLO's requirements for input format and dimensional consistency.
Implementation:
Bounding Box Conversion: Converts pixel-based bounding box coordinates into YOLO-compatible format. Adjusts for padding offsets and normalizes dimensions based on max_len. This process involves transforming pixel-based bounding box coordinates into a YOLO-compatible format. The raw coordinates (xmin, ymin, xmax, ymax) represent the position and size of the bounding box in the original image. These coordinates are normalized by dividing each value by a predefined maximum length (max_len) to ensure consistency across varying image dimensions. Additionally, if the original image dimensions are smaller than max_len, padding offsets are added to adjust the coordinates, maintaining alignment with the padded image.This conversion ensures YOLO's requirements for input dimensional consistency are met, enabling accurate training and evaluation.
Annotation Filtering: Filters and processes annotations to match the image id of the target image.
Output Preparation: Generates text files containing YOLO-format annotations, including class labels and normalized bounding boxes. The script iterates over all .tif images in the source directory, extracting and preprocessing their annotations. It outputs YOLO-compatible annotation files to a predefined target directory for tasks such as training and validation.
Key Functions and Libraries:
pixel_to_relative_coords: transforms pixel-based bounding box coordinates into normalized relative coordinates by adjusting for padding offsets when the original image dimensions are smaller than a specified max_len. Calculates the center coordinates (x_center, y_center), width, and height of bounding boxes relative to the maximum image dimension. Returns coordinates in YOLO format: <x_center> <y_center> <width> <height>, rounded to six decimal places.
For each .tif image, a corresponding .txt file is created in the target folder with YOLO-compatible annotations. Each annotation includes the class id and noramlized bounding box in the format of (<x_center> <y_center> <width> <height>)
### 2.2 - Class Filtering
The class filtering source code is in src/preprocessing/class_filtering.py.
The Class Filtering module ensures that the dataset is streamlined for training and evaluation by isolating and processing only the required classes. Irrelevant data is excluded, focusing the dataset on a predefined set of target classes as specified in the pipeline's configuration. This step enhances dataset efficiency and ensures compatibility with YOLO's requirements.
Implementation:
Class Filtering:
The primary function filter_classes(ctxt, target_class_ids) filters annotation data to retain only features with class IDs listed in target_class_ids. Irrelevant classes are excluded to maintain a concise and focused dataset.
Reindexing:
Class IDs are remapped to start from 0 in the order they are listed in the configuration. For example, target class IDs [21, 34, 55] are reindexed to [0, 1, 2], ensuring compatibility with YOLO, which expects zero-based indexing.
Output Preparation:
Writes the filtered annotations to a new GeoJSON file.
Updates YOLO configuration files with:
Class Count: The total number of retained classes.
Class Names: The names of the filtered classes.
Key Functions and Libraries:
filter_classes:
Filters and processes annotation data to retain only the specified classes.
Reindexes class IDs to start from 0 and prepares the filtered dataset.
write_to_annotations_filtered:
Saves the filtered annotations to a new GeoJSON file.
Configuration Updates:
update_data_config_class_names(ctxt, target_class_names): Updates YOLO configuration with the filtered class names.
update_data_config_class_count(ctxt, class_count): Updates YOLO configuration with the new class count.
Utility Functions:
get_class_name_from_id(ctxt, id_key): Retrieves class names corresponding to the filtered class IDs.
load_annotations_master(ctxt): Loads the original GeoJSON file containing master annotations.
### 2.3 - Tiling 
The preprocessing tiling source code is in src/preprocessing/tiling.py.

The Preprocessing Tiling module divides large images into smaller, manageable tiles while ensuring compatibility with YOLO requirements. This step is crucial for efficient model training, as it standardizes image dimensions and adjusts annotations to align with the tiled coordinates. By splitting images into tiles with overlap, the module ensures that no object is lost at the edges of tiles.
Implementation:
Tiling Images:
Processes large images by splitting them into smaller tiles of a fixed size (e.g., 640x640 pixels), as specified in the configuration.
Overlap between tiles is added (using a stride value) to prevent loss of objects located at the tile boundaries.
Bounding box annotations are adjusted to match the tiled image coordinates.
Integration:
This module integrates seamlessly into the preprocessing pipeline as an essential step.
Tiled outputs are subsequently used by other pipeline steps such as padding, filtering, and train-test splitting.
Output Preparation:
Tiled images are saved in the configured output directory.
Updated annotations, adjusted to match the tiled image dimensions, are stored alongside the tiles.




Key Functions and Libraries:
tiling:
Splits images into smaller tiles and saves both the tiles and their corresponding annotations.
Adjusts bounding box annotations for objects contained in each tile to ensure accuracy.
XViewTiler:
Handles the core tiling operations, including grouping annotations and managing tile overlap.
Provides methods such as get_class_wise_data to organize features by class and tile_image_and_save to split and save the tiles.
### 2.4 - Tiling to YOLO
The preprocessing “Tiling to YOLO” source code is in src/preprocessing/Tiling_images_to_yolo.py.
The Tiling to YOLO module specializes in preparing tiled images and annotations specifically for YOLO. It integrates YOLO-specific requirements such as class mappings, bounding box adjustments, and dataset formatting. This ensures that tiled images and their annotations are fully compatible with YOLO’s input specifications.
Implementation:
Class Mapping:
Updates class indices to match YOLO’s format using the xview_class2index mapping.
Ensures compatibility between xView class identifiers and YOLO’s zero-based indexing system.
YOLO-Compatible Tiling:
Divides images into tiles of a fixed size (e.g., 640x640 pixels) with overlap.
Ensures each tile adheres to YOLO’s input size and format requirements.
Annotation Adjustment:
Adjusts bounding box annotations to be relative to the tile’s dimensions.
Filters out objects that do not fit entirely within a tile after the split.
Output Directory Structure:
Images Directory: Stores YOLO-compatible tiled images.
Labels Directory: Stores YOLO-formatted annotation files for the corresponding tiles.

Key Functions and Libraries:
convert_bbox_to_yolo_format:
Converts bounding box coordinates into YOLO format normalized by the tile’s dimensions.
tile_image_and_save:
Splits images into tiles and adjusts bounding box annotations for each tile.
Saves tiled images and their YOLO-compatible labels to the specified output directories.
save_yolo_labels:
Saves bounding boxes and class IDs in YOLO format to .txt files.
Get_class_wise_data:
Filters and organizes annotations by class and image, preparing the data for tiling and formatting.
### 3 - Model Fine-Tuning Module
Unlike traditional machine-learning models, YOLO does not allow for hyper-parameter tuning after being fine-tuned on our dataset. Many of the hyperparameters directly affect the fine-tuning phase. Because of this, the combinations generated from the pipeline.YAML hyperparameter list must each be used to fine-tune the model to find the one that performs the best. 
Implementation Details
Grid-search Hyperparameter Tuning: This module iterates through all hyperparameter combinations, comparing the performance of each to find the optimal set of hyperparameters.
Hyperparameters: YOLO offers a variety of hyperparameters that when modified help tune the model to the specifics of the project.
Grid-search: Fine tuning is performed with each combination of hyperparameters at a time, until all have been tested.
Model Evaluation: The average Mean Average Precision (mAP) score is calculated across all classes for each hyperparameter combination. The one with the highest score moves forward to the evaluation and knee-detection phases. 
Full Factorial Search: Used to evaluate the performance of all hyperparameter combinations.
Fractional Factorial Search: Used to evaluate the performance of half of the hyperparameter combinations. Reduces run-time and memory usage while still capturing important relationships between the hyperparameters.
Key Functions and Libraries:
update_hyperparameters: Generates a CSV file containing the best hyperparameter combination to be used in the final report. 
run_hyperparameter_tuning: Iterates through the list of hyperparameters and performs fine tuning for each hyperparameter combination.
### 4 -	Detection and Knee Discovery Module
The Detection and Knee Discovery Module is a crucial part of our pipeline, designed to identify the optimal image resolution for object detection tasks. It automates the process of degrading image resolutions, evaluating model performance at each level, and applying an algorithm to detect the "knee" point in the performance curve where increasing resolution yields diminishing returns in detection accuracy.
Implementation Details
Image Degradation: The module systematically degrades images to simulate lower effective resolutions while maintaining the original dimensions required by the model. This is achieved through a two-step resizing process:
Downsampling: Images are resized to a lower degraded resolution using bilinear interpolation, reducing image detail and simulating a higher Ground Sample Distance (GSD).
Upsampling: The degraded images are then resized back to the original dimensions (e.g., from 320×320320 \times 320320×320 to 640×640640 \times 640640×640 pixels), resulting in blurrier images due to loss of detail.
Model Evaluation: The fine-tuned model is evaluated on each set of degraded images to measure the mAP for each object class. These values are recorded along with the corresponding degradation factors to construct the performance curve.
Knee Detection Algorithm: To identify the optimal resolution, the module employs an algorithm that combines spline interpolation and piecewise linear fitting:
Data Preparation:
Extract degradation factors and mAP values from the evaluation results.
Filter out data points with insignificant mAP values (below a certain threshold) to focus on meaningful performance metrics.
Spline Interpolation:
Apply cubic spline interpolation using the make_interp_spline function from the scipy.interpolate library.
This smooths the data and increases the number of data points, enhancing the accuracy of knee detection.
Piecewise Linear Fit:
Use the PiecewiseLinFit class from the pwlf library to fit a piecewise linear model with two segments to the interpolated data.
The breakpoint between the two segments is identified as the knee point.
Result Mapping:
Map the knee point back to the closest original data point for practical relevance.
Update the results accordingly for reporting and further analysis.
Key Functions and Libraries
degrade_images Function: Handles image degradation by resizing images to lower resolutions and then back to the original size, saving the degraded images for evaluation.
run_eval_on_degraded_images Function: Coordinates the evaluation of the model on degraded images across specified resolution ranges.
calculate_knee Function: Implements the knee detection algorithm using spline interpolation and piecewise linear fitting to identify the knee point in the performance curve.
By automating the detection of the knee point in the performance curve, this module enables the pipeline to balance detection accuracy with computational efficiency. It identifies the optimal image resolution for each object class, ensuring that higher resolutions are used only when they provide significant benefits, thereby optimizing resource utilization in object detection tasks.
### 5 -	Report Generation Module
The goal of the report generation module is to provide one or more files that present the results, obtained from the Detection and Knee Discovery Module, in a human-readable form. 
Implementation Details:
The report generation module generates a PDF report summarizing the results from the knee discovery module. It processes results data, calculates relevant metrics, visualizes performance curves, and formats the output into a structured report. 
Key Functions and Libraries:
generate_report: processes the IRPC Results CSV file to produce a comprehensive PDF report. It generates IRPC curves for all detected object classes, grouping no more than five curves per graph. Object classes are sorted by their average mAP, ensuring that those with similar performance metrics are displayed together. Additionally, the function extracts relevant columns from the IRPC Results CSV file to create detailed tables for each object class, summarizing key metrics such as degradation factors, mAP values, Ground Sample Distance (GSD), and knee point indicators. The resulting graphs and tables are combined into a structured and visually informative PDF report.
### 6 - Regression Testing Module 
Regression testing is a critical component of our pipeline to ensure that updates or modifications to the codebase do not unintentionally disrupt existing functionalities or alter expected outputs. In this project, the regression tests are specifically designed to validate the accuracy and consistency of the model outputs across different runs and to verify that changes in pipeline components maintain expected performance. The goal is to confirm that the pipeline consistently produces reliable results, especially for optimal resolution detection in satellite imagery.
Implementation Details:
Test Data Preparation: We begin by setting up a known configuration and a test dataset with known outputs. A tarball containing the input data, configuration files, and pre-existing output files (i.e. the IRP Results CSV File and the Hyperparameters CSV File) is used. This known dataset and configuration ensure consistency in testing, allowing for accurate comparisons between the expected and actual outcomes.
Automated Extraction and Setup: The test tarball is automatically extracted as part of the regression test process. The extracted files are then organized into the required directory structure to match the paths specified in the pipeline_config.yaml file, ensuring compatibility with the pipeline's configurations. Paths within the configuration file are kept relative to enhance flexibility and portability, preventing issues from absolute paths.
Pipeline Execution: Once the environment is set up, the entire pipeline is executed in a controlled manner. The pipeline processes the test data through each module, from data preprocessing to knee discovery and report generation, producing a new IRP Results CSV File, which contains the mAP results across different resolutions.
Comparison of Results: The core of the regression testing is the comparison between the generated output (the IRP Results CSV File) and the pre-existing expected output. Instead of treating any differences as a failure, the regression test now focuses on identifying and reporting specific differences. The script compares key columns—object_name, mAP, GSD, and knee—and reports which columns have differences. This approach acknowledges that certain changes in the pipeline may intentionally affect specific outputs, and it is important to inform the user about these differences without immediately flagging them as errors.
Handling Missing Output Files: If the IRP Results CSV File is missing after pipeline execution, the regression test reports this as a clear error. The absence of this file indicates that the pipeline may have failed to produce the expected output, necessitating further investigation.
Reporting Test Outcomes: Upon completion of the comparison, the regression test outputs a detailed report. If differences are found, it specifies which columns differ between the expected and actual results, allowing developers to understand the impact of their changes. If no differences are found, the test confirms that the pipeline produces consistent results. Importantly, the test does not automatically fail due to differences in data, recognizing that changes may be intentional. However, the missing IRP Results CSV File is treated as a failure condition, as it signifies a potential issue in the pipeline execution.
The regression testing framework thus plays a vital role in our pipeline, enabling consistent and reliable performance across iterations while accommodating intentional changes. By providing detailed feedback on specific differences and ensuring critical outputs are generated, the regression tests help maintain the robustness of optimal resolution detection. This layer of automated validation is particularly valuable for projects like ours, where precision in image processing and model outputs is crucial for practical deployment.
## irpc_results.csv 
To help the user make sense of the pipeline results, the irpc_results.csv file is included in the output. This file summarizes the key performance metrics and configurations as follows:
Original Resolution: Displays the original resolution's width and height. These values serve as the baseline when fine-tuning the YOLO model. 
Effective Resolution: Shows the degraded resolution's width and height. These values are used to evaluate the model's performance on the degraded evaluation images. 
mAP: Stands for mean average precision. This metric measures the accuracy of the model by evaluating how well it predicts bounding boxes and classifies objects.
Degradation Factor: Represents the effective resolution as a fraction of the original resolution. (e.g., 0.5 if the resolution is halved).
GSD: Stands for the ground sampling distance. It indicates the the distance between the centers of two consecutive pixels on the ground, typically in meters. 
Pixels on Target: Indicates the number of pixels covering the target object. This value typically increases as the image resolution improves.
Knee: A boolean value indicating whether a knee point (a significant drop in performance) has been detected at the current degradation level.
## hyperparams.csv
Similar to irpc_results.csv, the hyperparms.csv file is designed to clearly document the hyperparameters used to fine-tune the model. This file highlights the configuration that achieved the best performance. 

