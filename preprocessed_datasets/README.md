### Data Structure for Preprocessed Data

For use in YOLO models data should be structured as follows. This structure can store multiple variations of the preprocessed dataset at a time if applicable. 

Note also that for each dataset there must be a train and val directory each with directories "images/" and "labels/". There should be a corresponding annotation file (.txt) for each image (1:1). These are linked by naming convention (eg: image1.tif is to image1.txt).

```
preprocessed_datasets/
├── filtered_labels.json 
│   
├── baseline-unsplit/      # this is the full preprocessed set before train/test split
│   ├── images/
│   │   ├── image1.tif
│   │   ├── image2.tif
│   ├── labels/
│       ├── image1.txt
│       └── image2.txt
│   
├── baseline-<approach>-<max_dim>-<imgsz>/     # baseline set from which we degrade
│   │                                        # highest effective resolution possible
│   │                                        # split into test and train sets
│   │                                        # eg: degraded-padding-4224-1280
│   ├── train/
│   │   ├── images/
│   │   │   ├── image1.tif
│   │   │   ├── image2.tif
│   │   └── labels/
│   │       ├── image1.txt
│   │       ├── image2.txt
│   ├── val/
│       ├── images/
│       │   ├── image1.tif
│       └── labels/
│           ├── image1.txt
│   
├── degraded-<approach>-<max_dim>-<effective_imgsz>/      # baseline set from which we degrade    
│   │                                                   # eg: degraded-tiling-4224-608        
│   ├── train/
│   │   ├── images/
│   │   │   ├── image1.tif
│   │   │   ├── image2.tif
│   │   └── labels/
│   │       ├── image1.txt
│   │       ├── image2.txt
│   ├── val/
│       ├── images/
│       │   ├── image1.tif
│       └── labels/
│           ├── image1.txt
```