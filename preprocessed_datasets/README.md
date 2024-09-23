### Data Structure for Preprocessed Data

For use in YOLO models data should be structured as follows. This structure can store multiple variations of the preprocessed dataset at a time if applicable. 

Note also that for each dataset there must be a train and val directory each with directories "images/" and "labels/". There should be a corresponding annotation file (.txt) for each image (1:1). These are linked by naming convention (eg: image1.tif is to image1.txt).

```
preprocessed_datasets/
├── dataset_1/
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
├── dataset_2/
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