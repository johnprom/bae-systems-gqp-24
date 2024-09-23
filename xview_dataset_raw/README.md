### Data Structure for Preprocessed Data

Expects 3 folders: "train_images" containing images, the master annoation file: "xView_train.geojson", and "xview_class_labels.json" containing a mapping of numeric class label to class label.

Both the training images folder and "xView_train.geojson" file are currently both being ignored by git due to size constraints. Add them here locally in this structure to run the code:

```
xview_dataset_raw/
├── train_images/
│   ├── image1.tif
│   ├── image2.tif
├── xView_train.geojson
├── xview_class_labels.json

```