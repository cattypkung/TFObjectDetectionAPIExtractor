# TFObjectDetectionAPIExtractor
 A tool for extracting objects in images by using [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

# How to use
1. Put your `.jpg` and `.png` files into the `Images` folder.
2. Put your trained model from [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) into the `Model` folder.
3. Rename your `.pbtxt` as `label_map.pbtxt` and put it into the `Labels` folder.<br />
After these 3 steps, the folders should look like below:
```
.
├── Images
│   ├── *.jpg
│   ├── *.png
│   └── ...
├── Labels
│   └── lavel_map.pbtxt
├── Model
│   ├── checkpoint
│   ├── saved_model
│   └── pipeline.config
...
```
4. Run `python TFObjectDetectionAPIExtractor.py`

# Output
The output images will be saved in the `Images` folder by:
* All detected objects will be extracted and saved into `class name` folders, separately.
* All original images with Region of Interest will be saved into `ROI` folder.
* All original images will move into `Detect` or `Undetect` folders, depend on detection result.<br />
After run `TFObjectDetectionAPIExtractor.py`, the `Images` folder should look like below:
```
.
├── Images
│   ├── Object class
│   │     ├── *.jpg
│   │     ├── *.png
│   │     └── ...
│   ├── Detect
│   │     ├── *.jpg
│   │     ├── *.png
│   │     └── ...
│   ├── ROI
│   │     ├── *.jpg
│   │     ├── *.png
│   │     └── ...
│   ├── Undetect
│   │     ├── *.jpg
│   │     ├── *.png
│   │     └── ...
...
```
