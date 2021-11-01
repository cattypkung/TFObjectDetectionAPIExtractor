import numpy as np
import os
import time
import tensorflow as tf
import pathlib
import re
import shutil

from PIL import Image
from object_detection.protos.calibration_pb2 import ALL_CLASSES

# Object detection module
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from object_detection.builders import model_builder

# Set path directory
PATH_TO_CURRENT  = os.getcwd()
PATH_TO_MODEL    = os.path.join(PATH_TO_CURRENT,'Model')
PATH_TO_CFG      = os.path.join(PATH_TO_MODEL, 'pipeline.config')
PATH_TO_CKPT     = os.path.join(PATH_TO_MODEL, 'checkpoint')
PATH_TO_LABELS   = os.path.join(PATH_TO_CURRENT, 'Labels', 'label_map.pbtxt')
PATH_TO_IMAGES   = pathlib.Path(os.path.join(PATH_TO_CURRENT,'Images'))
FOLDER_LIST      = os.listdir(PATH_TO_IMAGES)

#######################
#   FUNCTION DECLARE  #
#######################
def get_files(folder,extensions):
    all_files = []
    folder = pathlib.Path(os.path.join(PATH_TO_IMAGES,folder))
    for ext in extensions:
        all_files.extend(folder.glob(ext))
    return all_files

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def get_class(label_path):
    all_classes = []
    regex = re.compile(r"'(\w+)'")
    file = open(label_path, 'r') 
    for line in file.readlines():
        # if line is contain name
        if "name" in line:
            all_classes.extend(regex.findall(line))
    file.close()
    return all_classes

def scan_folder(folder):
    PATH_TO_ROI      = os.path.join(PATH_TO_IMAGES,folder,'ROI')
    PATH_TO_DETECT   = os.path.join(PATH_TO_IMAGES,folder,'Detect')
    PATH_TO_UNDETECT = os.path.join(PATH_TO_IMAGES,folder,'Undetect')

    ## Check whether the specified path exists or not
    isRoiExist      = os.path.exists(PATH_TO_ROI)
    isDetectExist   = os.path.exists(PATH_TO_DETECT)
    isUndetectExist = os.path.exists(PATH_TO_UNDETECT)

    if not isRoiExist:
        os.makedirs(PATH_TO_ROI)
        print("The ROI directory in "+ folder +" is created!")

    if not isDetectExist:
        os.makedirs(PATH_TO_DETECT)
        print("The Detect directory in "+ folder +" is created!")

    if not isUndetectExist:
        os.makedirs(PATH_TO_UNDETECT)
        print("The Undetect directory in "+ folder +" is created!")

    ## Collect class names to make folders
    label_lists = get_class(PATH_TO_LABELS)
    for label in label_lists:
        # Create folders
        path_class = os.path.join(PATH_TO_IMAGES, folder, label)
        isFolderExist = os.path.exists(path_class)
        if not isFolderExist:
            os.makedirs(path_class)
            print("The class " + label + " directory is created!")

    ## Load images
    IMAGE_PATHS = get_files(folder,('*.jpg', '*.png'))

    for image_path in IMAGE_PATHS:
        print('Running inference for {}... '.format(image_path), end='')
        filename = os.path.basename(image_path)
        _, fileext = os.path.splitext(filename)

        # Load an image and save into a numpy array
        image_np = np.array(Image.open(image_path).convert('RGB'))

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # and The input also expects a batch of images,
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        
        # Run inference
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Visualization of the results of a detection.
        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        # Set parameters
        ## Coordinates of all boxes
        boxes = detections['detection_boxes']
        ## Get all boxes from an array
        max_boxes_to_draw = boxes.shape[0]
        ## get scores to get a threshold
        scores = detections['detection_scores']
        ## this is set as a default but feel free to adjust it to your needs
        Threshold = .30

        vis_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                boxes,
                detections['detection_classes']+label_id_offset,
                scores,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=Threshold,
                line_thickness=4,
                agnostic_mode=False)

        im_height = image_np_with_detections.shape[0]
        im_width  = image_np_with_detections.shape[1]
    
        # iterate over all objects found
        flag_detect = False
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > Threshold:

                if not flag_detect:
                    flag_detect = True

                class_num  = detections['detection_classes'][i] # this will return class number start from 0
                class_name = category_index[class_num + 1]['name']

                # boxes[i] is the box which will be drawn
                # boxes will return coordinates [ymin, xmin, ymax, xmax]
                boxes[i][0] = boxes[i][0] * im_height # left   or ymin
                boxes[i][1] = boxes[i][1] * im_width  # right  or xmin
                boxes[i][2] = boxes[i][2] * im_height # top    or ymax
                boxes[i][3] = boxes[i][3] * im_width  # bottom or xmax

                # Crop image
                ymin = boxes[i][0].astype(int)
                xmin = boxes[i][1].astype(int)
                ymax = boxes[i][2].astype(int)
                xmax = boxes[i][3].astype(int)
                image_crop_array = image_np[ymin:ymax, xmin:xmax]
                
                # Convert array to image
                image_crop = Image.fromarray(image_crop_array)
                filename_temp = filename.replace(fileext, '_object_' + str(i) + '_' + class_name + fileext)
                filename_temp = os.path.join(PATH_TO_IMAGES, folder, class_name, filename_temp)
                image_crop.save(filename_temp)
                
                # Save image with ROI
                if i == 0:
                    ROIname = os.path.basename(image_path)
                    ROIname = ROIname.replace(fileext, '_ROI' + fileext)
                    ROIname = os.path.join(PATH_TO_ROI,ROIname)
                    ROIimage = Image.fromarray(image_np_with_detections)
                    ROIimage.save(ROIname)

        # Move file if finish
        if flag_detect:
            destination = os.path.join(PATH_TO_DETECT,filename)
            shutil.move(image_path, destination)
        else:
            destination = os.path.join(PATH_TO_UNDETECT,filename)
            shutil.move(image_path, destination)

        print('Done')
    pass

#######################
# PRE PROCESSING STEP #
#######################

# Load model
print('Loading model... ', end='')
start_time      = time.time()

## Load pipeline config and build a detection model
configs         = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config    = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

## Restore checkpoint
ckpt            = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

## List of the strings that is used to add label for each box.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

#######################
#      EXECUTION      #
#######################
for folder in FOLDER_LIST:
    scan_folder(folder)