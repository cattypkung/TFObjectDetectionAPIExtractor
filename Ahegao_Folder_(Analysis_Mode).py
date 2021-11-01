import numpy as np
import os
import time
import tensorflow as tf
import pathlib
import re

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
    ## Collect class names to make folders
    label_lists   = get_class(PATH_TO_LABELS)
    label_counter = {}
    label_page    = {}
    for label in label_lists:
        # Create counter
        label_counter[label] = 0
        label_page[label]    = 0
        
    ## Load images
    IMAGE_PATHS = get_files(folder,('*.jpg', '*.png'))

    for image_path in IMAGE_PATHS:
        print('Running inference for {}... '.format(image_path), end='')

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
    
        # iterate over all objects found
        object_list = []
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > Threshold:

                class_num  = detections['detection_classes'][i] # this will return class number start from 0
                class_name = category_index[class_num + 1]['name']

                object_list.append(class_name)

                # count how many objects found in current page
                label_counter[class_name] = label_counter[class_name] + 1

        object_list = set(object_list)
        for obj in object_list:
            label_page[obj] = label_page[obj] + 1

    # Move file if finish
    print("COUNTER: {0}".format(label_counter))
    print("PAGE: {0}".format(label_page))

    analysis_result = os.path.join(PATH_TO_IMAGES,folder,'[Analysis result].txt')
    file = open(analysis_result, 'w')
    file.write("###########\n")
    file.write("# Objects #\n")
    file.write("###########\n\n")
    
    for key in label_counter:
        file.write("Class {0}: {1} objects\n".format(key,label_counter[key]))

    file.write("\n###########\n")
    file.write("#  Pages  #\n")
    file.write("###########\n\n")

    for key in label_page:
        file.write("Class {0}: {1} pages of {2}\n".format(key,label_counter[key],len(IMAGE_PATHS)))

    file.close()

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