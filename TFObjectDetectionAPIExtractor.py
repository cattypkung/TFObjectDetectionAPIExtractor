import numpy as np
import os
import time
import tensorflow as tf
import pathlib
import shutil

from PIL import Image

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
PATH_TO_ROI      = os.path.join(PATH_TO_IMAGES,'ROI')
PATH_TO_DETECT   = os.path.join(PATH_TO_IMAGES,'Detect')
PATH_TO_UNDETECT = os.path.join(PATH_TO_IMAGES,'Undetect')

#######################
#   FUNCTION DECLARE  #
#######################
def get_files(extensions):
    all_files = []
    for ext in extensions:
        all_files.extend(PATH_TO_IMAGES.glob(ext))
    return all_files

def detect_fn(detection_model,image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

#######################
#      EXECUTION      #
#######################
def main():
    ## Avoid out of memory by setting GPU memory consumption growth
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)

    ## Load model
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

    ## Load images
    image_list = get_files(('*.jpg', '*.png'))
    for image_path in image_list:

        # Set parameter
        filename = os.path.basename(image_path)
        _, fileext = os.path.splitext(filename)
        flag_detect = False

        print('Running inference for {}... '.format(image_path), end='')

        # Load an image and save into a numpy array
        image_np = np.array(Image.open(image_path).convert('RGB'))

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # and The input also expects a batch of images,
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        
        # Run inference
        detections = detect_fn(detection_model,input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
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
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > Threshold:

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
                filename_temp = os.path.join(PATH_TO_IMAGES, class_name, filename_temp)
                try:
                    image_crop.save(filename_temp)
                except OSError:
                    os.makedirs(os.path.join(PATH_TO_IMAGES,class_name))
                    image_crop.save(filename_temp)
                
                # Save image with ROI
                if i == 0:
                    flag_detect = True
                    image_ROI    = Image.fromarray(image_np_with_detections)                    
                    filename_ROI = os.path.basename(image_path)
                    filename_ROI = filename_ROI.replace(fileext, '_ROI' + fileext)
                    filename_ROI = os.path.join(PATH_TO_ROI,filename_ROI)
                    try:
                        image_ROI.save(filename_ROI)
                    except OSError:
                        os.makedirs(PATH_TO_ROI)
                        image_ROI.save(filename_ROI)

        # Move file if finish
        if flag_detect:
            destination = os.path.join(PATH_TO_DETECT,filename)
            try:
                shutil.move(image_path, destination)
            except OSError:
                os.makedirs(PATH_TO_DETECT)
                shutil.move(image_path, destination)
        else:
            destination = os.path.join(PATH_TO_UNDETECT,filename)
            try:
                shutil.move(image_path, destination)
            except OSError:
                os.makedirs(PATH_TO_UNDETECT)
                shutil.move(image_path, destination)

        print('Done!')

if __name__ == "__main__":
    main()