import numpy as np
import cv2 as cv
from time import time
import os
import pathlib
from windowcapture import WindowCapture
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import object_detection

w=1920
h=1080
wincap = WindowCapture(None,w,h)
#wincap.list_window_names()
FONT = cv.FONT_HERSHEY_SIMPLEX
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PATH_TO_SAVED_MODEL = "fine_tuned_model/save_model"
PATH_TO_CFG = "models/my_ssd_mobnet/pipeline.config"
PATH_TO_CKPT = os.path.join('models','my_ssd_mobnet')
LABELMAP = os.path.join('models', 'label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(LABELMAP)
LABELMAP_DICT = label_map_util.get_label_map_dict(LABELMAP)
print(LABELMAP_DICT)
tf.debugging.set_log_device_placement(True)
player = False
stats = False
mini = False

print(tf.__version__)
if tf.test.gpu_device_name(): print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:print("Please install GPU version of TF")
print('Loading model...', end='')
loop_time=time()


#detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time()
elapsed_time = end_time - loop_time
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
print('Done! Took {} seconds'.format(elapsed_time))

while (True):
    image_np_with_detections = wincap.get_screenshot()
    image_np = np.array(image_np_with_detections)
    
    if not (player and mini and stats):
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        #detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        #image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)
        bbox_array = [
                detections['detection_boxes'][index]
                for index,value in enumerate(detections['detection_scores'])
                if value > 0.8
                ]
        objects_array = [
                category_index.get(1+(detections['detection_classes'][index]))['name']
                for index,value in enumerate(detections['detection_scores'])
                if value > 0.8
                ]

        for index, value in enumerate(objects_array):
            if any(value in d.values() for d in category_index.values()):
                bbox = detections['detection_boxes'][index]
                (left, right, top, bottom) = (bbox[1] * w, bbox[3] * w,
                                              bbox[0] * h, bbox[2] * h)
                LABELMAP_DICT[value] = left, right, top, bottom
                if value == 'Player_Bar':
                    player = True
                if value == 'Stats_Display':
                    stats = True
                if value == 'Mini_Map':
                    mini = True
                
                image_np_with_detections = cv.putText(image_np_with_detections,
                                                      value, (50,(index*50)+50),
                                                      FONT, 1, (255,0,0), 2,
                                                      cv.LINE_AA)
    #[y_min, x_min, y_max, x_max]
    #(left, right, top, bottom) = (xmin * im_width, xmax * im_width, 
    #                           ymin * im_height, ymax * im_height)
    for index, value in LABELMAP_DICT.items():
        if type(value) == tuple:
            
            crop = image_np_with_detections[int(value[2]):int(value[3]),
                                            int(value[0]):int(value[1])]
            cv.imshow(index,crop)
    
    cv.imshow('computer vision', image_np_with_detections)
    
    print('FPS {}'.format(1/(time()-loop_time)))
    loop_time = time()
    if cv.waitKey(1) == ord('q'):
        print(LABELMAP_DICT)
        cv.destroyAllWindows()
        break
print('done')
