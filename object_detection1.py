
# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture(0)
sys.path.append("E:/project/Watch & Do Object Detection and Gaze Estimation mod1/object_detection/object_detection_webcam.py")



from gtts import gTTS
from playsound import playsound


# In[3]:

#from utils import label_map_util

#from utils import visualization_utils as vis_util


# In[4]:
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT ='E:/project/Watch & Do Object Detection and Gaze Estimation mod1/object_detection/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_complete_label_map.pbtxt')

NUM_CLASSES = 90

# In[5]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# In[9]:

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

IMAGE_SIZE = (12, 8)

# In[10]:
import pandas as pd
from datetime import datetime
import json
import time
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      df = pd.DataFrame(boxes.reshape(100, 4), columns=['y_min', 'x_min', 'y_max', 'x_max'])
      df1 = pd.DataFrame(classes.reshape(100, 1), columns=['classes'])
      df2 = pd.DataFrame(scores.reshape(100, 1), columns=['scores'])
      df5 = pd.concat([df, df1, df2], axis=1)
      df6 = df5.loc[df5['classes']==1]
      df9 = df5.loc[df5['classes'] == 1]
                    
                    
      df7 = df6.loc[df6['scores'] > 0.50]
      df7 = df6.loc[df6['scores'] > 0.50]

      if int(len(df7.index)) > 0:
          people = int(len(df7.index))
      else:
          people = 0

      columns = ['LocationID','DeviceID','DeviceTime', 'Class', 'Count']
      index = [0]
      timenow = datetime.utcnow()
      df_ = pd.DataFrame(index=index, columns=columns)
      df_.loc[0, 'LocationID'] = '1'
      df_.loc[0, 'DeviceID'] = '32'
      df_.loc[0, 'DeviceTime'] = timenow
      df_.loc[0, 'Class'] =1
      df_.loc[0, 'Count'] = people
                    
                    

      jn = df_.to_json(orient='records', lines=True)

      jn1 = json.loads(jn)
                    #print(jn)
      #print(jn1['Count'])
      
                    
      if jn1['Count']!=0:
          playsound("fan.mp3")
          print(jn1['Count'])
          
                    

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# In[ ]:



