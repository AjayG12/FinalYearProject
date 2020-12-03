import os
import datetime 
import time

import tensorflow as tf
import cv2

#from gtts import gTTS
from playsound import playsound

from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

#from firebase import firebase
#fixefixed_interval = 3

#firebase = firebase.FirebaseApplication('https://watch-and-do.firebaseio.com', None)

sess = tf.Session() 
my_head_pose_estimator = CnnHeadPoseEstimator(sess)

cam = cv2.VideoCapture(0) 
  
try: 
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
except OSError: 
    print ('Error: Creating directory of data') 
    
my_head_pose_estimator.load_yaw_variables(os.path.realpath("C:/Users/abhishek prakash/Downloads/Watch & Do Object Detection and Gaze Estimation mod2/deepgaze-master/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))


last_command=""
cur_cmd=""
currentframe = 0

while (True):
    
      ret,frame = cam.read()
        
      frame = cv2.resize(frame,(640,640))
      
      if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break
   
      ss = my_head_pose_estimator.return_yaw(frame)
      
      k=ss[0][0][0]
    
      if k>21 and k<100:
            
        print('right')
        print('TV')
        #playsound("tv.mp3")

        status="TV"
        data={"Status":status}
       # firebase.put('', 'WATCH AND DO/IOT device', data)
        
        cur_cmd="TV"
        
      elif k<(-19)and k>(-100):
        
        print('left')
        print('Refrigerator')
        #playsound("bulb.mp3")

        status="bulb"
        data={"Status":status}
        
        #firebase.put('', 'WATCH AND DO/IOT device', data)
        
        cur_cmd="bulb"
        
        
      elif k>(-20) and k<(20):
        #print(k)
        
        print('stright')
        print('FAN')
        
        status="fan"
        data={"Status":status}
        #firebase.put('', 'WATCH AND DO/IOT device', data)
        #playsound("fan.mp3")

        cur_cmd="Fan"
        
      cv2.putText(frame,cur_cmd, (200,600), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),6)
        
      cv2.imshow('frame',frame)
      

cam.release()
cv2.destroyAllWindows()







