import os
import cv2
import mediapipe as mp
import time 
import json


SAMPLE_VIDEO = "./videos/alcohol/01837.mp4"

def init_dict(point_dict,landmark_type,gloss):
    point_dict["type"] = landmark_type
    point_dict['gloss'] = gloss
    point_dict['data'] = list()
    return point_dict

landmark_dict = init_dict(dict(),'pose','animal')

cap = cv2.VideoCapture(SAMPLE_VIDEO)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
frame = 0

while True:
    try:
        success,img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        mpDraw = mp.solutions.drawing_utils
        frame +=1   
        if results.pose_landmarks:
            landmark = list()
            for id,lm in enumerate(results.pose_landmarks.landmark):
                landmark.append(
                    {'id':id,"x":lm.x,'y':lm.y,'z':lm.z}
                )
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(img, str(id), (cx,cy), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
       
            #print(landmark)
            landmark_dict['data'].append({'frame':frame,'landmark':landmark})

            #draw landmark point on image
            
        else:
            print('nono')
    except Exception as e:
        break

    cv2.imshow("Image", img)
    cv2.waitKey(1)
            