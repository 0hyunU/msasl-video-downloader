import os
import time
import json
from typing import Iterable
import numpy as np
import mediapipe as mp
import cv2

def video2vec(abs_video_path):
    cap = cv2.VideoCapture(abs_video_path)
    print(abs_video_path)
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    avail_frame = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        b = np.array([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.left_hand_landmarks:
                for id,pose_landmark in enumerate(results.left_hand_landmarks.landmark):
                    pass

                    #print(pose_landmark)

            # full body visible
            holistic_body_visible = ((results.left_hand_landmarks or results.right_hand_landmarks ) 
                                        and results.pose_landmarks and results.face_landmarks)
            
            if holistic_body_visible:
                #print(results.pose_landmarks.landmark)
                #print(type())
                
                # construct keypoint data

                # construct pose_landmark keypoint data
                b = concat_keypoint_data(results.pose_landmarks.landmark, b)



            # Drawing landmarks
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
            mp_drawing.draw_landmarks(
                image, results.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
            #print(dir(results))

            # show image with landmarks
            cv2.imshow('Raw Webcam Feed', image)
            # time.sleep(1)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        print(b.shape)
    print(f"frame satisfying the condition: {avail_frame}")
    cap.release()
    cv2.destroyAllWindows()


def concat_keypoint_data(landmark: Iterable, base:np.ndarray) -> np.ndarray:
    tmp = np.array([[i.x,i.y,i.z] for i in landmark])
    tmp = tmp[np.newaxis,:,:] # add dim at axis = 0
    base = np.vstack([base,tmp]) if base.shape[0] !=0 else tmp
    return base

if __name__ == "__main__":
    #video2vec(0)
    data_abs_path = os.path.abspath(f"./train_test/{os.listdir('./train_test')[1]}")
    
    print(data_abs_path)

    for i in os.listdir(data_abs_path):
        video2vec(os.path.join(data_abs_path,i))
   
#"C:\Users\user\temp\wlasl\WLASL\start_kit\train_test\all\69206.mp4"

