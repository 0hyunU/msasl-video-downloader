import time
import random
import traceback
from matplotlib.pyplot import draw
from vidaug.vidaug import VidAug
import vidaug.aug_vid as av
random.seed(random.random())

import numpy as np
import mediapipe as mp
import cv2
from keypoint import HolisticKeypoint as H
data = av.load_data()
#data random sampling
b = data[random.randint(0,len(data)-1)]['files'][random.randint(0,5)]
   

def draw_landmark(image,mp_drawing, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))

def test_keypoint_from_aug():
    print(b)
    a = VidAug(b)
    a = a.aug_vid_randomly()
    #print(f"{len(a)} frame aug function running time: ",av.get_func_running_time(av.aug_vid_randomly,a))
    print(a.shape)
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        st_time = time.time()
        results = [holistic.process(img) for img in a]
        print(f"{len(a)} process time: ",time.time()-st_time)
        mp_drawing = mp.solutions.drawing_utils
        
        assert len(results) == len(a)
        av.show_vid(a)
        
        try:
            for n,(i,r) in enumerate(zip(a,results)):
                i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
                draw_landmark(i,mp_drawing,r)
                a[n]=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(e)
            print(traceback.format_exc())

        av.show_vid(a)

def check_random_rate_hist():
    a = [random.uniform(0,0.2) for i in range(1000)]
    import matplotlib.pyplot as plt
    plt.hist(a,bins=10)
    plt.show()

if __name__ == "__main__":

    try:
        while True:
            test_keypoint_from_aug()
    except KeyboardInterrupt:
        print("end the program")
