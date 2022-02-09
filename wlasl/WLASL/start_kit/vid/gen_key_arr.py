import time
import traceback
from attr import dataclass
import mediapipe as mp
import numpy as np
import cv2
import dataclasses

from .aug_vid import *

import random
random.seed(random.random())

DRAW_VID = False
DRAW_KEYPOINT = True

@dataclasses.dataclass
class KeyArrGen():
    vid_arr: np.array

    def get_keyarr(self):
        key_res = self.keypoint_from_vid()
        return self.kpoint2arr(key_res)

    def minmaxscale_keypoint(self,keypoint):
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler()
        mms.fit(keypoint.reshape((-1,1)))
        res = mms.transform(keypoint.reshape((-1,1))).reshape(keypoint.shape)
        return res

    def draw_landmark(self,image,mp_drawing, results):
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
        mp_drawing.draw_landmarks(
            image, results.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))

    # stack keypoint data consequently
    def concat_keypoint_data(self,landmark: object, base:np.ndarray) -> np.ndarray:
        
        if landmark:
            tmp = np.array([[i.x,i.y] for i in landmark.landmark])
            return np.vstack([base,np.expand_dims(tmp,0)])
        else:
            """
            if landmark.landmark is None
            when keypoint disappear for a while
            interpolate(보간) data with previous keypoint
            """
            return np.vstack([base,np.expand_dims(base[-1],0)])

    def keypoint_from_vid(self):
        mp_holistic = mp.solutions.holistic

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            st_time = time.time()
            results = [holistic.process(img) for img in self.vid_arr]
            print(f"{len(self.vid_arr)} process time: ",time.time()-st_time)
            
            assert len(results) == len(self.vid_arr)
            
            if DRAW_VID:
                show_vid(self.vid_arr)
                mp_drawing = mp.solutions.drawing_utils
                
                try:
                    for n,(i,r) in enumerate(zip(self.vid_arr,results)):
                        i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
                        self.draw_landmark(i,mp_drawing,r)
                        self.vid_arr[n]=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())

                show_vid(self.vid_arr)
        return results

    def kpoint2arr(self,keypoint_res):
        face = pose = lh = rh= np.array([])
        avail_frame = 1 # start 1 base array stacked
        avail_flag = False
        for i in keypoint_res:
            holistic_body_visible = ((i.left_hand_landmarks or i.right_hand_landmarks) 
                                        and i.pose_landmarks and i.face_landmarks) is not None
            
            # start frame detected body keypoint
            # And init array
            if holistic_body_visible and not avail_flag: 
                avail_flag = True
                face = np.zeros((1,468,2))
                pose = np.zeros((1,33,2))
                lh = rh = np.zeros((1,21,2))

            if avail_flag:
                avail_frame +=1
                pose = self.concat_keypoint_data(i.pose_landmarks, pose)
                face = self.concat_keypoint_data(i.face_landmarks, face)
                lh = self.concat_keypoint_data(i.left_hand_landmarks, lh)
                rh = self.concat_keypoint_data(i.right_hand_landmarks, rh)
        
        face = np.array(face)
        pose = np.array(pose)
        lh = np.array(lh)
        rh = np.array(rh)

        # keypoint array length should be same with available frame
        try:
            assert len(face) == len(pose) == len(lh) == len(rh) == avail_frame
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("array len: ",len(face) ,len(pose) ,len(lh) ,len(rh) , avail_frame)

        # keypoint minmax scaling
        keypoint_concat = np.concatenate((face,pose,lh,rh),axis=1)
        print("keypoint before scale:", keypoint_concat.max(), keypoint_concat.min())   
        keypoint_concat = self.minmaxscale_keypoint(keypoint_concat)
        print("keypoint after scale", keypoint_concat.max(),keypoint_concat.max())

        if DRAW_KEYPOINT: 
            import draw_plot as v
            v.plot_2D_keypoint_every_move(keypoint_concat)

        return keypoint_concat