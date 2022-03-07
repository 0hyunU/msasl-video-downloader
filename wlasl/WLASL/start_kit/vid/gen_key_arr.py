import time
import traceback
import mediapipe as mp
import numpy as np
import cv2
import dataclasses
import copy
from aug_vid import *
import random
random.seed(random.random())

DRAW_VID = False
DRAW_KEYPOINT = False
VERBOSE =False 
FACE_FEATUERS = 468
POSE_FEATURES = 33
HAND_FEATURES = 21


@dataclasses.dataclass
class KeyArrGen():
    vid_arr: np.array

    def get_keyarr(self):
        key_res = self.keypoint_from_vid()
        return self.kpoint2arr(key_res)

    def minmaxscale_keypoint(self,keypoint):
        #todo x,y separate
        from sklearn.preprocessing import MinMaxScaler
        
        mms = MinMaxScaler()
        res = copy.deepcopy(keypoint)

        for i in range(res.shape[2]):
            res[:,:,i] =mms.fit_transform(keypoint[:,:,i].reshape((-1,1))).reshape(keypoint.shape[:2])

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
    def concat_keypoint_data(self,landmark: object, base:np.ndarray, hands_mean = 0, hand=True) -> np.ndarray:
        
        if landmark:
            tmp = np.array([[i.x,i.y,i.z] for i in landmark.landmark])
            if VERBOSE and np.any(hands_mean): print(f"distance: {np.sqrt(sum((tmp.mean(axis=0) - hands_mean) **2))}")
            if VERBOSE and (len(tmp) == 33): 
                print("elbow D:",np.sqrt(sum((tmp[13]-tmp[15])**2)))
                print("hand D:",np.sqrt(sum((tmp[21]-tmp[15])**2)))
            return np.vstack([base,np.expand_dims(tmp,0)])
        else:
            """
            if landmark.landmark is None
            when keypoint disappear for a while
            interpolate(보간) data with previous keypoint
            """
            # a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
            # if there's no data interploate
            # fill the data with pose's hand's data

            # interpolation case: 
            # when hands are not detected, fill with pose's hand data 
            # when hands are too far from pose's hand's coordinate
            if (np.any(hands_mean) and ~np.any(base[-1])) or \
            (np.any(hands_mean) and (np.sqrt(sum((base[-1].mean(axis=0) - hands_mean) **2)) > 0.5)):
                tmp = np.zeros(base[-1].shape)
                tmp[:,:] = hands_mean
                if VERBOSE and np.any(hands_mean): print(f"distance: {np.sqrt(sum((base[-1].mean(axis=0) - hands_mean) **2))}")
                
                if VERBOSE: print("filled with hand's avr")
                return np.vstack([base,np.expand_dims(tmp,0)])
            
            if VERBOSE and np.any(hands_mean): print(f"distance: {np.sqrt(sum((base[-1].mean(axis=0) - hands_mean) **2))}")
            
            return np.vstack([base,np.expand_dims(base[-1],0)])

    def keypoint_from_vid(self):
        mp_holistic = mp.solutions.holistic

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            st_time = time.time()
            results = [holistic.process(img) for img in self.vid_arr]
            if VERBOSE: print(f"{len(self.vid_arr)} process time: ",time.time()-st_time)
            
            assert len(results) == len(self.vid_arr)
            
            if DRAW_VID:
                show_vid(self.vid_arr, vid_title="Image without keypoint")
                mp_drawing = mp.solutions.drawing_utils
                
                try:
                    for n,(i,r) in enumerate(zip(self.vid_arr,results)):
                        i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
                        self.draw_landmark(i,mp_drawing,r)
                        self.vid_arr[n]=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())

                show_vid(self.vid_arr, vid_title="Image with keypoint")
        return results
    def modi_seq_length_rand(self,key_arr):
        if random.random() > 0.5: 
            res = self.shorten_seq(key_arr)
        else: 
            res = self.lengthen_seq(key_arr)
    
        return res
    def lengthen_seq(self,key_arr):
        res = np.expand_dims(key_arr[0],0)
        i = 1
        while i< len(key_arr[1:]):
            if random.random() <= 0.1:
                tmp = key_arr[i-1:i+1].mean(axis=0)
                res = np.vstack([res,np.expand_dims(tmp,0)])
                continue
                
            res = np.vstack([res,np.expand_dims(key_arr[i],0)])
            i += 1
        if VERBOSE: print("lengthen seq shapes:",res.shape, key_arr.shape)

        return res
    def shorten_seq(self, key_arr):
        res = np.expand_dims(key_arr[0],0)
        i = 1
        while i< len(key_arr[1:]):
            if random.random() <= 0.1:
                i+=1
                continue
            
            res = np.vstack([res, np.expand_dims(key_arr[i],0)])
            i += 1
        if VERBOSE: print("shorten seq shapes:",res.shape, key_arr.shape)
        return res


    def kpoint2arr(self,keypoint_res):
        face = pose = lh = rh= np.array([])
        avail_frame = 0 # start 1 base array stacked
        avail_flag = False
        for i in keypoint_res:
            holistic_body_visible = ((i.left_hand_landmarks or i.right_hand_landmarks) 
                                        and i.pose_landmarks and i.face_landmarks) is not None
            
            # start frame detected body keypoint
            # And init array
            if holistic_body_visible and not avail_flag: 
                avail_flag = True
                avail_frame += 1 # start 1 base array stacked
                face = np.zeros((1,FACE_FEATUERS,3))
                pose = np.zeros((1,POSE_FEATURES,3))
                lh = rh = np.zeros((1,HAND_FEATURES,3))

            if avail_flag:
                avail_frame +=1
                pose = self.concat_keypoint_data(i.pose_landmarks, pose)
                
                # data interpolation
                lh_indices =[15,17,19,21]
                rh_indices = [16,18,20,22]
                # for hands keypoint interpolation
                # in case: hands not detected or 
                # when too far two of each data which hands data and pose's hands data 
                lh_avr = pose[-1,lh_indices,:].mean(axis=0) 
                rh_avr = pose[-1,rh_indices,:].mean(axis=0)
                
                if avail_frame %2:
                    i.face_landmarks = None
                face = self.concat_keypoint_data(i.face_landmarks, face)
                lh = self.concat_keypoint_data(i.left_hand_landmarks, lh, lh_avr)
                rh = self.concat_keypoint_data(i.right_hand_landmarks, rh, rh_avr)
                #print(lh[-1].mean(axis=0) == lh_avr, lh_avr, lh[-1].mean(axis=0), lh[-1].shape)
        
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

        try:
            # keypoint minmax scaling
            keypoint_concat = np.concatenate((face,pose[:,:23,:],lh,rh),axis=1)
            if VERBOSE: print("keypoint before scale:",keypoint_concat.shape,keypoint_concat[:,:,0].min(), keypoint_concat[:,:,0].max(),keypoint_concat[:,:,1].min(), keypoint_concat[:,:,1].max())  
            keypoint_concat = self.minmaxscale_keypoint(keypoint_concat)
        except Exception as e:
            print("error occur, processed array len: ",len(face), avail_frame)
            print(e)
            print(traceback.format_exc())
            return np.zeros((1,))

        if DRAW_KEYPOINT: 
            from .draw_plot import plot_2D_keypoint_every_move
            plot_2D_keypoint_every_move(keypoint_concat)


        st_time = time.time()
        if VERBOSE : print("normalize before and after is same?:",np.allclose(keypoint_concat,keypoint_concat_))
        res = self.modi_seq_length_rand(keypoint_concat)
        if VERBOSE: print("numpy injection time: ",time.time()- st_time)

        if DRAW_KEYPOINT: 
            (res)

        return res