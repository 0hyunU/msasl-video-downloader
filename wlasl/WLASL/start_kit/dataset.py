import torch
import json
import torch.utils.data as data_utl
import cv2
import imutils
import numpy as np
import mediapipe as mp
from keypoint import HolisticKeypoint as H
import time

def load_data_json():
    path = "./train_test/data.json"
    with open(path,'r') as f:
        video_data = json.load(f)

    return video_data

def image_aug(image,aug_type,angle):
    if aug_type == "flip":
        image = cv2.flip(image,1)
    elif aug_type == 'rotate':
        image = imutils.rotate(image, angle)
    else: 
        return image
    return image

def keypoint_from_videos(frames, isAug= False):
    
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        pose = np.array([])
        face = np.array([])
        left_hand = np.array([])
        right_hand = np.array([])

        for f in frames:
            results = holistic.process(f)
            holistic_body_visible = ((results.left_hand_landmarks or results.right_hand_landmarks ) 
                                        and results.pose_landmarks and results.face_landmarks)
            if holistic_body_visible:                          
                
                # construct pose_landmark keypoint data
                pose = concat_keypoint_data(results.pose_landmarks, pose)
                face = concat_keypoint_data(results.face_landmarks, face)
                left_hand = concat_keypoint_data(results.left_hand_landmarks, left_hand)
                right_hand = concat_keypoint_data(results.right_hand_landmarks,right_hand)
    
        print(pose.shape, face.shape, left_hand.shape, right_hand.shape)

    return None

def concat_keypoint_data(landmark: object, base:np.ndarray) -> np.ndarray:
    try:
        tmp = np.array([[i.x,i.y,i.z] for i in landmark.landmark])
        tmp = tmp[np.newaxis,:,:] # add dim at axis = 0
        base = np.vstack([base,tmp]) if base.shape[0] !=0 else tmp
    
    except AttributeError as e: #interpolate(보간) data disappeared for a while
        tmp = np.vstack([base,base[-1][np.newaxis,:,:]]) if base.shape[0] != 0 else base
        #print(temp.shape)
        return tmp
    return base

def load_frames_from_videos(path):
    vid = cv2.VideoCapture(path)
    
    frames = []
    while vid.isOpened():
        suc,img = vid.read()
        if not suc: break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.asarray(frames)

class Time():
    def __init__(self) -> None:
        self.start = time.time()
        self.record = [self.start]
    
    def update(self, print_step = True):
        self.record.append(time.time())
        if print_step: print(self.record[-1] - self.record[-2])

        return self.record[-1] - self.record[-2]

    def running(self):
        return self.record[-1] - self.record[0]

class DataSeq(data_utl.Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = load_data_json()

    def __getitem__(self, index):
        gloss , path =  self.data[index]['gloss'], self.data[index]['data_path']
        frames = load_frames_from_videos(path)
        keypoint = keypoint_from_videos(frames)
        
        return gloss,path
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    a = DataSeq()
    t = Time()
    for i in range(len(a)):
        print(a[i])
        t.update()
    print(t.running())