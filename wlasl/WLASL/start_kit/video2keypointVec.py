import os
import time
import json

import numpy as np
import mediapipe as mp
import cv2
import imutils
import pickle
from keypoint import HolisticKeypoint as H
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


def image_aug(image,aug_type,angle):
    if aug_type == "flip":
        image = cv2.flip(image,1)
    elif aug_type == 'rotate':
        image = imutils.rotate(image, angle)
    else: 
        return image
    return image

def gen_keypoint(abs_video_path:str, is_aug=False,aug_type=None,rotate_angle=0, show_image=False) -> H:
    
    """ load video
        extract keypoint data from video
    """

    cap = cv2.VideoCapture(abs_video_path)
    
    keypoint_data = H(file_path=abs_video_path,
                file_name=abs_video_path.split(os.sep)[-1],
                avail_frame_len=0,
                action_class=abs_video_path.split(os.sep)[-2],
                left_hand=np.array([]), right_hand=np.array([]),
                pose=np.array([]), face=np.array([]))
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    avail_frame = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # load image by frame
        while cap.isOpened():
            ret, frame = cap.read()
            
            # frame ended
            if not ret: 
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # image augmentation flip
            if is_aug: 
                image = image_aug(image,aug_type=aug_type, angle=rotate_angle)
            
            # Make Detections
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # full body visible
            holistic_body_visible = ((results.left_hand_landmarks or results.right_hand_landmarks) 
                                        and results.pose_landmarks and results.face_landmarks)
            #print(type(results.left_hand_landmarks), type(results.right_hand_landmarks),type(results.pose_landmarks), type(results.face_landmarks))
            if holistic_body_visible:
                
                avail_frame +=1
                # construct keypoint data

                # construct pose_landmark keypoint data
                keypoint_data.pose = concat_keypoint_data(results.pose_landmarks, keypoint_data.pose)
                keypoint_data.face = concat_keypoint_data(results.face_landmarks, keypoint_data.face)
                keypoint_data.left_hand = concat_keypoint_data(results.left_hand_landmarks, keypoint_data.left_hand)
                keypoint_data.right_hand = concat_keypoint_data(results.right_hand_landmarks, keypoint_data.right_hand)

            # Drawing landmarks
            draw_landmark(image,mp_drawing,results)

            # show image with landmarks
            if show_image: 
                cv2.imshow('Raw Webcam Feed', image)
                #time.sleep(1)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
      
        keypoint_data.avail_frame_len = avail_frame
        keypoint_data = add_missing_data(keypoint_data)
        assert keypoint_data.get_face_data_len == keypoint_data.avail_frame_len

    cap.release()
    cv2.destroyAllWindows()

    return keypoint_data

def video2vec(abs_video_path):

    #video preprocessing
    AUG = False #flip augmentation
    # aug_type = "flip"
    # keypoint_data = gen_keypoint(abs_video_path,is_aug=AUG,aug_type=aug_type)
    # save_pickle(abs_video_path, keypoint_data,is_aug=AUG, aug_type=aug_type)
    tilt_range = [-20,-10,10,20] if AUG else [0]
    aug_type = 'rotate'
    
    for angle in tilt_range:
        keypoint_data = gen_keypoint(abs_video_path,is_aug=AUG,aug_type=aug_type,rotate_angle=angle)
        #save_pickle(abs_video_path, keypoint_data,is_aug=AUG, aug_type=aug_type + str(angle))

    print(f"frame satisfying the condition: {keypoint_data.avail_frame_len}")
    #print(f"{len(keypoint_data.pose)},{len(keypoint_data.face)},{len(keypoint_data.right_hand)},{len(keypoint_data.left_hand)}")
    

def assert_keypoint_dimension(key_point):
    assert key_point.left_hand.ndim == 3
    assert key_point.right_hand.ndim == 3
    assert key_point.pose.ndim == 3
    assert key_point.face.ndim == 3

# 길이가 다른 데이터(missing data) add(추가하기)
def add_missing_data(key_point):
    """
        ex> lh,rh,pose,face shape -> 50*21*3 , 25*21*3, 50*n*c, 50*k*c
        add_missing_data(key_point) -> 50*21*3 , 50*21*3, 50*n*c, 50*k*c
    """
    if key_point.get_lh_data_len < key_point.avail_frame_len:
        try:
            missing_len = key_point.avail_frame_len - key_point.get_lh_data_len
            temp = np.tile(key_point.left_hand[0],(missing_len,1,1))
            key_point.left_hand = np.vstack([temp,key_point.left_hand])
        except IndexError as e:
            print(e)
            # 아예 한 손만 사용하는 경우(case using only one hand -> zero padding)
            if key_point.get_lh_data_len == 0: 
                rh_shape = key_point.right_hand.shape
                key_point.left_hand = np.zeros((key_point.avail_frame_len,rh_shape[1],rh_shape[2]))
                print(key_point.left_hand.shape)

    if key_point.get_rh_data_len < key_point.avail_frame_len:
        try:
            missing_len = key_point.avail_frame_len - key_point.get_rh_data_len
            temp = np.tile(key_point.right_hand[0],(missing_len,1,1))
            key_point.right_hand = np.vstack([temp,key_point.right_hand])
        except IndexError as e:
            print(e)
            # 아예 한 손만 사용하는 경우(case using only one hand -> zero padding)
            if key_point.get_rh_data_len == 0: 
                rh_shape = key_point.left_hand.shape
                key_point.right_hand = np.zeros((key_point.avail_frame_len,rh_shape[1],rh_shape[2]))
                print(key_point.right_hand.shape)
    print(key_point.get_rh_data_len,key_point.avail_frame_len,key_point.get_lh_data_len)
    assert key_point.get_rh_data_len == key_point.avail_frame_len
    assert key_point.get_lh_data_len == key_point.avail_frame_len
    
    try:
        assert key_point.avail_frame_len != 0
    except Exception as e:
        print(e) 

    return key_point



#save KeyPoint object to pickle
def save_pickle(abs_video_path, keypoint_data,aug_type:str, is_aug=False):
    dir_name = os.path.dirname(abs_video_path)
    output_name = keypoint_data.file_name if not is_aug else keypoint_data.file_name + f"_{aug_type}"
    with open(os.path.join(dir_name,f"{output_name}.pickle"),'wb') as f:
        pickle.dump(keypoint_data,f)

def draw_landmark(image,mp_drawing, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))

# stack keypoint data consequently
def concat_keypoint_data(landmark: object, base:np.ndarray) -> np.ndarray:
    
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
        

if __name__ == "__main__":
    #video2vec(0)
    cls_list = os.listdir('./train_test')

    # video2vec(os.path.abspath("./train_test/who/63237.mp4"))

    # pass
    t = Time()

    for j in cls_list:
        data_abs_path = os.path.abspath(f"./train_test/{j}")
        
        print(data_abs_path)

        for i in os.listdir(data_abs_path):
            if ".mp4" in i[-4:]:
                video2vec(os.path.join(data_abs_path,i))
                t.update()
    print("--- %s seconds ---" % (time.time() - t.running))

