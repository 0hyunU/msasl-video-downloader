import time
import traceback
import mediapipe as mp
import numpy as np
import cv2
import os

from vidaug import VidAug
import aug_vid as av


import random
random.seed(random.random())

DRAW_VID = False
DRAW_KEYPOINT = True

def minmaxscale_keypoint(keypoint):
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    mms.fit(keypoint.reshape((-1,1)))
    res = mms.transform(keypoint.reshape((-1,1))).reshape(keypoint.shape)
    return res

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

def keypoint_from_aug(vid = None):
    
    if vid == None:
        #data random sampling
        #b="C:\\Users\\user\\Documents\\GitHub\\msasl-video-downloader\\wlasl\\WLASL\\start_kit\\train_test\\cool\\13201.mp4"
        data = av.load_data()
        vid_path = data[random.randint(0,len(data)-1)]['files'][random.randint(0,5)]
        print("video path: ",vid_path)
        vid = VidAug(vid_path)
        vid = vid.aug_vid_randomly()
    
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        st_time = time.time()
        results = [holistic.process(img) for img in vid]
        print(f"{len(vid)} process time: ",time.time()-st_time)
        
        assert len(results) == len(vid)
        
        if DRAW_VID:
            av.show_vid(vid)
            mp_drawing = mp.solutions.drawing_utils
            
            try:
                for n,(i,r) in enumerate(zip(vid,results)):
                    i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
                    draw_landmark(i,mp_drawing,r)
                    vid[n]=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(e)
                print(traceback.format_exc())

            av.show_vid(vid)
    y = vid_path.split(os.sep)[-2]
    return results, y

def get_kpoint_arr(keypoint_res):
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
            pose = concat_keypoint_data(i.pose_landmarks, pose)
            face = concat_keypoint_data(i.face_landmarks, face)
            lh = concat_keypoint_data(i.left_hand_landmarks, lh)
            rh = concat_keypoint_data(i.right_hand_landmarks, rh)
    
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
    keypoint_concat = minmaxscale_keypoint(keypoint_concat)
    print("keypoint after scale", keypoint_concat.max(),keypoint_concat.max())

    if DRAW_KEYPOINT: 
        import draw_plot as v
        v.plot_2D_keypoint_every_move(keypoint_concat)

    return keypoint_concat