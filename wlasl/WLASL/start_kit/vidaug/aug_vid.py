import os
import time
import json
import glob
import sys
import random
import traceback
from cv2 import resize

random.seed(random.random())
from matplotlib.pyplot import show

import numpy as np
import mediapipe as mp
import cv2
import imutils
import pickle
from keypoint import HolisticKeypoint as H

DATA_ROOT = './train_test'
IS_RANDOM = True

def split_origin_vid():
    train_json, test_json = list(), list()
    for dirpath, dirnames, filenames in os.walk('./train_test'):
    
        if dirpath == DATA_ROOT: continue

        gloss = dirpath.split(os.sep)[-1]
        filenames = [ os.path.abspath(os.path.join(DATA_ROOT,gloss,i)) for i in filenames if i.endswith('.mp4')] #origin vid
        
        p = int(len(filenames)*0.8)

        train, test = filenames[:p] , filenames[p:]
        train_json.append({"gloss":gloss, "files":train})
        test_json.append({"gloss":gloss, "files":test})

    json.dump( train_json,open("./train_test/train.json",'w'))
    json.dump(test_json,open( "./train_test/test.json",'w'))

#split_origin_vid()

def load_data(data:str = "train") -> list:
    """
        data = train or test
    """
    if data == "train":
        return json.load(open("./train_test/train.json",'r'))
    
    elif data == "test":
        return json.load(open("./train_test/test.json",'r'))

TEST_VID_PATH = load_data("train")[random.randint(0,5)]['files'][random.randint(0,5)]

def vid2arr(vid_path:str, save_obj:bool = False) -> np.array:
    cap = cv2.VideoCapture(vid_path)
    frame_list = list()
    while cap.isOpened():
        ret, frame = cap.read()

        # frame ended
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(image)

    frame_list = np.array(frame_list)

    if save_obj:
        save_path = vid_path + "Arr.pickle"
        pickle.dump(frame_list,open(save_path,'wb'))

    return frame_list

def show_vid(vid_arr,vid_path=TEST_VID_PATH) -> None:
    
    if vid_arr is None:
        vid_arr = vid2arr(vid_path)
    else:
        vid_arr = vid_arr

    for i in vid_arr:
        cv2.imshow("Image",cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)

def calcul_time_by_frame() -> None:
    st_time = time.time()
    a = glob.glob("./train_test/**/*.mp4")
    frame_sum = 0
    for i in a:
        frame_sum += len(vid2arr(i, True))

    end_time = time.time()
    running_time = end_time - st_time
    print("running time: ",running_time)
    print("running time by frame:", running_time/frame_sum)

def get_func_running_time(function, *args,**kargs):
    st_time = time.time()
    function(*args, **kargs)
    return time.time() - st_time

def load_vidArr(vid_path):
    arr_path = vid_path + "arr.pickle"
    try:
        if not os.path.isfile(arr_path): vid_arr = vid2arr(vid_path)
        else : vid_arr = pickle.load(open(arr_path,'rb'))
    except Exception as e:
        print(e)
        print(vid_path, arr_path)

    return vid_arr

def flip_vidArr(vid_arr, show=False):

    # vid_arr = vid_arr[:,:,::-1,:] # horizontal flip
    vid_arr = np.flip(vid_arr,2) #horizontal flip
    if show: show_vid(vid_arr)

    return vid_arr

def stretch_vidArr(vid_arr, show=False):
    cols,rows = vid_arr.shape[1:3]
    rate_y = 1 + random.uniform(-0.2,0.2)
    rate_x = 1 + random.uniform(-0.2,0.2)
    resize_arr = np.array([cv2.resize(img,(int(rows*rate_y),int(cols*rate_x))) for img in vid_arr])
    #print(resize_arr.shape)
    #show_vid(resize_arr)
    return resize_arr

def rotate_vidArr(vid_arr,angle=10,rot_bound=False, show=False):
    
    if IS_RANDOM: angle = random.uniform(-20,20)
    # rotate type
    if random.randint(0,1) % 2: 
        vid_arr_rot = np.array([imutils.rotate_bound(img,angle) for img in vid_arr])
    else:
        vid_arr_rot = np.array([imutils.rotate(image, angle) for image in vid_arr])

    if show: show_vid(vid_arr_rot)

    return vid_arr_rot



def hshift_vidArr(vid_arr, rate=0.1):
    if IS_RANDOM: rate = random.gauss(0,0.1)
    rows,cols = vid_arr.shape[1:3]

    M = np.float32([[1, 0, cols * rate], [0, 1, 1]])
    shifted_arr = np.array([cv2.warpAffine(img, M, (cols, rows)) for img in vid_arr])

    return shifted_arr

def vshift_vidArr(vid_arr,rate=0.1,show=False):
    if IS_RANDOM: rate = random.gauss(0,0.1)
    rows,cols = vid_arr.shape[1:3]

    M = np.float32([[1,0,1],[0,1,rows*rate]])
    shifted_arr = np.array([cv2.warpAffine(img, M, (cols, rows)) for img in vid_arr])
    if show: show_vid(shifted_arr)
    return shifted_arr  
    
def crop_vidArr(vid_arr, rate=0.1,show=False):
    if IS_RANDOM: 
        rate_x = random.uniform(0,0.2)
        rate_y = random.uniform(0,0.2)
    
    rows,cols = vid_arr.shape[1:3]
    print("crop rate: ",rate)

    # should not be zero
    crop_x = int(cols * rate_x / 2 + 1)
    crop_y = int(rows * rate_y / 2 +1)
    print("crop length",crop_x,crop_y)

    cropped_arr = vid_arr[:,crop_y:-crop_y,crop_x:-crop_x,:]

    crop_rows,crop_cols = cropped_arr.shape[1:3]
    if show: show_vid(cropped_arr)

    ## check crop assert
    try:
        assert crop_rows == rows - (crop_y*2)
        assert crop_cols == cols - (crop_x*2)
    except Exception as e:
        print(e)
        print(traceback.format_exc())

    print("cropped shape:",cropped_arr.shape)
    return cropped_arr

def aug_vid_randomly(vid_arr):

    a = [crop_vidArr,stretch_vidArr,hshift_vidArr,vshift_vidArr,rotate_vidArr,flip_vidArr]
    random.shuffle(a); print(a)
    #a = [rotate_vidArr]
    #a = [crop_vidArr]
    st_time = time.time()
    for i in a:
        if random.randint(0,10)>=0:
            #print(i)
            vid_arr = i(vid_arr) 
    print(time.time()-st_time)

    # image cols & rows length should be even
    # rows,cols = vid_arr.shape[1:3]
    # vid_arr  = vid_arr[:,(rows %2):,(cols%2):,:] 
    
    return vid_arr

def is_diff(a,b):
    #print(a[0].pose_landmarks)
    aa,bb=0,0
    if a[0].pose_landmarks:
        print(dir(a[0].pose_landmarks.landmark[0]))
        print(type(a[0].pose_landmarks.landmark[0]))
        print(a[0].pose_landmarks == b[0].pose_landmarks)
    if b[0].pose_landmarks:
        print(b[0].pose_landmarks.landmark[0])
    
    #print(b[0].pose_landmarks)

if __name__ =="__main__":
    #show_vid(aug_vid_randomly(vid_arr = load_vidArr(TEST_VID_PATH)))
    vid_arr = load_vidArr(TEST_VID_PATH)
    show_vid(vid_arr)
    stretch_vidArr(vid_arr)
