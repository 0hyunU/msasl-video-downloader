import os
import time
import json
import glob
import sys
import random
import traceback
from cv2 import resize
from vidaug import vidaug

random.seed(random.random())
from matplotlib.pyplot import show

import numpy as np
import cv2
import pickle

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

if __name__ =="__main__":
    try:
        train_data_dict = load_data()
        for i in train_data_dict:
            print(i['gloss'])
            test_p = i['files'][0]
            print(test_p)
            v = vidaug.VidAug(test_p)
            v.aug_vid_randomly()

    except Exception as e:
        print(e)
        print(traceback.format_exc())    
    pass
