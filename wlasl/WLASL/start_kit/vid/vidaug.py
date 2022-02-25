import traceback
import numpy as np
import cv2
import random
import pickle
import imutils
import os
import time

VERBOSE_SHAPE_DIMENSION = False
VERBOSE_AUG_CONTENTS = False

class VidAug():

    def __init__(self, vid_path, save =False) -> None:
        self.vid_path = vid_path
        self.save = save
        self.vid_arr = self.vid2arr()
        self.aug_time = 0

    def vid2arr(self):
        cap = cv2.VideoCapture(self.vid_path)
        frame_list = list()
        while cap.isOpened():
            ret, frame = cap.read()

            # frame ended
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(image)

        frame_list = np.array(frame_list)

        if self.save:
            save_path = self.vid_path + "Arr.pickle"
            pickle.dump(frame_list,open(save_path,'wb'))

        return frame_list
    def show_vid(self) -> None:
    
        if self.vid_arr is None:
            vid_arr = self.vid2arr(self.vid_path)
        else:
            vid_arr = self.vid_arr

        for i in vid_arr:
            cv2.imshow("Image",cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)
    
    def load_vidArr(self):
        arr_path = self.vid_path + "arr.pickle"
        try:
            if not os.path.isfile(arr_path): vid_arr = self.vid2arr(self.vid_path)
            else : vid_arr = pickle.load(open(arr_path,'rb'))
        except Exception as e:
            print(e)
            print(self.vid_path, arr_path)

        return vid_arr


    def crop_vidArr(self,vid_arr):
        rate_x = random.uniform(0,0.1)
        rate_y = random.uniform(0,0.1)
        
        rows,cols = vid_arr.shape[1:3]

        # should not be zero
        crop_x = int(cols * rate_x / 2 + 1)
        crop_y = int(rows * rate_y / 2 +1)

        cropped_arr = vid_arr[:,crop_y:-crop_y,crop_x:-crop_x,:]
        crop_rows,crop_cols = cropped_arr.shape[1:3]

        ## check crop assert
        try:
            assert crop_rows == rows - (crop_y*2)
            assert crop_cols == cols - (crop_x*2)
        except Exception as e:
            print(e)
            print(traceback.format_exc())

        if VERBOSE_SHAPE_DIMENSION:
            print("origin shape", vid_arr.shape)
            print("cropped shape:",cropped_arr.shape)
        return cropped_arr   
    
    def flip_vidArr(self,vid_arr):

        return np.flip(vid_arr,2)

    def rotate_vidArr(self,vid_arr):
    
        angle = random.uniform(-40,40)
        # rotate type
        if random.randint(0,1) % 2: 
            vid_arr_rot = np.array([imutils.rotate_bound(img,angle) for img in vid_arr])
        else:
            vid_arr_rot = np.array([imutils.rotate(image, angle) for image in vid_arr])

        return vid_arr_rot
    
    def hshift_vidArr(self,vid_arr):
        rate = random.gauss(0,0.2)
        rows,cols = vid_arr.shape[1:3]

        M = np.float32([[1, 0, cols * rate], [0, 1, 1]])
        shifted_arr = np.array([cv2.warpAffine(img, M, (cols, rows)) for img in vid_arr])

        return shifted_arr

    def vshift_vidArr(self,vid_arr):
        rate = random.uniform(-0.3,0.3)
        rows,cols = vid_arr.shape[1:3]

        M = np.float32([[1,0,1],[0,1,rows*rate]])
        return np.array([cv2.warpAffine(img, M, (cols, rows)) for img in vid_arr])
  

    def stretch_vidArr(self,vid_arr):
        cols,rows = vid_arr.shape[1:3]
        rate_y = 1 + random.uniform(-0.5,0.5)
        rate_x = 1 + random.uniform(-0.5,0.5)
        
        if VERBOSE_AUG_CONTENTS: print("x,y rates",rate_x,rate_y)
        
        return np.array([cv2.resize(img,(int(rows*rate_y),int(cols*rate_x))) for img in vid_arr])
        

    def get_randomly_aug_vid(self):

        aug_func_list = [self.crop_vidArr,self.stretch_vidArr,self.hshift_vidArr,
             self.vshift_vidArr,self.rotate_vidArr,self.flip_vidArr]
        random.shuffle(aug_func_list)
        #aug_func_list = [self.stretch_vidArr]
        if VERBOSE_AUG_CONTENTS: print("shuffled aug func: ",aug_func_list)

        st_time = time.time()
        vid_arr = self.vid_arr
        for i in aug_func_list:
            
            if random.randint(0,10)>=4:
                vid_arr = i(vid_arr)
        self.aug_time = time.time() - st_time
        if VERBOSE_AUG_CONTENTS:
             print("aug processing time: ",self.aug_time)
        
        return vid_arr
    
    
    