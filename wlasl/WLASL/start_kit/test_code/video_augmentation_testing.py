import os
import time
import json
from tkinter import image_names
from tkinter.filedialog import test
from typing import Iterable
from cv2 import KeyPoint
import PIL
import numpy as np
import mediapipe as mp
import cv2
import pickle
from keypoint import HolisticKeypoint as H
from PIL import Image

def main():
    test_video_path = "C:/Users/user/Documents/GitHub/msasl-video-downloader/wlasl/WLASL/start_kit/train_test/who/63219.mp4"
    cap = cv2.VideoCapture(test_video_path)
    print(type(cap))
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image)
        im_pil = im_pil.rotate(30)
        print(im_pil.size)
        # cv2.imshow('Raw Webcam Feed', frame)
        # cv2.imshow("flip image", cv2.flip(frame,1)) #좌우반전
        cv2.imshow("rotate image", cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE))
        h,w = frame.shape[:2]
        # M1 = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
        # rotation1 = cv2.wrapAffine(frame, M1, (w,h))

        # cv2.imshow("rotate 45 image",im_pil )


            #time.sleep(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    pass

if __name__ == "__main__":
    main()