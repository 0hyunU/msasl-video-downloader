import glob
import os
from pickle import FALSE
import traceback
import cv2
import mediapipe as mp
import time 
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

SAMPLE_VIDEO = "C:/Users/user/temp/wlasl/WLASL/start_kit/train_test/book/07070.mp4"

def init_dict(point_dict,landmark_type,gloss):
    point_dict["type"] = landmark_type
    point_dict['gloss'] = gloss

    return point_dict

def main(vid_path =os.path.join(os.getcwd(), "train_test", 'book/07074.mp4')):
    gloss = vid_path.split(os.sep)[-2]
    vid_num = vid_path.split(os.sep)[-1]
    cap = cv2.VideoCapture(vid_path)
    mpPose = mp.solutions.holistic
    pose = mpPose.Holistic()
    pTime = 0
    frame = 0
    X = np.random.rand(12, 3)
    
    plt.ion()

    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(0, -90) #X,Z direction
    ax.view_init(0,0) #Y,Z direction

    num = 0
    while True:
        try:
            success, img = cap.read()
            if not success: break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            mpDraw = mp.solutions.drawing_utils
            face = False
            hand = False

            frame +=1
            
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                kp = list()
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    if id>25: break
                    kp.append([lm.x,lm.y,lm.z])
                kp = kp[:23]

            if results.left_hand_landmarks:
                hand=True
                for id, lm in enumerate(results.left_hand_landmarks.landmark):
                    kp.append([lm.x,lm.y,lm.z(lm.z - kp[15][2])])
                mpDraw.draw_landmarks(img, results.left_hand_landmarks,mpPose.HAND_CONNECTIONS)

            if results.right_hand_landmarks: 
                hand=True
                for id, lm in enumerate(results.right_hand_landmarks.landmark):
                    kp.append([lm.x,lm.y,lm.z - (lm.z - kp[16][2])])
                mpDraw.draw_landmarks(img, results.right_hand_landmarks, mpPose.HAND_CONNECTIONS)
            if results.face_landmarks:
                face=True
                for id, lm in enumerate(results.face_landmarks.landmark):
                    kp.append([lm.x,lm.y,lm.z-(lm.z - kp[0][2])])
                mpDraw.draw_landmarks(img, results.face_landmarks, mpPose.FACE_CONNECTIONS)
            
            X = np.array(kp)
            # # print(X.shape)
            # landmark_dict[frame] = pose_landmark

        except Exception as e:
            print(traceback.format_exc())
            print(e)
            pass

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        plt.pause(0.001)
        X[:,2] *= -1
        X[:,2] -= X[11:13,2].mean()
         ## annotate
        if num%20 == 0:
            print(X.shape)
            for nn, txt in enumerate(X):
                if nn == 0: ax.text(X[nn,0],X[nn,1],X[nn,2],"NOSE",None)
                if nn == 21: ax.text(X[nn,0],X[nn,1],X[nn,2],"pose_hand",None)
                if hand and (nn == 24): ax.text(X[nn,0],X[nn,1],X[nn,2],"HANDS",None)
                if face and (nn == len(X[:-200])): ax.text(X[nn,0],X[nn,1],X[nn,2],"Face",None)
        sc._offsets3d = (X[:, 0], X[:, 1], X[:, 2])

        plt.draw()
        plt.savefig(f"./3d_scatter/Zscale{gloss}{vid_num}({num}).png")
        num+=1
        
        cv2.waitKey(1)
    plt.close()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    for i in glob.glob("./train_test/**/*.mp4"):
        main(i)