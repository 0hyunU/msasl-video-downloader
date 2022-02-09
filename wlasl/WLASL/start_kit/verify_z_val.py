import os
import cv2
import mediapipe as mp
import time 
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

SAMPLE_VIDEO = "C:/Users/user/temp/wlasl/WLASL/start_kit/train_test/all/01986.mp4"

def init_dict(point_dict,landmark_type,gloss):
    point_dict["type"] = landmark_type
    point_dict['gloss'] = gloss

    return point_dict

def main():
    gloss_list = os.listdir('videos')
    cap = cv2.VideoCapture(os.path.join(os.getcwd(), "videos", 'action/00857.mp4'))
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    pTime = 0
    frame = 0
    landmark_dict = init_dict(dict(),'pose','animal')
    #print(landmark_dict)
    X = np.random.rand(33, 3)
    
    plt.ion()

    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax.set_xlim([-1, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(0, -90)
    fig.show()

    while True:
        try:
            success, img = cap.read()
            if not success: break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            mpDraw = mp.solutions.drawing_utils
            frame +=1
            
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                pose_landmark = list()
                x = y = z = list()
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    if (id> 10) and (id < 23):
                        x.append(lm.x)
                        y.append(lm.y)
                        z.append(lm.z)
                X = np.c_[x,y,z]

            landmark_dict[frame] = pose_landmark

        except Exception as e:
            print(e)
            pass

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        plt.pause(0.1)
        sc._offsets3d = (X[:, 0], X[:, 1], X[:, 2])

        plt.draw()
        
        cv2.waitKey(1)

if __name__ == "__main__":
    main()