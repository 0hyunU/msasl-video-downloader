import copy
import glob
import os
import cv2
from cv2 import trace
import numpy as np
import mediapipe as mp
import traceback
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import vid.vidaug as v
import logging as log

log.basicConfig(filename=f"/Users/0hyun/Desktop/vid/all/{int(time.time())}.log", level=log.DEBUG)

Iter_num = 9
ANNOTE = False
NO_FACE_DOT = False
DRAW_KEYPOINT = True
FACE_FEATUERS = 468
POSE_FEATURES = 33
HAND_FEATURES = 21
VERBOSE = True


def plot_3d(arr, save_name = None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for i in range(len(arr)):
    for j in range(arr.shape[2]):
        arr[:,:,j] = MinMaxScaler().fit_transform(arr[:,:,j].reshape((-1,1))).reshape(arr[:,:,j].shape)

    sc = ax.scatter(arr[0,:, 0], arr[0,:, 1], arr[0,:, 2])
    print("x:", arr[:,:,0].max(), arr[:,:,0].min())
    print("y:", arr[:,:,1].max(), arr[:,:,1].min())
    print("z:", arr[:,:,2].max(), arr[:,:,2].min())
    ax.view_init(0,0) #z,y direction
    # ax.view_init(90,90) #x,y direction
    ax.set_xlim([arr[:,:,0].min(), arr[:,:,0].max()])
    ax.set_ylim([arr[:,:,1].min(), arr[:,:,1].max()])
    ax.set_zlim([arr[:,:,2].min(), arr[:,:,2].max()])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.invert_yaxis()

    for n,i in enumerate(arr):
        
        sc._offsets3d = (i[:,0], i[:,1], i[:,2])

        for n_,coordi in enumerate(i):
            if n_ == 0: ax.text(*(coordi),"Nose",None) 

        fig.canvas.draw_idle()
        plt.pause(0.1)
        if save_name is not None: plt.savefig(f"../vid/all/{save_name}3d({n}).png")
    plt.close()

def plot_2D_keypoint_every_move(keypoints, save_name=None, shoulder_scale=False,nose_scale=False):
    print("minmax:",keypoints.min(), keypoints.max())
    log.info(f"minmax: {keypoints.min()}, {keypoints.max()}")
    nose = keypoints[:,468:469,:]
    middle_of_the_shoulder = keypoints[:,479:481,:].mean(axis=1)[:,np.newaxis,:]
    print(nose.shape,middle_of_the_shoulder.shape)
    
    if nose_scale: keypoints = keypoints - nose
    if shoulder_scale: keypoints = keypoints - middle_of_the_shoulder
    # if scale:    
    #     for n,i in enumerate(keypoints):
    #         for j in range(keypoints.shape[2]):
    #             i[:,j] = MinMaxScaler().fit_transform(i[:,j].reshape((-1,1))).reshape(i[:,j].shape)
    try:
        # if NO_FACE_DOT: keypoints = keypoints[:,FACE_FEATUERS:,:]
        figure ,ax = plt.subplots(figsize=(8,6))
        axsca = ax.scatter(keypoints[0,:,0],keypoints[0,:,1])
        ax.set_xlim([keypoints[:,:,0].min(), keypoints[:,:,0].max()])
        ax.set_ylim([keypoints[:,:,1].min(), keypoints[:,:,1].max()])
        ax.invert_yaxis()


        for n,i in enumerate(keypoints):

            #print(i[:,:1].shape)
            axsca.set_offsets(i[:,:2])
            
            if(n%2):
                for nn, txt in enumerate(i[:,:2]):
                    if (nn==479) or (nn==480):
                        ax.annotate(nn, (i[nn,:2]))
            figure.canvas.draw_idle()
            plt.pause(0.001)
            if save_name is not None: 
                fig_save_names = f"../vid/all/{save_name}shoulder_scale_({shoulder_scale})nose_({nose_scale}) (*).png"
                plt.savefig(f"../vid/all/{save_name}shoulder_scale_({shoulder_scale})nose_({nose_scale}) ({n}).png")
        plt.close()
    except IndexError as e:
        print(e)
        return None

    # print(len(glob.glob(fig_save_names)))
    # print(fig_save_names)
    if save_name is not None: 
        png_to_gif(fig_save_names)

def png_to_gif(png_names):
    save_gif = png_names.split("(*).png")[0] + f"{str(Iter_num)}" + ".gif"
    cmd = f'convert "{png_names}" "{save_gif}"'
    log.info(f"convert command: {cmd}")
    os.system(cmd)

DRAW_VID = True

def draw_landmark(image,mp_drawing, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))

def keypoint_from_vid(vid_arr):
    mp_holistic = mp.solutions.holistic
    vid_arr = copy.deepcopy(vid_arr)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        results = [holistic.process(img) for img in vid_arr]
        
        assert len(results) == len(vid_arr)
        
        if DRAW_VID:
            show_vid(vid_arr, vid_title="Image without keypoint")
            mp_drawing = mp.solutions.drawing_utils
            
            try:
                for n,(i,r) in enumerate(zip(vid_arr,results)):
                    i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
                    draw_landmark(i,mp_drawing,r)
                    vid_arr[n]=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(e)
                print(traceback.format_exc())

            show_vid(vid_arr, vid_title="Image with keypoint")
            out = cv2.VideoWriter('./video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, vid_arr.shape[2:4])
            for i in vid_arr:
                out.write(i)
            out.release()
    return results


def show_vid(vid_arr, vid_title="Image") -> None:

    for i in vid_arr:
        cv2.imshow(f"{vid_title}",cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)
    cv2.destroyAllWindows()

def concat_keypoint_data(landmark: object, base:np.ndarray, hands_mean = 0) -> np.ndarray:
        
    if landmark:
        tmp = np.array([[i.x,i.y,i.z] for i in landmark.landmark])
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
        if base[-1].sum() == 0:
            tmp = np.zeros(base[-1].shape)
            tmp[:,:] = hands_mean
            return np.vstack([base,np.expand_dims(tmp,0)])

        return np.vstack([base,np.expand_dims(base[-1],0)])

def kpoint2arr(keypoint_res):
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
                pose = concat_keypoint_data(i.pose_landmarks, pose)
                
                # data interpolation
                lh_indices =[15,17,19,21]
                lh_avr = pose[-1,lh_indices,:].mean()
                rh_indices = [16,18,20,22]
                rh_avr = pose[-1,rh_indices,:].mean()

                face = concat_keypoint_data(i.face_landmarks, face)
                lh = concat_keypoint_data(i.left_hand_landmarks, lh, lh_avr)
                rh = concat_keypoint_data(i.right_hand_landmarks, rh, rh_avr)
        
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
            keypoint_concat = np.concatenate((face,pose[:,:25,:],lh,rh),axis=1)
            if VERBOSE: print("keypoint before scale:", keypoint_concat.max(), keypoint_concat.min())   

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return np.zeros((1,))
        
        return keypoint_concat

def read_vid(vid_path = "../vid/all/all.mp4"):
    cap = cv2.VideoCapture(vid_path)
    li = []
    while True:
        a,b = cap.read()
        if a != True: break
        li.append(cv2.cvtColor(b,cv2.COLOR_BGR2RGB)) 
        
    li = np.array(li)

    return li

def main():
    for k in range(10):
        global Iter_num
        Iter_num = k
        
        for i in glob.glob("/Users/0hyun/Desktop/vid/all/*.png"):
            import os
            os.remove(i)
        log.info("remained png removed")
        vid_path = "/Users/0hyun/Desktop/vid/all/all_2.mp4"

        try:
            va = v.VidAug(vid_path)
            print(va.vid_arr.shape)
            # show_vid(va.vid_arr)
            aug_vid =  va.get_randomly_aug_vid()
            pure_vid = va.vid_arr

            # show_vid(keypoint)
            keypoint_res, keypoint_res1 = gen_key_arr.KeyArrGen(aug_vid).get_keyarr()
            keypoint1_res, keypoint1_res1=gen_key_arr.KeyArrGen(pure_vid).get_keyarr()

            # result = keypoint_from_vid(li)
            # keypoint = kpoint2arr(result)


            plot_2D_keypoint_every_move(keypoint_res, "augvid")
            plot_2D_keypoint_every_move(keypoint_res1, "augvid_scale1")

            plot_2D_keypoint_every_move(keypoint_res, "augvid", nose_scale=True)
            plot_2D_keypoint_every_move(keypoint_res1, "augvid_scale1", nose_scale=True)

            plot_2D_keypoint_every_move(keypoint_res, "augvid", True)
            plot_2D_keypoint_every_move(keypoint_res1, "augvid_scale1", True)

            if k>0: 
                continue
            plot_2D_keypoint_every_move(keypoint1_res, 'purevid')
            plot_2D_keypoint_every_move(keypoint1_res1, 'purevid_scale1')
        
            plot_2D_keypoint_every_move(keypoint1_res, 'purevid', nose_scale=True)
            plot_2D_keypoint_every_move(keypoint1_res1, 'purevid_scale1', nose_scale=True)

            plot_2D_keypoint_every_move(keypoint1_res, 'purevid', True)
            plot_2D_keypoint_every_move(keypoint1_res1, 'purevid_scale1', True)

            if k==0: break

        except Exception:
            log.error(traceback.format_exc())
            continue
    
def check_streching_effect():

    vid_path = "/Users/0hyun/Desktop/vid/like/like.mp4"
    vid_arr = v.VidAug(vid_path).vid2arr()
    stretch_vidArr =v.VidAug("").stretch_vidArr(vid_arr)
    keypoint_from_vid(vid_arr)
    keypoint_from_vid(stretch_vidArr)


    a,b = gen_key_arr.KeyArrGen(vid_arr).get_keyarr()
    plot_2D_keypoint_every_move(a,'purevid')
    plot_2D_keypoint_every_move(b,'purevid_scale1')

    a,b = gen_key_arr.KeyArrGen(stretch_vidArr).get_keyarr()
    plot_2D_keypoint_every_move(a,'stretchvid')
    plot_2D_keypoint_every_move(b,'stretchvid_scale1')


def check_zvalues_distribution():
    vid_path = "/Users/0hyun/Desktop/vid/all/all_2.mp4"
    vid_arr = v.VidAug(vid_path).vid2arr()
    show_vid(vid_arr)
    keypoint_res, keypoint_res1 = gen_key_arr.KeyArrGen(vid_arr).get_keyarr()
    print(keypoint_res.shape, keypoint_res1.shape)
    plot_3d(keypoint_res[:,FACE_FEATUERS:,:],"test")
    

if __name__ == "__main__":
    st_time = time.time()


    check_streching_effect()
    
    #check_zvalues_distribution()


    log.info(f"{time.time() - st_time}")
