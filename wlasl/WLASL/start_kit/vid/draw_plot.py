
import matplotlib.pyplot as plt
from .gen_key_arr import FACE_FEATUERS
import numpy as np

PLOT_SHOW = True
def plot_2D_keypoint_every_move(keypoints, with_face = False, save=False,):

    if not with_face: keypoints = keypoints[:,FACE_FEATUERS:,:]
    figure ,ax = plt.subplots(figsize=(8,6))
    axsca = ax.scatter(keypoints[0,:,0],keypoints[0,:,1])
    ax.set_xlim([keypoints[:,:,0].min(),keypoints[:,:,0].max()])
    ax.set_ylim([keypoints[:,:,1].min(),keypoints[:,:,1].max()])
    ax.invert_yaxis()

    for n,k in enumerate(keypoints):
        #print(i[:,:1].shape)
        axsca.set_offsets(k[:,:2])
        figure.canvas.draw_idle()
        if PLOT_SHOW: plt.pause(0.001)
        if save and with_face: plt.savefig(f"./scatter_gif/{save} ({n}).png")
        elif save and not with_face: plt.savefig(f"./scatter_gif/{save}_without_face ({n}).png")
    plt.close()

def plot_3D_keypoint_every_move(keypoints, with_face = False, save= False):
    X = np.random.rand(12, 3)
    fig=plt.figure(figsize=(10,10))
    fig.show()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(0,0) #Y,Z direction
    # ax.view_init(90,90) #Y,X direction

    for n,key in enumerate(keypoints):
        sc._offsets3d = (key[:,0], key[:,1],key[:,2])

        if n == (len(keypoints)//2) or n == (len(keypoints) - 1):
            for nn,x in enumerate(key):
                if nn == 0:  ax.text(x[0],x[1],x[2],"FACE",None)
                if nn == FACE_FEATUERS +0: ax.text(x[0],x[1],x[2],"NOSE",None)
                if nn == FACE_FEATUERS +18: ax.text(x[0],x[1],x[2],"pose_hand",None)
                if nn == FACE_FEATUERS +19: ax.text(x[0],x[1],x[2],"pose_hand",None)
                if nn == FACE_FEATUERS + 26: ax.text(x[0],x[1],x[2],"lh_hand",None)
                if nn == FACE_FEATUERS + 25+21: ax.text(x[0],x[1],x[2],"rh_hand",None)
        plt.pause(0.001)
        plt.draw()

        if save and with_face: plt.savefig(f"./3d_scatter/{save} ({n}).png")
    
    plt.close()
