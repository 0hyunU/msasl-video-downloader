import matplotlib.pyplot as plt

def plot_2D_keypoint_every_move(keypoints):

    figure ,ax = plt.subplots(figsize=(8,6))
    axsca = ax.scatter(keypoints[0,:,0],keypoints[0,:,1])
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.invert_yaxis()

    for i in keypoints:
        #print(i[:,:1].shape)
        axsca.set_offsets(i[:,:2])
        figure.canvas.draw_idle()
        plt.pause(0.001)