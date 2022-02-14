import matplotlib.pyplot as plt

def plot_2D_keypoint_every_move(keypoints):

    figure ,ax = plt.subplots(figsize=(8,6))
    axsca = ax.scatter(keypoints[0,:,0],keypoints[0,:,1])
    ax.set_xlim([keypoints[:,:,0].min(),keypoints[:,:,0].max()])
    ax.set_ylim([keypoints[:,:,1].min(),keypoints[:,:,1].max()])
    ax.invert_yaxis()

    for i in keypoints:
        #print(i[:,:1].shape)
        axsca.set_offsets(i[:,:2])
        figure.canvas.draw_idle()
        plt.pause(0.001)
    plt.close()