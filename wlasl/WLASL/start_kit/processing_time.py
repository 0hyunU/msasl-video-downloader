import json
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import MAX_LEN, draw_plot
from secon_proto_model import load_test_data, load_train_data
from vid.aug_vid import load_data_json
from vid.draw_plot import plot_2D_keypoint_every_move

def processing_time():
    
    
    key = 0
    value = 0
    frame_list = []
    processing_log_files = ["processing_time",'more_vid_processing_time','own_vid_processing_time', 'processing_time_own_vid']
    for j in processing_log_files:
        content = json.load(open(f"./{j}.json"))
        print(len(content)," ", end=" ")
        
        for i in content:
            k,v = tuple(i.items())[0]
            key += int(k)
            value += v
            frame_list.append(int(k))
    print()
    print("entire frame count:",key)
    print("entire processing time:",value)
    print("processing time by frame: ",value/key)

    frame_list = []
    for i in glob.glob("./**/**/*.pickle"):
        frame_list.append(len(pickle.load(open(i,'rb'))))
        if len(pickle.load(open(i,'rb'))) > 100:
            pass
            # print(i)
    print(list(filter(lambda x: x>100, frame_list)))
    c,e,b = plt.hist(frame_list)
    # plt.bar_label(b)
    for pp in b:
        x = (pp._x0) + (pp._x1-pp._x0)*0.25
        y = pp._y1 + 1
        plt.text(x, y, int(pp._y1))

    # mean value
    plt.axvline(np.mean(frame_list), color='k', linestyle='dashed', linewidth=1)
    plt.text(np.mean(frame_list)*1.1, plt.ylim()[1]*0.8, 'Mean: {:.2f}'.format(np.mean(frame_list)))
    # median value
    plt.axvline(np.median(frame_list), color='k', linestyle='dashed', linewidth=1)
    plt.text(np.median(frame_list)*1.1, plt.ylim()[1]*0.6, 'Median: {:.2f}'.format(np.median(frame_list)))
    plt.show()

def compress_seq(arr):

    MAX_SEQ = 50
    if len(arr)<=MAX_SEQ: return arr

    window = i = ((len(arr) // MAX_SEQ) + 1)

    res = np.expand_dims(arr[:window].mean(axis=0),0)
    while i+window < len(arr):
        res = np.vstack([res,np.expand_dims(arr[i:window+i].mean(axis=0), 0) ])
        i += window

    return res


if __name__ == "__main__":
    processing_time()

    
    # x,y = load_train_data()
    # a = 0
    # for i,j in zip(x,y):
    #     print(len(i), compress_seq(i).shape, j)
    #     plot_2D_keypoint_every_move(compress_seq(i))
    
    # print(a)