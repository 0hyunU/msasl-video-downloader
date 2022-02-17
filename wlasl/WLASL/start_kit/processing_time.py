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
        
    content = json.load(open("./processing_time.json"))
    key = 0
    value = 0
    frame_list = []
    for i in content:
        k,v = tuple(i.items())[0]
        key += int(k)
        value += v
        frame_list.append(int(k))

    print("entire frame count",key)
    print("entire processing time:",value)
    print("processing time by frame: ",value/key)

    for i in glob.glob("./train/**/*.pickle"):
        if len(pickle.load(open(i,'rb'))) > 100:
            print(i)
    print(list(filter(lambda x: x>100, frame_list)))
    c,e,b = plt.hist(frame_list)
    # plt.bar_label(b)
    for pp in b:
        x = (pp._x0) + (pp._x1-pp._x0)*0.25
        y = pp._y1 + 1
        plt.text(x, y, int(pp._y1))
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
    x,y = load_train_data()
    a = 0
    for i,j in zip(x,y):
        print(len(i), compress_seq(i).shape, j)
        # if len(i) < 100: continue
        # else: a+=1
        #plot_2D_keypoint_every_move(i)
        plot_2D_keypoint_every_move(compress_seq(i))
    
    print(a)