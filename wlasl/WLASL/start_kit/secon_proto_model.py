from sklearn.preprocessing import LabelBinarizer
from model import cnn_model, dnn_flatten, lstm_model_proto, SEQ_LEN
from vid.aug_vid import load_data_json
import pickle
from sklearn.utils import shuffle
import numpy as np
import glob
import os


PAD_LEN = SEQ_LEN

def compress_seq(arr):
    ## method 1 -> using filter
    # i = 1
    # # sequence length - filter_length = 30
    # filter = len(arr) - 30
    # res = np.expand_dims(arr[:filter].mean(axis=0), 0)
    
    # while i+filter< len(arr):
    #     res = np.vstack([res, np.expand_dims(arr[i:filter+i].mean(axis=0),0)])
    #     i+=1

    
    # res = np.convolve(arr, np.ones(2), 'valid') / 2
    if len(arr)<=30: return arr
    elif len(arr) <=60:
        window = i = 2
    elif len(arr) <= 90:
        window = i = 3
    elif len(arr) <= 120:
        window = i = 4
    elif len(arr) <= 150:
        window = i = 5
    elif len(arr) <= 180:
        window = i = 6
    elif len(arr) <= 210:
        window = i = 7
    res = np.expand_dims(arr[:window].mean(axis=0),0)
    while i+window < len(arr):
        res = np.vstack([res,np.expand_dims(arr[i:window+i].mean(axis=0), 0) ])
        i += window

    return res

def load_train_data():
    x,y = [],[]
    for i in glob.glob("./train/**/*.pickle"):
        arr = pickle.load(open(i, 'rb'))
        #x.append(padding_seq(arr))
        x.append(arr)
        y.append(i.split(os.sep)[-2])

    return np.array(x, dtype="object"), np.array(y, dtype="object")

def load_test_data():
    x,y = [],[]
    for i in glob.glob("./test/**/*.pickle"):
        arr = pickle.load(open(i, 'rb'))
        #x.append(padding_seq(arr))
        x.append(arr)
        y.append(i.split(os.sep)[-2])

    return np.array(x, dtype="object"), np.array(y, dtype="object")

def padding_seq(data: np.array):
    l,f,d = data.shape
    try:
        if PAD_LEN > l:
            pad_len = PAD_LEN - l
            zero_pad = np.zeros((pad_len, f, d))
            padded_array = np.vstack([zero_pad,data])
            return padded_array
        else:
            start_p = l-PAD_LEN
            return data[start_p:]
    except ValueError as e:
        print(data.shape)

if __name__ == "__main__":
    x,y = load_train_data()
    x_val,y_val = load_test_data()
    
    #compression & padding
    x = [compress_seq(i) for i in x]
    x = np.array([padding_seq(i)for i in x])
    x_val = [compress_seq(i) for i in x_val]
    x_val = np.array([padding_seq(i) for i in x_val])
    print(x.shape, x_val.shape)

    y_labeling = LabelBinarizer().fit(y)
    y = y_labeling.transform(y).argmax(1)
    y_val = y_labeling.transform(y_val).argmax(1)
    # print(encoder)
    
    x,y = shuffle(x,y)
    x_val,y_val = shuffle(x_val,y_val)
    
    print(set(y_val))
    model = lstm_model_proto(x,y)
    model.summary()
    x = x.reshape((len(y),PAD_LEN,-1))
    x_val = x_val.reshape((len(y_val),PAD_LEN,-1))
    model.fit(x,y, epochs=300, validation_data = (x_val,y_val))
