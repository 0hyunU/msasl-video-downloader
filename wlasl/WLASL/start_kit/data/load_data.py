import os
import glob
import pickle
from cv2 import split
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

MAX_LEN = 150
FEATURES = 543
COLOR_DIM = 3
def padding_array(data: np.ndarray):
    if MAX_LEN > len(data):
        pad_len = MAX_LEN - len(data)
        zero_pad = np.zeros((pad_len, FEATURES, COLOR_DIM))
        padded_array = np.vstack([zero_pad,data])
        #print(len(data),np.vstack([zero_pad,data]).shape)

        # test 
        assert np.array_equal(data, padded_array[pad_len:])
        #assert data.all(padded_array[pad_len:])
        return padded_array
    
    elif MAX_LEN <= len(data):
        return data[:150]
        


def load_all_data():
    preprocessed_data = list()
    y_val = list()
    a,b = split_train_test()
    for i in glob.glob("./train_test/**/*.pickle",recursive=True):
        
        # load preprocessed data structure
        keypoint= pickle.load(open(i, 'rb'))
        raw_data = keypoint.get_tensor() #get 3D array
        preprocessed_data.append(padding_array(raw_data))
        y_val.append(keypoint.action_class)
    
    assert len(preprocessed_data) == len(y_val)

    return np.array(preprocessed_data), y_val


def load_train_test():
    train, test = split_train_test()
    train_x, train_y , test_x,test_y = list(), list(), list(), list()

    for i in train:
        # load preprocessed data structure
        keypoint= pickle.load(open(i, 'rb'))
        raw_data = keypoint.get_tensor() #get 3D array
        train_x.append(padding_array(raw_data))
        train_y.append(keypoint.action_class)

    for i in test:
        # load preprocessed data structure
        keypoint= pickle.load(open(i, 'rb'))
        raw_data = keypoint.get_tensor() #get 3D array
        test_x.append(padding_array(raw_data))
        test_y.append(keypoint.action_class)

    return np.array(train_x), np.array(test_x), train_y ,test_y


def split_train_test():
    preprocessed_data = list()
    y_val = list()
    test,train = list(),list()
    for i in glob.glob("./train_test/**/*.mp4",recursive=True):
        rand_split = random.randint(1,10)
        if rand_split >= 9:
            [test.append(j) for j in glob.glob(i+"*.pickle")]
        else:
            [train.append(j) for j in glob.glob(i+"*.pickle")]

    # preprocessed_data.append(i)
    # random.shuffle(preprocessed_data)
    # print(preprocessed_data)
    # split_point = int(len(preprocessed_data)*0.8)
    #train, test = preprocessed_data[:split_point], preprocessed_data[split_point:]
    return train, test
    

def try_PCA(n_components=3, saved_pca="pca_obj.pickle"):
    x,y = load_all_data()
    # for i,v in enumerate(x):
    #     x[i] = x[i].flatten()
    
    x = x.reshape((x.shape[0],-1))
    
    pca = PCA(n_components=n_components) # 주성분을 몇개로 할지 결정
    # printcipalComponents = pca.fit_transform(x)
    saved_pca = saved_pca
    if os.path.isfile(saved_pca):
        printcipalComponents = pickle.load(open(saved_pca,'rb'))
        print(printcipalComponents.shape)
        return printcipalComponents,y
    else: 
        printcipalComponents = pca.fit_transform(x)
        with open(f"{saved_pca}",'wb') as f:
            pickle.dump(printcipalComponents,f)
        
    print(printcipalComponents.shape)
    print(sum(pca.explained_variance_ratio_))
    print(len(pca.explained_variance_ratio_))
    return printcipalComponents, y

def draw_plot(x,y):
    print(x.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in set(y):
        condition = (np.array(y) == i)
        ax.scatter(x[condition][0],x[condition][1],x[condition][2])
    
    plt.title('3D PCA', fontsize=14)
    plt.xlabel('PCA1', fontsize=10)
    plt.ylabel('PCA2', fontsize=10)
    ax.set_zlabel('PCA3',fontsize=10)
    

    plt.savefig("scatter_afterPCA.png")
    plt.show()    

# 주성분으로 이루어진 데이터 프레임 구성

def draw_tsne(x,y):
    from sklearn.manifold import TSNE
    pca = PCA(n_components=0.95)
    print("pca start")
    x = x.reshape((x.shape[0],-1))
    x = pca.fit_transform(x)
    print("pca finish")
    x_embed = TSNE(n_components=2).fit(x)
    print("tnse finish")
   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in set(y):
        condition = (np.array(y) == i)
        ax.scatter(x_embed[condition][0],x_embed[condition][1])
    
    plt.title('TNSE', fontsize=14)

    plt.savefig("scatter_tnse.png")
    plt.show() 


def knn_classifier(train_x, test_x, train_y, test_y):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_x,train_y)
    acc_test = model.score(test_x,test_y)
    print(confusion_matrix(test_y, model.predict(test_x)))
    print("knn classfier acc:",acc_test)
    return acc_test

def svm_classifier(train_x, test_x, train_y, test_y):
    from sklearn.svm import SVC
    clf =SVC(gamma='auto')
    clf.fit(train_x,train_y)
    acc = clf.score(test_x,test_y)
    print("svm_classifier acc",acc)
    return acc

if __name__ == "__main__":
    #x,y = load_data()
    # x,y = try_PCA(0.9)
    # #draw_plot(x,y)
    # #draw_tsne(x,y)
    # y = np.array(y)
    # (train_x, test_x, train_y, test_y) = train_test_split(
	#                                                     x,y, test_size=0.2, random_state=42)
    # knn_acc = knn_classifier(train_x, test_x, train_y, test_y)
    # svm_acc = svm_classifier(train_x, test_x, train_y, test_y)

    # plt.plot(knn_acc,svm_acc)
    # a,b = split_train_test()
    # print(len(a),len(b))
    x,x_,y,y_ = load_train_test()
    print(len(set(y)),len(set(y_)))
    print(y)
    print(y_)

