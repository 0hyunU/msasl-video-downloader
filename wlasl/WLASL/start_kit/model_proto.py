# TensorFlow and tf.keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import *
from load_data import try_PCA, load_data
from sklearn.preprocessing import LabelBinarizer

def train():
    encoder = LabelBinarizer()

    x,y = try_PCA(0.9)
    # x,y = load_data()

    transfomed_label = encoder.fit_transform(y)
    y = transfomed_label.argmax(1)
    (train_x, test_x, train_y, test_y) = train_test_split(
	                                                    x,y, test_size=0.2, random_state=42)
    print(x.shape)
    model = dnn_model_after_pca()
    #tf.config.list_physical_devices('GPU')
    print(tf.test.is_gpu_available())
    print(model.summary())
    
    result = model.fit(train_x, train_y, validation_split=0.2,epochs=100 )
    return result
    



if __name__ == "__main__":
    
    # x,y = load_data()
    # print(len(y))
    result = train()
    draw_model_fitting_result(result,True, "dnn_model_save.png")
    #print(encoder.classes_)
    # a = tf.keras.utils.to_categorical(y, num_classes=30)
    # print(a)    
