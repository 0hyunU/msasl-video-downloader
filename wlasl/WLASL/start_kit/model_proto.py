# TensorFlow and tf.keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

from load_data import try_PCA, load_data

def dnn_model_after_pca():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(19, activation='relu', input_shape = (19,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(30, activation = 'softmax')
    ])

    model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
    return model

def train():
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()

    x,y = try_PCA(0.9)

    transfomed_label = encoder.fit_transform(y)
    y = transfomed_label.argmax(1)
    (train_x, test_x, train_y, test_y) = train_test_split(
	                                                    x,y, test_size=0.2, random_state=42)
    print(x.shape)
    model = dnn_model_after_pca()
    #tf.config.list_physical_devices('GPU')
    print(tf.test.is_gpu_available())
    print(model.summary())
    
    history = model.fit(train_x, train_y, epochs=100)
    


if __name__ == "__main__":
    
    # x,y = load_data()
    # print(len(y))
    train()

    #print(encoder.classes_)
    # a = tf.keras.utils.to_categorical(y, num_classes=30)
    # print(a)    
