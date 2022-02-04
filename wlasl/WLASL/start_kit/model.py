from gc import callbacks
from turtle import left
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob

from load_data import try_PCA, load_all_data, load_train_test
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, concatenate, BatchNormalization
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import LSTM


def draw_model_fitting_result(result, save=False, savefig_name='model_fitting_result_save.png'):
    plt.plot(result.history['loss'],label='loss')
    plt.plot(result.history['accuracy'],label='acc')
    plt.plot(result.history['val_accuracy'],label='val_acc')
    plt.plot(result.history['val_loss'],label='val_loss')
    plt.legend(loc='upper right')
    if save: plt.savefig(savefig_name)
    plt.show()

def dnn_model_after_pca():
    x,y = try_PCA(0.9)
    out_node = len(set(y))
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=x.shape[1:]),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(out_node, activation = 'softmax')
    ])

    model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
    return model

def dnn_flatten():
    x,y = load_all_data()
    out_node = len(set(y))
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=x.shape[1:]),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(out_node, activation = 'softmax')
    ])

    model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
    return model

def cnn_model():
    x,y = load_all_data()
    out_node = len(set(y))
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=x.shape[1:]))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(out_node, activation='softmax'))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

def lstm_model_proto():
    x,y = load_all_data()
    x = x.reshape((len(x),150,-1))

    output_node = len(set(y))
    model = models.Sequential()
    model.add(LSTM(64, return_sequences=True,input_shape=x.shape[1:]))
    model.add(LSTM(128, return_sequences=True))
    #model.add(layers.Dropout(0.5))
    model.add(LSTM(64, return_sequences=False))
    #model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_node, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model
import os
def mkdir_saved_model():
    base_path = "./saved_model"
    dir_count = len(os.listdir(base_path))
    mkdir_path = os.path.join(base_path, str(dir_count + 1))
    if not os.path.isdir(mkdir_path):
        os.mkdir(mkdir_path)
        print('mkdir complete')
    
    else: print("dir already exist")

    return mkdir_path
def ret_callback():
    
    base_path = mkdir_saved_model()
    modelpath=os.path.join(base_path,"{epoch:02d}-{val_loss:.4f}-{val_accuracy:.2f}.h5")
    print(modelpath)
    callbacks_list = [ \
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100),
    tf.keras.callbacks.ModelCheckpoint(filepath = modelpath, 
                                       monitor='val_loss', 
                                       verbose=1, save_best_only=True)]
    
    return callbacks_list

def train_cnn_model():
        
    cnn_model().summary()
    encoder = LabelBinarizer()
    x,y = load_all_data()

    transfomed_label = encoder.fit_transform(y)
    y = transfomed_label.argmax(1)
    (train_x, test_x, train_y, test_y) = train_test_split(
                                                        x,y, test_size=0.2, random_state=42)
    #print(x.shape)
    model = cnn_model()
    #tf.config.list_physical_devices('GPU')
    print(tf.test.is_gpu_available())
    print(model.summary())

    result = model.fit(train_x, train_y, validation_split=0.2,epochs=30)
    draw_model_fitting_result(result)

def train_lstm():
    model = lstm_model_proto()
    (train_x, test_x, train_y, test_y) = load_train_test()
    train_x = train_x.reshape((len(train_y),150,-1))
    test_x = test_x.reshape((len(test_y),150,-1))
    encoder = LabelBinarizer().fit(train_y)
    train_y = encoder.transform(train_y).argmax(1)
    test_y = encoder.transform(test_y).argmax(1)
    #(train_x, test_x, train_y, test_y) = train_test_split(x,y, test_size=0.2, random_state=42)
    
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    print(len(set(test_y)), len(set(train_y)))
    #result = model.fit(train_x, train_y, validation_split=0.2,
    #                  epochs=500,callbacks=ret_callback())
    tes_x, val_x, tes_y, val_y = train_test_split(test_x,test_y, test_size=0.5)
    
    import time
    st = time.time()
    result = model.fit(train_x, train_y, validation_data=(val_x,val_y),
                       epochs=500,callbacks=ret_callback())
    saved_model_path = len(os.listdir("./saved_model"))
    if glob.glob(f"./saved_model/{saved_model_path}/*.h5"):
        a = sorted(glob.glob(f"./saved_model/{saved_model_path}/*.h5"),key=lambda x: int(x[16:].split("-")[0]), reverse=True)[0]
        print("best saved model: ",a)
        saved_model = tf.keras.models.load_model(a)
        print(saved_model.evaluate(tes_x,tes_y, return_dict=True))
    print("training time: ", time.time() - st)
    draw_model_fitting_result(result)

def divide_segment(X, len_y):
    hands = X[:,-100:,:42,:].reshape((len_y,100,-1))
    face = X[:,-100:,42:510,:].reshape((len_y,100,-1))
    pose = X[:,-100:,-33:,:].reshape((len_y,100,-1))
    return hands, face, pose

def multimodal_data_load():
    (train_x, test_x, train_y, test_y) = load_train_test()
    hands = train_x[:,:,:42,:].reshape((len(train_y),150,-1))
    face = train_x[:,:,42:510,:].reshape((len(train_y),150,-1))
    pose = train_x[:,:,-33:,:].reshape((len(train_y),150,-1))
    
    return hands, face, pose, train_y

def lstm_multimodal(train_x,train_y):
    input1 = layers.Input(shape=train_x[0].shape[1:])
    input2 = layers.Input(shape=train_x[1].shape[1:])
    input3 = layers.Input(shape=train_x[2].shape[1:])

    lstm1  = LSTM(32,return_sequences=False)(input1)
    lstm2  = LSTM(32,return_sequences=False)(input2)
    lstm3  = LSTM(32,return_sequences=False)(input3)

    # lstm1  = LSTM(32,return_sequences=False)(input1)
    # lstm2  = LSTM(32,return_sequences=False)(input2)
    # lstm3  = LSTM(32,return_sequences=False)(input3)
    ## outputs
    # output1  = Dense(32, activation="relu", name='out1')(lstm1)
    # output2  = Dense(32, activation="relu", name='out2')(lstm2)
    # output3  = Dense(32, activation="relu", name='out3')(lstm3)
    concat = concatenate([lstm1, lstm2, lstm3])
    #output = BatchNormalization()(concat)
    # output = Dense(256, activation="relu")(concat)
    # output = layers.Dropout(0.5)(output)
    output = Dense(64, activation="relu")(concat)
    output = layers.Dropout(0.3)(output)
    #output = Dense(64, activation='relu')(output)
    output = Dense(len(set(train_y)), activation = 'softmax')(output)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=output)
    opt = tf.keras.optimizers.Adam( learning_rate=0.0003,clipnorm=1.0)
    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

    model.summary()

    return model
def train_multimodal():
    
    (train_x, test_x, train_y, test_y) = load_train_test()
    
    encoder = LabelBinarizer().fit(train_y)
    train_y = encoder.transform(train_y).argmax(1)
    test_y = encoder.transform(test_y).argmax(1)

    train_x = divide_segment(train_x, len(train_y))
    test_x = divide_segment(test_x, len(test_y))

    model = lstm_multimodal(train_x, train_y)
    
    #tes_x, val_x, tes_y, val_y = train_test_split(test_x,test_y, test_size=0.5)

    #print(len(tes_x))
    history = model.fit([train_x],train_y,validation_data=(test_x,test_y), epochs = 100,callbacks=ret_callback())
    draw_model_fitting_result(history)
train_multimodal()
#train_lstm()
#train_cnn_model()
# x,y = load_all_data()
# print(x[:,120:,:,:].shape)