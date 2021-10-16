import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras import backend as K
import pickle
import numpy as np

Xw = np.array(pickle.load(open('Xw', 'rb')))
yw = np.array(pickle.load(open('yw', 'rb')))
yw = (yw - yw.mean()) / yw.std()
Xb = np.array(pickle.load(open('Xb', 'rb')))
yb = np.array(pickle.load(open('yb', 'rb')))
yb = (yb - yb.mean()) / yb.std()

input_shape = (8, 8, 12)
K.set_image_data_format('channels_last')

def get_model(X, y):
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=6, strides=1, input_shape=input_shape, padding='same', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(filters=64, kernel_size=4, strides=1, input_shape=input_shape, padding='same', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))

    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['mse'])
    model.fit(X, y, batch_size=32, epochs=10, verbose=1)
    return model

get_model(Xw, yw).save('white_model')
get_model(Xb, yb).save('black_model')
