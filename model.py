#!/usr/bin/env python3
import os
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

data_path = os.path.join('data', 'datasets', 'micro_dataset.pkl')
with open(data_path, 'rb') as f:
    train_set, test_set = pickle.load(f)
X_train, y_train = train_set
X_test, y_test = test_set

model = Sequential()
model.add(Convolution2D(8, 3, 3, input_shape=(48, 48)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(X_train, y_train, batch_size=32, nb_epoch=1)
