#!/usr/bin/env python3
import os
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop

from visualize import AccuracyLossPlot, FirstWeightPlot

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(1, 48, 48)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(40))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop())

data_path = os.path.join('data', 'datasets', 'micro_dataset.pkl')
with open(data_path, 'rb') as f:
    train_set, test_set = pickle.load(f)

X_train, Y_train = train_set
X_test, Y_test = test_set
model.fit(X_train, Y_train, batch_size=32, nb_epoch=20,
          callbacks=[AccuracyLossPlot(), FirstWeightPlot(4, 8)],
          validation_split=0.1, show_accuracy=True)

score = model.evaluate(X_test, Y_test, show_accuracy=True)
print('Score: {}'.format(score[0]))
print('Accuracy: {}'.format(score[1]))
