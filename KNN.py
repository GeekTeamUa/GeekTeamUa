from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as K
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score


from keras.datasets import mnist
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
input_dim = 784 #28*28
X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 10)
Y_train = np_utils.to_categorical(y_train, 10)



model = KNeighborsClassifier(n_neighbors=3)


X = X_train
y = y_train
model.fit(X, y)
train_outputs = model.predict(X)
test_outputs = model.predict(X_test)
train_score_summary = metrics.accuracy_score(y, train_outputs)
test_score_summary = metrics.accuracy_score(y_test, test_outputs)


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])








