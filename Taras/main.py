from __future__ import print_function
import keras
import numpy
from xgboost import XGBClassifier
from keras.datasets import mnist

from keras import backend as K
import tensorflow
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score


batch_size  = 1000
num_classes = 10

epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) =  keras.datasets.mnist.load_data() #numpy load
tensorflow.keras.datasets.mnist.load_data()
mnist.load_data()
# print(x_train[8432])
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



xgb_classifier = XGBClassifier()
xgb_classifier.fit(x_train, y_train)
train_outputs = xgb_classifier.predict(x_train)
test_outputs = xgb_classifier.predict(x_train)
train_score = sklearn.metrics.accuracy_score(x_train, y_train)
test_score = sklearn.metrics.accuracy_score(x_train, y_train)
