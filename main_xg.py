from __future__ import print_function
import tensorflow
import keras
import numpy
import xgboost
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



model = xgboost.XGBClassifier()

model.fit(x_train, y_train)
train_outputs = model.predict(x_train)
test_outputs = model.predict(x_train)
train_score = sklearn.metrics.accuracy_score(x_train, y_train)
test_score = sklearn.metrics.accuracy_score(x_train, y_train)

print(train_score)
print(test_score)
