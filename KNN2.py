from __future__ import print_function
import glob
from sklearn.neighbors import KNeighborsClassifier
import json
from sklearn.metrics import accuracy_score
import os.path
from sklearn.model_selection import train_test_split
import numpy
from keras import backend as K





data = json.load(open("config.json"))
dataset = glob.glob(data["dataset"][0]+"*")



X = numpy.asarray([img[0] for img in dataset])
y = numpy.asarray([img[2] for img in dataset])

# the data, shuffled and split between train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)



img_rows, img_cols = 27, 35     # Set size of our images


if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 72)
Y_train = np_utils.to_categorical(y_train, 72)



model = KNeighborsClassifier(n_neighbors=3)


XX = X_train
yy = y_train
model.fit(XX, yy)
train_outputs = model.predict(XX)
test_outputs = model.predict(X_test)
train_score_summary = accuracy_score(yy, train_outputs)
test_score_summary = accuracy_score(y_test, test_outputs)




print('Test loss:', train_score_summary)
print('Test accuracy:', test_score_summary)









