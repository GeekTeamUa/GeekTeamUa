from __future__ import print_function
import glob
from sklearn.neighbors import KNeighborsClassifier
import json
from sklearn.metrics import accuracy_score
import os.path
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend as K
from data_reader import read_data



data = json.load(open("config.json"))

if os.path.isfile(data["features_path"]+"/.npy"):           # Check if we already have numpy with our data
    dataset = numpy.load(data["features_path"] + "/.npy")
else:
    dataset = read_data()   # Create and read numpy data (see data_reader.py)



X = numpy.asarray([img[0] for img in dataset])
y = numpy.asarray([img[2] for img in dataset])


# the data, shuffled and split between train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)




input_dim = 945 #27*35
X_train = X_train.reshape(X_train.shape[0], input_dim)
X_test = X_test.reshape(X_test.shape[0], input_dim)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 72)
y_test = np_utils.to_categorical(y_test, 72)

n_train = X_train.shape[0]

X_train = X_train[1:np.round(n_train*0.1),: ]

model = KNeighborsClassifier(n_neighbors=3, leaf_size=3)



XX = X_train
yy = y_train
model.fit(XX, yy)
train_outputs = model.predict(XX)
test_outputs = model.predict(X_test)
train_score_summary = accuracy_score(yy, train_outputs)
test_score_summary = accuracy_score(y_test, test_outputs)





print('Test loss:', train_score_summary)
print('Test accuracy:', test_score_summary)









