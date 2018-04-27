from __future__ import print_function

from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.metrics import accuracy_score
import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd




input_dim = 945 #27*35

train_df = pd.read_csv(r'C:Users/Valik/Downloads/fashionmnist/fashion-mnist_train.csv')
test_df = pd.read_csv(r'C:Users/Valik/Downloads/fashionmnist/fashion-mnist_test.csv')

train_data = np.asarray(train_df, dtype='float32')
test_data = np.asarray(test_df, dtype='float32')

X_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

X_test = test_data[:, 1:] / 255
y_test = test_data[:, 0]


X_train, x_validate, y_train, y_validate = train_test_split(X_train, y_train)



X_train = X_train.reshape(X_train.shape[0], input_dim)
X_test = X_test.reshape(X_test.shape[0], input_dim)




from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)




model= RandomForestClassifier(n_estimators=10)




XX = X_train
yy = y_train
model.fit(XX, yy)
train_outputs = model.predict(XX)
test_outputs = model.predict(X_test)
train_score_summary = accuracy_score(yy, train_outputs)
test_score_summary = accuracy_score(y_test, test_outputs)




print('Train accuracy:', train_score_summary)
print('Test accuracy:', test_score_summary)