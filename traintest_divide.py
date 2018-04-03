import numpy
import json
from sklearn.model_selection import train_test_split

data = json.load(open("config.json"))


img_features = numpy.load(data["features_path"]+"/.npy")

X = numpy.asarray([img[0] for img in img_features])
y = numpy.asarray([img[2] for img in img_features])

X_train, x_test, y_train, y_test = train_test_split(X, y)





