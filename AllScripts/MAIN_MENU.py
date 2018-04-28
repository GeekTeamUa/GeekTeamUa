from __future__ import print_function
import numpy
import json
from sklearn.model_selection import train_test_split
from reader_FashionMnist import load_mnist
from reader_UAMNIST import read_data
import os
from CNN_FORALL import SCRIPT_CONV_NEURAL_NETWORK
from XGBOOST_FORALL import SCRIPT_XG_BOOST
from RANDOMFOREST_FORALL import SCRIPT_RANDOM_FOREST
from LOGISTICREGRESSION_FORALL import SCRIPT_LOGISTIC_REGRESSION
from SVM_FORALL import SCRIPT_SVM

useUAMnist = True
useFashionMnist = False


useCNN = False
useRANDOM_FOREST = True
useXG_BOOST = False
useLOGISTICREGRESSION = False
useSVM = False



data = json.load(open("config.json"))

if useUAMnist:

    if os.path.isfile(data["features_path"] + "/.npy"):  # Check if we already have numpy with our data
        dataset = numpy.load(data["features_path"] + "/.npy")
    else:
        dataset = read_data()  # Create and read numpy data (see data_reader.py)

    X = numpy.asarray([img[0] for img in dataset])
    y = numpy.asarray([img[2] for img in dataset])

    x_train, x_test, y_train, y_test = train_test_split(X, y)  # Divide our data on train and test samples

    img_rows, img_cols = 27, 35  # Set size of our images
    input_dim = 945
    num_classes = 72  # Write number of classes 72

elif useFashionMnist:
    x_train, y_train = load_mnist(data["Mnist_dataset"], kind='train')
    x_test, y_test = load_mnist(data["Mnist_dataset"], kind='t10k')

    img_rows, img_cols = 28, 28  # Set size of our images
    input_dim = 784
    num_classes = 10



if useCNN:
    SCRIPT_CONV_NEURAL_NETWORK(x_train, y_train, x_test, y_test, img_rows, img_cols, input_dim, num_classes)
elif useRANDOM_FOREST:
    SCRIPT_RANDOM_FOREST(x_train, y_train, x_test, y_test, input_dim, num_classes)
elif useXG_BOOST:
    SCRIPT_XG_BOOST(x_train, y_train, x_test, y_test, input_dim)
elif useLOGISTICREGRESSION:
    SCRIPT_LOGISTIC_REGRESSION(x_train, y_train, x_test, y_test, input_dim)
elif useSVM:
    SCRIPT_SVM(x_train, y_train, x_test, y_test, input_dim)






