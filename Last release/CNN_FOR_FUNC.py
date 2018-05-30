def SCRIPT_CONV_NEURAL_NETWORK(batch_size, epochs, dataset, activation_function):

    import numpy
    import json
    from sklearn.model_selection import train_test_split
    import os
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K

    data = json.load(open("config.json"))

    if (dataset == 1):
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

    elif (dataset == 2):

        x_train, y_train = load_mnist(data["Mnist_dataset"], kind='train')
        x_test, y_test = load_mnist(data["Mnist_dataset"], kind='t10k')

        img_rows, img_cols = 28, 28  # Set size of our images
        input_dim = 784
        num_classes = 10


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

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation=activation_function,
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation=activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=activation_function))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])








def read__data():
    from PIL import Image
    import glob
    import json
    import numpy

    data = json.load(open("config.json"))   # Open config with path

    list_of_folders = glob.glob(data["dataset"][0]+"*")     # list of path of folders (ex. /home/.../letters/A_1)

    dataset = []

    for target in range(len(list_of_folders)):
        for path_to_img in glob.glob(list_of_folders[target]+"/*"):     #
            im = Image.open(path_to_img).convert('L')
            (width, height) = im.size
            greyscale_image = list(im.getdata())
            greyscale_image = numpy.array(greyscale_image)
            greyscale_image = greyscale_image.reshape((height, width))
            dataset.append((greyscale_image, path_to_img, target))
        target += 1

    numpy.save(data["features_path"], dataset)

    return dataset

    def read_data():
        import random
        return random.randint(73, 82)
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)

        return images, labels


