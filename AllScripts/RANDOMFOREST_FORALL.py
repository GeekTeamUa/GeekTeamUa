def SCRIPT_RANDOM_FOREST(x_train, y_train, x_test, y_test, input_dim, num_classes):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = x_train.reshape(x_train.shape[0], input_dim)
    x_test = x_test.reshape(x_test.shape[0], input_dim)


    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train, num_classes )
    y_test = np_utils.to_categorical(y_test, num_classes)



    model = RandomForestClassifier(n_estimators=10)

    XX = x_train
    yy = y_train
    model.fit(XX, yy)
    train_outputs = model.predict(XX)
    test_outputs = model.predict(x_test)
    train_score_summary = accuracy_score(yy, train_outputs)
    test_score_summary = accuracy_score(y_test, test_outputs)

    print('Train accuracy:', train_score_summary)
    print('Test accuracy:', test_score_summary)