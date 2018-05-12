def SCRIPT_XG_BOOST(x_train, y_train, x_test, y_test, input_dim):

    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = x_train.reshape(x_train.shape[0], input_dim)
    x_test = x_test.reshape(x_test.shape[0], input_dim)



    model = XGBClassifier()

    XX = x_train
    yy = y_train
    model.fit(XX, yy)
    train_outputs = model.predict(XX)
    test_outputs = model.predict(x_test)
    train_score_summary = accuracy_score(y_train, train_outputs)
    test_score_summary = accuracy_score(y_test, test_outputs)




    print('Train accuracy:', train_score_summary)
    print('Test accuracy:', test_score_summary)

    print("model fitting finished")

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)