from warnings import simplefilter
from matplotlib.pyplot import get
from numpy.core.arrayprint import DatetimeFormat
from config import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from keras.models import save_model


def get_data(path="data.pickle"):
    try:
        print("Fetching data ...")
        data = pickle.load(open(os.path.join(DATA_DIR, path), "rb"))
        X, Y = data["data"], data["labels"]
        X_new = signal_pipeline(X)
        del X
        gc.collect()
        print("Data Fetched- Successful") 
        return X_new, Y

    except Exception as ex:
        print("Data fetching failed due to: ", ex)
        sys.exit()

def get_features():
    """
    filter data and extract spectogram
    """
    X, Y = get_data()
    # return spectogram(X), Y
    return X,Y


def process_data(file="filtered.pickle"):
    """
    encode labels and perform train-test-val split
    """
    try:
        if(file in os.listdir(DATA_DIR)):
            print("Fetching data from pickle file")
            data = pickle.load(open(os.path.join(DATA_DIR, file), "rb"))
            X , Y = data["data"],data["labels"]
            del data
            gc.collect()
        else:
            X, Y = get_features()
            print("Saving filtered data in a pickle file ...")
            pickle.dump({"data": X, "labels": Y},
                        open(os.path.join(DATA_DIR,file), "wb"))
            print("Done !")
    except Exception as ex:
        print("Cannot load features pickle due to :",ex)
        sys.exit()


    print(type(X))
    print(type(Y))

    # encode labels
    print(Y[:10])
    Y.replace(list(MAPPINGS.keys()), list(MAPPINGS.values()), inplace=True)
    print(Y[:10])
    
    Y = Y.values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_id, test_id = next(splitter.split(X, Y))
    X_train, y_train, X_test, y_test = X[train_id], Y[train_id], X[test_id], Y[test_id]

    train_id, test_id = next(splitter.split(X_train, y_train))
    X_train, y_train, X_val, y_val = X_train[train_id], y_train[train_id], X_train[test_id], y_train[test_id]

    print("Shape of data instance: ",X_train[0].shape)
    print("Shape of train data: ",X_train.shape)
    print("Shape of val data: ",X_val.shape)
    print("Shape of test data: ",X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(INPUT_SHAPE):

    def CNN_1D(INPUT_SHAPE=INPUT_SHAPE, DROPOUT=0.3, learning_rate=0.0003, activation="relu", neurons=64, K_regulizer=0.001):
        model = keras.models.Sequential()
        # 1st conv layer
        model.add(keras.layers.Conv1D(32, (3), activation="relu", input_shape=INPUT_SHAPE,
                                      kernel_regularizer=tf.keras.regularizers.l2(K_regulizer)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (3), strides=(2), padding='same'))

        # 2nd conv layer
        model.add(tf.keras.layers.Conv1D(64, (3), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(K_regulizer)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (3), strides=(2), padding='same'))

        # 3rd conv layer
        model.add(tf.keras.layers.Conv1D(128, (2), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(K_regulizer)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (2), strides=(2), padding='same'))

        # flatten output and feed into dense layer
        model.add(tf.keras.layers.Flatten())
        tf.keras.layers.Dropout(DROPOUT)

        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        tf.keras.layers.Dropout(DROPOUT)

        model.add(tf.keras.layers.Dense(64, activation='relu'))
        tf.keras.layers.Dropout(DROPOUT)

        # softmax output layer
        model.add(tf.keras.layers.Dense(5, activation='softmax'))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        print(model.summary())
        return model

    def RNN(INPUT_SHAPE=INPUT_SHAPE, DROPOUT=0.3, learning_rate=0.0003, activation="relu", neurons=64, K_regulizer=0.001):

        model = keras.models.Sequential()

        # 1st conv layer
        model.add(keras.layers.Conv1D(32, (3), activation="relu", input_shape=INPUT_SHAPE,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (3), strides=(2), padding='same'))

        model.add(tf.keras.layers.Conv1D(64, (3), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (3), strides=(2), padding='same'))

        # 3rd conv layer
        model.add(tf.keras.layers.Conv1D(128, (2), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (2), strides=(2), padding='same'))

        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        tf.keras.layers.Dropout(0.3)
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        tf.keras.layers.Dropout(0.3)
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        tf.keras.layers.Dropout(0.3)

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(64, activation='relu'))
        tf.keras.layers.Dropout(0.3)

        # softmax output layer
        model.add(tf.keras.layers.Dense(5, activation='softmax'))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #     print(model.summary())
        return model

    def CNN_2D():
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=INPUT_SHAPE,kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

        # 3rd conv layer
        model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

        # flatten output and feed into dense layer
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(256, activation='relu'))
        tf.keras.layers.Dropout(0.3)

        model.add(tf.keras.layers.Dense(64, activation='relu'))
        tf.keras.layers.Dropout(0.3)

        # softmax output layer
        model.add(tf.keras.layers.Dense(5, activation='softmax'))

        print(model.summary())

        return model


    # return CNN_2D()
    return CNN_1D()


def main():
    # import dataset and process it

    X_train, y_train, X_val, y_val, X_test, y_test = process_data()

    INPUT_SHAPE = X_train[0].shape
    # return 0
    # build model
    model = build_model(INPUT_SHAPE)

    # compile model and run
    model.compile(optimizer="Adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", min_delta=0.001, patience=5)
    history = model.fit(X_train, y_train, epochs=30, batch_size=30, validation_data=(
        X_val, y_val), callbacks=[earlystop_callback])

    # see performance of model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    # save model
    try:
        print("Saving Model ... ")
        save_model(model,os.path.join(MODEL_DIR,"model1D.h5"))
        print("Model saved !!")

    except Exception as ex:
        print("Saving Model failed due to: ",ex)
        sys.exit()



if __name__ == "__main__":
    main()
    # model = keras.models.load_model(os.path.join(MODEL_DIR,"CNN2D_Adadelta2.h5"))
    # X_train, y_train, X_val, y_val, X_test, y_test = process_data()
    # model.predict()
    # print(model.summary())
    # test_loss, test_acc = model.evaluate(X_test, y_test)
    # print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
