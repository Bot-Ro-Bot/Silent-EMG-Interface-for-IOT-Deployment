from warnings import simplefilter
from matplotlib.pyplot import get
from numpy.core.arrayprint import DatetimeFormat
from config import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from keras.models import save_model
from keras.models import save_weights

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
    kr = keras.regularizers.l1(0.006)
    ar = keras.regularizers.l2(0.003)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3,3), activation="elu",  input_shape=INPUT_SHAPE))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2))) 

    model.add(keras.layers.Conv2D(64, (3,3), activation='elu', ))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # flatten output and feed into dense layer
    model.add(keras.layers.Flatten())
    # keras.layers.BatchNormalization(momentum=0.01, epsilon=1e-05)

    model.add(keras.layers.Dense(4096, activation='elu', kernel_regularizer=kr, activity_regularizer=ar)) 
    keras.layers.Dropout(0.3)
    # model.add(keras.layers.Dense(1024, activation='relu', kernel_regularizer=kr,activity_regularizer=ar)) 
    # keras.layers.Dropout(0.25)

    # keras.layers.BatchNormalization(momentum=0.01, epsilon=1e-05)
    model.add(keras.layers.Dense(512, activation='elu', kernel_regularizer=kr)) 
    keras.layers.Dropout(0.25)

    # keras.layers.BatchNormalization(momentum=0.01, epsilon=1e-05)
    model.add(keras.layers.Dense(64, activation='elu', kernel_regularizer=kr)) 
    keras.layers.Dropout(0.2)

    # softmax output layer
    model.add(keras.layers.Dense(5, activation='softmax'))
    keras.layers.BatchNormalization(momentum=0.001, epsilon=1e-05)

    return model


def main():
    # import dataset and process it

    X_train, y_train, X_val, y_val, X_test, y_test = process_data()

    INPUT_SHAPE = X_train[0].shape
    # return 0
    # build model
    model = build_model(INPUT_SHAPE)

    # compile model and run
    # model.compile(optimizer="Adam",
    #               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # earlystop_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor="accuracy", min_delta=0.001, patience=5)
    # history = model.fit(X_train, y_train, epochs=30, batch_size=30, validation_data=(
    #     X_val, y_val), callbacks=[earlystop_callback])

    epoch = 30
    lr = 0.00025   
    adadelta = keras.optimizers.Adadelta(learning_rate = lr, rho = 0.95, epsilon = 1e-07)

    model.compile(optimizer=adadelta, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # see performance of model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    # save model
    try:
        print("Saving Model ... ")
        # save_model(model,os.path.join(MODEL_DIR,"model1D.h5"))
        save_model(model,os.path.join(MODEL_DIR,"model1D_weights.h5"))

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
