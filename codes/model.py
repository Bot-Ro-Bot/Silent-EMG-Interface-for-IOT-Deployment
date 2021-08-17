from config import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


def get_data():
    pass


def parse_data():
    pass


def process_data():
    """
    filter data and encode labels, perform train-test-val split
    """
    pass


def build_model():

    def CNN_1D(INPUT_SHAPE, DROPOUT=0.3, learning_rate=0.0003, activation="relu", neurons=64, K_regulizer=0.001):
        model = keras.models.Sequential()
        # 1st conv layer
        model.add(keras.layers.Conv1D(64, (3), activation="relu", input_shape=INPUT_SHAPE,
                                      kernel_regularizer=tf.keras.regularizers.l2(K_regulizer)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (3), strides=(2), padding='same'))

        # 2nd conv layer
        model.add(tf.keras.layers.Conv1D(32, (3), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(K_regulizer)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (3), strides=(2), padding='same'))

        # 3rd conv layer
        model.add(tf.keras.layers.Conv1D(32, (2), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(K_regulizer)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (2), strides=(2), padding='same'))

        # flatten output and feed into dense layer
        model.add(tf.keras.layers.Flatten())
        tf.keras.layers.Dropout(DROPOUT)

        model.add(tf.keras.layers.Dense(64, activation='relu'))
        tf.keras.layers.Dropout(DROPOUT)

        # softmax output layer
        model.add(tf.keras.layers.Dense(5, activation='softmax'))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #     print(model.summary())
        return model

    def RNN(INPUT_SHAPE, DROPOUT=0.3, learning_rate=0.0003, activation="relu", neurons=64, K_regulizer=0.001):

        model = keras.models.Sequential()

        # 1st conv layer
        model.add(keras.layers.Conv1D(64, (3), activation="relu", input_shape=INPUT_SHAPE,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (3), strides=(2), padding='same'))

        model.add(tf.keras.layers.Conv1D(32, (3), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (3), strides=(2), padding='same'))

        # 3rd conv layer
        model.add(tf.keras.layers.Conv1D(32, (2), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(
            (2), strides=(2), padding='same'))

        model.add(LSTM(units=50, return_sequences=True))
        tf.keras.layers.Dropout(0.3)
        model.add(LSTM(units=50, return_sequences=True))
        tf.keras.layers.Dropout(0.3)
        model.add(LSTM(units=50, return_sequences=True))
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
        pass

    pass


def main():
    # import dataset and process it

    # build model

    # run model

    # test model

    pass


if __name__ == "__main__":
    main
