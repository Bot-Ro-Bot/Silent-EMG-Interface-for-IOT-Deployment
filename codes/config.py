# global imports
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import biosppy
import scipy.signal as sig
from scipy.signal import spectrogram, windows
import os
import sys
import pdb
import warnings
import tqdm
import gc
warnings.filterwarnings("ignore")


mlp.rc("xtick", labelsize=10)
mlp.rc("ytick", labelsize=10)
mlp.rc("axes", labelsize=11)
plt.rcParams["figure.figsize"] = [11, 5]
plt.rcParams["figure.dpi"] = 100


# directories  definition
MAIN_DIR = ".."
DATA_DIR = os.path.join(MAIN_DIR, "dataset")
FIG_DIR = os.path.join(MAIN_DIR, "figures")
MODEL_DIR = os.path.join(MAIN_DIR, "models")
TEST_DIR = os.path.join(MAIN_DIR, "test")
FONT_DIR = os.path.join(MAIN_DIR, "fonts")
FONT_PATH = os.path.join(FONT_DIR, "Kalimati Regular.otf")
font = mlp.font_manager.FontProperties(fname=FONT_PATH)

# data definitions
SAMPLING_RATE = SR = 250  # Hz
NUM_CHANNELS = 8
ADC_RESOLUTION = 24  # bits
ADC_GAIN = 24.0
REF_VOLTAGE = 4.5  # Volts
SCALE_FACTOR = (REF_VOLTAGE/float((pow(2, 23))-1) /
                ADC_GAIN)*1000000.0  # micro-volts
SENTENCES = ["अबको समय सुनाउ",
             "एउटा सङ्गित बजाउ",
             "आजको मौसम बताउ",
             "बत्तिको अवस्था बदल",
             "पङ्खाको स्तिथी बदल"]
LABELS = SENTENCES

MAPPINGS = {"अबको समय सुनाउ": 0.0,
            "एउटा सङ्गित बजाउ": 1.0,
            "आजको मौसम बताउ": 2.0,
            "बत्तिको अवस्था बदल": 3.0,
            "पङ्खाको स्तिथी बदल": 4.0}

LENGTH = 1654
# model parameters


# a function to save plotted figures
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(FIG_DIR, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def parser(files, DEPLOY=False, LENGTH=2000, SAMPLES=15):
    """
    parser function to extract utterances from .txt file and store them in a dictionary
    """

    dataset = {"data": [], "labels": [], "length": []}

    def get_data(file):
        signal, f_len = read_data(file)
           
        if(DEPLOY==True):
            dataset["data"].extend(signal)
            return
        
        if(len(signal) != SAMPLES):
            return

        dataset["data"].extend(signal)
        dataset["labels"].extend(LABELS if(SAMPLES == 15) else SENTENCE_LABEL)
        dataset["length"].extend(f_len)

    def read_data(file):
        f = open(file, 'r')
        contents = map(lambda x: x.strip(), f.readlines())
        # the file starts with '%' and some instruction before data and removing these data
        frames_original = list(
            filter(lambda x: x and x[0] != '%', contents))[1:]
        # the data row contains channels info digital trigger and accelerometer info separated by comma
        frames_original = list(
            map(lambda s: list(map(lambda ss: ss.strip(), s.split(','))), frames_original))
        # (8 channels) + digital triggers
        # the digital trigger is in a[16], used to indicate the utterance
        frames = list(
            map(lambda a: list(map(float, a[1:9])) + [float(a[16])], frames_original))
        frames = np.array(frames)
        indices = []
        signal = []
        f_len = []
        for index, f in enumerate(frames[:, -1]):
            if(bool(f) ^ bool(frames[(index+1) if ((index+1) < len(frames)) else index, -1])):
                indices.append(index)
                if len(indices) > 1 and len(indices) % 2 == 0:
                    frame_len = indices[len(indices)-1] - \
                        indices[len(indices)-2]
                    f_len.extend([frame_len])
                    if(frame_len < LENGTH):
                        pad = int(np.ceil((LENGTH - frame_len)/2))
                    else:
                        pad = 0
                    left_pad = indices[len(indices)-2] - pad
                    right_pad = indices[len(indices)-1] + pad
                    a_frame = (frames[left_pad:right_pad, :-1])[:LENGTH]
                    signal.append(a_frame)

        return np.array(signal), f_len

    for file in tqdm.tqdm(files, desc="PARSING DATA"):
        get_data(file)

    return dataset


def signal_pipeline(data, RICKER=False):
    """
    CORRECT DC DRIFT --> CORRECT DC BIAS --> SMOOTHING SIGNAL --> NORMALIZE DATA --> FILTER DATA 	
    """
    filter_data = []

    def digital_filter(data, HPF=0.5, LPF=10, H_ORDER=4, L_ORDER=4, SR=250):
        """
        HPF --> NOTCH --> LPF --> RICKER CONVOLUTION
        """
        # highpass filter
        f_signal = biosppy.signals.tools.filter_signal(
            data, ftype="butter", band="highpass", order=H_ORDER, sampling_rate=SR, frequency=HPF)
        # notch filter
        b, a = sig.iirnotch(50, 30, SR)
        f_signal = sig.lfilter(b, a, f_signal[0])

        # lowpass filter
        f_signal = biosppy.signals.tools.filter_signal(
            f_signal, ftype="butter", band="lowpass", order=L_ORDER, sampling_rate=SR, frequency=LPF)

        if(RICKER == True):
            # RICKER CONVOLUTION TO REMOVE HEARTBEAT ARTIFACTS
            ricker_width = 35 * SR // 250
            ricker_sigma = 4.0 * SR / 250
            ricker = sig.ricker(ricker_width, ricker_sigma)
            # normalize ricker
            ricker = np.array(ricker, np.float32) / np.sum(np.abs(ricker))
            convolution = sig.convolve(f_signal[0], ricker, mode="same")
            return (f_signal[0]-2*convolution)

        return f_signal[0]

    def process_signal(data):
        f_data = []
        for i in range(8):
            # correction of DC drift
            c_data = data[:, i] - data[0, i]

            # correct DC bias
            c_data = c_data - np.mean(c_data)

            # normalize and filter data
            c_data = digital_filter(c_data)
            f_data.append(c_data)

        return np.array(f_data).T

    for d in tqdm.tqdm(data, desc="FILTERING DATA: "):
        temp_data = process_signal(d)
        filter_data.extend([temp_data])

    return np.array(filter_data)


def spectogram(data):
    feature_data = []
    def getSpect(sdata):
        M = 60
        win = windows.hann(M, sym=False)
        return spectrogram(x=np.array(sdata), fs=SAMPLING_RATE, window=win, nperseg=len(win), noverlap=3*M/4, nfft=M)
    def process_signal(data):
        f_data = []
        for i in range(8):
            _, _, c_data = getSpect(data[:, i])
            f_data.append(c_data.T)
        return np.array(f_data).T
    
    for d in tqdm.tqdm(data,desc="EXTRACTING SPECTOGRAM: "):
        temp_data = process_signal(d)
        feature_data.extend([temp_data])
    return np.array(feature_data)
    

def CNN_1D(INPUT_SHAPE=(1654,8), DROPOUT=0.3, learning_rate=0.0003, activation="relu", neurons=64, K_regulizer=0.001):
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


if __name__ == "__main__":
    print("You are in config file.")
