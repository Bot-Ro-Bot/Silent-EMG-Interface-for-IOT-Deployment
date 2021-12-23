from config import LENGTH,SAMPLING_RATE
from inference_live import Inference
import time
import numpy as np

import socket_client_live

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

BoardShim.enable_dev_board_logger()

boardParameters = BrainFlowInputParams()
boardParameters.serial_port = '/dev/ttyUSB0'

FILENAME = "live_session/file"
TIME_DELAY = 2
# real board
cytonId = BoardIds.CYTON_BOARD.value 
board = BoardShim(cytonId, boardParameters)

def main():
    pass

def read_button():
    #read digital pin 17
    digital_pin_17 = board.get_current_board_data(1)[16]
    # print(digital_pin_17)
    return digital_pin_17

def save_file(data):
    DataFilter.write_file(data,FILENAME,"w")

def start_session():
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Starting Recording Session')
    board.prepare_session()
    board.start_stream()

def end_session():
    board.stop_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Ending Recording Session')
    board.release_session()
    

def record_data():
    # clear buffer
    end_session()
    #start stream
    start_session()
    time.sleep(TIME_DELAY)
    
    while(read_button()==True):
        pass

    # save_file(data)
    # end_session()
    return board.get_board_data()


def loop():
    start_session()

    while(True):
        if(read_button()==True):
            try:
                print("In record section")
                data = record_data()
                data = (np.transpose(data))[TIME_DELAY*SAMPLING_RATE:,1:9]
                data = data[:LENGTH,:]
                # print(data.shape)
                if (len(data)<LENGTH):
                    raise Exception("Length of data not sufficient!!!. Please try again with more samples.")
              
                #make prediction
                inf = Inference()
                prediction = inf.predict(data)
                print("Prediction: ", prediction[0])
                
                #control IOT device
                socket_client_live.send(prediction[1])

                key = input("Press any key to continue predicting or n to exit: ")
                if key.upper()=="N":
                    break
                else:
                    print("Onto a new session...")
                # start_session()
            except Exception as ex:
                print("Recording failed due to:",ex)
                return


if __name__=="__main__":
    # looper()
    # record_data()
    loop()
    pass