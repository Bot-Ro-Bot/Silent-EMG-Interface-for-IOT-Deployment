import time
import tqdm
import numpy as np
import pandas as pd
# from emg_lib import *


import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


BoardShim.enable_dev_board_logger()

boardParameters = BrainFlowInputParams()
boardParameters.serial_port = '/dev/ttyUSB0'

cytonId = BoardIds.SYNTHETIC_BOARD.value # BoardIds.CYTON_BOARD.value #
board = BoardShim(cytonId, boardParameters)

channels = board.get_emg_channels(cytonId)
print("Channels",channels)

board.prepare_session()

board.start_stream()
BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
for i in zip(tqdm.tqdm(range(5),desc="PARSING DATA")):
    time.sleep(1)
    # print(board.get_current_board_data(1))


data = board.get_board_data()
board.stop_stream()
board.release_session()

DataFilter.write_file(data,"file.txt","w")

# data = np.transpose(data)
print(data.shape)
# channel_data = data[:,channels[:8]]     #[:8] - for synthetic boards only remove this.
# print(channel_data.shape)
# rawdata = []
# rawdata.append(channel_data)
# print(len(rawdata))

# # filteredData = signal_pipeline(rawdata)
# # dataFeature = feature_pipeline_melspectrogram(filteredData)
# # dataFeature = reshapeChannelIndexToLast(dataFeature)
