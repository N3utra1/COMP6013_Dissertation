"""
    This is used to debug files written by the program. It is not needed when running the finished product
"""

import numpy as np 
import os
import glob


files_to_check = glob.glob("D:\\stft-chb-mit\\*\\*\\*.npy")

def check_shape():
    # checks if there is a file saved which is not the appropriate shape for the ml model
    for file in files_to_check:
        current_file = np.load(file)
        if not current_file.shape == (17, 3841, 2):
            print(f"bad shape for {file}")
            print(current_file.shape)
            input()
        del(current_file)

def check_3rd_dim():
    # check to see if the 3rd dim has data in it (debugging)
    for file in files_to_check:
        current_file = np.load(file)
        if current_file.shape[2] == 0:
            print(f"{file}")
        del(current_file)

check_shape()