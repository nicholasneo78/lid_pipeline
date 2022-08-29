from email.mime import audio
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import json
import librosa
from typing import List, Tuple
import random

class CompareAudioThreshold:
    def __init__(self, dir_root:str, dir_1: str, dir_2: str, dir_back: str) -> None:
        self.dir_root = dir_root
        self.dir_1 = dir_1 # the one with higher threshold (less data)
        self.dir_2 = dir_2 # the one with lower threshold (more data)
        self.dir_back = dir_back

    def compare(self) -> None:

        # get the count for the similar audio files
        count = 0

        # list out the audio files in directory 1
        for audio_1 in tqdm(os.listdir(f'{self.dir_root}/{self.dir_1}/{self.dir_back}/')):
            
            # list out the audio files in directory 2
            for audio_2 in os.listdir(f'{self.dir_root}/{self.dir_2}/{self.dir_back}/'):
                # check if the audio file in directory 1 is also in directory 2
                if audio_1 == audio_2:
                    count += 1
                    continue

        print(f'{count}/{len(os.listdir(f"{self.dir_root}/{self.dir_1}/{self.dir_back}/"))} are in both directory')
        
    def __call__(self):
        return self.compare()

if __name__ == '__main__':
    delta = CompareAudioThreshold(dir_root='/preproc/datasets/mms/mms_prediction',
                                  dir_1='mms_prediction_060', 
                                  dir_2='mms_prediction_055', 
                                  dir_back='mms_batch_2/mms_20220620/CH 10/en')

    delta()
