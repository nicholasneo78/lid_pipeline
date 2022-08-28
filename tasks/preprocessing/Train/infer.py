import os
import json
import logging
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pydub

import torch
from speechbrain.pretrained import EncoderClassifier

class BatchInfer:
    '''
        takes in a root directory containing .wav files and a pretrained speechbrain LID model for inference
        outputs the audio files into another directory, where the audio will be splitted into different languages and an additional "others" for the audio that did not hit the threshold stated for the classification
    '''

    def __init__(self, root_dir: str, pretrained_model_dir: str, silence_removed_dir: str, preprocessed_dir: str, threshold: Dict[str, float]) -> None:
        '''
            root_dir: the root directory where all the .wav files resides
            pretrained_model_dir: the pretrained speechbrain LID model path
            silence_removed_dir: the one level directory to search for the .wav files that is to be inferenced
            preprocessed_dir: the one directory in which the prediction of the audio file resides
            threshold: the dictionary that contains the language as the key and the confidence threshold as the value to place the audio into the predicted language directory or the "others" directory (model not very confident about the prediction) 
        '''    
        
        self.root_dir = root_dir
        self.pretrained_model_dir = pretrained_model_dir
        self.silence_removed_dir = silence_removed_dir
        self.preprocessed_dir = preprocessed_dir
        self.threshold = threshold

    def create_new_dir(self, directory: str) -> None:
        '''
            creates new directory and ignore already created ones

            directory: the directory path that is being created
        '''
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!

    def create_nested_dir(self, directory: str) -> str:
        '''
            creates nested directory to replicated the original folder's tree structure
        
            directory: the root directory of the audio file
        '''

        # split the directory based on the the delimiter '/'
        dir_split_list = directory.split('/')
        
        # remove the '' if there is (usually have when an absolute path is passed)
        dir_split_list.remove('')

        # remove the last element as it is not a directory, and save the last element as a separate variable
        dir_file = dir_split_list[-1]
        dir_split_list = dir_split_list[:-1]

        # get the index of the dir that is to be replaced with the new dir
        idx = [i for i,x in enumerate(dir_split_list) if x == self.silence_removed_dir]

        # replace the text to the edited folder path
        dir_split_list[max(idx)] = self.preprocessed_dir
         
        # merge the list into string again
        directory_edited = '/' + '/'.join(dir_split_list)

        # creating the nested dir from the list
        directory_edited_split_list_ = directory_edited.split('/')

        # appending '/' to each of the elements in the list
        directory_edited_split_list = [d + '/' for d in directory_edited_split_list_]

        # concatenate the first 4 elements in the list, this will be the creation of the base folder
        base_idx = 4
        #base_dir = ''.join(directory_edited_split_list[0:base_idx])

        # iterate the creation of the nested directory
        while(base_idx <= len(directory_edited_split_list)):
            self.create_new_dir(''.join(directory_edited_split_list[:base_idx]))

            # increment the count
            base_idx += 1

        # returns the edited file path and the audio filename
        return directory_edited, dir_file

    def predict_class(self, audio_path: str) -> str:
        '''
            takes in the audio filepath of the audio for inference

            returns the predicted class of the audio and the confidence of the prediction
        '''

        # load the pretrained model
        model = EncoderClassifier.from_hparams(source=self.pretrained_model_dir, savedir="./tmp")

        # load the audio for inference
        signal = model.load_audio(audio_path)

        # get the prediction
        prediction = model.classify_batch(signal)

        # The scores in the prediction[0] tensor can be interpreted as log-likelihoods that
        # the given utterance belongs to the given language (i.e., the larger the better)
        # The linear-scale likelihood can be retrieved using the following:
        # max_likelihood = prediction[1].exp()[0]

        # get the confidence score (max probability)
        probs = torch.nn.functional.softmax(prediction[0][0], dim=0)
        confidence = max(probs)

        print(confidence)

        # # check if the confidence score is above a certain threshold, if it is not, classify it as "others" for further investigation
        # if confidence >= self.threshold:
        #     # get the predicted class of the audio
        #     pred_class = prediction[3][0]
        # else:
        #     # label it as "others"
        #     pred_class = 'others'

        # get the prediction from the model
        pred_class = prediction[3][0]
        # check if the confidence score is above a certain threshold, if it is not, classify it as "others" for further investigation
        if confidence >= self.threshold[pred_class]:
            # get the predicted class of the audio by passing it
            pass
        else:
            # label it as "others"
            pred_class = 'others'

        # to remove unwanted temp .wav file in the root folder after the inference
        root_dir = '/lid/'
        for file in os.listdir(root_dir):
            if file.endswith('.wav'):
                os.remove(os.path.join(root_dir, file))
        
        return pred_class, confidence

    def batch_inference(self) -> None:
        '''
            loads all the audio files for inference and produce a manifest based on the root folder of where the audio files are residing

            returns the filepath of the manifest
        '''

        # read the filepaths of the audio file
        for root, subdirs, files in os.walk(self.root_dir):
            for file in tqdm(files):
                # change the target directory to a new one for the different audio classes
                initial_filepath = os.path.join(root, file)
                modified_filepath = initial_filepath.replace(self.silence_removed_dir, self.preprocessed_dir)

                # get all the .wav file only
                if file.endswith('.wav'):
                    # create the nested directory for the audio file to reside in
                    edited_dir, filename = self.create_nested_dir(f'{initial_filepath}')

                    # generate prediction
                    pred_class, confidence = self.predict_class(initial_filepath)

                    # create a new nested directory for the language in that particular channel and date
                    self.create_new_dir(f'{edited_dir}/{pred_class}/')

                    # load and export the audio
                    soundfile = pydub.AudioSegment.from_wav(initial_filepath)
                    soundfile.export(f'{edited_dir}/{pred_class}/{filename}', format='wav')

    def __call__(self) -> None:
        return self.batch_inference()

if __name__ ==  '__main__':
    inference = BatchInfer(root_dir='/lid/datasets/mms/sil_test/',
                           pretrained_model_dir='/lid/results/ECAPA-TDNN/1988/save/CKPT+2022-08-27+10-07-55+00/', 
                           silence_removed_dir='sil_test', 
                           preprocessed_dir='mms_prediction/mms_prediction_full_en_060', 
                           threshold={'en': 0.6, 'ms': 0.6}
                           )

    inference()