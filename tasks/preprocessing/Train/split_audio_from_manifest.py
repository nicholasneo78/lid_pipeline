import os
import json
import logging
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pydub

class SplitAudio:
    '''
        reads from the manifest and split the audio files into their respective language folders
    '''

    def __init__(self, manifest_dir: str, original_root_dir: str, replaced_root_dir: str, silence_removed_dir: str, preprocessed_dir: str, data_is_others: bool, is_clearml: bool) -> None:
        self.manifest_dir = manifest_dir
        self.original_root_dir = original_root_dir
        self.replaced_root_dir = replaced_root_dir
        self.silence_removed_dir = silence_removed_dir
        self.preprocessed_dir = preprocessed_dir
        self.data_is_others = data_is_others
        self.is_clearml = is_clearml

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

    def create_nested_dir_clearml(self, directory: str) -> str:
        '''
            creates nested directory to replicated the original folder's tree structure for clearml
        
            directory: the root directory of the audio file
        '''
        # print('before splitting')
        # print(directory)
        # split the directory based on the the delimiter '/'
        dir_split_list = directory.split('/')
        # print(dir_split_list)
        
        # print('splitted')
        # remove the '' if there is (usually have when an absolute path is passed)
        try:
            dir_split_list.remove('')
        except:
            print('WRONG VALUES')
            pass

        # remove the last element as it is not a directory, and save the last element as a separate variable
        dir_file = dir_split_list[-1]
        dir_split_list = dir_split_list[4:-1] # remove the clearml front path

        # merge the list into string again
        directory_edited = '/'.join(dir_split_list)

        # append the root preprocessed directory into the existing directory
        directory_edited_with_root = f'{self.preprocessed_dir}/{directory_edited}'

        # creating the nested dir from the list
        directory_edited_split_list_ = directory_edited_with_root.split('/')

        # appending '/' to each of the elements in the list
        directory_edited_split_list = [d + '/' for d in directory_edited_split_list_]

        # concatenate the first 4 elements in the list, this will be the creation of the base folder
        base_idx = 0

        # iterate the creation of the nested directory
        while(base_idx <= len(directory_edited_split_list)):
            self.create_new_dir(''.join(directory_edited_split_list[:base_idx]))
            # print(''.join(directory_edited_split_list[:base_idx]))

            # increment the count
            base_idx += 1

        # returns the edited file path and the audio filename
        return f'{self.preprocessed_dir}/{directory_edited}', dir_file

    def split(self) -> None:
        '''
            loads all the audio files based on the manifest and placed it in the folder based on the language predicted
        '''

        # if the json manifest is for the audio classified as 'others', as the json format differs for the ones classified as 'others' or for a particular language
        if self.data_is_others:

            with open(self.manifest_dir, mode='r', encoding='utf-8') as fr:
                for idx, line in tqdm(enumerate(fr.readlines())):
                    data = json.loads(line)

                    initial_filepath_raw = data['audio_filepath']

                    # replacing filepath from speechbrain format to the directory format
                    initial_filepath = initial_filepath_raw.replace(self.original_root_dir, self.replaced_root_dir)

                    if self.is_clearml:
                        # create the nested directory for the preprocessed audio
                        edited_dir, filename = self.create_nested_dir_clearml(f'{initial_filepath}')
                    else:
                        # create the nested directory for the preprocessed audio
                        edited_dir, filename = self.create_nested_dir(f'{initial_filepath}')

                    # create a new nested directory based on the the language for that channel
                    self.create_new_dir(f'{edited_dir}/others/')

                    # load and export the audio
                    soundfile = pydub.AudioSegment.from_wav(initial_filepath)
                    soundfile.export(f'{edited_dir}/others/{filename}', format='wav')

        # if the json manifest is for audio classified as a particular language
        else:
            # load the json manifest
            with open(self.manifest_dir, 'r') as f:
                data = json.load(f)

            # iterate the entries to get the data
            for entry in tqdm(data):
                initial_filepath_raw = data[entry]['wav']
                language = data[entry]['language']

                # replacing filepath from speechbrain format to the directory format
                initial_filepath = initial_filepath_raw.replace(self.original_root_dir, self.replaced_root_dir)

                # create the nested directory for the preprocessed audio
                edited_dir, filename = self.create_nested_dir_clearml(f'{initial_filepath}')

                # create a new nested directory based on the the language for that channel
                self.create_new_dir(f'{edited_dir}/{language}/')

                # load and export the audio
                soundfile = pydub.AudioSegment.from_wav(initial_filepath)
                soundfile.export(f'{edited_dir}/{language}/{filename}', format='wav')

    def __call__(self) -> None:
        return self.split()

if __name__ ==  '__main__':

    # for english data
    s_others = SplitAudio(manifest_dir='/lid/datasets/mms/mms_silence_removed/batch_2s_iteration_1_others.json', 
                   original_root_dir='/lid/datasets/mms/mms_silence_removed', 
                   replaced_root_dir='/lid/datasets/mms/mms_silence_removed', 
                   silence_removed_dir='mms_silence_removed',
                   preprocessed_dir='mms_prediction/testing', 
                   data_is_others=True,
                   is_clearml=False)
    s_others()

    s_en = SplitAudio(manifest_dir='/lid/datasets/mms/mms_silence_removed/batch_2s_iteration_1_en.json', 
                   original_root_dir='{data_root}', 
                   replaced_root_dir='/lid/datasets/mms/mms_silence_removed', 
                   silence_removed_dir='mms_silence_removed',
                   preprocessed_dir='mms_prediction/testing', 
                   data_is_others=False,
                   is_clearml=False)
    s_en()