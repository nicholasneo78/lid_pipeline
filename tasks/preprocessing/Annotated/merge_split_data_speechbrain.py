import os
from os.path import join
import numpy as np
import json
import librosa
from typing import List, Tuple
import random

# set up the seed number
SEED = 2022

class MergeAndSplit:
    def __init__(self, json_dir_list: str, train_dir: str, dev_dir: str, test_dir: str, sampling_mode: str) -> None:
        '''
            json_dir_list: the list of directories residing in the different language json files
            train_dir: the train manifest consisting of the train data directories 
            dev_dir: the dev manifest consisting of the dev data directories 
            test_dir: the test manifest consisting of the test data directories 
            sampling_mode: 'all' or 'equal', denoting which mode to process and combine the data during the train-dev-test split
        '''

        self.json_dir_list = json_dir_list
        self.train_dir = train_dir
        self.dev_dir = dev_dir
        self.test_dir = test_dir
        self.sampling_mode = sampling_mode

    def load_annotation(self, annotation_dir) -> List:
        '''
            load the annotation json file with the specified directory
        '''
        
        with open(annotation_dir, mode='r', encoding='utf-8') as f:
            # reading the whole json file
            data = json.load(f)

        return data

    def get_minimum_duration(self) -> Tuple[int, List]:
        '''
            takes in the various languages of the json annotations and check the duration in each of the json file

            returns the least number of duration among all the json annotation file
        '''

        # set up a list to store the number of entries
        audio_duration_list = []

        # set up the list to store the audio directories
        data_list = []

        # load the list of json files
        for json_dir in self.json_dir_list:
            data = self.load_annotation(json_dir)

            # append the manifest into a list
            data_list.append(data)

            # set up counter to count the duration
            audio_duration = 0

            for _, entry in enumerate(data):
                # append the duration into the counter
                audio_duration += entry['duration']

            # append the total duration per json file into the audio duration list
            audio_duration_list.append(audio_duration)

        return min(audio_duration_list), data_list

    def fix_equal_duration(self) -> List:
        '''
            calls get_minimum_duration to get the minimum duration and also to retrieve the data from the json file

            returns the minimum duration of each of the data according to the minimum duration
        '''

        # gets the min duration and the data list
        min_duration, annotation_list = self.get_minimum_duration()

        # set up a list to store the list of the selected file path for the final dataset per language
        selected_annotation_list = []

        # iterate through the different annotation files
        for annotation in annotation_list:

            # set up the list to store the selected file path for the final dataset per language
            selected_annotation = []

            # randomise the annotation in the list first
            random.Random(SEED).shuffle(annotation)

            # setup the counter for the audio duration
            audio_duration = 0
            
            # iterate through the annotation file
            for idx, entry in enumerate(annotation):

                # if the total duration is more than the threshold min_duration, break the loop
                if audio_duration >=  min_duration:
                    break
                
                audio_duration += entry['duration']
                
                # append the entries
                selected_annotation.append(entry)

            # append the list into the selected_annotation_list list
            selected_annotation_list.append(selected_annotation)

        return selected_annotation_list

    def load_all_data(self) -> List[List[str]]:
        '''
            load all the filepath into a list regardless of data imbalance

            returns a list of different languages of lists of audio filepaths 
        '''

        # set up the list to store the audio directories
        data_list = []

        # load the list of json files
        for json_dir in self.json_dir_list:
            data = self.load_annotation(json_dir)

            # append the manifest list into a list (nested list)
            data_list.append(data)

        return data_list

    def train_dev_test_split(self, train_ratio, dev_ratio) -> Tuple[List, List, List]:
        '''
            splits the data directories into train dev test set
        '''

        # get the annotations - either loading all data or equal data ('all' or 'equal')
        if self.sampling_mode == 'equal':
            selected_annotation_list = self.fix_equal_duration()
        elif self.sampling_mode == 'all':
            selected_annotation_list = self.load_all_data()

        # initiate the list 
        train_list = []
        dev_list = []
        test_list = []

        # iterate the list
        for idx, selected_annotation in enumerate(selected_annotation_list):
            # do the split
            train_list.extend(selected_annotation[:int(train_ratio*len(selected_annotation))])
            dev_list.extend(selected_annotation[int(train_ratio*len(selected_annotation)):int((train_ratio+dev_ratio)*len(selected_annotation))])
            test_list.extend(selected_annotation[int((train_ratio+dev_ratio)*len(selected_annotation)):])

        # shuffles the lists
        random.Random(SEED).shuffle(train_list)
        random.Random(SEED).shuffle(dev_list)
        random.Random(SEED).shuffle(test_list)

        # # write to json file
        # with open(f'{self.train_dir}', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(train_list, indent=2))

        # with open(f'{self.dev_dir}', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(dev_list, indent=2))

        # with open(f'{self.test_dir}', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(test_list, indent=2))

        return train_list, dev_list, test_list

    def convert_json_format(self, train_ratio, dev_ratio) -> Tuple[List, List, List]:
        '''
            converts the json format from a list format to a dictionary format with keys attach to it
        '''

        # import the lists
        train_list, dev_list, test_list = self.train_dev_test_split(train_ratio=train_ratio, dev_ratio=dev_ratio)

        # initiate the dictionaries for export
        train_dict, dev_dict, test_dict = {}, {}, {}
        dict_list = [train_dict, dev_dict, test_dict]

        # iterate the data list
        for data_list_idx, data_list in enumerate([train_list, dev_list, test_list]):
            
            # iterate the entries in the data list
            for _, entry in enumerate(data_list):
                dict_list[data_list_idx][entry['wav'].split('/')[-1]] = entry

        # write to json file
        with open(f'{self.train_dir}', 'w', encoding='utf-8') as f:
            f.write(json.dumps(train_dict, indent=2))

        with open(f'{self.dev_dir}', 'w', encoding='utf-8') as f:
            f.write(json.dumps(dev_dict, indent=2))

        with open(f'{self.test_dir}', 'w', encoding='utf-8') as f:
            f.write(json.dumps(test_dict, indent=2))

        return train_dict, dev_dict, test_dict

    def __call__(self, train_ratio=0.8, dev_ratio=0.1):
        return self.convert_json_format(train_ratio, dev_ratio)

if __name__ == '__main__':
    # BASE_DIR = '/lid/datasets/mms/mms_silence_removed'
    BASE_DIR = '/lid/datasets/daniel/vox-9langs-small'

    # l = MergeAndSplit(json_dir_list=[f'{BASE_DIR}/manifest_en.json', f'{BASE_DIR}/manifest_ms.json'], 
    #                   train_dir=f'{BASE_DIR}/train_manifest_sb.json', 
    #                   dev_dir=f'{BASE_DIR}/dev_manifest_sb.json', 
    #                   test_dir=f'{BASE_DIR}/test_manifest_sb.json',
    #                   sampling_mode='all')

    l = MergeAndSplit(json_dir_list=[f'{BASE_DIR}/{manifest_lang}' for manifest_lang in os.listdir(BASE_DIR) if manifest_lang.endswith('.json')], 
                      train_dir=f'{BASE_DIR}/train_manifest_sb.json', 
                      dev_dir=f'{BASE_DIR}/dev_manifest_sb.json', 
                      test_dir=f'{BASE_DIR}/test_manifest_sb.json',
                      sampling_mode='all')

    train_dict, dev_dict, test_dict = l(train_ratio=0.8, dev_ratio=0.1)
    print(len(train_dict))
    print(len(dev_dict))
    print(len(test_dict))