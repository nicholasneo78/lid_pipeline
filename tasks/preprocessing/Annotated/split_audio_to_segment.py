import os
import json
import logging
from tqdm import tqdm
import numpy as np
import pydub
from typing import List
from pathlib import Path

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

class SplitAudioToSegment:
    '''
        Split the audio files based on the json annotation
    '''
    
    def __init__(self, annotation_dir: str, raw_base_dir: str, replaced_base_dir: str, original_folder: str, edited_folder: str, sr: int=16000) -> None:
        '''
            annotation_dir: the directory of the annotation json file
            raw_base_dir: the base directory in the json file entries that is to be replaced
            replaced_base_dir: the base directory that is replaced, where the splitted file is produced at
            original_folder: the folder name where the original data resides
            edited_folder: the folder name where the segmented data resides
            sr: sampling rate of the audio
        '''

        self.annotation_dir = annotation_dir
        self.raw_base_dir = raw_base_dir
        self.replaced_base_dir = replaced_base_dir
        self.original_folder = original_folder
        self.edited_folder = edited_folder
        self.sr = sr

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
        idx = [i for i,x in enumerate(dir_split_list) if x == self.original_folder]

        # replace the text to the edited folder path
        dir_split_list[max(idx)] = self.edited_folder
         
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

    def load_annotation(self) -> List:
        '''
            load the annotation json file with the specified directory
        '''
        
        with open(self.annotation_dir, mode='r', encoding='utf-8') as f:
            # reading the whole json file
            data = json.load(f)

        return data

    def split_to_segment(self) -> None:
        '''
            load the raw audio, based on the json annotation, do the split and classify the language 
        '''

        # read the annotation
        for entry in tqdm(self.load_annotation()):
            # do a check on the entry['language'], to see if it is a string (one segment only) or a list of string (more than one segment)
            if isinstance(entry['language'], list):
                pass
            elif isinstance(entry['language'], str):
                entry['language'] = [entry['language']]

            # change the directory in the json annotation to the directory that points to the audio file
            full_audio_dir = entry['audio'].replace(self.raw_base_dir, self.replaced_base_dir) 

            # create the nested directory for the audio file to reside in
            edited_dir, filename = self.create_nested_dir(full_audio_dir)

            # import the audio file based on the modified audio directory
            sound_file = pydub.AudioSegment.from_wav(full_audio_dir)
            sound_file_value = np.array(sound_file.get_array_of_samples())

            # iterate the number of segments to obtain the start and end
            for idx_seg, language in enumerate(entry['language']):
                start = int(entry['labels'][idx_seg]['start']*self.sr)
                end = int(entry['labels'][idx_seg]['end']*self.sr)

                if language == 'english':
                    lang_id = 'en'
                elif language == 'bahasa':
                    lang_id = 'ms'

                self.create_new_dir(f'{edited_dir}/{lang_id}/')

                audio_segment = pydub.AudioSegment(sound_file_value[start:end].tobytes(), 
                                                   frame_rate=sound_file.frame_rate, 
                                                   sample_width=sound_file.sample_width,
                                                   channels=1)

                audio_segment.export(f'{edited_dir}/{lang_id}/' + Path(filename).stem + f'_{idx_seg}.wav', format='wav')

    def __call__(self) -> None:
        return self.split_to_segment()


if __name__ == '__main__':
    split = SplitAudioToSegment(annotation_dir='/lid/datasets/mms/mms/mms_batch_1/annotations/mms_20220417.json', 
                                raw_base_dir='/data/local-files/?d=local_data/', 
                                replaced_base_dir='/lid/datasets/mms/mms/mms_batch_1/', 
                                original_folder='mms', 
                                edited_folder='mms_split_audio_segment', 
                                sr=16000)

    split()
