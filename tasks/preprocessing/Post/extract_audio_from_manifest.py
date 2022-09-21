'''
To extract the audio based on the manifest
'''

import os
import json
from tqdm import tqdm
import pydub
from typing import List, Dict

class ExtractAudioFromManifest:
    def __init__(self, manifest_path: str, root_audio_dir: str, batch_dir: List[str], date_dir: Dict[str, List[str]], channel_list: List[str], language: str) -> None:
        self.manifest_path = manifest_path
        self.root_audio_dir = root_audio_dir
        self.batch_dir = batch_dir
        self.date_dir = date_dir
        self.channel_list = channel_list
        self.language = language

    def create_new_dir(self, directory: str) -> None:
        '''
            creates new directory and ignore already created ones

            directory: the directory path that is being created
        '''
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!

    def extract(self) -> None:
        # read the manifest file
        with open(self.manifest_path, mode='r', encoding='utf-8') as fr:
            for idx, line in tqdm(enumerate(fr.readlines())):
                try:
                    d = json.loads(line)

                    # get the audio filepath from the manifest file
                    audio_path = d['audio_filepath']

                    # load the audio and save it to another directory
                    audio = pydub.AudioSegment.from_wav(audio_path)

                    # determine which batch the data belongs to
                    for batch in self.batch_dir:
                        if audio_path.find(batch) != -1:
                            for date in self.date_dir[batch]:
                                if audio_path.find(date) != -1:
                                    for channel in self.channel_list:
                                        if audio_path.find(channel) != -1:
                                            self.create_new_dir(f'{self.root_audio_dir}/{batch}')
                                            self.create_new_dir(f'{self.root_audio_dir}/{batch}/{date}')
                                            self.create_new_dir(f'{self.root_audio_dir}/{batch}/{date}/{channel}')
                                            self.create_new_dir(f'{self.root_audio_dir}/{batch}/{date}/{channel}/{self.language}')
                                            audio.export(f'{self.root_audio_dir}/{batch}/{date}/{channel}/{self.language}/{os.path.basename(audio_path)}', format='wav')
                                        else:
                                            continue
                                else:
                                    continue
                        else:
                            continue
                except:
                    print('error in getting the audio file')

    def __call__(self) -> None:
        return self.extract()

if __name__ ==  '__main__':
    date_dict= {
        'mms_batch_0': ['mms_20220110', 'mms_20220130', 'mms_20220131', 'mms_20220201', 'mms_20220202', 'mms_20220204', 'mms_20220304'],
        'mms_batch_1': ['mms_20220404', 'mms_20220501', 'mms_20220520', 'mms_20220529', 'mms_20220610'],
        'mms_batch_2': ['mms_20220612', 'mms_20220620', 'mms_20220624', 'mms_20220627', 'mms_20220628', 'mms_20220629', 'mms_20220630'],
        'mms_batch_3': ['mms_20220705', 'mms_20220706', 'mms_20220710', 'mms_20220714', 'mms_20220715', 'mms_20220716', 'mms_20220725', 'mms_20220726', 'mms_20220727'],
        'mms_batch_4': ['mms_20220502', 'mms_20220507', 'mms_20220510', 'mms_20220513', 'mms_20220801', 'mms_20220802'],
        'mms_batch_5': ['mms_20220807', 'mms_20220809', 'mms_20220811', 'mms_20220812', 'mms_20220813', 'mms_20220814', 'mms_20220815', 'mms_20220816', 'mms_20220817']
        }

    channel_list = ['CH 10', 'CH 14', 'CH 16', 'CH 73']

    e = ExtractAudioFromManifest(manifest_path='/lid/datasets/mms/mms_final/ms_speech/ms_full_manifest.json', 
                                 root_audio_dir='/lid/datasets/mms/mms_ms', 
                                 batch_dir=['mms_batch_0', 'mms_batch_1', 'mms_batch_2', 'mms_batch_3', 'mms_batch_4', 'mms_batch_5'],
                                 date_dir=date_dict,
                                 channel_list=channel_list,
                                 language='ms')

    e()

