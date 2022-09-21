'''
To extract the audio based on the manifest
'''

import os
import json
from tqdm import tqdm
import pydub
from typing import List

class ExtractAudioFromManifest:
    def __init__(self, manifest_path: str, root_audio_dir: str, batch_dir: List[str]) -> None:
        self.manifest_path = manifest_path
        self.root_audio_dir = root_audio_dir
        self.batch_dir = batch_dir

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
                        if audio_path.find(batch) != 1:
                            self.create_new_dir(f'{self.root_audio_dir}/{batch}')
                            audio.export(f'{self.root_audio_dir}/{batch}/{os.path.basename(audio_path)}')
                        else:
                            continue
                except:
                    print('error in getting the audio file')

    def __call__(self) -> None:
        return self.extract()

if __name__ ==  '__main__':
    e = ExtractAudioFromManifest(manifest_path='/lid/datasets/mms/mms_final/ms_speech/ms_full_manifest.json', 
                                 root_audio_dir='/lid/datasets/mms/mms_ms', 
                                 batch_dir=['mms_batch_0', 'mms_batch_1', 'mms_batch_2', 'mms_batch_3', 'mms_batch_4', 'mms_batch_5'])

    e()

