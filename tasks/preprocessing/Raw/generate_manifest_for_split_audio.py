import os
from os.path import join
import numpy as np
import json
import librosa
from pathlib import Path
from tqdm import tqdm

class GenerateManifestForSplitAudio:
    '''
        to produce a general manifest for each batch of data and place that manifest in the root of the folder
    '''

    def __init__(self, root_folder: str, manifest_filename: str) -> None:
        '''
            root_folder: the topmost directory to traverse the audio file
            manifest_filename: the manifest generated in the root folder, place it in the root folder directory
        '''
        self.root_folder = root_folder
        self.manifest_filename = manifest_filename

    def get_manifest(self):
        # walk the root directory and search for the individual manifest files (.json)
        for root, subdirs, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith('.json'):
                    # load the json first
                    with open(os.path.join(root, file), mode='r', encoding='utf-8') as fr:
                        for idx, line in tqdm(enumerate(fr.readlines())):
                            d = json.loads(line)
                            try:
                                data = {
                                    'audio_filepath': os.path.join(root, d['audio_filepath']),
                                    'duration': librosa.get_duration(filename=os.path.join(root, d['audio_filepath'])),
                                    'text': ""
                                }

                                # write to the json manifest file
                                with open(f'{self.manifest_filename}', 'a+', encoding='utf-8') as f:
                                    f.write(json.dumps(data) + '\n')
                            except FileNotFoundError:
                                continue

        return f'{self.manifest_filename}'

    def __call__(self):
        return self.get_manifest()

if __name__ == '__main__':
    g = GenerateManifestForSplitAudio(root_folder='/lid/datasets/mms/mms_silence_removed/mms_batch_2s/', 
                                      manifest_filename='/lid/datasets/mms/mms_silence_removed/mms_batch_2s/manifest.json')

    manifest_path = g()