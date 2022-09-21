'''
merge the json manifests into one for easy loading of data afterwards
'''

import json
import os
from tqdm import tqdm

class MergeJSONManifest:
    '''
        a class that merges the json manifests for easier preprocessing of data later
    '''

    def __init__(self, manifest_root: str, output_manifest: str, raw_root_dir: str, replaced_dir: str) -> None:
        self.manifest_root = manifest_root
        self.output_manifest = output_manifest
        self.raw_root_dir = raw_root_dir
        self.replaced_dir = replaced_dir

    def merge(self) -> None:
        # iterate the root folder to find all the json files
        for root, subdirs, files in os.walk(self.manifest_root):
            for file in files:
                if file.endswith('.json'):
                    # preprocess the json to extract the filepath
                    with open(f'{root}/{file}', 'r') as f:
                        data_dict = json.load(f)

                        # iterate the dictionary
                        for key in tqdm(data_dict):
                            # replace the filepath root
                            filepath_raw = data_dict[key]['wav']
                            filepath_new = filepath_raw.replace(self.raw_root_dir, self.replaced_dir)

                            # temp dictionary to be stored in the json file
                            temp_dict = {'audio_filepath': filepath_new,
                                         'language': data_dict[key]['language'],
                                         'duration': data_dict[key]['duration']}

                            # write to the final manifest 
                            with open(f'{self.output_manifest}', 'a+', encoding='utf-8') as f:
                                f.write(json.dumps(temp_dict) + '\n')

    def __call__(self):
        return self.merge()

if __name__ == '__main__':
    m = MergeJSONManifest(manifest_root='/lid/datasets/mms/mms_final/ms_speech', 
                          output_manifest='/lid/datasets/mms/mms_final/ms_speech/ms_full_manifest.json', 
                          raw_root_dir='{data_root}', 
                          replaced_dir='/lid/datasets/mms/mms_silence_removed')

    m()