import os
import json
import logging
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict

class CombineTrainManifest:
    def __init__(self, manifest_list:List[str], output_manifest:str) -> None:
        self.manifest_list = manifest_list
        self.output_manifest = output_manifest

    def combine(self) -> None:
        # initialise a dict to store all the entries
        final_data_dict = {}

        # iterate through the manifests for the train set
        for manifest in self.manifest_list:

            # iterate the entries and store into the data_list
            with open(manifest, mode='r', encoding='utf-8') as f:
                data_dict = json.load(f)

                for data_key in tqdm(data_dict):
                    final_data_dict[data_key] = data_dict[data_key]



        # export to the final json format
        with open(f'{self.output_manifest}', 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_data_dict, indent=2))

    def __call__(self):
        return self.combine()

if __name__ == '__main__':
    MANIFEST_ROOT = '/lid/datasets/mms/mms_silence_removed'
    FILELIST = ['train_manifest_sb', 'batch_1s_iteration_1_en', 'batch_2s_iteration_1_en']

    c = CombineTrainManifest(manifest_list=[f'{MANIFEST_ROOT}/{f}.json' for f in FILELIST],
                             output_manifest=f'{MANIFEST_ROOT}/train_manifest_sb_2.json')

    c()