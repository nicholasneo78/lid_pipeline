import os
import json
import logging
import shutil
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import shutil

import torch
from speechbrain.pretrained import EncoderClassifier

class InferManifest:
    '''
        takes in a manifest file containing all the unannotated audio clips, produce the various manifest files based on languages that is based on confidence threshold
    '''

    def __init__(self, input_manifest_dir: str, pretrained_model_root: str, ckpt_folder: str, threshold: Dict[str, float], root_dir_remove_tmp: str, old_dir: str, replaced_dir: str, output_manifest_dir: str, data_batch: str, iteration_num: int) -> None:
        '''
            input_manifest_dir: the manifest file with all the directories to the audio files
            pretrained_model_root: the pretrained speechbrain LID model root, ends with '.../save'
            ckpt_folder: the folder where the pretrained model will be in (full path to the model will be '<pretrained_model_root>/<ckpt_folder>')
            threshold: the dictionary that contains the language as the key and the confidence threshold as the value to place the audio into the predicted language directory or the "others" directory (model not very confident about the prediction) 
            root_dir_remove_tmp: the root directory to remove all the unwanted .wav tmp files
            old_dir: the root or fixed path that is being replaced for the speechbrain format
            replaced_dir: the root directory that is to be replaced for the speechbrain format
            output_manifest_dir: the final manifest directory of the confident prediction
            data_batch: the data batch that is being inferenced
            iteration_num: the number of times the data has iterate already in the training loop
        '''
        self.input_manifest_dir = input_manifest_dir
        self.pretrained_model_root = pretrained_model_root
        self.ckpt_folder = ckpt_folder
        self.threshold = threshold
        self.root_dir_remove_tmp = root_dir_remove_tmp
        self.old_dir = old_dir
        self.replaced_dir = replaced_dir
        self.output_manifest_dir = output_manifest_dir
        self.data_batch = data_batch
        self.iteration_num = iteration_num

        # full path to the model will be '<pretrained_model_root>/<ckpt_folder>'
        self.pretrained_model_path = f'{self.pretrained_model_root}/{self.ckpt_folder}'

    def load_json_manifest(self) -> List[str]:
        '''
            loads the json manifest with all the audio filepaths of the topmost directory to traverse for audio files 
        '''
        # store the manifest entries into a list
        manifest_list = []

        with open(self.input_manifest_dir, mode='r', encoding='utf-8') as fr:
            for idx, line in tqdm(enumerate(fr.readlines())):
                d = json.loads(line)

                manifest_list.append(d)

        return manifest_list

    def shift_language_encoder_file(self) -> None:
        '''
            shifts and rename the language encoder text file to the correct file directory for the code to work 
        '''
        try:
            # shift the encoder text file
            shutil.move(f'{self.pretrained_model_root}/language_encoder.txt', f'{self.pretrained_model_path}/label_encoder.txt')
        except FileNotFoundError:
            # the language encoder file has been shifted
            pass

    def predict_class(self, audio_path: str) -> str:
        '''
            takes in the audio filepath of the audio for inference

            returns the predicted class of the audio and the confidence of the prediction
        '''

        # load the pretrained model, overrides the filepath in the yaml file to the correct one
        model = EncoderClassifier.from_hparams(source=self.pretrained_model_path, overrides={'pretrained_path': self.pretrained_model_path}, savedir='tmp')

        # load the audio for inference
        signal = model.load_audio(audio_path)

        # get the prediction
        prediction = model.classify_batch(signal)

        # get the confidence score (max probability)
        probs = torch.nn.functional.softmax(prediction[0][0], dim=0)
        confidence = max(probs)

        print(confidence)

        # get the prediction from the model
        pred_class = prediction[3][0]
        # check if the confidence score is above a certain threshold, if it is not, classify it as "others" for further investigation
        if confidence >= self.threshold[pred_class]:
            pass
        else:
            pred_class = 'others'

        # to remove unwanted temp .wav file in the root folder after the inference
        root_dir = f'{self.root_dir_remove_tmp}'
        for file in os.listdir(root_dir):
            if file.endswith('.wav'):
                os.remove(os.path.join(root_dir, file))
        
        return pred_class, confidence

    def get_predicted_manifests(self):
        '''
            loads the json manifest to get the audio filepaths, load the audio pointed from the manifest entries, then do the predictions and separate the predictions into their respective language manifests including 'others'
        '''

        # load the manifest list
        manifest_list = self.load_json_manifest()

        # shift the language encoder file into the correct place
        self.shift_language_encoder_file()

        # iterate the list to get the individual entries
        for entry in tqdm(manifest_list):
            # generate prediction
            pred_class, confidence = self.predict_class(entry['audio_filepath'])

            # change the filepath to the speechbrain format
            edit_path = entry['audio_filepath'].replace(self.old_dir, self.replaced_dir)

            # reorganise the dictionary storing the data into the speechbrain format
            data = {'wav': edit_path,
                    'language': pred_class,
                    'duration': entry['duration']}

            # write to json file
            with open(f'{self.output_manifest_dir}/{self.data_batch}_iteration_{self.iteration_num}_{pred_class}.json', 'a+', encoding='utf-8') as f:
                f.write(json.dumps(data) + '\n')

    def __call__(self):
        return self.get_predicted_manifests()

class ConvertToStandardJSON:
    '''
        convert the produced json file into standard format from InferManifest object
    '''

    def __init__(self, input_manifest: str, output_manifest: str) -> None:
        '''
            input_manifest: the manifest needed to be processed
            output_manifest: the manifest produced
        '''

        self.input_manifest = input_manifest
        self.output_manifest = output_manifest

    def convert(self) -> None:
        '''
            actual conversion to be done
        '''

        # initialise a dict to store all the entries
        data_dict = {}

        # iterate the entries and store into the data_list
        with open(self.input_manifest, mode='r', encoding='utf-8') as fr:
            for idx, line in tqdm(enumerate(fr.readlines())):
                d = json.loads(line)
                data_dict[os.path.basename(d['wav'])] = {'wav': d['wav'],
                                                         'language': d['language'],
                                                         'duration': d['duration']}

        # export to the final json format
        with open(f'{self.output_manifest}', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_dict, indent=2))

    def __call__(self) -> None:
        return self.convert()

if __name__ == '__main__':
    infer = InferManifest(input_manifest_dir='/lid/datasets/mms/mms_silence_removed/mms_batch_2s/manifest.json', 
                          pretrained_model_root='/lid/tasks/preprocessing/Train/results/ECAPA-TDNN/2010/save', 
                          ckpt_folder='CKPT+2022-08-31+02-28-36+00',
                          threshold={'en': 0.6, 'ms': 0.6}, 
                          root_dir_remove_tmp='/lid/tasks/preprocessing/Train/', 
                          old_dir='/lid/datasets/mms/mms_silence_removed/', 
                          replaced_dir='{data_root}/', 
                          output_manifest_dir='/lid/datasets/mms/mms_silence_removed/', 
                          data_batch='batch_1s', 
                          iteration_num=1)

    EN_PATH = '/lid/datasets/mms/mms_silence_removed/batch_1s_iteration_1_en.json'
    OTHERS_PATH = '/lid/datasets/mms/mms_silence_removed/batch_1s_iteration_1_others.json'

    c_en = ConvertToStandardJSON(input_manifest=EN_PATH, output_manifest=EN_PATH)
    c_others = ConvertToStandardJSON(input_manifest=OTHERS_PATH, output_manifest=OTHERS_PATH)

    infer()
    c_en()
    c_others()