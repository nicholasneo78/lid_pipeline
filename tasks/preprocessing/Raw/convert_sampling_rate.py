import soundfile as sf
import librosa
import os
import json
from tqdm import tqdm
import logging

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

class ConvertSamplingRate:
    def __init__(self, manifest_path: str, input_data_dir: str, output_data_dir: str, target_sr: int) -> None:
        self.manifest_path = manifest_path
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.target_sr = target_sr

    def convert(self) -> None:
        with open(self.manifest_path, mode='r', encoding='utf-8') as fr:
            for idx, line in tqdm(enumerate(fr.readlines())):
                try:
                    d = json.loads(line)

                    # get the audio filepath from the manifest file
                    audio_path = d['audio_filepath']
                    speech_array, sr = librosa.load(os.path.join(self.input_data_dir , audio_path), sr=None)
                    speech_array_16k = librosa.resample(speech_array, orig_sr=sr, target_sr=self.target_sr)
                    
                    # overwrite the sound with the target sample rate
                    sf.write(os.path.join(self.output_data_dir, audio_path), speech_array_16k, self.target_sr, subtype='FLOAT')
                except:
                    logging.getLogger('No audio').info(f'No audio files found in this directory')
                    continue

    def __call__(self) -> None:
        return self.convert()

if __name__ == '__main__':

    # other configs
    dataset_dir = 'mms'
    batch = 'mms_batch_5'
    root_dir = f'/lid/datasets/mms/{dataset_dir}/{batch}'
    batch_date_list = ['mms_20220814', 'mms_20220815', 'mms_20220816', 'mms_20220817']
    # batch_date_list =  [d for d in os.listdir(root_dir)] # ['mms_20220404']
    channel_list = ['CH 10', 'CH 14', 'CH 16', 'CH 73']

    for batch_date in batch_date_list:
        for channel in channel_list:
            try:
                c = ConvertSamplingRate(manifest_path=f'/lid/datasets/mms/mms/{batch}/{batch_date}/{channel}/manifest.json',
                                        input_data_dir=f'/lid/datasets/mms/mms/{batch}/{batch_date}/{channel}',
                                        output_data_dir=f'/lid/datasets/mms/mms/{batch}/{batch_date}/{channel}',
                                        target_sr=16000)
                c()

                logging.getLogger('Progress').info(f'Completed: {dataset_dir} - {batch} - {batch_date} - {channel}')

            except FileNotFoundError:
                logging.getLogger('No audio folder').info(f'{dataset_dir} - {batch} - {batch_date} - {channel}: No audio files found in this directory')
                continue
    
    logging.getLogger('DONE').info('Done')