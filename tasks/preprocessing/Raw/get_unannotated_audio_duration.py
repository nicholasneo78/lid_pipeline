import os
import json
import logging

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

# gets audio durations from a specific channel via the generated manifest
class GetAudioInfo:
    def __init__(self, audio_dir: str, manifest_dir: str) -> None:
        self.audio_dir = audio_dir
        self.manifest_dir = manifest_dir

    def get_audio_duration_from_manifest(self):
        audio_counts = 0
        audio_duration_counts = 0

        with open(self.manifest_dir, mode='r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr.readlines()):
                d = json.loads(line)

                # append to the audio duration counts
                audio_duration_counts += d['duration']

                # add on to the count to the number of audio files in this folder (with sound)
                audio_counts += 1

        # convert total number of seconds of audio collected to number of hours collected
        audio_duration_counts_hr = audio_duration_counts/3600

        return audio_duration_counts, audio_duration_counts_hr, audio_counts

    # get the number of audio files (including the empty audio files -> audio files with 0 second)
    def get_actual_number_of_audio_files(self):
        return len([s for s in os.listdir(self.audio_dir) if s.endswith('.wav')])

    def __call__(self):
        audio_s, audio_hr, audio_count_no_empty_audio = self.get_audio_duration_from_manifest()
        audio_count_actual = self.get_actual_number_of_audio_files()
        return audio_s, audio_hr, audio_count_no_empty_audio, audio_count_actual

if __name__ == '__main__':

    # choose the mode to get the channel informations
    MODE = 'one_date' # one_dir or one_date

    # other configs
    # dataset_dir = 'mms' # 'data_to_i2r' 
    dataset_dir = 'mms_silence_removed'
    batch = 'mms_batch_7' # 'mms_set_1' 
    batch_date = 'mms_20220831' # 'en'

    # check info for a single directory, specific date and channel
    if MODE == 'one_dir':

        channel = 'CH 10'
        AUDIO_DIR = f'/lid/datasets/mms/{dataset_dir}/{batch}/{batch_date}/{channel}'
        MANIFEST_DIR = f'/lid/datasets/mms/{dataset_dir}/{batch}/{batch_date}/{channel}/manifest.json'

        audio_details = GetAudioInfo(audio_dir=AUDIO_DIR, 
                                     manifest_dir=MANIFEST_DIR)

        audio_length_s, audio_length_hr, num_audio, actual_num_audio = audio_details()

        print()
        # logging.getLogger('Number of Audio').info(f'{dataset_dir} - {batch} - {batch_date} - {channel} (without empty audio): {num_audio}')
        logging.getLogger('Number of Audio').info(f'{dataset_dir} - {batch} - {batch_date} - {channel} (all audio files in the folder): {actual_num_audio}')
        # logging.getLogger('Total Audio Length').info(f'{dataset_dir} - {batch} - {batch_date} - {channel} (s): {audio_length_s}')
        logging.getLogger('Total Audio Length').info(f'{dataset_dir} - {batch} - {batch_date} - {channel}  (h): {audio_length_hr}')

        print()
    
    # check info for a single date
    elif MODE == 'one_date':

        channel_list = ['CH 10', 'CH 14', 'CH 16', 'CH 73']
        # channel_list = ['CH 10/en', 'CH 10/ms', 'CH 10/others', 'CH 14/en', 'CH 14/ms', 'CH 14/others', 'CH 16/en', 'CH 16/ms', 'CH 16/others', 'CH 73/en', 'CH 73/ms', 'CH 73/others']
        for channel in channel_list:
            try:
                AUDIO_DIR = f'/lid/datasets/mms/{dataset_dir}/{batch}/{batch_date}/{channel}'
                MANIFEST_DIR = f'/lid/datasets/mms/{dataset_dir}/{batch}/{batch_date}/{channel}/manifest.json'

                audio_details = GetAudioInfo(audio_dir=AUDIO_DIR, 
                                             manifest_dir=MANIFEST_DIR)

                audio_length_s, audio_length_hr, num_audio, actual_num_audio = audio_details()

                print()
                # logging.getLogger('Number of Audio').info(f'{dataset_dir} - {batch} - {batch_date} - {channel} (without empty audio): {num_audio}')
                logging.getLogger('Number of Audio').info(f'{dataset_dir} - {batch} - {batch_date} - {channel} (all audio files in the folder): {actual_num_audio}')
                # logging.getLogger('Total Audio Length').info(f'{dataset_dir} - {batch} - {batch_date} - {channel} (s): {audio_length_s}')
                logging.getLogger('Total Audio Length').info(f'{dataset_dir} - {batch} - {batch_date} - {channel}  (h): {audio_length_hr}')
            
            except FileNotFoundError:
                print()
                logging.getLogger('EMPTY FOLDER ALERT').info(f'{dataset_dir} - {batch} - {batch_date} - {channel}: Channel with no audio files')
                continue
        
        print()

