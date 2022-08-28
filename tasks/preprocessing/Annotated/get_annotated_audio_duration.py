import json
import logging
from tqdm import tqdm

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

class GetAnnotatedAudioDuration:
    '''
        Get the duration of the annotated audio in the target language, with the json annotation file
    '''

    def __init__(self, annotation_filepath: str, channel: str, target_language: str, display_duration: str) -> None:
        '''
            annotation_filepath: the directory of the json file containing the language labelling
            channel: the channel of interest to calculate the duration of the audio
            target_language: the language of interest to calculate the duration of the audio
            display_duration: to display the duration of audio collected in h, m or s
        '''
        
        self.annotation_filepath = annotation_filepath
        self.channel = channel
        self.target_language = target_language
        self.display_duration = display_duration

    def language_duration(self) -> None:
        
        # initiate the audio duration count for that particular language
        specific_audio_duration = 0
        total_audio_duration = 0
        annotated_clip_count = 0

        with open(self.annotation_filepath, mode='r', encoding='utf-8') as f:
            # # reading an entry
            # for _, line in tqdm(enumerate(f.readlines())):
            d = json.load(f)

            for entry in tqdm(d):
                # do a check on the entry['language'], to see if it is a string (one segment only) or a list of string (more than one segment)
                if isinstance(entry['language'], list):
                    pass
                elif isinstance(entry['language'], str):
                    entry['language'] = [entry['language']]

                # iterate the number of segments
                for idx_seg, segment in enumerate(entry['language']):
                    # check for the specified channel (find from the filepath stated in the json file)
                    if entry['audio'].find(self.channel) != -1:
                        annotated_clip_count += 1
                        # get the total audio duration here (no matter what language it is being specified)
                        total_audio_duration += (entry['labels'][idx_seg]['end'] - entry['labels'][idx_seg]['start'])

                        # check for the specified target language
                        if segment == self.target_language:
                            specific_audio_duration += (entry['labels'][idx_seg]['end'] - entry['labels'][idx_seg]['start'])

        # get the total audio duration that is being annotated
        logging.getLogger('Total Audio Length').info(f'audio duration for {self.channel} (h): {total_audio_duration/3600}')

        # to display in h, m or s format
        if self.display_duration == 's':
            logging.getLogger('Specific Audio Length').info(f'audio duration for {self.channel} - {self.target_language} (s): {specific_audio_duration}')
        elif self.display_duration == 'm':
            logging.getLogger('Specific Audio Length').info(f'audio duration for {self.channel} - {self.target_language} (min): {specific_audio_duration/60}')
        elif self.display_duration == 'h':
            logging.getLogger('Specific Audio Length').info(f'audio duration for {self.channel} - {self.target_language} (h): {specific_audio_duration/3600}')
        else:
            logging.getLogger('Value Error').info('check display parameter again, either h, m or s only')

        logging.getLogger('Annotated clip count').info(f'number of clips annotated on {self.channel}: {annotated_clip_count}')

    def __call__(self):
        return self.language_duration()

if __name__ == '__main__':

    JSON_FILE = '/lid/datasets/mms/mms/mms_batch_1/annotations/mms_20220430.json'
    CHANNEL_LIST = ['CH 10', 'CH 14', 'CH 16', 'CH 73']

    print()
    
    for ch in CHANNEL_LIST:
        # english
        l = GetAnnotatedAudioDuration(annotation_filepath=JSON_FILE, 
                                      channel=ch,
                                      target_language='english',
                                      display_duration='h')

        l()

        # bahasa
        l = GetAnnotatedAudioDuration(annotation_filepath=JSON_FILE, 
                                      channel=ch,
                                      target_language='bahasa',
                                      display_duration='m')

        l()

        print()