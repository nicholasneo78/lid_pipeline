import os
import re
import json
from tqdm import tqdm
from num2words import num2words
from typing import Dict, List

class DigitToTextConversion:
    '''
        reads in an audio manifest file, get the annotation and convert the numbers into words, taking into the account of the audio language as well
    '''

    def __init__(self, input_manifest_dir: str, output_manifest_dir: str, language: str) -> None:
        '''
            manifest_dir: the directory of the manifest that is being preprocessed
            language: the language of the annotation
        '''

        self.input_manifest_dir = input_manifest_dir
        self.output_manifest_dir =output_manifest_dir
        self.language = language

    def normalise_annotation(self, text: str) -> str:
        '''
            to remove unwanted characters in the corpus
        '''
        
        text = re.sub(r'[^A-Za-z\' ]+', ' ', text)

        # remove trailing whitespaces
        text = " ".join(re.split("\s+", text, flags=re.UNICODE))

        # remove the starting and ending whitespace if applicable
        text = text.strip()

        return text

    def replace_numbers(self, text: str) -> str:
        '''
            replaces the numbers to the respective words, taking into the account of the audio language
        '''
        
        re_results = re.findall('\d+', text)
        for term in re_results:
            num = int(term)

            if self.language == 'ms':
                lang = 'id'
                text = text.replace(term, num2words(num, lang=lang))
                text = text.replace('delapan', 'lapan')
                text = text.replace('nol', 'kosong')

            else:
                text = text.replace(term, num2words(num, lang=self.language))

        return text.upper()

    def convert(self) -> List[Dict[str, str]]:
        '''
            main function to do the text preprocesing
        '''

        # load the manifest file
        with open(self.input_manifest_dir, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
        items = [json.loads(line.strip('\r\n')) for line in lines]

        # create a list to store all output entries
        output_list = []
        
        # iterate the list of dictionaries
        for item in items:
            
            # initialise an empty dictionary
            output_dict = {}

            text = self.replace_numbers(item['text'])
            text = self.normalise_annotation(text)

            # append dictionary
            output_dict['audio_filepath'] = item['audio_filepath']
            output_dict['duration'] = item['duration']
            output_dict['text'] = text

            # append entries
            output_list.append(output_dict)

        return output_list

    def __call__(self):

        # get the list of dictionaries
        items = self.convert()

        # export to json file
        with open(self.output_manifest_dir, 'w', encoding='utf-8') as fw:
            for item in items:
                fw.write(json.dumps(item)+'\n')

if __name__ == '__main__':
    c = DigitToTextConversion(input_manifest_dir='/lid/datasets/jtubespeech/ms_2/annotated_data/manifest.json', 
                              output_manifest_dir='/lid/datasets/jtubespeech/ms_2/annotated_data/manifest_updated.json',
                              language='ms')

    c()