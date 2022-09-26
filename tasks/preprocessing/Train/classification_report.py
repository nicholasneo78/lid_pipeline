'''
To get the filename mapping of the index class and print out the classification report
'''

import enum
import os
import pickle
import torch
from torchmetrics import ConfusionMatrix
from sklearn.metrics import classification_report
from typing import Dict, List

class CustomMetrics:
    '''
        a class to take in the actual and predicted class of the audio, produces the confusion matrix and gets the Precision, Recall and F1 for each of the classes
    '''

    def __init__(self, predicted: torch.Tensor, target: torch.Tensor, mapping: List[str]=None) -> None:
        '''
            predicted: the tensor of all the predictions of the audio
            target: the tensor of all the actual label of the audio
        '''
        self.predicted = predicted.to('cpu')
        self.target = target.to('cpu')
        self.mapping = mapping

    def get_confusion_matrix(self) -> torch.Tensor:
        '''
            returns the confusion matrix
        '''
        conf_matrix = ConfusionMatrix(num_classes=len(self.target.unique()))
        print(conf_matrix(self.predicted, self.target))

    def get_f1_report(self) -> None:
        '''
            get the f1 score for all classes
        '''
        print(classification_report(self.target, self.predicted, target_names=self.mapping))

class ClassificationReport:
    def __init__(self, pkl_dir: str, mapping_dir: str) -> None:
        self.pkl_dir = pkl_dir
        self.mapping_dir = mapping_dir

    def get_language_mapping(self) -> Dict[str, int]:
        # read the text file for the mapping
        with open(self.mapping_dir, 'r') as f:
            lines = f.readlines()

        # get the number of languages from the txt file mapping, according to speechbrain LID
        num_lang = len(lines)-2

        # initiate the mapping dictionary
        lang_mapping = {}

        # iterate to form the dictionary mapping
        for idx, line in enumerate(lines):
            if idx < num_lang:
                lang_mapping[int(line[8])] = str(line[1:3])

        return lang_mapping

    def report(self) -> None:
        # load the pickle file
        with open(self.pkl_dir, 'rb') as f:
            results = pickle.load(f)

        # get the language mapping
        lang_mapping = self.get_language_mapping()

        # print the language mapping
        print('LANGUAGE MAPPING\n')
        print(lang_mapping)
        print

        # get the confusion matrix
        metric = CustomMetrics(predicted=results['predicted'], target=results['target'])
        print('CONFUSION MATRIX\n')
        metric.get_confusion_matrix()
        print()

        print('CLASSIFICATION REPORT\n')
        f_metric = CustomMetrics(predicted=results['predicted'], target=results['target'], mapping=[lang_mapping[idx] for idx in lang_mapping])
        f_metric.get_f1_report()
        print()

if __name__ == '__main__':
    cr = ClassificationReport(pkl_dir='output.pkl', 
                              mapping_dir='language_encoder.txt')
    cr.report()