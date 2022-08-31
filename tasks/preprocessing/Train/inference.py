import os
import subprocess
from typing import Union, List
from speechbrain.pretrained import EncoderClassifier

def standardize(audiopath: str, channels: int=1, samplerate: int = 16000, outdir: str = './tmp6') -> str:
    outpath = os.path.join(outdir, os.path.basename(audiopath))
    subprocess.run([
        'sox', audiopath, 
        '-c', str(channels),
        '-r', str(samplerate),
        outpath
    ])
    return outpath

def load_model(source: str, savedir: str = 'tmp6', device: str = 'cuda') -> EncoderClassifier:
    if not os.path.exists(source):
        pass
    else:
        return EncoderClassifier.from_hparams(source, savedir=savedir,  run_opts={'device': device})

def infer(model: EncoderClassifier, audiopaths: Union[str, List[str]], channels: int = 1, samplerate: int = 16000, batchsize: int = 8, device: str ='cuda') -> str:

    if type(audiopaths) is str:
        # if only one audio path
        audiopaths = [audiopaths,]

    standardized_audiopaths = []
    for audiopath in audiopaths:
        standardized_audiopaths.append(standardize(audiopath, channels, samplerate))

    signals = [model.load_audio(p) for p in standardized_audiopaths]
    predictions = []

    for idx, signal in enumerate(signals):
        print(f'Processing index {idx}', end='\r')    
        scores, best_llh, best_pos, best_langs =  model.classify_batch(signal.to(device))
        predictions.append(best_langs[0])

    print('\n')
    # print(predictions)

    return predictions

if __name__ == '__main__':

    SOURCE_MODEL = '/lid/tasks/preprocessing/Train/results/ECAPA-TDNN/2010/save/CKPT+2022-08-31+02-28-36+00/' # 'speechbrain/lang-id-voxlingua107-ecapa'
    SIGNAL_PATH = '/lid/datasets/mms/mms_silence_removed/mms_batch_1s/mms_20220404/CH 10/0000000000_CHDIR_495_2022-04-04_00_02_08_0.wav' #'https://omniglot.com/soundfiles/udhr/udhr_th.mp3'

    model = load_model(SOURCE_MODEL)
    lang_prediction = infer(model, SIGNAL_PATH)

    print(lang_prediction)