import os
import json
import soundfile as sf
import librosa
from typing import List
from pydub import AudioSegment
from pydub.silence import split_on_silence

class SilenceSplitter():
    def __init__(self, thresh: int = 16, min_silence_len: int = 500) -> None:
        '''
        thresh: the silence threshold according to decibels relative to full scale (dBFS)
        min_silence_len: The minimum length of silence before splitting (in ms)
        '''
        self.thresh = thresh
        self.min_silence_len = min_silence_len

    def silence_split(self, path: str) -> List[AudioSegment]:
        '''
        path: string path to the audio file
        '''

        # resample the audio to 16k to fulfil the wav2vec2 framework
        audio = AudioSegment.from_wav(path)

        # decibels relative to full scale
        dBFS = audio.dBFS
        silence_thresh = dBFS - self.thresh
        chunks = split_on_silence(
            audio,
            min_silence_len = self.min_silence_len,
            silence_thresh = silence_thresh
        )

        return chunks

    def batch_silence_split(self, input_dir: str, output_dir: str, manifest_path: str)-> str:
        '''
        input_dir: the input directory
        output_dir: the output directory
        manifest_path: path to manifest file from the input_dir
        thresh: the silence threshold according to decibels relative to full scale (dBFS)
        min_silence_len: The minimum length of silence before splitting (in ms)
        '''
        os.makedirs(output_dir, exist_ok=True)
        print('Batch processing of silence splits...')
        with open(os.path.join(input_dir, manifest_path), mode='r', encoding='utf-8') as fr, \
            open(os.path.join(output_dir, manifest_path), mode='w', encoding='utf-8') as fw:
            total_num_chunks = 0
            del_chunks = 0

            for idx, line in enumerate(fr.readlines()):
                if idx % 100 == 0 and idx != 0:
                    print(f'no. of files processed: {idx}')
                d = json.loads(line)
                orig_path = d['audio_filepath']
                os.makedirs(os.path.join(output_dir, os.path.dirname(d['audio_filepath'])), exist_ok=True)

                audio_chunks = self.silence_split(os.path.join(input_dir, orig_path))
                total_num_chunks += len(audio_chunks)
                for chunk_idx, chunk in enumerate(audio_chunks):
                    chunk_path = orig_path.replace('.wav', f'_{str(chunk_idx)}.wav')

                    # export the chunk as a wav file
                    chunk.export(
                        os.path.join(output_dir, chunk_path),
                        format = 'wav'
                    )

                    # check if the audio < 1 second, if it is, delete and do not append to the manifest
                    if librosa.get_duration(filename=os.path.join(output_dir, chunk_path)) < 1.0:
                        os.remove(os.path.join(output_dir, chunk_path))
                        del_chunks+=1                
                        continue

                    # replace audio_filepath with chunk path, and its length
                    d['audio_filepath'] = chunk_path
                    d['duration'] = round(
                        librosa.get_duration(filename=os.path.join(output_dir, chunk_path)), 3)
                    fw.write(json.dumps(d) + '\n')

        print('total no. of files processed:', idx+1)
        print('total no. of files produced:', total_num_chunks-del_chunks)
        
        # returns dataset dir and new manifest file path
        return output_dir, os.path.join(output_dir, manifest_path)

    def __call__(self, input_dir: str, output_dir: str = 'temp', manifest_path: str = 'manifest.json'):
        return self.batch_silence_split(input_dir, output_dir, manifest_path)

if __name__ == '__main__':

    dataset_input_dir = 'mms'
    dataset_output_dir = 'mms_silence_removed'
    batch = 'mms_batch_8'
    # batch_date = [date for date in os.listdir(f'/lid/datasets/mms/{dataset_input_dir}/{batch}')]
    # batch_date = ['mms_20221001', 'mms_20221002', 'mms_20221003', 'mms_202 21004', 'mms_20221005']
    batch_date = ['mms_20221006', 'mms_20221007', 'mms_20221008', 'mms_20221009', 'mms_20221010']
    channel_list = ['CH 10', 'CH 14', 'CH 16', 'CH 73']

    for date in batch_date:
        for channel in channel_list:
            try:
                LOCAL_DIR = f'/lid/datasets/mms/{dataset_input_dir}/{batch}/{date}/{channel}'
                OUTPUT_DIR = f'/lid/datasets/mms/{dataset_output_dir}/{batch}/{date}/{channel}'

                s = SilenceSplitter(thresh=16, min_silence_len=500)
                s(LOCAL_DIR, OUTPUT_DIR)
                print()

            except FileNotFoundError:
                print('No audio found in this directory')
                print()
                continue
