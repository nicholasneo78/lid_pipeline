import os
import numpy as np
import shutil
from tqdm import tqdm

def create_new_dir(directory: str) -> None:
    '''
        creates new directory and ignore already created ones

        directory: the directory path that is being created
        ---
        returns None
    '''
    try:
        os.mkdir(directory)
    except OSError as error:
        pass # directory already exists!

root_dir = '/mnt/d/datasets/mms/data/'

for audio_dir in tqdm(os.listdir(root_dir)):
    if audio_dir[11:20] == 'CHDIR_495':
        create_new_dir(os.path.join(root_dir, 'channel_1'))
        shutil.move(os.path.join(root_dir, audio_dir), os.path.join(root_dir, 'channel_1'))
        
    elif audio_dir[11:20] == 'CHDIR_497':
        create_new_dir(os.path.join(root_dir, 'channel_2'))
        shutil.move(os.path.join(root_dir, audio_dir), os.path.join(root_dir, 'channel_2'))
        
    elif audio_dir[11:20] == 'CHDIR_154':
        create_new_dir(os.path.join(root_dir, 'channel_3'))
        shutil.move(os.path.join(root_dir, audio_dir), os.path.join(root_dir, 'channel_3'))
        
    elif audio_dir[11:20] == 'CHDIR_155':
        create_new_dir(os.path.join(root_dir, 'channel_4'))
        shutil.move(os.path.join(root_dir, audio_dir), os.path.join(root_dir, 'channel_4'))