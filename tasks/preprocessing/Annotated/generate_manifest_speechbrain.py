import os
import json
import librosa

class GenerateManifestSpeechBrain:
    '''
        build a class to produce the manifest to take in the data in speechbrain format
    '''

    def __init__(self, root_folder, manifest_filename, removed_dir, replaced_dir, target_language, audio_ext):
        self.root_folder = root_folder
        self.manifest_filename = manifest_filename
        self.removed_dir = removed_dir
        self.replaced_dir = replaced_dir
        self.target_language = target_language
        self.audio_ext = audio_ext

    # check if the json file name already existed (if existed, need to throw error or else the new json manifest will be appended to the old one, hence causing a file corruption)
    def json_existence(self):
        assert not os.path.isfile(f'{self.manifest_filename}'), "json filename exists! Please remove old json file!"

    # create the json manifest
    def create_json_manifest(self):
        
        # check if the json filename have existed in the directory
        self.json_existence()

        error_count = 0

        # create a list to store the dictionaries
        data_list = []

        # read the filepaths of the audio file
        for root, subdirs, files in os.walk(self.root_folder):
            for file in files:

                # remove the front part of the directory to suit the speechbrain case
                modified_root = root.replace(self.removed_dir, self.replaced_dir)

                if file.endswith(self.audio_ext):
                    if f'/{self.target_language}/' in os.path.join(root, file):
                        language = self.target_language
                    else:
                        continue

                    try:
                        # create the dictionary that is to be appended to the json file
                        data = {
                                'wav': os.path.join(modified_root, file),
                                'language': language,
                                'duration': librosa.get_duration(filename=os.path.join(root, file))
                            }

                        # append the data
                        data_list.append(data)
            
                    # for corrupted file of the target extension                
                    except:
                        error_count+=1
                        print(f"Error loading {file}")
                        continue

        # write to json file
        with open(f'{self.manifest_filename}', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_list, indent=2))
        
        print(f'Total number of errors: {error_count}')
        return f'{self.manifest_filename}'

    def __call__(self):
        return self.create_json_manifest()


if __name__ == '__main__':

    LANGUAGES = ['ms', 'en']

    for lang in LANGUAGES:
        get_manifest = GenerateManifestSpeechBrain(root_folder='/lid/datasets/mms/mms_split_audio_segment/', 
                                                manifest_filename=f'/lid/datasets/mms/mms_split_audio_segment/manifest_{lang}.json', 
                                                removed_dir='/lid/datasets/mms/mms_split_audio_segment/',
                                                replaced_dir='{data_root}/', #'/workspace/datasets/mms/', #'{root}/',
                                                target_language=lang,
                                                audio_ext='.wav')

        get_manifest()