from clearml import Task, Dataset

# start clearml
task = Task.init(project_name='LID', task_name='infer', output_uri='s3://experiment-logging')
task.set_base_docker(
    docker_image='nicholasneo78/sb_lid:v0.0.2',
)

# execute clearml
task.execute_remotely(queue_name='compute', exit_process=True)

from preprocessing.Train.infer_manifest import InferManifest, ConvertToStandardJSON
import shutil
import os

# the clearml dataset ID for all the datasets
PRETRAINED_MODEL_ID = '23ae6fbf80ec489ca3c17591552d6427'
HYPERPARAMS_YAML_ID = '22ffe0aba2594c3abaf3251f2b16af5c'
DATASET_ID = 'a8872c8f04444a75b7e1436a72a534e4'
CKPT_PATH = 'CKPT+2022-08-31+11-25-40+00'
MANIFEST_ROOT = 'output'

# load the model that is trained previously
get_model = Dataset.get(dataset_id=PRETRAINED_MODEL_ID)
model_root_path = get_model.get_local_copy()

# load the hyperparams config file that is needed for the inference
get_hyperparams = Dataset.get(dataset_id=HYPERPARAMS_YAML_ID)
hyperparams_root_path = get_hyperparams.get_local_copy()

# load the dataset for inference
get_dataset = Dataset.get(dataset_id=DATASET_ID)
dataset_root_path = get_dataset.get_local_copy()

# move the config file into the save folder in order for the code to run correctly
shutil.move(f'{hyperparams_root_path}/hyperparams.yaml', f'{model_root_path}/save/{CKPT_PATH}/hyperparams.yaml')

# create a new directory in the remote folder to store the classified json file
os.mkdir(f'{MANIFEST_ROOT}/')

### executing the code ###
THRESHOLD_DICT = {'en': 0.6, 'ms': 0.6}
OLD_DIR = '/lid/datasets/mms/mms_silence_removed/'
DATA_BATCH = 'batch_1s'
ITERATION = 'iteration_1'
LANG_LIST = ['en', 'ms', 'others']

infer = InferManifest(input_manifest_dir=f'{dataset_root_path}/mms_{DATA_BATCH}/manifest.json', 
                      pretrained_model_root=f'{model_root_path}/save', 
                      ckpt_folder=CKPT_PATH,
                      threshold=THRESHOLD_DICT, 
                      root_dir_remove_tmp=dataset_root_path, 
                      old_dir=OLD_DIR, # the manifest path where the data path is being uploaded locally
                      inference_replaced_dir=f'{dataset_root_path}/',
                      new_manifest_replaced_dir='{data_root}/', 
                      output_manifest_dir=MANIFEST_ROOT, 
                      data_batch=DATA_BATCH,
                      iteration=ITERATION)

infer()

# convert json to standard format
for lang in LANG_LIST:
    LANG_PATH = f'{MANIFEST_ROOT}/{DATA_BATCH}_{ITERATION}_{lang}.json'

    # in case there is no prediction for a certain language, hence no file produced
    try:
        c = ConvertToStandardJSON(input_manifest=LANG_PATH, output_manifest=LANG_PATH)
        c()
    except FileNotFoundError:
        print(f'no audio predicted for this language - {lang}')
        continue

# create dataset to store the new manifests
dataset = Dataset.create(
    dataset_project='datasets/LID',
    dataset_name=f'inference_{ITERATION}',
    parent_datasets=[DATASET_ID]
)

dataset.add_files(path=F'{MANIFEST_ROOT}/')
dataset.upload(output_url="s3://experiment-logging")
dataset.finalize()

print('Done')