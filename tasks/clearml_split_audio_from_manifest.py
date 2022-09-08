from clearml import Task, Dataset

### configs for starting clearml ###
PROJ_NAME = 'LID'
ITER = 2
TASK_NAME = 'split_audio_from_manifest'
DOCKER_IMG = 'nicholasneo78/sb_lid:v0.0.2'
QUEUE = 'compute'
####################################

### configs to get the clearml dataset ID ###
DATASET_ID = 'a8872c8f04444a75b7e1436a72a534e4'
TRAIN_MANIFEST_ID = 'abeee357e9f14dbbb3729cbfaa032d4c'
MANIFEST_ID_LIST = ['d54ffb239345410fb7e37e0a43b44ea3', 
                    '898dcc04c9ad476e8071b0d7ffd3b45b']

MANIFEST_ROOT = 'output'
#############################################

### configs to store the new dataset to save the updated train manifest ###
DATASET_PROJ_NAME = 'datasets/LID'
DATASET_NAME = 'split_audio'
###########################################################################

# start clearml
task = Task.init(project_name=PROJ_NAME, task_name=TASK_NAME, output_uri='s3://experiment-logging')
task.set_base_docker(
    docker_image=DOCKER_IMG,
)

# execute clearml
task.execute_remotely(queue_name=QUEUE, exit_process=True)

from preprocessing.Train.split_audio_from_manifest import SplitAudio
import os

# initiate a list to store the manifest root path for each of the batches
root_path_list = []

# get the dataset path
get_dataset = Dataset.get(dataset_id=DATASET_ID)
get_dataset_root_path = get_dataset.get_local_copy()

# load the train manifest which includes the english dataset
get_train_manifest = Dataset.get(dataset_id=TRAIN_MANIFEST_ID)
get_train_manifest_root_path = get_train_manifest.get_local_copy()

# load the inferred manifest from the previous inference task
for manifest in MANIFEST_ID_LIST:
    get_manifest = Dataset.get(dataset_id=manifest)
    get_manifest_root_path = get_manifest.get_local_copy()

    # append the remaining root path batches
    root_path_list.append(get_manifest_root_path)

# create a list to store the full path to the json manifest files
full_manifest_path_list = []

# iterate the root path list to append the full manifest file path
for idx, root_path in enumerate(root_path_list):
    full_manifest_path_list.append(os.path.join(root_path, 'others.json'))

# create an output folder to save the output as a clearml dataset
os.mkdir(f'{MANIFEST_ROOT}/')

# call the class to do the shifting of files for 'en' class
shift_en = SplitAudio(manifest_dir=f'{get_train_manifest_root_path}/train_manifest_sb_iteration_{ITER}.json', 
                      original_root_dir='{data_root}', 
                      replaced_root_dir=f'{get_dataset_root_path}', 
                      silence_removed_dir=None,
                      preprocessed_dir=f'{MANIFEST_ROOT}', 
                      data_is_others=False,
                      is_clearml=True)

shift_en()

# call the class to do the shifting of files for 'others' class
for path in full_manifest_path_list:
    shift_others = SplitAudio(manifest_dir=path, 
                              original_root_dir='/lid/datasets/mms/mms_silence_removed', 
                              replaced_root_dir=f'{get_dataset_root_path}', 
                              silence_removed_dir=None,
                              preprocessed_dir=f'{MANIFEST_ROOT}', 
                              data_is_others=True,
                              is_clearml=True)
    shift_others()

# create a datasets to store the data in the final edited form
dataset = Dataset.create(
    dataset_project=DATASET_PROJ_NAME,
    dataset_name=DATASET_NAME,
)

dataset.add_files(path=f'{MANIFEST_ROOT}/')
dataset.upload(output_url="s3://experiment-logging")
dataset.finalize()

print('Done')