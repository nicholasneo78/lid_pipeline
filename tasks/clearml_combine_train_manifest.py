from clearml import Task, Dataset

### configs for starting clearml ###
PROJ_NAME = 'LID'
ITER = 2
TASK_NAME = f'combine_manifest_iteration_{ITER}'
DOCKER_IMG = 'nicholasneo78/sb_lid:v0.0.2'
QUEUE = 'compute'
#####################################

### configs to get the clearml dataset ID (predicted json manifest file) ###
# id of the original train data
ORIGINAL_TRAIN_ID = '71ab138ab92b4969a4e05a9691ef9066'

# id of the batches of data from the previous inference
MANIFEST_ID_LIST = ['d54ffb239345410fb7e37e0a43b44ea3', 
                    '898dcc04c9ad476e8071b0d7ffd3b45b']

# for the first iter 
if ITER == 1:
    SUFFIX = ''
# for the subsequent iteration of the train data, to get the previous train manifest
else:
    SUFFIX = f'_iteration_{ITER-1}' 
MANIFEST_ROOT = 'output'
######################################

### configs to store the new dataset to save the updated train manifest ###
DATASET_PROJ_NAME = 'datasets/LID'
DATASET_NAME = f'combine_iteration_{ITER}'
PARENT_DATASET_ID = '71ab138ab92b4969a4e05a9691ef9066' #'a8872c8f04444a75b7e1436a72a534e4'
######################################

# start clearml
task = Task.init(project_name=PROJ_NAME, task_name=TASK_NAME, output_uri='s3://experiment-logging')
task.set_base_docker(
    docker_image=DOCKER_IMG,
)

# execute clearml
task.execute_remotely(queue_name=QUEUE, exit_process=True)

from preprocessing.Train.combine_train_manifest import CombineTrainManifest
import os

# initiate a list to store the manifest root path for each of the batches
root_path_list = []

# load the original train manifest filepath
get_original_train_manifest =  Dataset.get(dataset_id=ORIGINAL_TRAIN_ID)
get_original_train_manifest_root_path = get_original_train_manifest.get_local_copy()

# append the root path of the original train manifest
root_path_list.append(get_original_train_manifest_root_path)

# load the generated manifest filepaths from the previous inference task
for man_id in MANIFEST_ID_LIST:
    get_manifest = Dataset.get(dataset_id=man_id)
    get_manifest_root_path = get_manifest.get_local_copy()

    # append the remaining root path batches
    root_path_list.append(get_manifest_root_path)

# create a list to store the full path to the json manifest files
full_manifest_path_list = []

# iterate the root path list to append the full manifest file path
for idx, root_path in enumerate(root_path_list):
    # the first entry will be the original train manifest path
    if idx == 0:
        full_manifest_path_list.append(os.path.join(root_path, f'train_manifest_sb{SUFFIX}.json'))
    else:
        full_manifest_path_list.append(os.path.join(root_path, 'en.json'))

# create an output folder to save the output as a clearml dataset
os.mkdir(f'{MANIFEST_ROOT}/')

# call the class to do the combining
c = CombineTrainManifest(manifest_list=full_manifest_path_list,
                         output_manifest=f'{MANIFEST_ROOT}/train_manifest_sb_iteration_{ITER}.json')
c()

# create dataset to store the new train manifest
dataset = Dataset.create(
    dataset_project=DATASET_PROJ_NAME,
    dataset_name=DATASET_NAME,
    parent_datasets=[PARENT_DATASET_ID]
)

dataset.add_files(path=f'{MANIFEST_ROOT}/')
dataset.upload(output_url="s3://experiment-logging")
dataset.finalize()

print('Done')