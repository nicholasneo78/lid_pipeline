from clearml import Task, Dataset
import os

### configs to start clearml ###
PROJ_NAME = 'LID'
TASK_NAME = 'combined_dataset'
DOCKER_IMG = 'nicholasneo78/sb_lid:v0.0.2'
QUEUE = 'compute'
#####################################

### configs to get the clearml id of the individual datasets ###
# DATASET_ID_LIST = ['dd4117a41f2841ff9238648d191cc015', 'a098c93c37ee4181871dbd4120552ad9', 'b65dafd1fac14a5ea88a0535b13502c2']

DATASET_ID_DICT = {'mms_batch_train': 'dd4117a41f2841ff9238648d191cc015',
                   'mms_batch_1s': 'a098c93c37ee4181871dbd4120552ad9',
                   'mms_batch_2s': 'b65dafd1fac14a5ea88a0535b13502c2'}
################################################################

### configs to get the clearml dataset ID #############
DATASET_PROJ_NAME = 'datasets/LID'
DATASET_NAME = 'combined_dataset'
MANIFEST_ROOT = 'output'
#######################################################

# start clearml
task = Task.init(project_name=PROJ_NAME, task_name=TASK_NAME, output_uri='s3://experiment-logging')
task.set_base_docker(
    docker_image=DOCKER_IMG,
)

# execute clearml
task.execute_remotely(queue_name=QUEUE, exit_process=True)

# # create a new directory in the remote folder to store the datasets
os.mkdir(f'{MANIFEST_ROOT}/')

# retrieve all the clearml dataset root path
#clearml_root_path_list = []

dataset = Dataset.create(
    dataset_project=DATASET_PROJ_NAME,
    dataset_name=DATASET_NAME,
    parent_datasets=None
)

for dataset_key in DATASET_ID_DICT:
    dataset_ = Dataset.get(dataset_id=DATASET_ID_DICT[dataset_key])
    dataset_root_path = dataset_.get_local_copy()
    dataset.add_files(path=f'{dataset_root_path}')

# upload the datasets
dataset.upload(output_url="s3://experiment-logging")
dataset.finalize()

print('Done')