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

# load the model that is trained previously
get_model = Dataset.get(dataset_id='23ae6fbf80ec489ca3c17591552d6427')
model_root_path = get_model.get_local_copy()

ckpt_path = 'CKPT+2022-08-31+11-25-40+00'

# load the hyperparams config file that is needed for the inference
get_hyperparams = Dataset.get(dataset_id='22ffe0aba2594c3abaf3251f2b16af5c')
hyperparams_root_path = get_hyperparams.get_local_copy()

# load the dataset for inference
get_dataset = Dataset.get(dataset_id='a8872c8f04444a75b7e1436a72a534e4')
dataset_root_path = get_dataset.get_local_copy()

# move the config file into the save folder in order for the code to run correctly
shutil.move(f'{hyperparams_root_path}/hyperparams.yaml', f'{model_root_path}/save/{ckpt_path}/hyperparams.yaml')

### running the code
infer = InferManifest(input_manifest_dir=f'{dataset_root_path}/mms_batch_1s/manifest.json', 
                      pretrained_model_root=f'{model_root_path}/save', 
                      ckpt_folder=ckpt_path,
                      threshold={'en': 0.6, 'ms': 0.6}, 
                      root_dir_remove_tmp=dataset_root_path, 
                      old_dir='/lid/datasets/mms/mms_silence_removed/', # the manifest path where the data path is being uploaded locally
                      inference_replaced_dir=f'{dataset_root_path}/mms_batch_1s/',
                      new_manifest_replaced_dir='{data_root}/', 
                      output_manifest_dir='output/', 
                      data_batch='batch_1s',
                      iteration_num=1)

EN_PATH = 'output/batch_1s_iteration_1_en.json'
OTHERS_PATH = 'output/batch_1s_iteration_1_others.json'

c_en = ConvertToStandardJSON(input_manifest=EN_PATH, output_manifest=EN_PATH)
c_others = ConvertToStandardJSON(input_manifest=OTHERS_PATH, output_manifest=OTHERS_PATH)

infer()
c_en()
c_others()

# create dataset to store the new manifests
dataset = Dataset.create(
    dataset_project='datasets/LID',
    dataset_name='inference_iteration_1',
    parent_datasets=['a8872c8f04444a75b7e1436a72a534e4']
)

dataset.add_files(path='output/')
dataset.upload(output_url="s3://experiment-logging")
dataset.finalize()

print('Done')