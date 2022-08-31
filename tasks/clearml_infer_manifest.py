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
from hyperpyyaml import load_hyperpyyaml

# load the model that is trained previously
get_model = Dataset.get(dataset_id='380af3a250354166baa15fedb046de58')
get_model_root_path = get_model.get_local_copy()

# load the hyperparams config file that is needed for the inference
get_hyperparams = Dataset.get(dataset_id='a0e6290d17eb4407a5b753c0f4934a7f')
get_hyperparams_root_path = get_hyperparams.get_local_copy()

# load the yaml file to get the parameters
with open(f'{get_hyperparams_root_path}/hyperparams.yaml') as fin:
    hparams = load_hyperpyyaml(fin)

ckpt_path = 'save/CKPT+2022-08-30+03-45-50+00'

# overwrite the values in the hyperparams file to point to the clearml dataset
MODEL_ROOT_PATH = f'{get_model_root_path}/{ckpt_path}'
hparams['pretrained_path'] = MODEL_ROOT_PATH

# hparams['pretrainer'].paths['embedding_model'] = f'{MODEL_ROOT_PATH}/embedding_model.ckpt'
# hparams['pretrainer'].paths['normalizer_input'] = f'{MODEL_ROOT_PATH}/normalizer_input.ckpt'
# hparams['pretrainer'].paths['classifier'] = f'{MODEL_ROOT_PATH}/classifier.ckpt'
# hparams['pretrainer'].paths['label_encoder'] = f'{MODEL_ROOT_PATH}/label_encoder.ckpt'

# save the edited hyperparams into the correct folder where it should reside


# move the config file into the save folder
shutil.move(f'{get_hyperparams_root_path}/hyperparams.yaml', f'{get_model_root_path}/{ckpt_path}/hyperparams.yaml')

# change the input of the hyperparams.yaml file
