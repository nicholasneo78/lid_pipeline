from clearml import Task, Dataset
import sys
#import speechbrain as sb    
from preprocessing.Modules.parse_args import parse_arguments
from preprocessing.Modules.ddp import ddp_init_group
# from hyperpyyaml import load_hyperpyyaml
from preprocessing.Modules.load_hyperpyyaml import load_hyperpyyaml

# Reading command line arguments
hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])

# Initialize ddp (useful only for multi-GPU DDP training).
ddp_init_group(run_opts)

# Load hyperparameters file with command-line overrides.
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

### configs for starting clearml ###
PROJ_NAME = 'LID'
ITER = 2
TASK_NAME = f'train_iteration_{ITER}'
DOCKER_IMG = 'nicholasneo78/sb_lid:v0.0.2'
QUEUE = 'compute'
####################################

### configs to get the clearml dataset ID #############
PRETRAINED_EMBEDDING_ID = '45e011de2c0d4c87b39656e0e3f61a24'
DATASET_ID = 'a8872c8f04444a75b7e1436a72a534e4'
MANIFEST_ID = '71ab138ab92b4969a4e05a9691ef9066'
DATASET_PROJ_NAME = 'datasets/LID'
DATASET_NAME = f'trained_model_iteration_{ITER}'
#######################################################

### manifest filename ###
TRAIN_MANIFEST = f'train_manifest_sb_iteration_{ITER-1}.json'
DEV_MANIFEST = 'dev_manifest_sb.json'
TEST_MANIFEST = 'test_manifest_sb.json'

# start clearml
task = Task.init(project_name=PROJ_NAME, task_name=TASK_NAME, output_uri='s3://experiment-logging')
task.set_base_docker(
    docker_image=DOCKER_IMG,
)

# execute clearml
task.execute_remotely(queue_name=QUEUE, exit_process=True)

from preprocessing.Train.train import LID, dataio_prep

# get the pretrained embedding model
pretrained_embedding = Dataset.get(dataset_id=PRETRAINED_EMBEDDING_ID)
pretrained_embedding_path = pretrained_embedding.get_local_copy()

# overwrite the embedding path 
print(f'Before overriding: {hparams["embedding_model_path"]}')
hparams['embedding_model_path'] = f'{pretrained_embedding_path}/embedding_model.ckpt'
print(f'After overriding: {hparams["embedding_model_path"]}')

# get the dataset
dataset_small = Dataset.get(dataset_id=DATASET_ID)
dataset_small_path = dataset_small.get_local_copy()

# get the manifests to the dataset
manifest = Dataset.get(dataset_id=MANIFEST_ID)
manifest_path = manifest.get_local_copy()

# overwrite the dataset path
hparams['data_folder'] = dataset_small_path

# load dataset manifest path, the train dev and test set
hparams['train_annotation'] = f'{manifest_path}/{TRAIN_MANIFEST}'
hparams['dev_annotation'] = f'{manifest_path}/{DEV_MANIFEST}'
hparams['test_annotation'] = f'{manifest_path}/{TEST_MANIFEST}'

### start running the code ###

# Create experiment directory
sb.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
    overrides=overrides,
)

if hparams["use_tensorboard"]:
    from speechbrain.utils.train_logger import TensorboardLogger

    hparams["tensorboard_train_logger"] = TensorboardLogger(
        hparams["tensorboard_logs"]
    )

# Create dataset objects "train", "dev", and "test" and language_encoder
datasets, language_encoder = dataio_prep(hparams)
hparams['pretrainer'].paths['embedding_model'] = hparams['embedding_model_path']

# Fetch and load pretrained modules
sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
hparams["pretrainer"].load_collected(device=run_opts["device"])

# Initialize the Brain object to prepare for mask training.
lid_brain = LID(
    modules=hparams["modules"],
    opt_class=hparams["opt_class"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],
)

# freezing certain layers
for name, child in lid_brain.modules.named_children():
    if name == "embedding_model":
        # print(dir(child))
        child.requires_grad_ = False

lid_brain.fit(
    epoch_counter=lid_brain.hparams.epoch_counter,
    train_set=datasets["train"],
    valid_set=datasets["dev"],
    train_loader_kwargs=hparams["train_dataloader_options"],
    valid_loader_kwargs=hparams["test_dataloader_options"],
)

# Load the best checkpoint for evaluation
test_stats = lid_brain.evaluate(
    test_set=datasets["test"],
    min_key="error",
    test_loader_kwargs=hparams["test_dataloader_options"],
)

# create the dataset to store the pretrained model
dataset = Dataset.create(
    dataset_project=DATASET_PROJ_NAME,
    dataset_name=DATASET_NAME,
    parent_datasets=[DATASET_ID]
)

dataset.add_files(path=hparams['output_folder'])
dataset.upload(output_url="s3://experiment-logging")
dataset.finalize()

print('Done')