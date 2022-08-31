from clearml import Task, Dataset
import sys
import speechbrain as sb    
from hyperpyyaml import load_hyperpyyaml

# Reading command line arguments
hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

# Initialize ddp (useful only for multi-GPU DDP training).
sb.utils.distributed.ddp_init_group(run_opts)

# Load hyperparameters file with command-line overrides.
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

# start clearml
task = Task.init(project_name='LID', task_name='train', output_uri='s3://experiment-logging')
task.set_base_docker(
    docker_image='nicholasneo78/sb_lid:v0.0.2',
)

# execute clearml
task.execute_remotely(queue_name='compute', exit_process=True)

from preprocessing.Train.train import LID, dataio_prep

# get the pretrained embedding model
pretrained_embedding = Dataset.get(dataset_id='45e011de2c0d4c87b39656e0e3f61a24')
pretrained_embedding_path = pretrained_embedding.get_local_copy()

# overwrite the embedding path 
print(hparams['embedding_model_path'])
hparams['embedding_model_path'] = f'{pretrained_embedding_path}/embedding_model.ckpt'
print(hparams['embedding_model_path'])

# get the dataset
dataset_small = Dataset.get(dataset_id='a8c2b7fc535c4cd3b7d900721a76a2b3')
dataset_small_path = dataset_small.get_local_copy()
# overwrite the dataset path
hparams['data_folder'] = dataset_small_path

dataset_task = Task.get_task(dataset_small.id)

# load dataset manifest path, the train dev and test set
hparams['train_annotation'] = f'{dataset_small_path}/train_manifest_sb.json'
hparams['dev_annotation'] = f'{dataset_small_path}/dev_manifest_sb.json'
hparams['test_annotation'] = f'{dataset_small_path}/test_manifest_sb.json'

### start running the code

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
    dataset_project='datasets/LID',
    dataset_name='trained_model',
    parent_datasets=['a8c2b7fc535c4cd3b7d900721a76a2b3']
)

dataset.add_files(path=hparams['output_folder'])
dataset.upload(output_url="s3://experiment-logging")
dataset.finalize()

print('Done')