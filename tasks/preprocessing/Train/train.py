import os
import sys
import torch
import logging
import torchaudio
import speechbrain as sb    
from hyperpyyaml import load_hyperpyyaml
from tasks.preprocessing.Modules.metric_stats_override import BinaryMetricStats

"""Recipe for training a LID system with CommonLanguage.
To run this recipe, do the following:
> python train.py hparams/train_ecapa_tdnn.yaml
Author
------
 * Mirco Ravanelli 2021
 * Pavlo Ruban 2021
"""

logger = logging.getLogger(__name__)

# Brain class for Language ID training
class LID(sb.Brain):
    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.
        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            # wavs_noise = self.modules.env_corrupt(wavs, lens)
            wavs = torch.cat([wavs, wavs], dim=0)
            lens = torch.cat([lens, lens], dim=0)
            wavs = self.hparams.augmentation(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.
        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs

        # squeeze the "predictions" tensor for the binary_metrics and EER to work with as the prediction and the target has to be in the same dimension
        predictions_squeeze_raw = torch.squeeze(predictions, dim=1)
        # argmax the tensor
        # predictions_squeeze = torch.argmax(predictions_squeeze_raw, dim=1)
        # unsqueeze the tensor again to the correct form
        # predictions_binary = torch.unsqueeze(predictions_squeeze, dim=1)
        # predictions_binary = predictions_squeeze
        predictions_binary = torch.argmax(predictions_squeeze_raw, dim=1)
        targets = batch.language_encoded.data

        if stage == sb.Stage.TRAIN:
            print(f'Pred: {predictions}')
            print(f'Target: {targets}')
            # Concatenate labels (due to data augmentation)
            targets = torch.cat([targets, targets], dim=0)
            lens = torch.cat([lens, lens], dim=0)

        else:
            print(f'Pred-edit: {predictions_binary}')
            targets_squeeze = torch.squeeze(targets, dim=1)
            print(f'Target: {targets_squeeze}')

        # print(f'Lens: {lens}')
        # print(f'Batch ID: {batch.id}')

        loss = self.hparams.compute_cost(predictions, targets)

        print(f'Loss: {loss}')

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

            # self.binary_metrics.append(batch.id, predictions, targets, lens)
            self.binary_metrics.append(batch.id, predictions_binary, targets_squeeze)
            # print(batch.id, predictions_binary, targets)

            self.acc_metric.append(
                batch.id, predict=predictions, target=targets, lengths=lens
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            # self.binary_metrics = sb.utils.metric_stats.BinaryMetricStats(positive_label=0)
            self.binary_metrics = BinaryMetricStats(positive_label=0)

            def accuracy_value(predict, target, lengths):
                """Computes Accuracy"""
                nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                    predict, target, lengths
                )
                acc = torch.tensor([nbr_correct / nbr_total])
                return acc

            self.acc_metric = sb.utils.metric_stats.MetricStats(metric=accuracy_value, n_jobs=1)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_stats = {
                "loss": stage_loss
            }

        # Summarize the statistics from the stage for record-keeping.
        else:
            print('DEBUG')
            print(self.binary_metrics.summarize(field='F-score'))
            print('DEBUG')
            valid_stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
                "acc": self.acc_metric.summarize("average"),
                "F1": self.binary_metrics.summarize(field='F-score')[0],
                "EER": self.binary_metrics.summarize(field='DER')[0],
                "precision": self.binary_metrics.summarize(field='precision')[0],
                "recall": self.binary_metrics.summarize(field='recall')[0],
                "TP": self.binary_metrics.summarize(field='TP')[0],
                "TN": self.binary_metrics.summarize(field='TN')[0],
                "FP": self.binary_metrics.summarize(field='FP')[0],
                "FN": self.binary_metrics.summarize(field='FN')[0],
            }
        

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.

            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch, "lr": old_lr},
                    train_stats=self.train_stats,
                    valid_stats=valid_stats
                )
            else:
                self.hparams.train_logger.log_stats(
                    {"Epoch": epoch, "lr": old_lr},
                    train_stats=self.train_stats,
                    valid_stats=valid_stats,
                )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=valid_stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=valid_stats,
            )
            # plot confusion matrix
            # print(
            #     {"Confusion Matrix:": torch.tensor([[],[]])}
            # )


def dataio_prep(hparams):
    """ This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_common_language` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'lang01': 0, 'lang02': 1, ..)
    language_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig, _ = torchaudio.load(wav)
        sig = sig.transpose(0, 1).squeeze(1)

        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("language")
    @sb.utils.data_pipeline.provides("language", "language_encoded")
    def label_pipeline(language):
        yield language
        language_encoded = language_encoder.encode_label_torch(language)
        yield language_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "language_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    language_encoder_file = os.path.join(
        hparams["save_folder"], "language_encoder.txt"
    )
    
    language_encoder.load_or_create(
        path=language_encoder_file,
        from_didatasets=[datasets["train"]],
        output_key="language",
    )

    return datasets, language_encoder


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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

    # # Data preparation, to be run on only one process.
    # sb.utils.distributed.run_on_main(
    #     prepare_common_language,
    #     kwargs={
    #         "data_folder": hparams["data_folder"],
    #         "save_folder": hparams["save_folder"],
    #         "skip_prep": hparams["skip_prep"],
    #     },
    # )

    # Create dataset objects "train", "dev", and "test" and language_encoder
    datasets, language_encoder = dataio_prep(hparams)

    # Fetch and laod pretrained modules
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

    # freezing starts

    count = 0

    for name, child in lid_brain.modules.named_children():
        if name == "embedding_model":
            # print(dir(child))
            child.requires_grad_ = False

    # freezing ends

            # for param in child.parameters():
            #     param.requires_grad = False
            # count+=1
            # if count == 4:

            #     break

    # # print(lid_brain.modules)
    # # print()
    # # print()
    # print(lid_brain.modules)

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
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