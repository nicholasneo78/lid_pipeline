import torch
import logging
import os

logger = logging.getLogger(__name__)

def ddp_init_group(run_opts):
    """This function will initialize the ddp group if
    distributed_launch bool is given in the python command line.
    The ddp group will use distributed_backend arg for setting the
    DDP communication protocol. `RANK` Unix variable will be used for
    registering the subprocess to the ddp group.
    Arguments
    ---------
    run_opts: list
        A list of arguments to parse, most often from `sys.argv[1:]`.
    """
    if run_opts["distributed_launch"]:
        if "local_rank" not in run_opts:
            raise ValueError(
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.launch [args]\n\t"
                "experiment.py hyperparams.yaml --distributed_launch "
                "--distributed_backend=nccl"
            )
        else:
            if run_opts["local_rank"] + 1 > torch.cuda.device_count():
                raise ValueError(
                    "Killing process " + str() + "\n"
                    "Not enough GPUs available!"
                )
        if "RANK" in os.environ is None or os.environ["RANK"] == "":
            raise ValueError(
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.launch [args]\n\t"
                "experiment.py hyperparams.yaml --distributed_launch "
                "--distributed_backend=nccl"
            )
        rank = int(os.environ["RANK"])

        if run_opts["distributed_backend"] == "nccl":
            if not torch.distributed.is_nccl_available():
                raise ValueError("NCCL is not supported in your machine.")
        elif run_opts["distributed_backend"] == "gloo":
            if not torch.distributed.is_gloo_available():
                raise ValueError("GLOO is not supported in your machine.")
        elif run_opts["distributed_backend"] == "mpi":
            if not torch.distributed.is_mpi_available():
                raise ValueError("MPI is not supported in your machine.")
        else:
            logger.info(
                run_opts["distributed_backend"]
                + " communcation protocol doesn't exist."
            )
            raise ValueError(
                run_opts["distributed_backend"]
                + " communcation protocol doesn't exist."
            )
        # rank arg is used to set the right rank of the current process for ddp.
        # if you have 2 servers with 2 gpu:
        # server1:
        #   GPU0: local_rank=device=0, rank=0
        #   GPU1: local_rank=device=1, rank=1
        # server2:
        #   GPU0: local_rank=device=0, rank=2
        #   GPU1: local_rank=device=1, rank=3
        torch.distributed.init_process_group(
            backend=run_opts["distributed_backend"], rank=rank
        )
    else:
        logger.info(
            "distributed_launch flag is disabled, "
            "this experiment will be executed without DDP."
        )
        if "local_rank" in run_opts and run_opts["local_rank"] > 0:
            raise ValueError(
                "DDP is disabled, local_rank must not be set.\n"
                "For DDP training, please use --distributed_launch. "
                "For example:\n\tpython -m torch.distributed.launch "
                "experiment.py hyperparams.yaml "
                "--distributed_launch --distributed_backend=nccl"
            )