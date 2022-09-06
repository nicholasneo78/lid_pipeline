import logging
import torch
import os
import sys
import inspect
import shutil
import yaml
from datetime import date
from load_hyperpyyaml import resolve_references
import collections
import subprocess

logger = logging.getLogger(__name__)
DEFAULT_LOG_CONFIG = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_CONFIG = os.path.join(DEFAULT_LOG_CONFIG, "log-config.yaml")
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 7

def create_experiment_directory(
    experiment_directory,
    hyperparams_to_save=None,
    overrides={},
    log_config=DEFAULT_LOG_CONFIG,
    save_env_desc=True,
):
    """Create the output folder and relevant experimental files.
    Arguments
    ---------
    experiment_directory : str
        The place where the experiment directory should be created.
    hyperparams_to_save : str
        A filename of a yaml file representing the parameters for this
        experiment. If passed, references are resolved, and the result is
        written to a file in the experiment directory called "hyperparams.yaml".
    overrides : dict
        A mapping of replacements made in the yaml file, to save in yaml.
    log_config : str
        A yaml filename containing configuration options for the logger.
    save_env_desc : bool
        If True, an environment state description is saved to the experiment
        directory, in a file called env.log in the experiment directory.
    """
    try:
        # all writing command must be done with the main_process
        if if_main_process():
            if not os.path.isdir(experiment_directory):
                os.makedirs(experiment_directory)

            # Write the parameters file
            if hyperparams_to_save is not None:
                hyperparams_filename = os.path.join(
                    experiment_directory, "hyperparams.yaml"
                )
                with open(hyperparams_to_save) as f:
                    resolved_yaml = resolve_references(f, overrides)
                with open(hyperparams_filename, "w") as w:
                    print("# Generated %s from:" % date.today(), file=w)
                    print("# %s" % os.path.abspath(hyperparams_to_save), file=w)
                    print("# yamllint disable", file=w)
                    shutil.copyfileobj(resolved_yaml, w)

            # Copy executing file to output directory
            module = inspect.getmodule(inspect.currentframe().f_back)
            if module is not None:
                callingfile = os.path.realpath(module.__file__)
                shutil.copy(callingfile, experiment_directory)

            # Log exceptions to output automatically
            log_file = os.path.join(experiment_directory, "log.txt")
            logger_overrides = {
                "handlers": {"file_handler": {"filename": log_file}}
            }
            setup_logging(log_config, logger_overrides)
            sys.excepthook = _logging_excepthook

            # Log beginning of experiment!
            logger.info("Beginning experiment!")
            logger.info(f"Experiment folder: {experiment_directory}")

            # Save system description:
            if save_env_desc:
                description_str = get_environment_description()
                with open(
                    os.path.join(experiment_directory, "env.log"), "w"
                ) as fo:
                    fo.write(description_str)
    finally:
        # wait for main_process if ddp is used
        ddp_barrier()

def _logging_excepthook(exc_type, exc_value, exc_traceback):
    """Interrupt exception raising to log the error."""
    logger.error("Exception:", exc_info=(exc_type, exc_value, exc_traceback))

def if_main_process():
    """Checks if the current process is the main process and authorized to run
    I/O commands. In DDP mode, the main process is the one with RANK == 0.
    In standard mode, the process will not have `RANK` Unix var and will be
    authorized to run the I/O commands.
    """
    if "RANK" in os.environ:
        if os.environ["RANK"] == "":
            return False
        else:
            if int(os.environ["RANK"]) == 0:
                return True
            return False
    return True

def ddp_barrier():
    """In DDP mode, this function will synchronize all processes.
    torch.distributed.barrier() will block processes until the whole
    group enters this function.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def setup_logging(
    config_path="log-config.yaml", overrides={}, default_level=logging.INFO,
):
    """Setup logging configuration.
    Arguments
    ---------
    config_path : str
        The path to a logging config file.
    default_level : int
        The level to use if the config file is not found.
    overrides : dict
        A dictionary of the same structure as the config dict
        with any updated values that need to be applied.
    """
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        recursive_update(config, overrides)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_environment_description():
    """Returns a string describing the current Python / SpeechBrain environment.
    Useful for making experiments as replicable as possible.
    Returns
    -------
    str
        The string is formatted ready to be written to a file.
    Example
    -------
    >>> get_environment_description().splitlines()[0]
    'SpeechBrain system description'
    """
    python_version_str = "Python version:\n" + sys.version + "\n"
    try:
        freezed, _, _ = run_shell("pip freeze")
        python_packages_str = "Installed Python packages:\n"
        python_packages_str += freezed.decode(errors="replace")
    except OSError:
        python_packages_str = "Could not list python packages with pip freeze"
    try:
        git_hash, _, _ = run_shell("git rev-parse --short HEAD")
        git_str = "Git revision:\n" + git_hash.decode(errors="replace")
    except OSError:
        git_str = "Could not get git revision"
    if torch.cuda.is_available():
        if torch.version.cuda is None:
            cuda_str = "ROCm version:\n" + torch.version.hip
        else:
            cuda_str = "CUDA version:\n" + torch.version.cuda
    else:
        cuda_str = "CUDA not available"
    result = "SpeechBrain system description\n"
    result += "==============================\n"
    result += python_version_str
    result += "==============================\n"
    result += python_packages_str
    result += "==============================\n"
    result += git_str
    result += "==============================\n"
    result += cuda_str
    return result

def recursive_update(d, u, must_match=False):
    """Similar function to `dict.update`, but for a nested `dict`.
    From: https://stackoverflow.com/a/3233356
    If you have to a nested mapping structure, for example:
        {"a": 1, "b": {"c": 2}}
    Say you want to update the above structure with:
        {"b": {"d": 3}}
    This function will produce:
        {"a": 1, "b": {"c": 2, "d": 3}}
    Instead of:
        {"a": 1, "b": {"d": 3}}
    Arguments
    ---------
    d : dict
        Mapping to be updated.
    u : dict
        Mapping to update with.
    must_match : bool
        Whether to throw an error if the key in `u` does not exist in `d`.
    Example
    -------
    >>> d = {'a': 1, 'b': {'c': 2}}
    >>> recursive_update(d, {'b': {'d': 3}})
    >>> d
    {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    # TODO: Consider cases where u has branch off k, but d does not.
    # e.g. d = {"a":1}, u = {"a": {"b": 2 }}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping) and k in d:
            recursive_update(d.get(k, {}), v)
        elif must_match and k not in d:
            raise KeyError(
                f"Override '{k}' not found in: {[key for key in d.keys()]}"
            )
        else:
            d[k] = v

def run_shell(cmd):
    """This function can be used to run a command in the bash shell.
    Arguments
    ---------
    cmd : str
        Shell command to run.
    Returns
    -------
    bytes
        The captured standard output.
    bytes
        The captured standard error.
    int
        The returncode.
    Raises
    ------
    OSError
        If returncode is not 0, i.e., command failed.
    Example
    -------
    >>> out, err, code = run_shell("echo 'hello world'")
    >>> _ = out.decode(errors="ignore")
    """

    # Executing the command
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    # Capturing standard output and error
    (output, err) = p.communicate()

    if p.returncode != 0:
        raise OSError(err.decode(errors="replace"))

    # Adding information in the logger
    msg = output.decode(errors="replace") + "\n" + err.decode(errors="replace")
    logger.debug(msg)

    return output, err, p.returncode

def run_on_main(
    func,
    args=None,
    kwargs=None,
    post_func=None,
    post_args=None,
    post_kwargs=None,
    run_post_on_main=False,
):
    """Runs a function with DPP (multi-gpu) support.
    The main function is only run on the main process.
    A post_function can be specified, to be on non-main processes after the main
    func completes. This way whatever the main func produces can be loaded on
    the other processes.
    Arguments
    ---------
    func : callable
        Function to run on the main process.
    args : list, None
        Positional args to pass to func.
    kwargs : dict, None
        Keyword args to pass to func.
    post_func : callable, None
        Function to run after func has finished on main. By default only run on
        non-main processes.
    post_args : list, None
        Positional args to pass to post_func.
    post_kwargs : dict, None
        Keyword args to pass to post_func.
    run_post_on_main : bool
        Whether to run post_func on main process as well. (default: False)
    """
    # Handle the mutable data types' default args:
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if post_args is None:
        post_args = []
    if post_kwargs is None:
        post_kwargs = {}

    if if_main_process():
        # Main comes here
        try:
            func(*args, **kwargs)
        finally:
            ddp_barrier()
    else:
        # Others go here
        ddp_barrier()
    if post_func is not None:
        if run_post_on_main:
            # Just run on every process without any barrier.
            post_func(*post_args, **post_kwargs)
        elif not if_main_process():
            # Others go here
            try:
                post_func(*post_args, **post_kwargs)
            finally:
                ddp_barrier()
        else:
            # But main comes here
            ddp_barrier()