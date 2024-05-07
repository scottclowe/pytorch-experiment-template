"""
Input/output utilities.
"""

import os
from inspect import getsourcefile

import torch

PACKAGE_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


def get_project_root() -> str:
    return os.path.dirname(PACKAGE_DIR)


def safe_save_model(modules, checkpoint_path=None, config=None, **kwargs):
    """
    Save a model to a checkpoint file, along with any additional data.

    Parameters
    ----------
    modules : dict
        A dictionary of modules to save. The keys are the names of the modules
        and the values are the modules themselves.
    checkpoint_path : str, optional
        Path to the checkpoint file. If not provided, the path will be taken
        from the config object.
    config : :class:`argparse.Namespace`, optional
        A configuration object containing the checkpoint path.
    **kwargs
        Additional data to save to the checkpoint file.
    """
    if checkpoint_path is not None:
        pass
    elif config is not None and hasattr(config, "checkpoint_path"):
        checkpoint_path = config.checkpoint_path
    else:
        raise ValueError("No checkpoint path provided")
    print(f"\nSaving model to {checkpoint_path}")
    # Create the directory if it doesn't already exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    # Save to a temporary file first, then move the temporary file to the target
    # destination. This is to prevent clobbering the checkpoint with a partially
    # saved file, in the event that the saving process is interrupted. Saving
    # the checkpoint takes a little while and can be disrupted by preemption,
    # whereas moving the file is an atomic operation.
    tmp_a, tmp_b = os.path.split(checkpoint_path)
    tmp_fname = os.path.join(tmp_a, ".tmp." + tmp_b)
    data = {k: v.state_dict() for k, v in modules.items()}
    data.update(kwargs)
    if config is not None:
        data["config"] = config

    torch.save(data, tmp_fname)
    os.rename(tmp_fname, checkpoint_path)
    print(f"Saved model to  {checkpoint_path}")
