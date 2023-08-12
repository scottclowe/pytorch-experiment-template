import os
import random
import secrets
import string

import numpy as np
import torch


def init_or_resume_wandb_run(output_dir, basename="wandb_runid.txt", **kwargs):
    r"""
    Initialize a wandb run, resuming if one already exists for this job.

    Parameters
    ----------
    output_dir : str
        Path to output directory for this job, where the wandb run id file
        will be stored.
    basename : str, default="wandb_runid.txt"
        Basename of wandb run id file.
    **kwargs
        Additional parameters to be passed through to ``wandb.init``.
        Examples include, ``"project"``, ``"name"``, ``"config"``.

    Returns
    -------
    run : :class:`wandb.sdk.wandb_run.Run`
        A wandb Run object, as returned by :func:`wandb.init`.
    """
    import wandb

    if not output_dir:
        wandb_id_file_path = None
    else:
        wandb_id_file_path = os.path.join(output_dir, basename)
    if wandb_id_file_path and os.path.isfile(wandb_id_file_path):
        # If the run_id was previously saved, get the id and resume it
        with open(wandb_id_file_path, "r") as f:
            resume_id = f.read()
        run = wandb.init(resume=resume_id, **kwargs)
    else:
        # If the run id file doesn't exist, create a new wandb run
        run = wandb.init(**kwargs)
        if wandb_id_file_path:
            # Write the run id to the expected file for resuming later
            with open(wandb_id_file_path, "w") as f:
                f.write(run.id)

    return run


def set_rng_seeds_fixed(seed, all_gpu=True):
    r"""
    Seed pseudo-random number generators throughout python's random module, numpy.random, and pytorch.

    Parameters
    ----------
    seed : int
        The random seed to use. Should be between 0 and 4294967295 to ensure
        unique behaviour for numpy, and between 0 and 18446744073709551615 to
        ensure unique behaviour for pytorch.
    all_gpu : bool, default=True
        Whether to set the torch seed on every GPU. If ``False``, only the
        current GPU has its seed set.

    Returns
    -------
    None
    """
    # Note that random, numpy, and pytorch all use different RNG methods/
    # implementations, so you'll get different outputs from each of them even
    # if you use the same seed for them all.
    # We use modulo with the maximum values permitted for np.random and torch.
    # If your seed exceeds 4294967295, numpy will have looped around to a
    random.seed(seed)
    np.random.seed(seed % 0xFFFF_FFFF)
    torch.manual_seed(seed % 0xFFFF_FFFF_FFFF_FFFF)
    if all_gpu:
        torch.cuda.manual_seed_all(seed % 0xFFFF_FFFF_FFFF_FFFF)
    else:
        torch.cuda.manual_seed(seed % 0xFFFF_FFFF_FFFF_FFFF)


def worker_seed_fn(worker_id):
    r"""
    Seed builtin :mod:`random` and :mod:`numpy`.

    A worker initialization function for :class:`torch.utils.data.DataLoader`
    objects which seeds builtin :mod:`random` and :mod:`numpy` with the
    torch seed for the worker.

    Parameters
    ----------
    worker_id : int
        The ID of the worker.
    """
    worker_seed = torch.utils.data.get_worker_info().seed
    random.seed(worker_seed)
    np.random.seed(worker_seed % 0xFFFF_FFFF)


def determine_epoch_seed(seed, epoch):
    r"""
    Determine the seed to use for the random number generator for a given epoch.

    Parameters
    ----------
    seed : int
        The original random seed, used to generate the sequence of seeds for
        the epochs.
    epoch : int
        The epoch for which to determine the seed.

    Returns
    -------
    epoch_seed : int
        The seed to use for the random number generator for the given epoch.
    """
    if epoch == 0:
        raise ValueError("Epoch must be indexed from 1, not 0.")
    random.seed(seed)
    # Generate a seed for every epoch so far. We have to traverse the
    # series of RNG calls to reach the next value (our next seed). The final
    # value is the one for our current epoch.
    # N.B. We use random.randint instead of torch.randint because torch.randint
    # only supports int32 at most (max value of 0xFFFF_FFFF).
    for _ in range(epoch):
        epoch_seed = random.randint(0, 0xFFFF_FFFF_FFFF_FFFF)
    return epoch_seed


def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 string of `length` digits.

    Parameters
    ----------
    length : int, default=8
        Length of the string to generate.

    Returns
    -------
    id : str
        The randomly generated id.
    """
    # Borrowed from https://github.com/wandb/wandb/blob/0e00efd/wandb/sdk/lib/runid.py
    # under the MIT license.
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def count_parameters(model, only_trainable=True):
    r"""
    Count the number of (trainable) parameters within a model and its children.

    Parameters
    ----------
    model : torch.nn.Model
        The parametrized model.
    only_trainable : bool, optional
        Whether the count should be restricted to only trainable parameters
        (default), otherwise all parameters are included.
        Default is ``True``.

    Returns
    -------
    int
        Total number of (trainable) parameters possessed by the model.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def concat_all_gather(tensor, **kwargs):
    r"""
    Gather a tensor over all processes and concatenate them into one.

    Similar to :func:`torch.distributed.all_gather`, except this function
    concatenates the result into a single tensor instead of a list of tensors.

    Parameters
    ----------
    tensor : torch.Tensor
        The distributed tensor on the current process.
    group : ProcessGroup, optional
        The process group to work on. If ``None``, the default process group
        will be used.
    async_op : bool, default=False
        Whether this op should be an async op.

    Returns
    -------
    gathered_tensor : torch.Tensor
        The contents of ``tensor`` from every distributed process, gathered
        together. None of the entries support a gradient.

    Warning
    -------
    As with :func:`torch.distributed.all_gather`, this has no gradient.
    """
    world_size = torch.distributed.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, **kwargs)
    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def concat_all_gather_ragged(tensor, **kwargs):
    r"""
    Gather a tensor over all processes and concatenate them into one.

    This version supports ragged tensors where the first dimension is not the
    same across all processes.

    Parameters
    ----------
    tensor : torch.Tensor
        The distributed tensor on the current process. The equivalent tensors
        on the other processes may differ in shape only in their first
        dimension.
    group : ProcessGroup, optional
        The process group to work on. If ``None``, the default process group
        will be used.
    async_op : bool, default=False
        Whether this op should be an async op.

    Returns
    -------
    gathered_tensor : torch.Tensor
        The contents of ``tensor`` from every distributed process, gathered
        together. None of the entries support a gradient.

    Warning
    -------
    As with :func:`torch.distributed.all_gather`, this has no gradient.
    """
    world_size = torch.distributed.get_world_size()
    # Gather the lengths of the tensors from all processes
    local_length = torch.tensor(tensor.shape[0], device=tensor.device)
    all_length = [torch.zeros_like(local_length) for _ in range(world_size)]
    torch.distributed.all_gather(all_length, local_length, **kwargs)
    # We will have to pad them to be the size of the longest tensor
    max_length = max(x.item() for x in all_length)

    # Pad our tensor on the current process
    length_diff = max_length - local_length.item()
    if length_diff:
        pad_size = (length_diff, *tensor.shape[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding), dim=0)

    # Gather the padded tensors from all processes
    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(all_tensors_padded, tensor, **kwargs)
    # Remove padding
    all_tensors = []
    for tensor_i, length_i in zip(all_tensors_padded, all_length):
        all_tensors.append(tensor_i[:length_i])

    # Concatenate the tensors
    output = torch.cat(all_tensors, dim=0)
    return output
