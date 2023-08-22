"""
Handlers for various image datasets.
"""

import os
import socket
import warnings

import numpy as np
import sklearn.model_selection
import torch
import torchvision.datasets


def determine_host():
    r"""
    Determine which compute server we are on.

    Returns
    -------
    host : str, one of {"vaughan", "mars"}
        An identifier for the host compute system.
    """
    hostname = socket.gethostname()
    slurm_submit_host = os.environ.get("SLURM_SUBMIT_HOST")
    slurm_cluster_name = os.environ.get("SLURM_CLUSTER_NAME")

    if slurm_cluster_name and slurm_cluster_name.startswith("vaughan"):
        return "vaughan"
    if slurm_submit_host and slurm_submit_host in ["q.vector.local", "m.vector.local"]:
        return "mars"
    if hostname and hostname in ["q.vector.local", "m.vector.local"]:
        return "mars"
    if hostname and hostname.startswith("v"):
        return "vaughan"
    if slurm_submit_host and slurm_submit_host.startswith("v"):
        return "vaughan"
    return ""


def image_dataset_sizes(dataset):
    r"""
    Get the image size and number of classes for a dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    Returns
    -------
    num_classes : int
        Number of classes in the dataset.
    img_size : int or None
        Size of the images in the dataset, or None if the images are not all
        the same size. Images are assumed to be square.
    num_channels : int
        Number of colour channels in the images in the dataset. This will be
        1 for greyscale images, and 3 for colour images.
    """
    dataset = dataset.lower().replace("-", "").replace("_", "").replace(" ", "")

    if dataset == "cifar10":
        num_classes = 10
        img_size = 32
        num_channels = 3

    elif dataset == "cifar100":
        num_classes = 100
        img_size = 32
        num_channels = 3

    elif dataset in ["imagenet", "imagenet1k", "ilsvrc2012"]:
        num_classes = 1000
        img_size = None
        num_channels = 3

    elif dataset.startswith("imagenette"):
        num_classes = 10
        img_size = None
        num_channels = 3

    elif dataset.startswith("imagewoof"):
        num_classes = 10
        img_size = None
        num_channels = 3

    elif dataset == "mnist":
        num_classes = 10
        img_size = 28
        num_channels = 1

    elif dataset == "svhn":
        num_classes = 10
        img_size = 32
        num_channels = 3

    else:
        raise ValueError("Unrecognised dataset: {}".format(dataset))

    return num_classes, img_size, num_channels


def fetch_image_dataset(
    dataset,
    root=None,
    transform_train=None,
    transform_eval=None,
    download=False,
):
    r"""
    Fetch a train and test dataset object for a given image dataset name.

    Parameters
    ----------
    dataset : str
        Name of dataset.
    root : str, optional
        Path to root directory containing the dataset.
    transform_train : callable, optional
        A function/transform that takes in an PIL image and returns a
        transformed version, to be applied to the training dataset.
    transform_eval : callable, optional
        A function/transform that takes in an PIL image and returns a
        transformed version, to be applied to the evaluation dataset.
    download : bool, optional
        Whether to download the dataset to the expected directory if it is not
        there. Only supported by some datasets. Default is ``False``.
    """
    dataset = dataset.lower().replace("-", "").replace("_", "").replace(" ", "")
    host = determine_host()

    if dataset == "cifar10":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd002/datasets/"
        elif host == "mars":
            root = "/scratch/gobi1/datasets/"
        else:
            root = "~/Datasets"
        dataset_train = torchvision.datasets.CIFAR10(
            os.path.join(root, dataset),
            train=True,
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.CIFAR10(
            os.path.join(root, dataset),
            train=False,
            transform=transform_eval,
            download=download,
        )

    elif dataset == "cifar100":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd002/datasets/"
        else:
            root = "~/Datasets"
        dataset_train = torchvision.datasets.CIFAR100(
            os.path.join(root, dataset),
            train=True,
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.CIFAR100(
            os.path.join(root, dataset),
            train=False,
            transform=transform_eval,
            download=download,
        )

    elif dataset in ["imagenet", "imagenet1k", "ilsvrc2012"]:
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/"
        elif host == "mars":
            root = "/scratch/gobi1/datasets/"
        else:
            root = "~/Datasets"
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(root, "imagenet", "train"),
            transform=transform_train,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(root, "imagenet", "val"),
            transform=transform_eval,
        )

    elif dataset == "imagenette":
        if root:
            root = os.path.join(root, "imagenette")
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/imagenette2/full/"
        else:
            root = "~/Datasets/imagenette/"
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(root, "train"),
            transform=transform_train,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(root, "val"),
            transform=transform_eval,
        )

    elif dataset == "imagewoof":
        if root:
            root = os.path.join(root, "imagewoof")
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/imagewoof2/full/"
        else:
            root = "~/Datasets/imagewoof/"
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(root, "train"),
            transform=transform_train,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(root, "val"),
            transform=transform_eval,
        )

    elif dataset == "mnist":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/"
        else:
            root = "~/Datasets"
        # Will read from [root]/MNIST/processed
        dataset_train = torchvision.datasets.MNIST(
            root,
            train=True,
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.MNIST(
            root,
            train=False,
            transform=transform_eval,
            download=download,
        )

    elif dataset == "svhn":
        # SVHN has:
        #  73,257 digits for training,
        #  26,032 digits for testing, and
        # 531,131 additional, less difficult, samples to use as extra training data
        # We don't use the extra split here, only train. There are original
        # images which are large and have bounding boxes, but the pytorch class
        # just uses the 32px cropped individual digits.
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd002/datasets/"
        elif host == "mars":
            root = "/scratch/gobi1/datasets/"
        else:
            root = "~/Datasets"
        dataset_train = torchvision.datasets.SVHN(
            os.path.join(root, dataset),
            split="train",
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.SVHN(
            os.path.join(root, dataset),
            split="test",
            transform=transform_eval,
            download=download,
        )

    else:
        raise ValueError("Unrecognised dataset: {}".format(dataset))

    return dataset_train, dataset_val, dataset_test


def fetch_dataset(
    dataset,
    root=None,
    prototyping=False,
    transform_train=None,
    transform_eval=None,
    protoval_split_rate=0.1,
    protoval_split_id=0,
    download=False,
):
    r"""
    Fetch a train and test dataset object for a given dataset name.

    Parameters
    ----------
    dataset : str
        Name of dataset.
    root : str, optional
        Path to root directory containing the dataset.
    prototyping : bool, default=False
        Whether to return a validation split distinct from the test split.
        If ``False``, the validation split will be the same as the test split
        for datasets which don't intrincally have a separate val and test
        partition.
        If ``True``, the validation partition is carved out of the train
        partition (resulting in a smaller training set) when there is no
        distinct validation partition available.
    transform_train : callable, optional
        A function/transform that takes in an PIL image and returns a
        transformed version, to be applied to the training dataset.
    transform_eval : callable, optional
        A function/transform that takes in an PIL image and returns a
        transformed version, to be applied to the evaluation dataset.
    protoval_split_rate : float, default=0.1
        The fraction of the train data to use for validating when in
        prototyping mode.
    protoval_split_id : int, default=0
        The identity of the random split used for the train/val partitioning.
        This controls the seed of the folds used for the split, and which
        fold to use for the validation set.
        The seed is equal to ``int(protoval_split_id * protoval_split_rate)``
        and the fold is equal to ``protoval_split_id % (1 / protoval_split_rate)``.
    download : bool, optional
        Whether to download the dataset to the expected directory if it is not
        there. Only supported by some datasets. Default is ``False``.

    Returns
    -------
    dataset_train : torch.utils.data.Dataset
        The training dataset.
    dataset_val : torch.utils.data.Dataset
        The validation dataset.
    dataset_test : torch.utils.data.Dataset
        The test dataset.
    distinct_val_test : bool
        Whether the validation and test partitions are distinct (True) or
        identical (False).
    """
    dataset_train, dataset_val, dataset_test = fetch_image_dataset(
        dataset=dataset,
        root=root,
        transform_train=transform_train,
        transform_eval=transform_eval,
        download=download,
    )

    # Handle the validation partition
    if dataset_val is not None:
        distinct_val_test = True
    elif not prototyping:
        dataset_val = dataset_test
        distinct_val_test = False
    else:
        # Create our own train/val split.
        #
        # For the validation part, we need a copy of dataset_train with the
        # evaluation transform instead.
        # The transform argument is *probably* going to be set to an attribute
        # on the dataset object called transform and called from there. But we
        # can't be completely sure, so to be completely agnostic about the
        # internals of the dataset class let's instantiate the dataset again!
        dataset_val = fetch_dataset(
            dataset,
            root=root,
            prototyping=False,
            transform_train=transform_eval,
        )[0]
        # dataset_val is a copy of the full training set, but with the transform
        # changed to transform_eval
        # Create the train/val split using these dataset objects.
        dataset_train, dataset_val = create_train_val_split(
            dataset_train,
            dataset_val,
            split_rate=protoval_split_rate,
            split_id=protoval_split_id,
        )
        distinct_val_test = True

    return (
        dataset_train,
        dataset_val,
        dataset_test,
        distinct_val_test,
    )


def create_train_val_split(
    dataset_train,
    dataset_val=None,
    split_rate=0.1,
    split_id=0,
):
    r"""
    Create a train/val split of a dataset.

    Parameters
    ----------
    dataset_train : torch.utils.data.Dataset
        The full training dataset with training transforms.
    dataset_val : torch.utils.data.Dataset, optional
        The full training dataset with evaluation transforms.
        If this is not given, the source for the validation set will be
        ``dataset_test`` (with the same transforms as the training partition).
        Note that ``dataset_val`` must have the same samples as
        ``dataset_train``, and the samples must be in the same order.
    split_rate : float, default=0.1
        The fraction of the train data to use for the validation split.
    split_id : int, default=0
        The identity of the split to use.
        This controls the seed of the folds used for the split, and which
        fold to use for the validation set.
        The seed is equal to ``int(split_id * split_rate)``
        and the fold is equal to ``split_id % (1 / split_rate)``.

    Returns
    -------
    dataset_train : torch.utils.data.Dataset
        The training subset of the dataset.
    dataset_val : torch.utils.data.Dataset
        The validation subset of the dataset.
    """
    if dataset_val is None:
        dataset_val = dataset_train
    # Now we need to reduce it down to just a subset of the training set.
    # Let's use K-folds so subsequent prototype split IDs will have
    # non-overlapping validation sets. With split_rate = 0.1,
    # there will be 10 folds.
    n_splits = round(1.0 / split_rate)
    if (1.0 / n_splits) != split_rate:
        warnings.warn(
            "The requested train/val split rate is not possible when using"
            " dataset into K folds. The actual split rate will be"
            f" {1.0 / n_splits} instead of {split_rate}.",
            UserWarning,
            stacklevel=2,
        )
    split_seed = int(split_id * split_rate)
    fold_id = split_id % n_splits
    print(
        f"Creating prototyping train/val split #{split_id}."
        f" Using fold {fold_id} of {n_splits} folds, generated with seed"
        f" {split_seed}."
    )
    # Try to do a stratified split.
    classes = get_dataset_labels(dataset_train)
    if classes is None:
        warnings.warn(
            "Creating prototyping splits without stratification.",
            UserWarning,
            stacklevel=2,
        )
        splitter_ftry = sklearn.model_selection.KFold
    else:
        splitter_ftry = sklearn.model_selection.StratifiedKFold

    # Create our splits. Assuming the dataset objects are always loaded
    # the same way, since a given split ID will always be the same
    # fold from the same seeded KFold splitter, it will yield the same
    # train/val split on each run.
    splitter = splitter_ftry(n_splits=n_splits, shuffle=True, random_state=split_seed)
    splits = splitter.split(np.arange(len(dataset_train)), classes)
    # splits is an iterable and we want to take the n-th fold from it.
    for i, (train_indices, val_indices) in enumerate(splits):  # noqa: B007
        if i == fold_id:
            break
    dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val, val_indices)
    return dataset_train, dataset_val


def get_dataset_labels(dataset):
    r"""
    Get the class labels within a :class:`torch.utils.data.Dataset` object.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset object.

    Returns
    -------
    array_like or None
        The class labels for each sample.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        # For a dataset subset, we need to get the full set of labels from the
        # interior subset and then reduce them down to just the labels we have
        # in the subset.
        labels = get_dataset_labels(dataset.dataset)
        if labels is None:
            return labels
        return np.array(labels)[dataset.indices]

    labels = None
    if hasattr(dataset, "targets"):
        # MNIST, CIFAR, ImageFolder, etc
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        # STL10, SVHN
        labels = dataset.labels
    elif hasattr(dataset, "_labels"):
        # Flowers102
        labels = dataset._labels

    return labels
