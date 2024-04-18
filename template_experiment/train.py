#!/usr/bin/env python

import builtins
import copy
import os
import shutil
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from socket import gethostname

import torch
import torch.optim
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from template_experiment import data_transformations, datasets, encoders, utils
from template_experiment.evaluation import evaluate

BASE_BATCH_SIZE = 128


def check_is_distributed():
    r"""
    Check if the current job is running in distributed mode.

    Returns
    -------
    bool
        Whether the job is running in distributed mode.
    """
    return (
        "WORLD_SIZE" in os.environ
        and "RANK" in os.environ
        and "LOCAL_RANK" in os.environ
        and "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
    )


def setup_slurm_distributed():
    r"""
    Use SLURM environment variables to set up environment variables needed for DDP.

    Note: This is not used when using torchrun, as that sets RANK etc. for us,
    but is useful if you're using srun without torchrun (i.e. using srun within
    the sbatch file to lauching one task per GPU).
    """
    if "WORLD_SIZE" in os.environ:
        pass
    elif "SLURM_NNODES" in os.environ and "SLURM_GPUS_ON_NODE" in os.environ:
        os.environ["WORLD_SIZE"] = str(int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_GPUS_ON_NODE"]))
    elif "SLURM_NPROCS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        if int(os.environ["RANK"]) > 0 and "WORLD_SIZE" not in os.environ:
            raise EnvironmentError(
                f"SLURM_PROCID is {os.environ['SLURM_PROCID']}, implying"
                " distributed training, but WORLD_SIZE could not be determined."
            )
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "MASTER_ADDR" not in os.environ and "SLURM_NODELIST" in os.environ:
        os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split("-")[0]
    if "MASTER_PORT" not in os.environ and "SLURM_JOB_ID" in os.environ:
        os.environ["MASTER_PORT"] = str(49152 + int(os.environ["SLURM_JOB_ID"]) % 16384)


def run(config):
    r"""
    Run training job (one worker if using distributed training).

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # DISTRIBUTION ============================================================
    # Setup for distributed training
    setup_slurm_distributed()
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.distributed = check_is_distributed()
    if config.world_size > 1 and not config.distributed:
        raise EnvironmentError(
            f"WORLD_SIZE is {config.world_size}, but not all other required"
            " environment variables for distributed training are set."
        )
    # Work out the total batch size depending on the number of GPUs we are using
    config.batch_size = config.batch_size_per_gpu * config.world_size

    if config.distributed:
        # For multiprocessing distributed training, gpu rank needs to be
        # set to the global rank among all the processes.
        config.global_rank = int(os.environ["RANK"])
        config.local_rank = int(os.environ["LOCAL_RANK"])
        print(
            f"Rank {config.global_rank} of {config.world_size} on {gethostname()}"
            f" (local GPU {config.local_rank} of {torch.cuda.device_count()})."
            f" Communicating with master at {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        torch.distributed.init_process_group(backend="nccl")
    else:
        config.global_rank = 0

    # Suppress printing if this is not the master process for the node
    if config.distributed and config.global_rank != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.")

    # Check which device to use
    use_cuda = not config.no_cuda and torch.cuda.is_available()

    if config.distributed and not use_cuda:
        raise EnvironmentError("Distributed training with NCCL requires CUDA.")
    if not use_cuda:
        device = torch.device("cpu")
    elif config.local_rank is not None:
        device = f"cuda:{config.local_rank}"
    else:
        device = "cuda"

    print(f"Using device {device}")

    # RESTORE OMITTED CONFIG FROM RESUMPTION CHECKPOINT =======================
    checkpoint = None
    config.model_output_dir = None
    if config.checkpoint_path:
        config.model_output_dir = os.path.dirname(config.checkpoint_path)
    if not config.checkpoint_path:
        # Not trying to resume from a checkpoint
        pass
    elif not os.path.isfile(config.checkpoint_path):
        # Looks like we're trying to resume from the checkpoint that this job
        # will itself create. Let's assume this is to let the job resume upon
        # preemption, and it just hasn't been preempted yet.
        print(f"Skipping premature resumption from preemption: no checkpoint file found at '{config.checkpoint_path}'")
    else:
        print(f"Loading resumption checkpoint '{config.checkpoint_path}'")
        # Map model parameters to be load to the specified gpu.
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        keys = vars(get_parser().parse_args("")).keys()
        keys = set(keys).difference(["resume", "gpu", "global_rank", "local_rank", "cpu_workers"])
        for key in keys:
            if getattr(checkpoint["config"], key, None) is None:
                continue
            if getattr(config, key) is None:
                print(f"  Restoring config value for {key} from checkpoint: {getattr(checkpoint['config'], key)}")
                setattr(config, key, getattr(checkpoint["config"], key, None))
            elif getattr(config, key) != getattr(checkpoint["config"], key):
                print(
                    f"  Warning: config value for {key} differs from checkpoint:"
                    f" {getattr(config, key)} (ours) vs {getattr(checkpoint['config'], key)} (checkpoint)"
                )

    if checkpoint is None:
        # Our epochs go from 1 to n_epoch, inclusive
        start_epoch = 1
    else:
        # Continue from where we left off
        start_epoch = checkpoint["epoch"] + 1
        if config.seed is not None:
            # Make sure we don't get the same behaviour as we did on the
            # first epoch repeated on this resumed epoch.
            utils.set_rng_seeds_fixed(config.seed + start_epoch, all_gpu=False)

    # MODEL ===================================================================

    # Encoder -----------------------------------------------------------------
    # Build our Encoder.
    # We have to build the encoder before we load the dataset because it will
    # inform us about what size images we should produce in the preprocessing pipeline.
    n_class, raw_img_size, img_channels = datasets.image_dataset_sizes(config.dataset_name)
    if img_channels > 3 and config.freeze_encoder:
        raise ValueError(
            "Using a dataset with more than 3 image channels will require retraining the encoder"
            ", but a frozen encoder was requested."
        )
    if config.arch_framework == "timm":
        encoder, encoder_config = encoders.get_timm_encoder(config.arch, config.pretrained, in_chans=img_channels)
    elif config.arch_framework == "torchvision":
        # It's trickier to implement this for torchvision models, because they
        # don't have the same naming conventions for model names as in timm;
        # need us to specify the name of the weights when loading a pretrained
        # model; and don't support changing the number of input channels.
        raise NotImplementedError(f"Unsupported architecture framework: {config.arch_framework}")
    else:
        raise ValueError(f"Unknown architecture framework: {config.arch_framework}")

    if config.freeze_encoder and not config.pretrained:
        warnings.warn(
            "A frozen encoder was requested, but the encoder is not pretrained.",
            UserWarning,
            stacklevel=2,
        )

    if config.image_size is None:
        if "input_size" in encoder_config:
            config.image_size = encoder_config["input_size"][-1]
            print(f"Setting model input image size to encoder's expected input size: {config.image_size}")
        else:
            config.image_size = 224
            print(f"Setting model input image size to default: {config.image_size}")
            if raw_img_size:
                warnings.warn(
                    "Be aware that we are using a different input image size"
                    f" ({config.image_size}px) to the raw image size in the"
                    f" dataset ({raw_img_size}px).",
                    UserWarning,
                    stacklevel=2,
                )
    elif "input_size" in encoder_config and config.pretrained and encoder_config["input_size"][-1] != config.image_size:
        warnings.warn(
            f"A different image size {config.image_size} than what the model was"
            f" pretrained with {encoder_config['input_size'][-1]} was suplied",
            UserWarning,
            stacklevel=2,
        )

    # Classifier -------------------------------------------------------------
    # Build our classifier head
    classifier = nn.Linear(encoder_config["n_feature"], n_class)

    # Configure model for distributed training --------------------------------
    print("\nEncoder architecture:")
    print(encoder)
    print("\nClassifier architecture:")
    print(classifier)
    print()

    if config.cpu_workers is None:
        config.cpu_workers = utils.get_num_cpu_available()

    if not use_cuda:
        print("Using CPU (this will be slow)")
    elif config.distributed:
        # Convert batchnorm into SyncBN, using stats computed from all GPUs
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        classifier = nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        # For multiprocessing distributed, the DistributedDataParallel
        # constructor should always set a single device scope, otherwise
        # DistributedDataParallel will use all available devices.
        encoder.to(device)
        classifier.to(device)
        torch.cuda.set_device(device)
        encoder = nn.parallel.DistributedDataParallel(
            encoder, device_ids=[config.local_rank], output_device=config.local_rank
        )
        classifier = nn.parallel.DistributedDataParallel(
            classifier, device_ids=[config.local_rank], output_device=config.local_rank
        )
    else:
        if config.local_rank is not None:
            torch.cuda.set_device(config.local_rank)
        encoder = encoder.to(device)
        classifier = classifier.to(device)

    # DATASET =================================================================
    # Transforms --------------------------------------------------------------
    transform_args = {}
    if config.dataset_name in data_transformations.VALID_TRANSFORMS:
        transform_args["normalization"] = config.dataset_name

    if "mean" in encoder_config:
        transform_args["mean"] = encoder_config["mean"]
    if "std" in encoder_config:
        transform_args["std"] = encoder_config["std"]

    transform_train, transform_eval = data_transformations.get_transform(
        config.transform_type, config.image_size, transform_args
    )

    # Dataset -----------------------------------------------------------------
    dataset_args = {
        "dataset": config.dataset_name,
        "root": config.data_dir,
        "prototyping": config.prototyping,
        "download": config.allow_download_dataset,
    }
    if config.protoval_split_id is not None:
        dataset_args["protoval_split_id"] = config.protoval_split_id
    (
        dataset_train,
        dataset_val,
        dataset_test,
        distinct_val_test,
    ) = datasets.fetch_dataset(
        **dataset_args,
        transform_train=transform_train,
        transform_eval=transform_eval,
    )
    eval_set = "Val" if distinct_val_test else "Test"

    # Dataloader --------------------------------------------------------------
    dl_train_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": True,
        "sampler": None,
        "shuffle": True,
        "worker_init_fn": utils.worker_seed_fn,
    }
    dl_test_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": False,
        "sampler": None,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
    }
    if use_cuda:
        cuda_kwargs = {"num_workers": config.cpu_workers, "pin_memory": True}
        dl_train_kwargs.update(cuda_kwargs)
        dl_test_kwargs.update(cuda_kwargs)

    dl_val_kwargs = copy.deepcopy(dl_test_kwargs)

    if config.distributed:
        # The DistributedSampler breaks up the dataset across the GPUs
        dl_train_kwargs["sampler"] = DistributedSampler(
            dataset_train,
            shuffle=True,
            seed=config.seed if config.seed is not None else 0,
            drop_last=False,
        )
        dl_train_kwargs["shuffle"] = None
        dl_val_kwargs["sampler"] = DistributedSampler(
            dataset_val,
            shuffle=False,
            drop_last=False,
        )
        dl_val_kwargs["shuffle"] = None
        dl_test_kwargs["sampler"] = DistributedSampler(
            dataset_test,
            shuffle=False,
            drop_last=False,
        )
        dl_test_kwargs["shuffle"] = None

    dataloader_train = torch.utils.data.DataLoader(dataset_train, **dl_train_kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, **dl_val_kwargs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **dl_test_kwargs)

    # OPTIMIZATION ============================================================
    # Optimizer ---------------------------------------------------------------
    # Set up the optimizer

    # Bigger batch sizes mean better estimates of the gradient, so we can use a
    # bigger learning rate. See https://arxiv.org/abs/1706.02677
    # Hence we scale the learning rate linearly with the total batch size.
    config.lr = config.lr_relative * config.batch_size / BASE_BATCH_SIZE

    # Freeze the encoder, if requested
    if config.freeze_encoder:
        for m in encoder.parameters():
            m.requires_grad = False

    # Set up a parameter group for each component of the model, allowing
    # them to have different learning rates (for fine-tuning encoder).
    params = []
    if not config.freeze_encoder:
        params.append(
            {
                "params": encoder.parameters(),
                "lr": config.lr * config.lr_encoder_mult,
                "name": "encoder",
            }
        )
    params.append(
        {
            "params": classifier.parameters(),
            "lr": config.lr * config.lr_classifier_mult,
            "name": "classifier",
        }
    )

    # Fetch the constructor of the appropriate optimizer from torch.optim
    optimizer = getattr(torch.optim, config.optimizer)(params, lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler ---------------------------------------------------------------
    # Set up the learning rate scheduler
    if config.scheduler.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [p["lr"] for p in optimizer.param_groups],
            epochs=config.epochs,
            steps_per_epoch=len(dataloader_train),
        )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} not supported.")

    # Loss function -----------------------------------------------------------
    # Set up loss function
    criterion = nn.CrossEntropyLoss()

    # LOGGING =================================================================
    # Setup logging and saving

    # If we're using wandb, initialize the run, or resume it if the job was preempted.
    if config.log_wandb and config.global_rank == 0:
        wandb_run_name = config.run_name
        if wandb_run_name is not None and config.run_id is not None:
            wandb_run_name = f"{wandb_run_name}__{config.run_id}"
        EXCLUDED_WANDB_CONFIG_KEYS = [
            "log_wandb",
            "wandb_entity",
            "wandb_project",
            "global_rank",
            "local_rank",
            "run_name",
            "run_id",
            "model_output_dir",
        ]
        utils.init_or_resume_wandb_run(
            config.model_output_dir,
            name=wandb_run_name,
            id=config.run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=wandb.helper.parse_config(config, exclude=EXCLUDED_WANDB_CONFIG_KEYS),
            job_type="train",
            tags=["prototype" if config.prototyping else "final"],
        )
        # If a run_id was not supplied at the command prompt, wandb will
        # generate a name. Let's use that as the run_name.
        if config.run_name is None:
            config.run_name = wandb.run.name
        if config.run_id is None:
            config.run_id = wandb.run.id

    # If we still don't have a run name, generate one from the current time.
    if config.run_name is None:
        config.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.run_id is None:
        config.run_id = utils.generate_id()

    # If no checkpoint path was supplied, but models_dir was, we will automatically
    # determine the path to which we will save the model checkpoint.
    # If both are empty, we won't save the model.
    if not config.checkpoint_path and config.models_dir:
        config.model_output_dir = os.path.join(
            config.models_dir,
            config.dataset_name,
            f"{config.run_name}__{config.run_id}",
        )
        config.checkpoint_path = os.path.join(config.model_output_dir, "checkpoint_latest.pt")
        if config.log_wandb and config.global_rank == 0:
            wandb.config.update({"checkpoint_path": config.checkpoint_path}, allow_val_change=True)

    if config.checkpoint_path is None:
        print("Model will not be saved.")
    else:
        print(f"Model will be saved to '{config.checkpoint_path}'")

    # RESUME ==================================================================
    # Now that everything is set up, we can load the state of the model,
    # optimizer, and scheduler from a checkpoint, if supplied.

    # Initialize step related variables as if we're starting from scratch.
    # Their values will be overridden by the checkpoint if we're resuming.
    total_step = 0
    n_samples_seen = 0

    best_stats = {"max_accuracy": 0, "best_epoch": 0}

    if checkpoint is not None:
        print(f"Loading state from checkpoint (epoch {checkpoint['epoch']})")
        # Map model to be loaded to specified single gpu.
        total_step = checkpoint["total_step"]
        n_samples_seen = checkpoint["n_samples_seen"]
        encoder.load_state_dict(checkpoint["encoder"])
        classifier.load_state_dict(checkpoint["classifier"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_stats["max_accuracy"] = checkpoint.get("max_accuracy", 0)
        best_stats["best_epoch"] = checkpoint.get("best_epoch", 0)

    # TRAIN ===================================================================
    print()
    print("Configuration:")
    print()
    print(config)
    print()

    # Ensure modules are on the correct device
    encoder = encoder.to(device)
    classifier = classifier.to(device)

    # Stack the encoder and classifier together to create an overall model.
    # At inference time, we don't need to make a distinction between modules
    # within this stack.
    model = nn.Sequential(encoder, classifier)

    timing_stats = {}
    t_end_epoch = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        t_start_epoch = time.time()
        if config.seed is not None:
            # If the job is resumed from preemption, our RNG state is currently set the
            # same as it was at the start of the first epoch, not where it was when we
            # stopped training. This is not good as it means jobs which are resumed
            # don't do the same thing as they would be if they'd run uninterrupted
            # (making preempted jobs non-reproducible).
            # To address this, we reset the seed at the start of every epoch. Since jobs
            # can only save at the end of and resume at the start of an epoch, this
            # makes the training process reproducible. But we shouldn't use the same
            # RNG state for each epoch - instead we use the original seed to define the
            # series of seeds that we will use at the start of each epoch.
            epoch_seed = utils.determine_epoch_seed(config.seed, epoch=epoch)
            # We want each GPU to have a different seed to the others to avoid
            # correlated randomness between the workers on the same batch.
            # We offset the seed for this epoch by the GPU rank, so every GPU will get a
            # unique seed for the epoch. This means the job is only precisely
            # reproducible if it is rerun with the same number of GPUs (and the same
            # number of CPU workers for the dataloader).
            utils.set_rng_seeds_fixed(epoch_seed + config.global_rank, all_gpu=False)
            if isinstance(getattr(dataloader_train, "generator", None), torch.Generator):
                # Finesse the dataloader's RNG state, if it is not using the global state.
                dataloader_train.generator.manual_seed(epoch_seed + config.global_rank)
            if isinstance(getattr(dataloader_train.sampler, "generator", None), torch.Generator):
                # Finesse the sampler's RNG state, if it is not using the global RNG state.
                dataloader_train.sampler.generator.manual_seed(config.seed + epoch + 10000 * config.global_rank)

        if hasattr(dataloader_train.sampler, "set_epoch"):
            # Handling for DistributedSampler.
            # Set the epoch for the sampler so that it can shuffle the data
            # differently for each epoch, but synchronized across all GPUs.
            dataloader_train.sampler.set_epoch(epoch)

        # Train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Note the number of samples seen before this epoch started, so we can
        # calculate the number of samples seen in this epoch.
        n_samples_seen_before = n_samples_seen
        # Run one epoch of training
        train_stats, total_step, n_samples_seen = train_one_epoch(
            config=config,
            encoder=encoder,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            dataloader=dataloader_train,
            device=device,
            epoch=epoch,
            n_epoch=config.epochs,
            total_step=total_step,
            n_samples_seen=n_samples_seen,
        )
        t_end_train = time.time()

        timing_stats["train"] = t_end_train - t_start_epoch
        n_epoch_samples = n_samples_seen - n_samples_seen_before
        train_stats["throughput"] = n_epoch_samples / timing_stats["train"]

        print(f"Training epoch {epoch}/{config.epochs} summary:")
        print(f"  Steps ..............{len(dataloader_train):8d}")
        print(f"  Samples ............{n_epoch_samples:8d}")
        if timing_stats["train"] > 172800:
            print(f"  Duration ...........{timing_stats['train']/86400:11.2f} days")
        elif timing_stats["train"] > 5400:
            print(f"  Duration ...........{timing_stats['train']/3600:11.2f} hours")
        elif timing_stats["train"] > 120:
            print(f"  Duration ...........{timing_stats['train']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['train']:11.2f} seconds")
        print(f"  Throughput .........{train_stats['throughput']:11.2f} samples/sec")
        print(f"  Loss ...............{train_stats['loss']:14.5f}")
        print(f"  Accuracy ...........{train_stats['accuracy']:11.2f} %")

        # Validate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate on validation set
        t_start_val = time.time()

        eval_stats = evaluate(
            dataloader=dataloader_val,
            model=model,
            device=device,
            partition_name=eval_set,
            is_distributed=config.distributed,
        )
        t_end_val = time.time()
        timing_stats["val"] = t_end_val - t_start_val
        eval_stats["throughput"] = len(dataloader_val.dataset) / timing_stats["val"]

        # Check if this is the new best model
        if eval_stats["accuracy"] >= best_stats["max_accuracy"]:
            best_stats["max_accuracy"] = eval_stats["accuracy"]
            best_stats["best_epoch"] = epoch

        # Save model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        t_start_save = time.time()
        if config.model_output_dir and (not config.distributed or config.global_rank == 0):
            print(f"\nSaving model to {config.checkpoint_path}")
            # Save to a temporary file first, then move the temporary file to the target
            # destination. This is to prevent clobbering the checkpoint with a partially
            # saved file, in the event that the saving process is interrupted. Saving
            # the checkpoint takes a little while and can be disrupted by preemption,
            # whereas moving the file is an atomic operation.
            tmp_a, tmp_b = os.path.split(config.checkpoint_path)
            tmp_fname = os.path.join(tmp_a, ".tmp." + tmp_b)
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "total_step": total_step,
                    "n_samples_seen": n_samples_seen,
                    "config": config,
                    "encoder_config": encoder_config,
                    "transform_args": transform_args,
                    **best_stats,
                },
                tmp_fname,
            )
            os.rename(tmp_fname, config.checkpoint_path)
            print(f"Saved model to  {config.checkpoint_path}")

            if config.save_best_model and best_stats["best_epoch"] == epoch:
                ckpt_path_best = os.path.join(config.model_output_dir, "best_model.pt")
                print(f"Copying model to {ckpt_path_best}")
                shutil.copyfile(config.checkpoint_path, ckpt_path_best)

        t_end_save = time.time()
        timing_stats["saving"] = t_end_save - t_start_save

        # Log to wandb ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Overall time won't include uploading to wandb, but there's nothing
        # we can do about that.
        timing_stats["overall"] = time.time() - t_end_epoch
        t_end_epoch = time.time()

        # Send training and eval stats for this epoch to wandb
        if config.log_wandb and config.global_rank == 0:
            pre = "Training/epochwise"
            wandb.log(
                {
                    "Training/stepwise/epoch": epoch,
                    "Training/stepwise/epoch_progress": epoch,
                    "Training/stepwise/n_samples_seen": n_samples_seen,
                    f"{pre}/epoch": epoch,
                    **{f"{pre}/Train/{k}": v for k, v in train_stats.items()},
                    **{f"{pre}/{eval_set}/{k}": v for k, v in eval_stats.items()},
                    **{f"{pre}/duration/{k}": v for k, v in timing_stats.items()},
                },
                step=total_step,
            )
            # Record the wandb time as contributing to the next epoch
            timing_stats = {"wandb": time.time() - t_end_epoch}
        else:
            # Reset timing stats
            timing_stats = {}
        # Print with flush=True forces the output buffer to be printed immediately
        print("", flush=True)

    if start_epoch > config.epochs:
        print("Training already completed!")
    else:
        print(f"Training complete! (Trained epochs {start_epoch} to {config.epochs})")
    print(
        f"Best {eval_set} accuracy was {best_stats['max_accuracy']:.2f}%,"
        f" seen at the end of epoch {best_stats['best_epoch']}"
    )

    # TEST ====================================================================
    print(f"\nEvaluating final model (epoch {config.epochs}) performance")
    # Evaluate on test set
    print("\nEvaluating final model on test set...")
    eval_stats = evaluate(
        dataloader=dataloader_test,
        model=model,
        device=device,
        partition_name="Test",
        is_distributed=config.distributed,
    )
    # Send stats to wandb
    if config.log_wandb and config.global_rank == 0:
        wandb.log({**{f"Eval/Test/{k}": v for k, v in eval_stats.items()}}, step=total_step)

    if distinct_val_test:
        # Evaluate on validation set
        print(f"\nEvaluating final model on {eval_set} set...")
        eval_stats = evaluate(
            dataloader=dataloader_val,
            model=model,
            device=device,
            partition_name=eval_set,
            is_distributed=config.distributed,
        )
        # Send stats to wandb
        if config.log_wandb and config.global_rank == 0:
            wandb.log(
                {**{f"Eval/{eval_set}/{k}": v for k, v in eval_stats.items()}},
                step=total_step,
            )

    # Create a copy of the train partition with evaluation transforms
    # and a dataloader using the evaluation configuration (don't drop last)
    print("\nEvaluating final model on train set under test conditions (no augmentation, dropout, etc)...")
    dataset_train_eval = datasets.fetch_dataset(
        **dataset_args,
        transform_train=transform_eval,
        transform_eval=transform_eval,
    )[0]
    dl_train_eval_kwargs = copy.deepcopy(dl_test_kwargs)
    if config.distributed:
        # The DistributedSampler breaks up the dataset across the GPUs
        dl_train_eval_kwargs["sampler"] = DistributedSampler(
            dataset_train_eval,
            shuffle=False,
            drop_last=False,
        )
        dl_train_eval_kwargs["shuffle"] = None
    dataloader_train_eval = torch.utils.data.DataLoader(dataset_train_eval, **dl_train_eval_kwargs)
    eval_stats = evaluate(
        dataloader=dataloader_train_eval,
        model=model,
        device=device,
        partition_name="Train",
        is_distributed=config.distributed,
    )
    # Send stats to wandb
    if config.log_wandb and config.global_rank == 0:
        wandb.log({**{f"Eval/Train/{k}": v for k, v in eval_stats.items()}}, step=total_step)


def train_one_epoch(
    config,
    encoder,
    classifier,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    device="cuda",
    epoch=1,
    n_epoch=None,
    total_step=0,
    n_samples_seen=0,
):
    r"""
    Train the encoder and classifier for one epoch.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The global config object.
    encoder : torch.nn.Module
        The encoder network.
    classifier : torch.nn.Module
        The classifier network.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    criterion : torch.nn.Module
        The loss function.
    dataloader : torch.utils.data.DataLoader
        A dataloader for the training set.
    device : str or torch.device, default="cuda"
        The device to use.
    epoch : int, default=1
        The current epoch number (indexed from 1).
    n_epoch : int, optional
        The total number of epochs scheduled to train for.
    total_step : int, default=0
        The total number of steps taken so far.
    n_samples_seen : int, default=0
        The total number of samples seen so far.

    Returns
    -------
    results: dict
        A dictionary containing the training performance for this epoch.
    total_step : int
        The total number of steps taken after this epoch.
    n_samples_seen : int
        The total number of samples seen after this epoch.
    """
    # Put the model in train mode
    encoder.train()
    classifier.train()

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    loss_epoch = 0
    acc_epoch = 0

    if config.print_interval is None:
        # Default to printing to console every time we log to wandb
        config.print_interval = config.log_interval

    t_end_batch = time.time()
    t_start_wandb = t_end_wandb = None
    for batch_idx, (stimuli, y_true) in enumerate(dataloader):
        t_start_batch = time.time()
        batch_size_this_gpu = stimuli.shape[0]

        # Move training inputs and targets to the GPU
        stimuli = stimuli.to(device)
        y_true = y_true.to(device)

        # Forward pass --------------------------------------------------------
        # Perform the forward pass through the model
        t_start_encoder = time.time()
        # N.B. To accurately time steps on GPU we need to use torch.cuda.Event
        ct_forward = torch.cuda.Event(enable_timing=True)
        ct_forward.record()
        with torch.no_grad() if config.freeze_encoder else nullcontext():
            h = encoder(stimuli)
        logits = classifier(h)
        # Reset gradients
        optimizer.zero_grad()
        # Measure loss
        loss = criterion(logits, y_true)

        # Backward pass -------------------------------------------------------
        # Now the backward pass
        ct_backward = torch.cuda.Event(enable_timing=True)
        ct_backward.record()
        loss.backward()

        # Update --------------------------------------------------------------
        # Use our optimizer to update the model parameters
        ct_optimizer = torch.cuda.Event(enable_timing=True)
        ct_optimizer.record()
        optimizer.step()

        # Step the scheduler each batch
        scheduler.step()

        # Increment training progress counters
        total_step += 1
        batch_size_all = batch_size_this_gpu * config.world_size
        n_samples_seen += batch_size_all

        # Logging -------------------------------------------------------------
        # Log details about training progress
        t_start_logging = time.time()
        ct_logging = torch.cuda.Event(enable_timing=True)
        ct_logging.record()

        # Update the total loss for the epoch
        if config.distributed:
            # Fetch results from other GPUs
            loss_batch = torch.mean(utils.concat_all_gather(loss.reshape((1,))))
            loss_batch = loss_batch.item()
        else:
            loss_batch = loss.item()
        loss_epoch += loss_batch

        # Compute accuracy
        with torch.no_grad():
            y_pred = torch.argmax(logits, dim=-1)
            is_correct = y_pred == y_true
            acc = 100.0 * is_correct.sum() / len(is_correct)
            if config.distributed:
                # Fetch results from other GPUs
                acc = torch.mean(utils.concat_all_gather(acc.reshape((1,))))
            acc = acc.item()
            acc_epoch += acc

        if epoch <= 1 and batch_idx == 0:
            # Debugging
            print("stimuli.shape =", stimuli.shape)
            print("y_true.shape  =", y_true.shape)
            print("y_pred.shape  =", y_pred.shape)
            print("logits.shape  =", logits.shape)
            print("loss.shape    =", loss.shape)
            # Debugging intensifies
            print("y_true =", y_true)
            print("y_pred =", y_pred)
            print("logits[0] =", logits[0])
            print("loss =", loss.detach().item())

        # Log sample training images to show on wandb
        if config.log_wandb and batch_idx <= 1:
            # Log 8 example training images from each GPU
            img_indices = [offset + relative for offset in [0, batch_size_this_gpu // 2] for relative in [0, 1, 2, 3]]
            img_indices = sorted(set(img_indices))
            log_images = stimuli[img_indices]
            if config.distributed:
                # Collate sample images from each GPU
                log_images = utils.concat_all_gather(log_images)
            if config.global_rank == 0:
                wandb.log(
                    {"Training/stepwise/Train/stimuli": wandb.Image(log_images)},
                    step=total_step,
                )

        # Log to console
        if batch_idx <= 2 or batch_idx % config.print_interval == 0 or batch_idx >= len(dataloader) - 1:
            print(
                f"Train Epoch:{epoch:4d}" + (f"/{n_epoch}" if n_epoch is not None else ""),
                " Step:{:4d}/{}".format(batch_idx + 1, len(dataloader)),
                " Loss:{:8.5f}".format(loss_batch),
                " Acc:{:6.2f}%".format(acc),
                " LR: {}".format(scheduler.get_last_lr()),
            )

        # Log to wandb
        if config.log_wandb and config.global_rank == 0 and batch_idx % config.log_interval == 0:
            # Create a log dictionary to send to wandb
            # Epoch progress interpolates smoothly between epochs
            epoch_progress = epoch - 1 + (batch_idx + 1) / len(dataloader)
            # Throughput is the number of samples processed per second
            throughput = batch_size_all / (t_start_logging - t_end_batch)
            log_dict = {
                "Training/stepwise/epoch": epoch,
                "Training/stepwise/epoch_progress": epoch_progress,
                "Training/stepwise/n_samples_seen": n_samples_seen,
                "Training/stepwise/Train/throughput": throughput,
                "Training/stepwise/Train/loss": loss_batch,
                "Training/stepwise/Train/accuracy": acc,
            }
            # Track the learning rate of each parameter group
            for lr_idx in range(len(optimizer.param_groups)):
                if "name" in optimizer.param_groups[lr_idx]:
                    grp_name = optimizer.param_groups[lr_idx]["name"]
                elif len(optimizer.param_groups) == 1:
                    grp_name = ""
                else:
                    grp_name = f"grp{lr_idx}"
                if grp_name != "":
                    grp_name = f"-{grp_name}"
                grp_lr = optimizer.param_groups[lr_idx]["lr"]
                log_dict[f"Training/stepwise/lr{grp_name}"] = grp_lr
            # Synchronize ensures everything has finished running on each GPU
            torch.cuda.synchronize()
            # Record how long it took to do each step in the pipeline
            pre = "Training/stepwise/duration"
            if t_start_wandb is not None:
                # Record how long it took to send to wandb last time
                log_dict[f"{pre}/wandb"] = t_end_wandb - t_start_wandb
            log_dict[f"{pre}/dataloader"] = t_start_batch - t_end_batch
            log_dict[f"{pre}/preamble"] = t_start_encoder - t_start_batch
            log_dict[f"{pre}/forward"] = ct_forward.elapsed_time(ct_backward) / 1000
            log_dict[f"{pre}/backward"] = ct_backward.elapsed_time(ct_optimizer) / 1000
            log_dict[f"{pre}/optimizer"] = ct_optimizer.elapsed_time(ct_logging) / 1000
            log_dict[f"{pre}/overall"] = time.time() - t_end_batch
            t_start_wandb = time.time()
            log_dict[f"{pre}/logging"] = t_start_wandb - t_start_logging
            # Send to wandb
            wandb.log(log_dict, step=total_step)
            t_end_wandb = time.time()

        # Record the time when we finished this batch
        t_end_batch = time.time()

    results = {
        "loss": loss_epoch / len(dataloader),
        "accuracy": acc_epoch / len(dataloader),
    }
    return results, total_step, n_samples_seen


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import sys

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train image classification model.",
        add_help=False,
    )
    # Help arg ----------------------------------------------------------------
    group = parser.add_argument_group("Help")
    group.add_argument(
        "--help",
        "-h",
        action="help",
        help="Show this help message and exit.",
    )
    # Dataset args ------------------------------------------------------------
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="cifar10",
        help="Name of the dataset to learn. Default: %(default)s",
    )
    group.add_argument(
        "--prototyping",
        dest="protoval_split_id",
        nargs="?",
        const=0,
        type=int,
        help=(
            "Use a subset of the train partition for both train and val."
            " If the dataset doesn't have a separate val and test set with"
            " public labels (which is the case for most datasets), the train"
            " partition will be reduced in size to create the val partition."
            " In all cases where --prototyping is enabled, the test set is"
            " never used during training. Generally, you should use"
            " --prototyping throughout the model exploration and hyperparameter"
            " optimization phases, and disable it for your final experiments so"
            " they can run on a completely held-out test set."
        ),
    )
    group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory within which the dataset can be found."
            " Default is ~/Datasets, except on Vector servers where it is"
            " adjusted as appropriate depending on the dataset's location."
        ),
    )
    group.add_argument(
        "--allow-download-dataset",
        action="store_true",
        help="Attempt to download the dataset if it is not found locally.",
    )
    group.add_argument(
        "--transform-type",
        type=str,
        default="cifar",
        help="Name of augmentation stack to apply to training data. Default: %(default)s",
    )
    group.add_argument(
        "--image-size",
        type=int,
        help="Size of images to use as model input. Default: encoder's default.",
    )
    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Architecture")
    group.add_argument(
        "--model",
        "--encoder",
        "--arch",
        "--architecture",
        dest="arch",
        type=str,
        default="resnet18",
        help="Name of model architecture. Default: %(default)s",
    )
    group.add_argument(
        "--pretrained",
        action="store_true",
        help="Use default pretrained model weights, taken from hugging-face hub.",
    )
    mx_group = group.add_mutually_exclusive_group()
    mx_group.add_argument(
        "--torchvision",
        dest="arch_framework",
        action="store_const",
        const="torchvision",
        default="timm",
        help="Use model architecture from torchvision (default is timm).",
    )
    mx_group.add_argument(
        "--timm",
        dest="arch_framework",
        action="store_const",
        const="timm",
        default="timm",
        help="Use model architecture from timm (default).",
    )
    group.add_argument(
        "--freeze-encoder",
        action="store_true",
    )
    # Optimization args -------------------------------------------------------
    group = parser.add_argument_group("Optimization routine")
    group.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train for. Default: %(default)s",
    )
    group.add_argument(
        "--lr",
        dest="lr_relative",
        type=float,
        default=0.01,
        help=(
            f"Maximum learning rate, set per {BASE_BATCH_SIZE} batch size."
            " The actual learning rate used will be scaled up by the total"
            " batch size (across all GPUs). Default: %(default)s"
        ),
    )
    group.add_argument(
        "--lr-encoder-mult",
        type=float,
        default=1.0,
        help="Multiplier for encoder learning rate, relative to overall LR.",
    )
    group.add_argument(
        "--lr-classifier-mult",
        type=float,
        default=1.0,
        help="Multiplier for classifier head's learning rate, relative to overall LR.",
    )
    group.add_argument(
        "--weight-decay",
        "--wd",
        dest="weight_decay",
        type=float,
        default=0.0,
        help="Weight decay. Default: %(default)s",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Name of optimizer (case-sensitive). Default: %(default)s",
    )
    group.add_argument(
        "--scheduler",
        type=str,
        default="OneCycle",
        help="Learning rate scheduler. Default: %(default)s",
    )
    # Output checkpoint args --------------------------------------------------
    group = parser.add_argument_group("Output checkpoint")
    group.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="PATH",
        help="Output directory for all models. Ignored if --checkpoint is set. Default: %(default)s",
    )
    group.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        help=(
            "Save and resume partially trained model and optimizer state from this checkpoint."
            " Overrides --models-dir and --models-dir."
        ),
    )
    group.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save a copy of the model with best validation performance.",
    )
    # Reproducibility args ----------------------------------------------------
    group = parser.add_argument_group("Reproducibility")
    group.add_argument(
        "--seed",
        type=int,
        help="Random number generator (RNG) seed. Default: not controlled",
    )
    group.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable non-deterministic features of cuDNN.",
    )
    # Hardware configuration args ---------------------------------------------
    group = parser.add_argument_group("Hardware configuration")
    group.add_argument(
        "--batch-size",
        dest="batch_size_per_gpu",
        type=int,
        default=BASE_BATCH_SIZE,
        help=(
            "Batch size per GPU. The total batch size will be this value times"
            " the total number of GPUs used. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--cpu-workers",
        "--workers",
        dest="cpu_workers",
        type=int,
        help="Number of CPU workers per node. Default: number of CPUs available on device.",
    )
    group.add_argument(
        "--no-cuda",
        action="store_true",
        help="Use CPU only, no GPUs.",
    )
    group.add_argument(
        "--gpu",
        "--local-rank",
        dest="local_rank",
        default=None,
        type=int,
        help="Index of GPU to use when training a single process. (Ignored for distributed training.)",
    )
    # Logging args ------------------------------------------------------------
    group = parser.add_argument_group("Debugging and logging")
    group.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Number of batches between each log to wandb (if enabled). Default: %(default)s",
    )
    group.add_argument(
        "--print-interval",
        type=int,
        default=None,
        help="Number of batches between each print to STDOUT. Default: same as LOG_INTERVAL.",
    )
    group.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log results with Weights & Biases https://wandb.ai",
    )
    group.add_argument(
        "--disable-wandb",
        "--no-wandb",
        dest="disable_wandb",
        action="store_true",
        help="Overrides --log-wandb and ensures wandb is always disabled.",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        help=(
            "The entity (organization) within which your wandb project is"
            ' located. By default, this will be your "default location" set on'
            " wandb at https://wandb.ai/settings"
        ),
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="template-experiment",
        help="Name of project on wandb, where these runs will be saved. Default: %(default)s",
    )
    group.add_argument(
        "--run-name",
        type=str,
        help="Human-readable identifier for the model run or job. Used to name the run on wandb.",
    )
    group.add_argument(
        "--run-id",
        type=str,
        help="Unique identifier for the model run or job. Used as the run ID on wandb.",
    )

    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    # Set protoval_split_id from prototyping, and turn prototyping into a bool
    config.prototyping = config.protoval_split_id is not None
    return run(config)


if __name__ == "__main__":
    cli()
