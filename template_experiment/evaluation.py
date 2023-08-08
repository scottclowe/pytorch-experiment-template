"""
Evaluation routines.
"""

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F

from . import utils


def evaluate(
    dataloader,
    model,
    device,
    partition_name="Val",
    verbosity=1,
    is_distributed=False,
):
    """
    Evaluate model performance on a dataset.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset to evaluate on.
    model : torch.nn.Module
        Model to evaluate.
    device : torch.device
        Device to run the model on.
    partition_name : str, default="Val"
        Name of the partition being evaluated.
    verbosity : int, default=1
        Verbosity level.
    is_distributed : bool, default=False
        Whether the model is distributed across multiple GPUs.

    Returns
    -------
    results : dict
        Dictionary of evaluation results.
    """
    model.eval()

    positives = 0
    total_xent = 0
    total = 0

    y_true_all = []
    y_pred_all = []

    for stimuli, y_true in dataloader:
        stimuli = stimuli.to(device)
        y_true = y_true.to(device)
        with torch.no_grad():
            logits = model(stimuli)
            xent = F.cross_entropy(logits, y_true, reduction="sum")
            y_pred = torch.argmax(logits, dim=-1)
            is_correct = y_pred == y_true

        if is_distributed:
            # Fetch results from other GPUs
            xent = torch.sum(utils.concat_all_gather(xent.reshape((1,))))
            y_true = utils.concat_all_gather_ragged(y_true)
            y_pred = utils.concat_all_gather_ragged(y_pred)
            is_correct = utils.concat_all_gather_ragged(is_correct)

        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())
        positives += torch.sum(is_correct).item()
        total_xent += xent
        total += is_correct.shape[0]

    # Take the mean of the cross-entropy
    xent = xent / total
    # Concatenate the targets and predictions from each batch
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    # Create results dictionary
    results = {}
    results["count"] = total
    results["cross-entropy"] = xent
    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * positives / total
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(
        y_true, y_pred
    )
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(
        y_true, y_pred, average="micro"
    )
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(
        y_true, y_pred, average="macro"
    )
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(
        y_true, y_pred, average="weighted"
    )
    # Could expand to other metrics too

    if verbosity >= 1:
        print(f"\n{partition_name} evaluation results:")
        for k, v in results.items():
            if k == "count":
                print(f"  {k + ' ':.<21s}{v:7d}")
            elif "entropy" in k:
                print(f"  {k + ' ':.<24s} {v:9.5f} nat")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")

    return results
