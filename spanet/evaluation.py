from glob import glob
from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map

from rich import progress

from spanet import JetReconstructionModel, Options
from spanet.dataset.types import Evaluation, Outputs, Source
from spanet.network.jet_reconstruction.jet_reconstruction_network import extract_predictions

from collections import defaultdict
import os
import re


def dict_concatenate(tree):
    output = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            output[key] = dict_concatenate(value)
        else:
            output[key] = np.concatenate(value)

    return output

def select_best_checkpoint(base_folder_path, metric_name, mode='min'):
    """
    Selects the checkpoint file with the best metric value.
    Works for metric names with 0, 1, or multiple slashes.
    
    Args:
        base_folder_path (str): Base output path (e.g., spanet output version folder).
        metric_name (str): Metric name like 'total_loss', 'validation_loss/total_loss', etc.
        mode (str): 'min' to select lowest value, 'max' to select highest value.

    Returns:
        str: Full path to the best checkpoint file.
    """
    metric_parts = metric_name.split('/')

    if len(metric_parts) == 1:
        # No slashes, simple metric
        metric_folder = ''
        metric_filename = metric_parts[0]
    else:
        # Slashes exist
        metric_folder = os.path.join(*metric_parts[:-1])
        metric_filename = metric_parts[-1]

    # Build full folder path
    if metric_folder:
        folder_path = os.path.join(base_folder_path, 'checkpoints', metric_folder)
    else:
        folder_path = os.path.join(base_folder_path, 'checkpoints')

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist!")

    safe_metric = re.escape(metric_filename)
    pattern = re.compile(rf'{safe_metric}=(-?[0-9.]+)(?:-v(\d+))?\.ckpt')

    files = os.listdir(folder_path)

    metric_files = []

    for filename in files:
        match = pattern.search(filename)
        if match:
            value = float(match.group(1))
            version = int(match.group(2)) if match.group(2) else -1  # No version = -1
            metric_files.append((value, version, filename))

    if not metric_files:
        raise ValueError(f"No checkpoint files found matching metric '{metric_filename}' in {folder_path}.")

    if mode == 'min':
        best_file = min(metric_files, key=lambda x: (x[0], -x[1]))[2]
    elif mode == 'max':
        best_file = max(metric_files, key=lambda x: (x[0], x[1]))[2]
    else:
        raise ValueError("mode must be 'min' or 'max'.")

    # Return full path to the best checkpoint
    return os.path.join(folder_path, best_file)


def tree_concatenate(trees):
    leaves = []
    for tree in trees:
        data, tree_spec = tree_flatten(tree)
        leaves.append(data)

    results = [np.concatenate(l) for l in zip(*leaves)]
    return tree_unflatten(results, tree_spec)


def load_model(
    log_directory: str,
    testing_file: Optional[str] = None,
    event_info_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    cuda: bool = False,
    fp16: bool = False,
    checkpoint: Optional[str] = None
) -> JetReconstructionModel:

    # Load the options that were used for this run and set the testing-dataset value
    options = Options.load(f"{log_directory}/options.json")
    # Load the best-performing checkpoint on validation data
    if checkpoint is None:
        checkpoint = select_best_checkpoint(log_directory, options.checkpoint_metric, mode=options.checkpoint_mode)
        print(f"Loading: {checkpoint}")

    checkpoint = torch.load(checkpoint, map_location='cpu')
    checkpoint = checkpoint["state_dict"]
    if fp16:
        checkpoint = tree_map(lambda x: x.half(), checkpoint)

    

    # Override options from command line arguments
    if testing_file is not None:
        options.testing_file = testing_file

    if event_info_file is not None:
        options.event_info_file = event_info_file

    if batch_size is not None:
        options.batch_size = batch_size

    # Create model and disable all training operations for speed
    model = JetReconstructionModel(options)
    model.load_state_dict(checkpoint)
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if cuda:
        model = model.cuda()

    return model


def evaluate_on_test_dataset(
        model: JetReconstructionModel,
        progress=progress,
        return_full_output: bool = False,
        fp16: bool = False
) -> Union[Evaluation, Tuple[Evaluation, Outputs]]:
    full_assignments = defaultdict(list)
    full_assignment_probabilities = defaultdict(list)
    full_detection_probabilities = defaultdict(list)

    full_classifications = defaultdict(list)
    full_regressions = defaultdict(list)

    full_outputs = []

    dataloader = model.test_dataloader()
    if progress:
        dataloader = progress.track(model.test_dataloader(), description="Evaluating Model")

    for batch in dataloader:
        sources = tuple(Source(x[0].to(model.device), x[1].to(model.device)) for x in batch.sources)

        with torch.cuda.amp.autocast(enabled=fp16):
            outputs = model.forward(sources)

        assignment_indices = extract_predictions([
            np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
            for assignment in outputs.assignments
        ])

        detection_probabilities = np.stack([
            torch.sigmoid(detection).cpu().numpy()
            for detection in outputs.detections
        ])

        classifications = {
            key: torch.softmax(classification, 1).cpu().numpy()
            for key, classification in outputs.classifications.items()
        }

        regressions = {
            key: value.cpu().numpy()
            for key, value in outputs.regressions.items()
        }

        assignment_probabilities = []
        dummy_index = torch.arange(assignment_indices[0].shape[0])
        for assignment_probability, assignment, symmetries in zip(
            outputs.assignments,
            assignment_indices,
            model.event_info.product_symbolic_groups.values()
        ):
            # Get the probability of the best assignment.
            # Have to use explicit function call here to construct index dynamically.
            assignment_probability = assignment_probability.__getitem__((dummy_index, *assignment.T))

            # Convert from log-probability to probability.
            assignment_probability = torch.exp(assignment_probability)

            # Multiply by the symmetry factor to account for equivalent predictions.
            assignment_probability = symmetries.order() * assignment_probability

            # Convert back to cpu and add to database.
            assignment_probabilities.append(assignment_probability.cpu().numpy())

        for i, name in enumerate(model.event_info.product_particles):
            full_assignments[name].append(assignment_indices[i])
            full_assignment_probabilities[name].append(assignment_probabilities[i])
            full_detection_probabilities[name].append(detection_probabilities[i])

        for key, regression in regressions.items():
            full_regressions[key].append(regression)

        for key, classification in classifications.items():
            full_classifications[key].append(classification)

        if return_full_output:
            full_outputs.append(tree_map(lambda x: x.cpu().numpy(), outputs))

    evaluation = Evaluation(
        dict_concatenate(full_assignments),
        dict_concatenate(full_assignment_probabilities),
        dict_concatenate(full_detection_probabilities),
        dict_concatenate(full_regressions),
        dict_concatenate(full_classifications)
    )

    if return_full_output:
        return evaluation, tree_concatenate(full_outputs)

    return evaluation

