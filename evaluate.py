# 
import numpy as np
from skimage.measure import label

def evaluatio(preidction, ground_truth):

    dice = get_dice(preidction, ground_truth)
    avd = get_avd(preidction, ground_truth)
    instance_recall, instance_precision = get_recall_and_precision(preidction, ground_truth)

    return dice, avd, instance_recall, instance_precision

def get_dice(preidction, ground_truth):
    """
    Calculate the Dice coefficient between two 3D binary volumes.

    Parameters:
    - true_volume (array_like): The ground truth 3D binary volume.
    - pred_volume (array_like): The predicted 3D binary volume.

    Returns:
    - float: The Dice coefficient.
    """
    true_volume = np.asarray(ground_truth).astype(bool)
    pred_volume = np.asarray(preidction).astype(bool)

    # Logical AND to find the intersection
    intersection = np.logical_and(true_volume, pred_volume)

    # Calculate Dice coefficient
    dice = 2. * intersection.sum() / (true_volume.sum() + pred_volume.sum())

    return dice

def get_avd(prediction, ground_truth):
    """
    Calculate the Normalized Absolute Volume Difference (AVD) between two binary volumes.
    The AVD is normalized by the number of voxels in the ground truth.

    Parameters:
    - prediction (array_like): The predicted binary volume.
    - ground_truth (array_like): The ground truth binary volume.

    Returns:
    - float: The normalized AVD, a value between 0 and 1.
    """
    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)

    # Calculate the volume difference
    volume_difference = np.abs(prediction.sum() - ground_truth.sum())
    
    # Normalize by the number of voxels in the ground truth
    normalized_avd = volume_difference / ground_truth.sum()

    return normalized_avd

def get_recall_and_precision(prediction, ground_truth):
    """
    Calculate the instance-level recall and precision for 3D volumes using connected component labeling.

    Parameters:
    - prediction (array_like): The predicted 3D binary volume.
    - ground_truth (array_like): The ground truth 3D binary volume.

    Returns:
    - tuple: (instance-level recall, instance-level precision)
    """
    # Label connected components in 3D
    prediction_labels = label(prediction, connectivity=3)
    ground_truth_labels = label(ground_truth, connectivity=3)

    # Calculate True Positives (TP)
    true_positives = 0
    for i in range(1, np.max(ground_truth_labels) + 1):
        ground_truth_instance = (ground_truth_labels == i)
        if np.any(prediction_labels[ground_truth_instance] > 0):
            true_positives += 1

    # Calculate False Negatives (FN)
    false_negatives = 0
    for i in range(1, np.max(ground_truth_labels) + 1):
        ground_truth_instance = (ground_truth_labels == i)
        if not np.any(prediction_labels[ground_truth_instance] > 0):
            false_negatives += 1

    # Calculate False Positives (FP)
    false_positives = 0
    for i in range(1, np.max(prediction_labels) + 1):
        prediction_instance = (prediction_labels == i)
        if not np.any(ground_truth_labels[prediction_instance] > 0):
            false_positives += 1

    num_instances_ground_truth = np.max(ground_truth_labels)
    num_instances_prediction = np.max(prediction_labels)

    if num_instances_ground_truth == 0:
        recall = 0  # Avoid division by zero if no instances in ground truth
    else:
        recall = true_positives / num_instances_ground_truth

    if num_instances_prediction == 0:
        precision = 0  # Avoid division by zero if no instances in prediction
    else:
        precision = true_positives / num_instances_prediction

    return (recall, precision)