import numpy as np
import nibabel as nib
from evaluate import get_recall_and_precision, get_dice, get_avd


def load_and_evaluate(ground_truth, prediction):
    # Load synthetic data
    prediction = nib.load(ground_truth).get_fdata()
    ground_truth = nib.load(prediction).get_fdata()

    binarize = lambda x: (x > 0).astype(np.uint8)
    prediction = binarize(prediction)
    ground_truth = binarize(ground_truth)

    # Evaluate recall and precision
    recall, precision = get_recall_and_precision(prediction >= 0.5, ground_truth >= 0.5)
    dice = get_dice(prediction >= 0.5, ground_truth >= 0.5)
    avd = get_avd(prediction >= 0.5, ground_truth >= 0.5)
    print("Recall:", recall)
    print("Precision:", precision)
    print("Dice:", dice)
    print("AVD:", avd)

load_and_evaluate()
