import numpy as np
import nibabel as nib
from evaluate import get_recall_and_precision, get_dice, get_avd


def load_and_evaluate(ground_truth, prediction):
    # Load synthetic data
    print(f"Ground truth path: {ground_truth_path}")
    print(f"Prediction path: {prediction_path}")

    if '\x00' in ground_truth_path or '\x00' in prediction_path:
        raise ValueError("File path contains a null character")

    ground_truth = nib.load(ground_truth_path).get_fdata()
    prediction = nib.load(prediction_path).get_fdata()
    
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

if __name__ == "__main__":

    ground_truth_path = "/Users/yileiwu/EPVS_challenge/labelsTr/EPVSCHALLENGE_ED_01_05.nii.gz"
    prediction_path = "/Users/yileiwu/EPVS_challenge/validation_2d_3/EPVSCHALLENGE_ED_01_05.nii.gz"
    load_and_evaluate(ground_truth_path, prediction_path)
    # Ground truth path: /Users/yileiwu/EPVS_challenge/labelsTr/EPVSCHALLENGE_ED_01_05.nii.gz
    # Prediction path: /Users/yileiwu/EPVS_challenge/validation_2d_3/EPVSCHALLENGE_ED_01_05.nii.gz
    # Recall: 0.6216216216216216
    # Precision: 0.5985130111524164
    # Dice: 0.5603349828701941
    # AVD: 0.16436663233779608