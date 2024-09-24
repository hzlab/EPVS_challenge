import numpy as np
import nibabel as nib
from evaluate import get_recall_and_precision, get_dice, get_avd
from test_evaluate import load_and_evaluate
import glob
import os


def get_atlas_base_path(filename):
    if "ED_01" in filename:
        return "/Volumes/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/SHIVA_PVS/challenge_data/ED_01"
    elif "ED_02" in filename:
        return "/Volumes/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/SHIVA_PVS/challenge_data/ED_02"
    elif "ED_03" in filename:
        return "/Volumes/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/SHIVA_PVS/challenge_data/ED_03"
    elif "ED_04" in filename:
        return "/Volumes/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/SHIVA_PVS/challenge_data/ED_04"
    elif "SGSG" in filename:
        return "/Volumes/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/SHIVA_PVS/challenge_data/SG70"
    elif "HD" in filename:
        return "/Volumes/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/SHIVA_PVS/challenge_data/MACC"
    else:
        raise ValueError("Unknown filename pattern: {}".format(filename))


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--folder_name", type=str, required=True)
    folder_name = args.parse_args().folder_name
    print(f"Folder name: {folder_name}")    
    participant_pred = glob.glob(f"/Users/yileiwu/Downloads/{folder_name}/*")
    gts = "/Volumes/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/EPVS_challenge_data/Val"

    for each in participant_pred:
        filename = glob.glob(each + "/*.nii.gz")[0]
        pred = nib.load(filename).get_fdata()

        filename = os.path.basename(filename).split(".")[0]
        # print(f"Processing {filename}")
        gt = nib.load(os.path.join(gts, filename[:-4], filename + '.nii.gz')).get_fdata()

        atlas_base = get_atlas_base_path(filename)
        cso_atlas = nib.load(os.path.join(atlas_base, filename[:-4], 'cso_native_space.nii.gz')).get_fdata()
        lbg_atlas = nib.load(os.path.join(atlas_base, filename[:-4], 'lbg_native_space.nii.gz')).get_fdata()
        rbg_atlas = nib.load(os.path.join(atlas_base, filename[:-4], 'rbg_native_space.nii.gz')).get_fdata()
        bg_atlas = np.where((lbg_atlas + rbg_atlas) > 0.5, 1, 0)
        cso_atlas = np.where(cso_atlas > 0.5, 1, 0)

        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}, filename: {filename}"
        assert pred.shape == cso_atlas.shape, f"Shape mismatch: {pred.shape} vs {cso_atlas.shape}"
        assert pred.shape == bg_atlas.shape, f"Shape mismatch: {pred.shape} vs {bg_atlas.shape}"

        pred_cso = np.zeros_like(pred)
        pred_bg = np.zeros_like(pred)
        pred_cso[(pred != 0) & (cso_atlas != 0)] = 1
        pred_bg[(pred != 0) & (bg_atlas != 0)] = 1

        gt_cso = np.zeros_like(gt)
        gt_bg = np.zeros_like(gt)
        gt_cso[(gt != 0) & (cso_atlas != 0)] = 1
        gt_bg[(gt != 0) & (bg_atlas != 0)] = 1

        recall_cso, precision_cso = get_recall_and_precision(pred_cso, gt_cso)
        dice_cso = get_dice(pred_cso, gt_cso)
        avd_cso = get_avd(pred_cso, gt_cso)

        recall_bg, precision_bg = get_recall_and_precision(pred_bg, gt_bg)
        dice_bg = get_dice(pred_bg, gt_bg)
        avd_bg = get_avd(pred_bg, gt_bg)

        print("{} CSO: Recall: {:.4f}, Precision: {:.4f}, Dice: {:.4f}, AVD: {:.4f} BG: Recall: {:.4f}, Precision: {:.4f}, Dice: {:.4f}, AVD: {:.4f}".format(
            filename, recall_cso, precision_cso, dice_cso, avd_cso, recall_bg, precision_bg, dice_bg, avd_bg))
