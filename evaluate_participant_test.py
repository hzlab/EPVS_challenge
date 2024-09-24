import numpy as np
import nibabel as nib
from evaluate import get_recall_and_precision, get_dice, get_avd
from test_evaluate import load_and_evaluate
import glob
import os
from scipy.ndimage import label, binary_erosion



def remove_small_components(tensor, n_voxel):
    """
    Removes all connected components in a 3D binary tensor with fewer than n voxels.
    
    Parameters:
    - tensor (np.ndarray): 3D binary tensor with foreground labeled as 1.
    - n_voxel (int): Minimum number of voxels required to retain a connected component.
    
    Returns:
    - np.ndarray: 3D binary tensor with small components removed.
    """
    # Label connected components in the binary tensor
    labeled_tensor, num_features = label(tensor)
    
    # Create an empty array to store the filtered components
    filtered_tensor = np.zeros_like(tensor)
    
    # Iterate through each connected component
    for i in range(1, num_features + 1):
        # Extract the current component
        component = (labeled_tensor == i)
        
        # Check the number of voxels in the component
        if np.sum(component) >= n_voxel:
            # If the component has at least n_voxels, keep it
            filtered_tensor = np.logical_or(filtered_tensor, component)

    # Convert the filtered tensor back to binary (0/1)
    return filtered_tensor.astype(int)

def erode_components(tensor, n_erode=1):
    """
    Erodes each connected component in the XY plane of a 3D binary tensor by a specified number of voxels.
    
    Parameters:
    - tensor (np.ndarray): 3D binary tensor with foreground labeled as 1.
    - n_erode (int): Number of voxels to erode for each connected component in the XY plane.
    
    Returns:
    - np.ndarray: Eroded 3D binary tensor in the XY plane.
    """
    # Label connected components in the binary tensor
    labeled_tensor, num_features = label(tensor)
    
    # Create an empty array to store the eroded components
    eroded_tensor = np.zeros_like(tensor)
    
    # Iterate through each slice in the Z-axis
    for z in range(tensor.shape[2]):
        # Extract the 2D slice
        slice_2d = tensor[:, :, z]
        
        # Label connected components in the 2D slice
        labeled_slice, num_features_2d = label(slice_2d)
        
        # Iterate through each connected component in the 2D slice
        for i in range(1, num_features_2d + 1):
            # Extract the current component
            component = (labeled_slice == i)
            
            # Apply erosion multiple times based on n_erode
            eroded_component = component
            for _ in range(n_erode):
                eroded_component = binary_erosion(eroded_component)
                if not eroded_component.any():  # If the component is completely eroded, stop
                    break
            
            # Add the eroded component back to the final tensor
            eroded_tensor[:, :, z] = np.logical_or(eroded_tensor[:, :, z], eroded_component)
    
    # Convert the eroded tensor back to binary (0/1)
    return eroded_tensor.astype(int)



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
        return None


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--folder_name", type=str, required=True)
    folder_name = args.parse_args().folder_name
    print(f"Folder name: {folder_name}")    
    participant_pred = glob.glob(f"/Users/yileiwu/Downloads/{folder_name}/*")
    gts = "/Users/yileiwu/Downloads/Test"

    for each in sorted(participant_pred):
        filename = glob.glob(each + "/*.nii.gz")[0]
        pred = nib.load(filename).get_fdata()

        filename = os.path.basename(filename).split(".")[0]
        # print(f"Processing {filename}")
        gt = nib.load(os.path.join(gts, filename[:-5], filename[:-4] + 'PVS.nii.gz')).get_fdata()

        # if filename is all digit
        # if filename[0].isdigit():
        #     print(filename)
        #     gt = erode_components(gt, n_erode=1)
            # gt = remove_small_components(gt, n_voxel=1)

        atlas_base = get_atlas_base_path(filename)
        
        if atlas_base:
            cso_atlas = nib.load(os.path.join(atlas_base, filename[:-5], 'cso_native_space.nii.gz')).get_fdata()
            lbg_atlas = nib.load(os.path.join(atlas_base, filename[:-5], 'lbg_native_space.nii.gz')).get_fdata()
            rbg_atlas = nib.load(os.path.join(atlas_base, filename[:-5], 'rbg_native_space.nii.gz')).get_fdata()
            bg_atlas = np.where((lbg_atlas + rbg_atlas) > 0.5, 1, 0)
            cso_atlas = np.where(cso_atlas > 0.5, 1, 0)

            assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}, filename: {filename}"
            assert pred.shape == cso_atlas.shape, f"Shape mismatch: {pred.shape} vs {cso_atlas.shape}"
            assert pred.shape == bg_atlas.shape, f"Shape mismatch: {pred.shape} vs {bg_atlas.shape}"
        else:
            # bg_atlas is the cneter of the brain
            # cso_atlas is the exclueded region
            cso_atlas = np.zeros_like(pred)
            bg_atlas = np.zeros_like(pred)
            # center half of the brain
            bg_atlas[3*pred.shape[0]//7:4*pred.shape[0]//7, 3*pred.shape[1]//7:4*pred.shape[1]//7, 4*pred.shape[2]//9:5*pred.shape[2]//9] = 1
            
            cso_atlas[2*pred.shape[0]//7:5*pred.shape[0]//7, 2*pred.shape[1]//7:5*pred.shape[1]//7, 3*pred.shape[2]//9:6*pred.shape[2]//9] = 1
            cso_atlas = cso_atlas - bg_atlas
            # exlucede the boundary 

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
        avd_cso = min(1, get_avd(pred_cso, gt_cso))

        recall_bg, precision_bg = get_recall_and_precision(pred_bg, gt_bg)
        dice_bg = get_dice(pred_bg, gt_bg)
        avd_bg = min(1, get_avd(pred_bg, gt_bg))

        print("{} CSO: Recall: {:.4f} Precision: {:.4f} Dice: {:.4f} AVD: {:.4f} BG: Recall: {:.4f} Precision: {:.4f} Dice: {:.4f} AVD: {:.4f}".format(
            filename, recall_cso, precision_cso, dice_cso, avd_cso, recall_bg, precision_bg, dice_bg, avd_bg))
