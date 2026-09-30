#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import nibabel as nib
import cv2

def levelset2originalRF_new_flip():
    original_file = sys.argv[1]
    levelset_file = sys.argv[2]
    OUTPUT_DIRECTORY = sys.argv[3]

    original_nib = nib.load(original_file)
    original_data = original_nib.get_fdata()

    print("For the file {}".format(levelset_file))
    print("I am in levelset2originalRF_new_flip()")
    print("original_file_nib_data.shape[1]: {}".format(original_data.shape[1]))
    match_and_flip_to_original_xy(original_file, levelset_file, OUTPUT_DIRECTORY=OUTPUT_DIRECTORY)

def levelset2originalRF_new_flip_py(original_file,levelset_file,OUTPUT_DIRECTORY):
    # original_file = sys.argv[1]
    # levelset_file = sys.argv[2]
    # OUTPUT_DIRECTORY = sys.argv[3]

    original_nib = nib.load(original_file)
    original_data = original_nib.get_fdata()

    print("For the file {}".format(levelset_file))
    print("I am in levelset2originalRF_new_flip()")
    print("original_file_nib_data.shape[1]: {}".format(original_data.shape[1]))
    match_and_flip_to_original_xy(original_file, levelset_file, OUTPUT_DIRECTORY=OUTPUT_DIRECTORY)

def whenOFsize512x512_new_flip(levelset_file, original_file, OUTPUT_DIRECTORY):
    print('original_file.shape::{}'.format(original_file))
    print('levelset_file.shape::{}'.format(levelset_file))

    original_nib = nib.load(original_file)
    levelset_nib = nib.load(levelset_file)
    levelset_data = levelset_nib.get_fdata()

    print('levelset_data.shape::{}'.format(levelset_data.shape))

    # Flip along vertical axis (axis 0)
    for z in range(levelset_data.shape[2]):
        levelset_data[:, :, z] = cv2.flip(levelset_data[:, :, z], 0)

    print("I am in whenOFsize512x512_new()")
    array_mask = nib.Nifti1Image(levelset_data, affine=original_nib.affine, header=original_nib.header)
    out_path = os.path.join(OUTPUT_DIRECTORY, os.path.basename(levelset_file))
    nib.save(array_mask, out_path)
    return "X"

def whenOFsize512x5xx_new_flip(original_file, levelset_file, OUTPUT_DIRECTORY="./"):
    original_nib = nib.load(original_file)
    original_data = original_nib.get_fdata()

    levelset_nib = nib.load(levelset_file)
    levelset_data = levelset_nib.get_fdata()

    x, y, z = levelset_data.shape
    pad_crop_data = np.copy(levelset_data)

    # Padding or cropping X
    target_x = 512
    diff_x = target_x - x
    if diff_x > 0:
        pad_x = [(math.floor(diff_x/2), math.ceil(diff_x/2)), (0, 0), (0, 0)]
        pad_crop_data = np.pad(pad_crop_data, pad_width=pad_x, mode='constant', constant_values=np.min(levelset_data))
    elif diff_x < 0:
        crop_x = -diff_x
        start = math.floor(crop_x / 2)
        pad_crop_data = pad_crop_data[start:start+512, :, :]

    # Padding or cropping Y
    target_y = 512
    diff_y = target_y - pad_crop_data.shape[1]
    if diff_y > 0:
        pad_y = [(0, 0), (math.floor(diff_y/2), math.ceil(diff_y/2)), (0, 0)]
        pad_crop_data = np.pad(pad_crop_data, pad_width=pad_y, mode='constant', constant_values=np.min(levelset_data))
    elif diff_y < 0:
        crop_y = -diff_y
        start = math.floor(crop_y / 2)
        pad_crop_data = pad_crop_data[:, start:start+512, :]

    # Flip vertically (axis 0)
    for z in range(pad_crop_data.shape[2]):
        pad_crop_data[:, :, z] = cv2.flip(pad_crop_data[:, :, z], 0)

    final_mask = nib.Nifti1Image(pad_crop_data, affine=original_nib.affine, header=original_nib.header)
    out_path = os.path.join(OUTPUT_DIRECTORY, os.path.basename(levelset_file))
    nib.save(final_mask, out_path)

    print("Final size: {} vs Original: {}".format(pad_crop_data.shape, original_data.shape))
    return "X"

def match_and_flip_to_original_xy(original_file, levelset_file, OUTPUT_DIRECTORY="./"):
    # Load NIfTI files
    original_nib = nib.load(original_file)
    original_data = original_nib.get_fdata()
    orig_x, orig_y, orig_z = original_data.shape

    levelset_nib = nib.load(levelset_file)
    levelset_data = levelset_nib.get_fdata()
    lvl_x, lvl_y, lvl_z = levelset_data.shape

    adjusted_data = np.copy(levelset_data)

    # Only crop/pad if X or Y dimensions are not equal to original
    if (lvl_x != orig_x) or (lvl_y != orig_y):
        # --- X dimension ---
        diff_x = orig_x - lvl_x
        if diff_x > 0:
            # Pad symmetrically
            pad_x = (math.floor(diff_x/2), math.ceil(diff_x/2))
            adjusted_data = np.pad(adjusted_data, ((pad_x[0], pad_x[1]), (0, 0), (0, 0)),
                                   mode='constant', constant_values=np.min(levelset_data))
        elif diff_x < 0:
            # Crop center
            crop = -diff_x
            start = math.floor(crop / 2)
            adjusted_data = adjusted_data[start:start + orig_x, :, :]

        # --- Y dimension ---
        diff_y = orig_y - adjusted_data.shape[1]
        if diff_y > 0:
            pad_y = (math.floor(diff_y/2), math.ceil(diff_y/2))
            adjusted_data = np.pad(adjusted_data, ((0, 0), (pad_y[0], pad_y[1]), (0, 0)),
                                   mode='constant', constant_values=np.min(levelset_data))
        elif diff_y < 0:
            crop = -diff_y
            start = math.floor(crop / 2)
            adjusted_data = adjusted_data[:, start:start + orig_y, :]

    # Flip vertically (axis 0) for each slice
    for z in range(adjusted_data.shape[2]):
        adjusted_data[:, :, z] = cv2.flip(adjusted_data[:, :, z], 0)

    # Save result
    final_mask = nib.Nifti1Image(adjusted_data, affine=original_nib.affine, header=original_nib.header)
    out_path = os.path.join(OUTPUT_DIRECTORY, os.path.basename(levelset_file))
    nib.save(final_mask, out_path)

    print(f"Saved flipped and dimension-matched mask to: {out_path}")
    print("Final size:", adjusted_data.shape, "| Original size:", original_data.shape)
    return out_path

def levelset2originalRF_new_flip_with_params(original_file,levelset_file,OUTPUT_DIRECTORY) : #original_file,levelset_file,OUTPUT_DIRECTORY="./"):
    #     print(sys.argv[1])

    # original_file=sys.argv[1]
    # levelset_file=sys.argv[2]
    # OUTPUT_DIRECTORY=sys.argv[3]
    original_file_nib=nib.load(original_file)

    original_file_nib_data=original_file_nib.get_fdata()
    print("For the file {}".format(levelset_file))
    print("I am in levelset2originalRF_new_flip()")
    print("original_file_nib_data.shape[1]: {}".format(original_file_nib_data.shape[1]))
    match_and_flip_to_original_xy(original_file, levelset_file, OUTPUT_DIRECTORY=OUTPUT_DIRECTORY)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 levelset2originalRF_new_flip_utilities.py levelset2originalRF_new_flip <original_file> <levelset_file> <output_dir>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "levelset2originalRF_new_flip":
        if len(sys.argv) != 5:
            print("Usage:")
            print("  python3 levelset2originalRF_new_flip_utilities.py levelset2originalRF_new_flip <original_file> <levelset_file> <output_dir>")
            sys.exit(1)

        original_file = sys.argv[2]
        levelset_file = sys.argv[3]
        output_dir = sys.argv[4]

        levelset2originalRF_new_flip_py(
            original_file,
            levelset_file,
            output_dir
        )

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
if __name__ == "__main__":
    main()