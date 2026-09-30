#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import nibabel as nib


def _compute_crop_pad(current_size: int, target_size: int):
    """
    Return crop_before, crop_after, pad_before, pad_after
    for centered cropping/padding.
    """
    if current_size > target_size:
        total_crop = current_size - target_size
        crop_before = total_crop // 2
        crop_after = total_crop - crop_before
        pad_before = 0
        pad_after = 0
    else:
        total_pad = target_size - current_size
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        crop_before = 0
        crop_after = 0

    return crop_before, crop_after, pad_before, pad_after


def center_crop_or_pad_array(array: np.ndarray, target_shape, pad_value=0):
    """
    Center-crop and/or center-pad a 3D array to target_shape.

    Returns
    -------
    output : np.ndarray
        Array with shape == target_shape.
    voxel_offset : np.ndarray, shape (3,)
        Mapping offset such that:
            input_index = output_index + voxel_offset

        This is useful for updating the NIfTI affine.
    """
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")

    if len(target_shape) != 3:
        raise ValueError(f"target_shape must have 3 dimensions, got {target_shape}")

    crop_slices = []
    pad_width = []
    voxel_offset = []

    for current_size, target_size in zip(array.shape, target_shape):
        crop_before, crop_after, pad_before, pad_after = _compute_crop_pad(
            current_size, target_size
        )

        start = crop_before
        stop = current_size - crop_after if crop_after > 0 else current_size

        crop_slices.append(slice(start, stop))
        pad_width.append((pad_before, pad_after))

        # input_index = output_index + crop_before - pad_before
        voxel_offset.append(crop_before - pad_before)

    cropped = array[tuple(crop_slices)]

    output = np.pad(
        cropped,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    )

    return output, np.asarray(voxel_offset, dtype=float)


def update_affine_for_crop_pad(affine: np.ndarray, voxel_offset):
    """
    Update affine after cropping/padding so that physical coordinates
    of the original voxels remain unchanged.

    voxel_offset must satisfy:
        input_index = output_index + voxel_offset
    """
    voxel_offset = np.asarray(voxel_offset, dtype=float)

    new_affine = affine.copy()
    new_affine[:3, 3] = (
        affine[:3, :3] @ voxel_offset + affine[:3, 3]
    )

    return new_affine


def match_mask_to_grayscale(
    mask_nifti_path,
    grayscale_nifti_path,
    output_nifti_path,
    pad_value=0,
):
    """
    Center crop/pad a NIfTI mask so its 3D matrix size matches
    the grayscale NIfTI matrix size.

    This function does NOT resample or register the images.
    It only trims/pads voxel arrays.

    Parameters
    ----------
    mask_nifti_path : str or Path
        Input segmentation mask NIfTI.
    grayscale_nifti_path : str or Path
        Reference grayscale NIfTI whose shape will be used.
    output_nifti_path : str or Path
        Output path for the adjusted mask.
    pad_value : int or float
        Value used for new padded voxels. Usually 0 for masks.

    Returns
    -------
    Path
        Output NIfTI path.
    """
    mask_nifti_path = Path(mask_nifti_path)
    grayscale_nifti_path = Path(grayscale_nifti_path)
    output_nifti_path = Path(output_nifti_path)

    mask_img = nib.load(str(mask_nifti_path))
    gray_img = nib.load(str(grayscale_nifti_path))

    mask_data = np.asanyarray(mask_img.dataobj)

    if mask_data.ndim != 3:
        raise ValueError(
            f"Mask must be 3D. Got mask shape: {mask_data.shape}"
        )

    if len(gray_img.shape) < 3:
        raise ValueError(
            f"Grayscale image must have at least 3 dimensions. "
            f"Got shape: {gray_img.shape}"
        )

    target_shape = tuple(gray_img.shape[:3])

    print(f"Mask shape      : {mask_data.shape}")
    print(f"Grayscale shape : {target_shape}")

    adjusted_mask, voxel_offset = center_crop_or_pad_array(
        mask_data,
        target_shape,
        pad_value=pad_value,
    )

    new_affine = update_affine_for_crop_pad(
        mask_img.affine,
        voxel_offset,
    )

    # Preserve the mask datatype/header where possible.
    new_header = mask_img.header.copy()
    new_header.set_data_shape(target_shape)
    new_header.set_data_dtype(mask_data.dtype)

    output_img = nib.Nifti1Image(
        adjusted_mask.astype(mask_data.dtype, copy=False),
        new_affine,
        header=new_header,
    )

    output_nifti_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_img, str(output_nifti_path))

    print(f"Voxel offset    : {voxel_offset.astype(int)}")
    print(f"Output shape    : {adjusted_mask.shape}")
    print(f"Saved           : {output_nifti_path}")

    return output_nifti_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Center crop/pad a NIfTI segmentation mask so its matrix "
            "size matches a reference grayscale NIfTI."
        )
    )

    parser.add_argument(
        "mask_nifti",
        help="Input mask NIfTI (.nii or .nii.gz)",
    )

    parser.add_argument(
        "grayscale_nifti",
        help="Reference grayscale NIfTI (.nii or .nii.gz)",
    )

    parser.add_argument(
        "output_nifti",
        help="Output adjusted mask NIfTI (.nii or .nii.gz)",
    )

    parser.add_argument(
        "--pad-value",
        type=float,
        default=0,
        help="Padding value; default is 0",
    )

    args = parser.parse_args()

    match_mask_to_grayscale(
        mask_nifti_path=args.mask_nifti,
        grayscale_nifti_path=args.grayscale_nifti,
        output_nifti_path=args.output_nifti,
        pad_value=args.pad_value,
    )
