"""
Dataset Adapter for HDF5 Slice-based BraTS format
===================================================
Handles datasets stored as volume_X_slice_Y.h5 files,
where each H5 file contains a 2D slice with multiple channels
(image modalities + segmentation mask).

This module reconstructs 3D volumes from 2D slice files and
provides them in a format compatible with the pipeline.
"""
import os
import re
import json
import numpy as np
import h5py
import SimpleITK as sitk


def discover_volumes(data_dir: str) -> dict:
    """
    Scan the data directory and discover all available volumes.
    Returns a dict: {volume_id: [list of (slice_idx, filepath)]}
    """
    pattern = re.compile(r'volume_(\d+)_slice_(\d+)\.h5')
    volumes = {}

    for f in os.listdir(data_dir):
        match = pattern.match(f)
        if match:
            vol_id = int(match.group(1))
            slice_idx = int(match.group(2))
            if vol_id not in volumes:
                volumes[vol_id] = []
            volumes[vol_id].append((slice_idx, os.path.join(data_dir, f)))

    # Sort slices by index
    for vol_id in volumes:
        volumes[vol_id].sort(key=lambda x: x[0])

    return volumes


def inspect_h5_file(filepath: str) -> dict:
    """Inspect the structure of an H5 file to understand its keys and shapes."""
    info = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            ds = f[key]
            info[key] = {
                'shape': ds.shape,
                'dtype': str(ds.dtype),
            }
    return info


def reconstruct_volume(slice_files: list, data_key: str = 'image',
                       mask_key: str = 'mask') -> tuple:
    """
    Reconstruct a 3D volume from a list of (slice_idx, filepath) tuples.
    Returns (image_volume, mask_volume) as numpy arrays.
    image_volume shape: (num_channels, D, H, W) or (D, H, W) depending on data.
    mask_volume shape: (D, H, W)
    """
    # First, inspect the first non-empty file to understand structure
    sample_info = None
    h5_keys = None
    for _, fp in slice_files:
        with h5py.File(fp, 'r') as f:
            h5_keys = list(f.keys())
            if len(h5_keys) > 0:
                sample_info = {k: f[k].shape for k in h5_keys}
                break

    if sample_info is None or h5_keys is None:
        raise ValueError("Could not find valid H5 files in slice list")

    print(f"  H5 keys: {h5_keys}")
    print(f"  Sample shapes: {sample_info}")

    # Determine the actual key names by inspecting available keys
    # Common key patterns: 'image', 'mask', 'data', 'label', 'seg'
    img_key = None
    seg_key = None
    for k in h5_keys:
        k_lower = k.lower()
        if k_lower in ['image', 'data', 'img', 'volume']:
            img_key = k
        elif k_lower in ['mask', 'seg', 'label', 'segmentation']:
            seg_key = k

    # If standard keys not found, use positional
    if img_key is None and len(h5_keys) >= 1:
        img_key = h5_keys[0]
    if seg_key is None and len(h5_keys) >= 2:
        seg_key = h5_keys[1]

    print(f"  Using image key: '{img_key}', mask key: '{seg_key}'")

    # Collect all slices
    image_slices = []
    mask_slices = []
    num_slices = len(slice_files)

    for slice_idx, fp in slice_files:
        with h5py.File(fp, 'r') as f:
            if img_key and img_key in f:
                image_slices.append(f[img_key][:])
            if seg_key and seg_key in f:
                mask_slices.append(f[seg_key][:])

    if len(image_slices) > 0:
        image_volume = np.stack(image_slices, axis=0)  # (D, H, W) or (D, C, H, W)
    else:
        image_volume = None

    if len(mask_slices) > 0:
        mask_volume = np.stack(mask_slices, axis=0)  # (D, H, W)
    else:
        mask_volume = None

    return image_volume, mask_volume, h5_keys


def save_reconstructed_volume(image_volume: np.ndarray, mask_volume: np.ndarray,
                               patient_id: str, output_dir: str) -> tuple:
    """
    Save reconstructed volumes as NIfTI files in a patient directory.
    Returns (patient_dir, dict of saved paths).
    """
    patient_dir = os.path.join(output_dir, "reconstructed", patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    saved = {}

    if image_volume is not None:
        # If image has multiple channels, split them
        if image_volume.ndim == 4:
            # Detect channel dimension: channels are the smallest spatial dim
            # H5 BraTS data is (D, H, W, C) with C=4 modalities
            if image_volume.shape[-1] <= 4:
                # Channels-last: (D, H, W, C)
                num_channels = image_volume.shape[-1]
                get_channel = lambda vol, idx: vol[:, :, :, idx]
            else:
                # Channels-first: (D, C, H, W)
                num_channels = image_volume.shape[1]
                get_channel = lambda vol, idx: vol[:, idx, :, :]

            modality_names = ['t1', 't1ce', 't2', 'flair']
            for ch_idx in range(num_channels):
                ch_name = modality_names[ch_idx] if ch_idx < len(modality_names) else f'ch{ch_idx}'
                ch_data = get_channel(image_volume, ch_idx)
                img_sitk = sitk.GetImageFromArray(ch_data.astype(np.float32))
                path = os.path.join(patient_dir, f"{patient_id}_{ch_name}.nii.gz")
                sitk.WriteImage(img_sitk, path)
                saved[ch_name] = path
        elif image_volume.ndim == 3:
            # Single channel or combined
            img_sitk = sitk.GetImageFromArray(image_volume.astype(np.float32))
            path = os.path.join(patient_dir, f"{patient_id}_image.nii.gz")
            sitk.WriteImage(img_sitk, path)
            saved['image'] = path

    if mask_volume is not None:
        if mask_volume.ndim == 3:
            mask_sitk = sitk.GetImageFromArray(mask_volume.astype(np.int32))
        elif mask_volume.ndim == 4:
            # Multi-channel mask: combine across channels
            # Detect channel dim: channels-last (D,H,W,C) vs channels-first (D,C,H,W)
            if mask_volume.shape[-1] <= 4:
                # Channels-last: (D, H, W, C) — max across last dim
                mask_2d = np.max(mask_volume, axis=-1)
            else:
                # Channels-first: (D, C, H, W) — max across dim 1
                mask_2d = mask_volume[:, 0, :, :] if mask_volume.shape[1] == 1 else np.max(mask_volume, axis=1)
            mask_sitk = sitk.GetImageFromArray(mask_2d.astype(np.int32))
        else:
            mask_sitk = sitk.GetImageFromArray(mask_volume.astype(np.int32))
        path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
        sitk.WriteImage(mask_sitk, path)
        saved['seg'] = path

    return patient_dir, saved


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data"

    print("Discovering volumes...")
    volumes = discover_volumes(data_dir)
    print(f"Found {len(volumes)} volumes")

    if volumes:
        first_vol = min(volumes.keys())
        print(f"\nInspecting volume {first_vol}...")
        slices = volumes[first_vol]
        print(f"  Number of slices: {len(slices)}")

        # Inspect first non-empty slice
        for _, fp in slices:
            info = inspect_h5_file(fp)
            if info:
                print(f"  H5 structure: {json.dumps(info, indent=2, default=str)}")
                break
