"""
MRI Preprocessing Pipeline Node
================================
Handles: N4 Bias Field Correction, Skull Stripping, Intensity Normalization,
         Voxel Resampling, Modality Alignment, Multi-channel Stacking.
"""
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib


def n4_bias_field_correction(image: sitk.Image) -> sitk.Image:
    """Apply N4 Bias Field Correction to an MRI image."""
    mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
    input_image = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected = corrector.Execute(input_image, mask_image)
    return corrected


def skull_strip(image: sitk.Image) -> sitk.Image:
    """
    Basic skull stripping using Otsu thresholding.
    Many curated MRI datasets are already skull-stripped, so this acts as a safety step.
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mask = otsu_filter.Execute(image)
    stripped = sitk.Mask(image, mask)
    return stripped


def normalize_intensity(image: sitk.Image) -> sitk.Image:
    """Z-score intensity normalization (zero mean, unit variance) on non-zero voxels."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    non_zero = arr[arr > 0]
    if len(non_zero) > 0:
        mean_val = np.mean(non_zero)
        std_val = np.std(non_zero)
        if std_val > 0:
            arr[arr > 0] = (arr[arr > 0] - mean_val) / std_val
    result = sitk.GetImageFromArray(arr)
    result.CopyInformation(image)
    return result


def resample_volume(image: sitk.Image, target_spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """Resample voxels to isotropic target spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(image)


def preprocess_modality(nifti_path: str) -> sitk.Image:
    """Full preprocessing pipeline for a single MRI modality."""
    image = sitk.ReadImage(nifti_path, sitk.sitkFloat32)
    image = n4_bias_field_correction(image)
    image = skull_strip(image)
    image = normalize_intensity(image)
    image = resample_volume(image, target_spacing=(1.0, 1.0, 1.0))
    return image


def find_modality_file(base_dir: str, modality: str) -> str:
    """
    Locate a specific modality NIfTI file in a patient directory.
    Handles naming variations like 'T1.nii.gz', 't1.nii.gz', 'subject_..._t1.nii.gz'.
    """
    modality_lower = modality.lower()
    for f in os.listdir(base_dir):
        if f.endswith('.nii.gz') or f.endswith('.nii'):
            fname_lower = f.lower()
            # Match exact modality tag at end before extension
            # e.g., '_t1.nii.gz', '_t1ce.nii.gz', '_flair.nii.gz', '_t2.nii.gz'
            stem = fname_lower.replace('.nii.gz', '').replace('.nii', '')
            if stem.endswith(f"_{modality_lower}") or stem == modality_lower:
                return os.path.join(base_dir, f)
    return None


def run_preprocessing(state: dict) -> dict:
    """
    LangGraph node: Preprocess all MRI modalities for a patient.
    Stacks T1, T1CE, T2, FLAIR into a multi-channel numpy array.
    """
    patient_id = state["patient_id"]
    base_dir = state["base_dir"]
    output_dir = state["output_dir"]
    errors = list(state.get("errors", []))

    prep_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(prep_dir, exist_ok=True)

    modalities = ["t1", "t1ce", "t2", "flair"]
    channels = []

    print(f"[Preprocessing] Processing patient: {patient_id}")

    for mod in modalities:
        path = find_modality_file(base_dir, mod)
        if path is None:
            msg = f"Modality {mod} not found for patient {patient_id} in {base_dir}"
            print(f"  [WARNING] {msg}")
            errors.append(msg)
            continue

        print(f"  Processing {mod}: {os.path.basename(path)}")
        try:
            processed = preprocess_modality(path)
            arr = sitk.GetArrayFromImage(processed)
            channels.append(arr)
        except Exception as e:
            msg = f"Error preprocessing {mod} for {patient_id}: {e}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)

    if len(channels) == len(modalities):
        # Determine target shape (use the first modality as reference)
        target_shape = channels[0].shape
        aligned_channels = []
        for ch in channels:
            if ch.shape != target_shape:
                # Pad or crop to match reference shape
                padded = np.zeros(target_shape, dtype=np.float32)
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(ch.shape, target_shape))
                padded[slices] = ch[slices]
                aligned_channels.append(padded)
            else:
                aligned_channels.append(ch)

        stacked = np.stack(aligned_channels, axis=0)  # Shape: (4, D, H, W)
        save_path = os.path.join(prep_dir, f"{patient_id}_preprocessed.npy")
        np.save(save_path, stacked)
        print(f"  Saved preprocessed volume: {save_path} | shape: {stacked.shape}")
        return {**state, "preprocessed_path": save_path, "errors": errors}
    else:
        errors.append(f"Incomplete modalities for {patient_id}, got {len(channels)}/{len(modalities)}")
        return {**state, "preprocessed_path": None, "errors": errors}
