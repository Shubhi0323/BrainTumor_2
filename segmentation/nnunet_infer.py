"""
Tumor Segmentation Node — nnU-Net Pretrained
==============================================
Uses the official nnU-Net pretrained model for BraTS 2020 brain tumor
segmentation. Falls back to ground-truth masks if nnU-Net inference fails.

BraTS seg labels:
  0 = background
  1 = necrotic / non-enhancing tumor core (NCR/NET)
  2 = peritumoral edema (ED)
  4 = GD-enhancing tumor (ET)

We produce a binary mask: 0 = healthy, 1 = tumor (any label > 0).
"""
import os
import subprocess
import numpy as np
import SimpleITK as sitk

# nnU-Net environment variables
NNUNET_RESULTS = os.environ.get("nnUNet_results", "/workspace/nnUNet_results")
NNUNET_RAW = os.environ.get("nnUNet_raw", "/workspace/nnUNet_raw")
NNUNET_PREPROCESSED = os.environ.get("nnUNet_preprocessed", "/workspace/nnUNet_preprocessed")

# BraTS task config for nnU-Net
BRATS_DATASET_ID = "Dataset137_BraTS2021"  # Standard nnU-Net BraTS task name
BRATS_TRAINER = "nnUNetTrainer"
BRATS_CONFIG = "3d_fullres"
BRATS_FOLDS = "0"


def setup_nnunet_env():
    """Set up nnU-Net environment variables."""
    os.environ["nnUNet_results"] = NNUNET_RESULTS
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    for d in [NNUNET_RESULTS, NNUNET_RAW, NNUNET_PREPROCESSED]:
        os.makedirs(d, exist_ok=True)


def check_nnunet_available() -> bool:
    """Check if nnU-Net is installed and pretrained weights are available."""
    try:
        import nnunetv2
        # Check if model weights exist
        model_dir = os.path.join(
            NNUNET_RESULTS, BRATS_DATASET_ID, BRATS_TRAINER + "__nnUNetPlans__" + BRATS_CONFIG
        )
        if os.path.isdir(model_dir):
            return True
        else:
            print(f"  nnU-Net installed but model weights not found at: {model_dir}")
            return False
    except ImportError:
        return False


def download_nnunet_weights():
    """
    Download pretrained nnU-Net weights for BraTS segmentation.
    Uses the nnU-Net model from Zenodo or MIC-DKFZ repository.
    """
    setup_nnunet_env()
    model_dir = os.path.join(
        NNUNET_RESULTS, BRATS_DATASET_ID, BRATS_TRAINER + "__nnUNetPlans__" + BRATS_CONFIG
    )

    if os.path.isdir(model_dir):
        print("  nnU-Net weights already downloaded.")
        return True

    print("  Downloading pretrained nnU-Net BraTS weights...")
    try:
        # Install nnU-Net if not present
        subprocess.run(
            ["pip", "install", "nnunetv2"],
            capture_output=True, text=True, check=True
        )

        # Download pretrained weights using nnU-Net's built-in download
        # The official way to get pretrained weights:
        subprocess.run(
            ["nnUNetv2_download_pretrained_model_by_url",
             "-url", "https://zenodo.org/records/10782801/files/Dataset137_BraTS2021.zip",
             "-o", NNUNET_RESULTS],
            capture_output=True, text=True, timeout=600
        )

        if os.path.isdir(model_dir):
            print("  nnU-Net weights downloaded successfully.")
            return True
        else:
            print("  Direct download didn't create expected folder. Trying alternative...")
            # Alternative: manual download
            os.makedirs(model_dir, exist_ok=True)
            subprocess.run([
                "wget", "-q",
                "https://zenodo.org/records/10782801/files/Dataset137_BraTS2021.zip",
                "-O", "/tmp/nnunet_brats.zip"
            ], capture_output=True, text=True, timeout=600)
            subprocess.run([
                "unzip", "-q", "-o", "/tmp/nnunet_brats.zip",
                "-d", NNUNET_RESULTS
            ], capture_output=True, text=True)
            return os.path.isdir(model_dir)

    except Exception as e:
        print(f"  [WARNING] Failed to download nnU-Net weights: {e}")
        return False


def prepare_nnunet_input(base_dir: str, patient_id: str) -> str:
    """
    Prepare input files in nnU-Net expected format.
    nnU-Net expects: {case_id}_0000.nii.gz, {case_id}_0001.nii.gz, etc.
    For BraTS: 0000=T1, 0001=T1CE, 0002=T2, 0003=FLAIR
    """
    input_dir = os.path.join("/tmp", "nnunet_input", patient_id)
    os.makedirs(input_dir, exist_ok=True)

    modality_map = {
        "t1": "0000",
        "t1ce": "0001",
        "t2": "0002",
        "flair": "0003",
    }

    found_count = 0
    for f in os.listdir(base_dir):
        f_lower = f.lower()
        if not (f_lower.endswith('.nii.gz') or f_lower.endswith('.nii')):
            continue
        stem = f_lower.replace('.nii.gz', '').replace('.nii', '')

        for mod, suffix in modality_map.items():
            if stem.endswith(f"_{mod}") or stem == mod:
                src = os.path.join(base_dir, f)
                dst = os.path.join(input_dir, f"{patient_id}_{suffix}.nii.gz")
                # Copy or symlink
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
                found_count += 1
                break

    # Also check for generic 'image' file (from H5 reconstruction)
    if found_count == 0:
        for f in os.listdir(base_dir):
            f_lower = f.lower()
            if 'image' in f_lower and f_lower.endswith('.nii.gz'):
                src = os.path.join(base_dir, f)
                dst = os.path.join(input_dir, f"{patient_id}_0000.nii.gz")
                import shutil
                shutil.copy2(src, dst)
                found_count = 1
                break

    if found_count > 0:
        return input_dir
    return None


def run_nnunet_inference(input_dir: str, output_dir: str) -> str:
    """Run nnU-Net inference on prepared input."""
    setup_nnunet_env()
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", BRATS_DATASET_ID,
        "-c", BRATS_CONFIG,
        "-tr", BRATS_TRAINER,
        "-f", BRATS_FOLDS,
        "--disable_tta",
    ]

    print(f"  Running nnU-Net inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        print(f"  [ERROR] nnU-Net inference failed: {result.stderr[:500]}")
        return None

    # Find output file
    for f in os.listdir(output_dir):
        if f.endswith('.nii.gz'):
            return os.path.join(output_dir, f)
    return None


def find_seg_file(base_dir: str) -> str:
    """Locate the segmentation mask NIfTI file in a patient directory."""
    for f in os.listdir(base_dir):
        fname_lower = f.lower()
        if fname_lower.endswith('.nii.gz') or fname_lower.endswith('.nii'):
            stem = fname_lower.replace('.nii.gz', '').replace('.nii', '')
            if stem.endswith('_seg') or stem == 'seg':
                return os.path.join(base_dir, f)
    return None


def load_and_binarize_mask(seg_path: str) -> tuple:
    """Load a segmentation mask and binarize it (any label > 0 → tumor)."""
    seg_image = sitk.ReadImage(seg_path, sitk.sitkInt32)
    seg_arr = sitk.GetArrayFromImage(seg_image)
    binary_mask = (seg_arr > 0).astype(np.int32)
    binary_sitk = sitk.GetImageFromArray(binary_mask)
    binary_sitk.CopyInformation(seg_image)
    return binary_mask, binary_sitk


def run_segmentation(state: dict) -> dict:
    """
    LangGraph node: Extract tumor segmentation mask.
    Priority order:
      1. nnU-Net pretrained inference
      2. Ground-truth BraTS segmentation (fallback)
    """
    patient_id = state["patient_id"]
    base_dir = state["base_dir"]
    output_dir = state["output_dir"]
    errors = list(state.get("errors", []))

    seg_dir = os.path.join(output_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)

    print(f"[Segmentation] Processing patient: {patient_id}")

    # --- Strategy 1: nnU-Net pretrained inference ---
    nnunet_available = check_nnunet_available()
    if not nnunet_available:
        print("  nnU-Net weights not found. Attempting download...")
        nnunet_available = download_nnunet_weights()

    if nnunet_available:
        print("  Attempting nnU-Net inference...")
        input_dir = prepare_nnunet_input(base_dir, patient_id)
        if input_dir:
            nnunet_out_dir = os.path.join("/tmp", "nnunet_output", patient_id)
            seg_result = run_nnunet_inference(input_dir, nnunet_out_dir)
            if seg_result:
                binary_mask, binary_sitk = load_and_binarize_mask(seg_result)
                tumor_voxels = np.sum(binary_mask > 0)
                total_voxels = binary_mask.size
                print(f"  nnU-Net segmentation: {tumor_voxels}/{total_voxels} tumor voxels "
                      f"({100*tumor_voxels/total_voxels:.2f}%)")
                save_path = os.path.join(seg_dir, f"{patient_id}_seg.nii.gz")
                sitk.WriteImage(binary_sitk, save_path)
                print(f"  Saved nnU-Net segmentation: {save_path}")
                return {**state, "segmentation_path": save_path, "errors": errors}
            else:
                errors.append(f"nnU-Net inference failed for {patient_id}, falling back to GT")
        else:
            errors.append(f"Could not prepare nnU-Net input for {patient_id}")

    # --- Strategy 2: Ground-truth segmentation mask ---
    seg_path = find_seg_file(base_dir)
    if seg_path is not None:
        print(f"  Using ground-truth segmentation: {os.path.basename(seg_path)}")
        try:
            binary_mask, binary_sitk = load_and_binarize_mask(seg_path)
            tumor_voxels = np.sum(binary_mask > 0)
            total_voxels = binary_mask.size
            print(f"  GT tumor voxels: {tumor_voxels}/{total_voxels} ({100*tumor_voxels/total_voxels:.2f}%)")

            save_path = os.path.join(seg_dir, f"{patient_id}_seg.nii.gz")
            sitk.WriteImage(binary_sitk, save_path)
            print(f"  Saved GT segmentation: {save_path}")
            return {**state, "segmentation_path": save_path, "errors": errors}
        except Exception as e:
            msg = f"Error loading GT seg for {patient_id}: {e}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)

    msg = f"No segmentation available for {patient_id}"
    errors.append(msg)
    return {**state, "segmentation_path": None, "errors": errors}
