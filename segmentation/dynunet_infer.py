"""
Tumor Segmentation Node — SegResNet (MONAI)
=============================================
Uses MONAI's SegResNet architecture with official pretrained weights from
the MONAI Model Zoo (brats_mri_segmentation bundle) for BraTS brain tumor
segmentation.

Falls back to ground-truth masks if model weights are not available.

BraTS seg labels:
  0 = background
  1 = necrotic / non-enhancing tumor core (NCR/NET)
  2 = peritumoral edema (ED)
  4 = GD-enhancing tumor (ET)

Model outputs 3 channels: TC, WT, ET.
We produce a binary mask: 0 = healthy, 1 = tumor (any channel > 0).
"""
import os
import subprocess
import numpy as np
import SimpleITK as sitk
import torch

from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
    Activationsd,
    AsDiscreted,
)

# --- SegResNet BraTS Architecture Config ---
# Matches the official MONAI Model Zoo brats_mri_segmentation bundle
IN_CHANNELS = 4      # T1, T1CE, T2, FLAIR
OUT_CHANNELS = 3     # TC (tumor core), WT (whole tumor), ET (enhancing tumor)
BLOCKS_DOWN = [1, 2, 2, 4]
BLOCKS_UP = [1, 1, 1]
INIT_FILTERS = 16
DROPOUT_PROB = 0.2

# Sliding window inference config (matches official bundle)
ROI_SIZE = (240, 240, 160)
SW_BATCH_SIZE = 1
OVERLAP = 0.5

# Default path for pretrained weights
WEIGHTS_DIR = os.environ.get("DYNUNET_WEIGHTS_DIR", "weights")
WEIGHTS_FILE = os.environ.get("DYNUNET_WEIGHTS_FILE", "model_brats_mri_segmentation.pt")

# Official MONAI Model Zoo weights URL
MONAI_BRATS_WEIGHTS_URL = (
    "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
    "model_zoo/model_brats_mri_segmentation.pt"
)


def build_segresnet_model() -> SegResNet:
    """Build SegResNet model with BraTS architecture configuration."""
    model = SegResNet(
        blocks_down=BLOCKS_DOWN,
        blocks_up=BLOCKS_UP,
        init_filters=INIT_FILTERS,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        dropout_prob=DROPOUT_PROB,
    )
    return model


def find_weights_path() -> str | None:
    """
    Locate pretrained SegResNet weights.
    Searches in order:
      1. $DYNUNET_WEIGHTS_DIR/$DYNUNET_WEIGHTS_FILE (env-configurable)
      2. $DYNUNET_WEIGHTS_DIR/model_brats_mri_segmentation.pt (MONAI bundle name)
      3. weights/ directory relative to this file
      4. ~/.monai/ user-level cache
    """
    monai_name = "model_brats_mri_segmentation.pt"
    candidates = [
        os.path.join(WEIGHTS_DIR, WEIGHTS_FILE),
        os.path.join(WEIGHTS_DIR, monai_name),
        os.path.join(os.path.dirname(__file__), "..", "weights", WEIGHTS_FILE),
        os.path.join(os.path.dirname(__file__), "..", "weights", monai_name),
        os.path.join(os.path.expanduser("~"), ".monai", WEIGHTS_FILE),
        os.path.join(os.path.expanduser("~"), ".monai", monai_name),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            return path
    return None


def download_weights() -> str | None:
    """
    Download official MONAI BraTS SegResNet pretrained weights.
    Returns the path to the downloaded file, or None on failure.
    """
    dest_dir = os.path.abspath(WEIGHTS_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "model_brats_mri_segmentation.pt")

    if os.path.isfile(dest_path):
        file_size = os.path.getsize(dest_path)
        if file_size > 1_000_000:  # >1MB = likely valid
            print(f"  Weights already downloaded: {dest_path} ({file_size / 1e6:.1f} MB)")
            return dest_path
        else:
            os.remove(dest_path)  # Remove corrupt/partial file

    print("  Downloading MONAI BraTS SegResNet weights...")
    print(f"  URL: {MONAI_BRATS_WEIGHTS_URL}")
    print(f"  Destination: {dest_path}")

    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", dest_path, MONAI_BRATS_WEIGHTS_URL],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0 and os.path.isfile(dest_path):
            file_size = os.path.getsize(dest_path)
            print(f"  Download complete: {file_size / 1e6:.1f} MB")
            return dest_path
        else:
            print(f"  wget failed (exit {result.returncode}): {result.stderr[:200]}")
    except Exception as e:
        print(f"  Download error: {e}")

    # Fallback: try Python urllib
    try:
        print("  Trying Python urllib fallback...")
        import urllib.request
        urllib.request.urlretrieve(MONAI_BRATS_WEIGHTS_URL, dest_path)
        if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            print(f"  Download complete: {os.path.getsize(dest_path) / 1e6:.1f} MB")
            return dest_path
    except Exception as e:
        print(f"  urllib fallback failed: {e}")

    # Clean up partial downloads
    if os.path.isfile(dest_path):
        os.remove(dest_path)

    return None


def load_model_with_weights(device: torch.device) -> SegResNet | None:
    """
    Build the SegResNet model and load pretrained weights if available.
    Attempts download if weights are not found locally.
    Returns None if no weights can be loaded.
    """
    weights_path = find_weights_path()

    if weights_path is None:
        print("  No pretrained weights found locally. Attempting download...")
        weights_path = download_weights()

    if weights_path is None:
        print("  [INFO] No pretrained SegResNet weights available.")
        print(f"  [INFO] To enable model inference, place weights at:")
        print(f"         {os.path.abspath(os.path.join(WEIGHTS_DIR, WEIGHTS_FILE))}")
        return None

    print(f"  Loading SegResNet weights from: {weights_path}")
    model = build_segresnet_model()

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Support various checkpoint formats
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("  SegResNet model loaded successfully.")
    return model


def build_preprocessing_transforms(keys: list[str]):
    """Build MONAI preprocessing transforms for BraTS MRI modalities."""
    return Compose([
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=keys, name="image"),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


def find_modality_files(base_dir: str) -> dict[str, str] | None:
    """
    Find BraTS modality files (T1, T1CE, T2, FLAIR) in a patient directory.
    Returns dict mapping modality keys to file paths, or None if not found.
    """
    modality_map: dict[str, str | None] = {"t1": None, "t1ce": None, "t2": None, "flair": None}

    for f in os.listdir(base_dir):
        f_lower = f.lower()
        if not (f_lower.endswith(".nii.gz") or f_lower.endswith(".nii")):
            continue
        stem = f_lower.replace(".nii.gz", "").replace(".nii", "")
        # Skip segmentation masks
        if stem.endswith("_seg") or stem == "seg":
            continue
        for mod in modality_map:
            if stem.endswith(f"_{mod}") or stem == mod:
                modality_map[mod] = os.path.join(base_dir, f)
                break

    if all(v is not None for v in modality_map.values()):
        return modality_map
    # Report which modalities are missing
    missing = [k for k, v in modality_map.items() if v is None]
    print(f"  Missing modalities: {missing}")
    return None


def run_segresnet_inference(model: SegResNet, base_dir: str,
                            device: torch.device) -> np.ndarray | None:
    """
    Run SegResNet inference on a patient's MRI volumes.
    Returns a binary tumor mask as a numpy array, or None on failure.
    """
    modality_files = find_modality_files(base_dir)
    if modality_files is None:
        return None

    keys = ["t1", "t1ce", "t2", "flair"]
    transforms = build_preprocessing_transforms(keys)

    data = {mod: modality_files[mod] for mod in keys}

    try:
        transformed = transforms(data)
    except Exception as e:
        print(f"  [ERROR] Preprocessing failed: {e}")
        return None

    image_tensor = transformed["image"].unsqueeze(0).to(device)  # (1, 4, D, H, W)

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        overlap=OVERLAP,
        mode="gaussian",
    )

    with torch.no_grad():
        output = inferer(image_tensor, model)

    # Post-process: sigmoid → threshold → combine channels to binary mask
    post = Compose([
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ])
    post_data = post({"pred": output.squeeze(0)})
    pred = post_data["pred"].cpu().numpy()  # (3, D, H, W)

    # Combine TC/WT/ET channels into a single binary mask
    binary_mask = (pred.sum(axis=0) > 0).astype(np.int32)
    return binary_mask


def find_seg_file(base_dir: str) -> str | None:
    """Locate the ground-truth segmentation mask NIfTI file in a patient directory."""
    for f in os.listdir(base_dir):
        fname_lower = f.lower()
        if fname_lower.endswith(".nii.gz") or fname_lower.endswith(".nii"):
            stem = fname_lower.replace(".nii.gz", "").replace(".nii", "")
            if stem.endswith("_seg") or stem == "seg":
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
      1. SegResNet pretrained inference (MONAI)
      2. Ground-truth BraTS segmentation (fallback)
    """
    patient_id = state["patient_id"]
    base_dir = state["base_dir"]
    output_dir = state["output_dir"]
    errors = list(state.get("errors", []))

    seg_dir = os.path.join(output_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)

    print(f"[Segmentation] Processing patient: {patient_id}")

    # --- Strategy 1: SegResNet (MONAI) pretrained inference ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model = load_model_with_weights(device)
    if model is not None:
        print("  Attempting SegResNet inference...")
        binary_mask = run_segresnet_inference(model, base_dir, device)
        if binary_mask is not None:
            tumor_voxels = np.sum(binary_mask > 0)
            total_voxels = binary_mask.size
            print(f"  SegResNet segmentation: {tumor_voxels}/{total_voxels} tumor voxels "
                  f"({100 * tumor_voxels / total_voxels:.2f}%)")

            # Save using SimpleITK to preserve NIfTI format/metadata
            # Try to copy spatial info from an input modality
            ref_image = None
            modality_files = find_modality_files(base_dir)
            if modality_files:
                ref_path = next(iter(modality_files.values()))
                ref_image = sitk.ReadImage(ref_path)

            binary_sitk = sitk.GetImageFromArray(binary_mask)
            if ref_image is not None:
                binary_sitk.SetOrigin(ref_image.GetOrigin())
                binary_sitk.SetSpacing(ref_image.GetSpacing())
                binary_sitk.SetDirection(ref_image.GetDirection())

            save_path = os.path.join(seg_dir, f"{patient_id}_seg.nii.gz")
            sitk.WriteImage(binary_sitk, save_path)
            print(f"  Saved SegResNet segmentation: {save_path}")
            return {**state, "segmentation_path": save_path, "errors": errors}
        else:
            errors.append(f"SegResNet inference failed for {patient_id}, falling back to GT")

    # --- Strategy 2: Ground-truth segmentation mask ---
    seg_path = find_seg_file(base_dir)
    if seg_path is not None:
        print(f"  Using ground-truth segmentation: {os.path.basename(seg_path)}")
        try:
            binary_mask, binary_sitk = load_and_binarize_mask(seg_path)
            tumor_voxels = np.sum(binary_mask > 0)
            total_voxels = binary_mask.size
            print(f"  GT tumor voxels: {tumor_voxels}/{total_voxels} "
                  f"({100 * tumor_voxels / total_voxels:.2f}%)")

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
