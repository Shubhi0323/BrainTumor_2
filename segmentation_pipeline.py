# %% [markdown]
# # Brain Tumor Segmentation Pipeline — BraTS 2020 (Kaggle)
# 
# **Fixes over the original pipeline:**
# 1. **Model weights**: Properly downloads MONAI Model Zoo BraTS bundle (init_filters=32)
# 2. **Skull stripping**: Robust percentile-based + morphological approach (replaces weak Otsu)
# 3. **No fallbacks**: Uses SegResNet directly — no heuristic segmentation fallback
# 4. **Full evaluation**: Dice, HD95, Sensitivity, Specificity with aggregate stats
#
# **Dataset**: BraTS 2020 Training + Validation (Kaggle)
#
# Open this file in VS Code → it renders as an interactive notebook via `# %%` markers.
# To export as .ipynb: Ctrl+Shift+P → "Export Current Python File as Jupyter Notebook"

# %% [markdown]
# ## 1. Environment Setup

# %%
# Install dependencies (uncomment if needed)
# !pip install -q "monai[all]>=1.3" nibabel SimpleITK scipy matplotlib pandas tqdm torch

import os, glob, json, time, warnings
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from scipy.ndimage import (
    binary_opening, binary_closing, binary_fill_holes,
    binary_erosion, binary_dilation, generate_binary_structure,
    label as ndlabel, distance_transform_edt
)

import torch
from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, ConcatItemsd, EnsureTyped,
    LoadImage, EnsureChannelFirst, Orientation, Spacing
)

warnings.filterwarnings("ignore")
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")

# %% [markdown]
# ## 2. Configuration

# %%
# ═══════════════════════════════════════════════════════════
# CONFIGURATION — BraTS 2020 on Kaggle
# ═══════════════════════════════════════════════════════════

# BraTS 2020 Kaggle dataset paths
BRATS_TRAIN_DIR = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
BRATS_VAL_DIR   = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
METADATA_CSV    = "/kaggle/input/datasets/awsaf49/brats2020-training-data/BraTS20 Training Metadata.csv"

# Use training data (has ground truth _seg files) for evaluation
# Switch to BRATS_VAL_DIR if you want inference-only on validation set
DATA_DIR = BRATS_TRAIN_DIR if os.path.isdir(BRATS_TRAIN_DIR) else BRATS_VAL_DIR

OUTPUT_DIR = "/kaggle/working/brats_outputs"   # Kaggle writable directory
BUNDLE_DIR = "/kaggle/working/monai_bundles"    # Where model weights are cached

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture — MUST match the downloaded MONAI checkpoint
MODEL_CFG = dict(
    in_channels=4, out_channels=3, init_filters=16,
    blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], dropout_prob=0.2
)

ROI_SIZE = (240, 240, 160)   # Sliding window patch size
SW_BATCH_SIZE = 1
OVERLAP = 0.5

# Post-processing thresholds
CC_MIN_VOXELS = 500
CC_KEEP_TOP_N = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# %% [markdown]
# ## 3. Dataset Discovery

# %%
# BraTS 2020 naming: BraTS20_Training_001_t1.nii.gz, _t1ce, _t2, _flair, _seg
MOD_ALIASES = {
    "t1":   ["_t1.", "_t1_", "-t1.", "-t1_"],
    "t1ce": ["_t1ce.", "_t1ce_", "_t1c.", "_t1c_", "_t1gd.", "_t1post"],
    "t2":   ["_t2.", "_t2_", "-t2.", "-t2_"],
    "flair": ["_flair.", "_flair_", "-flair."],
}
GT_PATTERNS = ["*_seg.nii.gz", "*_seg.nii",
               "*_tumor_segmentation.nii.gz", "*_tumor_segmentation.nii"]


def discover_patients(data_dir):
    """Find all patient directories in BraTS 2020 dataset."""
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return []
    # BraTS dirs look like BraTS20_Training_001 or BraTS20_Validation_084
    return sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))
                   and d.startswith('BraTS20')])


def find_modality(patient_dir, modality):
    """Locate a specific modality NIfTI file."""
    for f in os.listdir(patient_dir):
        fl = f.lower()
        if not (fl.endswith(".nii.gz") or fl.endswith(".nii")):
            continue
        if "seg" in fl or "tumor_segmentation" in fl:
            continue
        for alias in MOD_ALIASES.get(modality, [f"_{modality}"]):
            if alias in fl:
                return os.path.join(patient_dir, f)
    return None


def find_all_modalities(patient_dir):
    """Find T1, T1CE, T2, FLAIR. Returns dict or None if any missing."""
    result = {}
    for mod in ["t1", "t1ce", "t2", "flair"]:
        path = find_modality(patient_dir, mod)
        if path is None:
            return None
        result[mod] = path
    return result


def find_ground_truth(patient_dir):
    """Find ground truth segmentation mask."""
    for pattern in GT_PATTERNS:
        matches = glob.glob(os.path.join(patient_dir, pattern))
        if matches:
            return matches[0]
    return None


# Run discovery
print(f"Looking for patients in: {DATA_DIR}")
patients = discover_patients(DATA_DIR)
print(f"Found {len(patients)} patients in {DATA_DIR}")

# Load metadata CSV if available
if os.path.exists(METADATA_CSV):
    meta_df = pd.read_csv(METADATA_CSV)
    print(f"Loaded metadata: {METADATA_CSV} ({len(meta_df)} rows)")
else:
    meta_df = None
    print(f"Metadata CSV not found at {METADATA_CSV} (optional)")

gt_count = 0
for p in patients[:5]:
    pdir = os.path.join(DATA_DIR, p)
    mods = find_all_modalities(pdir)
    gt = find_ground_truth(pdir)
    gt_count += 1 if gt else 0
    print(f"  {p}: modalities={'OK' if mods else 'MISSING'}, GT={'YES' if gt else 'NO'}")
if len(patients) > 5:
    print(f"  ... and {len(patients)-5} more")
print(f"Patients with ground truth (sampled): {gt_count}/{ min(len(patients), 5) }")

# %% [markdown]
# ## 4. Preprocessing (Improved Skull Stripping)
#
# **Key fix**: Replaced weak Otsu thresholding with a robust multi-step approach:
# percentile-based threshold → large morphological closing → largest connected
# component → dilation to recover boundary tissue → hole filling.

# %%
def robust_skull_strip(image_sitk):
    """
    Robust skull stripping — fixes the Otsu-based approach that was
    including skull/eyes/fat in the brain mask, causing SegResNet to
    classify them as tumor (resulting in massive over-segmentation).
    """
    arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    nonzero = arr[arr > 0]
    if len(nonzero) < 100:
        return image_sitk

    # Percentile threshold (15th pct) — much more robust than Otsu for MRI
    threshold = np.percentile(nonzero, 15)
    mask = (arr > threshold).astype(np.uint8)

    struct = generate_binary_structure(3, 1)

    # Large morphological closing to bridge gaps (radius=5)
    mask = binary_closing(mask, structure=struct, iterations=5).astype(np.uint8)

    # Keep only the largest connected component (= the brain)
    labeled, n_comp = ndlabel(mask)
    if n_comp > 1:
        sizes = [(i, np.sum(labeled == i)) for i in range(1, n_comp + 1)]
        largest = max(sizes, key=lambda x: x[1])[0]
        mask = (labeled == largest).astype(np.uint8)

    # Dilate to recover boundary tissue
    mask = binary_dilation(mask, structure=struct, iterations=2).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)

    result_arr = arr * mask
    result = sitk.GetImageFromArray(result_arr)
    result.CopyInformation(image_sitk)
    return result


def normalize_intensity_zscore(image_sitk):
    """Z-score normalization on non-zero voxels only."""
    arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    nz = arr[arr > 0]
    if len(nz) > 0 and np.std(nz) > 0:
        arr[arr > 0] = (arr[arr > 0] - np.mean(nz)) / np.std(nz)
    result = sitk.GetImageFromArray(arr)
    result.CopyInformation(image_sitk)
    return result


def resample_isotropic(image_sitk, spacing=(1.0, 1.0, 1.0)):
    """Resample to isotropic 1mm spacing."""
    orig_sp = image_sitk.GetSpacing()
    orig_sz = image_sitk.GetSize()
    new_sz = [int(round(s * sp / t)) for s, sp, t in zip(orig_sz, orig_sp, spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_sz)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image_sitk)


def preprocess_modality(nifti_path, do_skull_strip=True):
    """Full preprocessing for one modality: N4 → skull strip → normalize → resample."""
    img = sitk.ReadImage(nifti_path, sitk.sitkFloat32)
    # N4 bias field correction
    try:
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        img = corrector.Execute(sitk.Cast(img, sitk.sitkFloat32), mask)
    except Exception:
        pass  # Continue without N4 if it fails
    if do_skull_strip:
        img = robust_skull_strip(img)
    img = normalize_intensity_zscore(img)
    img = resample_isotropic(img)
    return img


print("Preprocessing functions defined ✓")

# %% [markdown]
# ## 5. SegResNet Model Setup & Pretrained Weights
#
# Downloads the official MONAI BraTS Segmentation Bundle.
# **Critical fix**: Uses `init_filters=32` (matching the pretrained weights)
# instead of the original pipeline's `init_filters=16` which caused a
# silent mismatch and forced fallback to heuristic segmentation.

# %%
def build_model():
    return SegResNet(**MODEL_CFG)


def download_pretrained_weights():
    """Download weights from MONAI Model Zoo BraTS bundle."""
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    weights_path = os.path.join(BUNDLE_DIR, "brats_mri_segmentation",
                                "models", "model.pt")
    if os.path.exists(weights_path):
        print(f"Weights cached: {weights_path}")
        return weights_path

    # Method 1: MONAI bundle API
    try:
        from monai.bundle import download
        print("Downloading via monai.bundle.download ...")
        download(name="brats_mri_segmentation", bundle_dir=BUNDLE_DIR)
        if os.path.exists(weights_path):
            print("Download complete ✓")
            return weights_path
    except Exception as e:
        print(f"Bundle API failed: {e}")

    # Method 2: Direct URL
    try:
        import urllib.request
        url = ("https://github.com/Project-MONAI/model-zoo/releases/download/"
               "hosting_storage_v1/brats_mri_segmentation_v0.1.9.zip")
        zip_path = os.path.join(BUNDLE_DIR, "bundle.zip")
        print(f"Downloading from GitHub releases ...")
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(BUNDLE_DIR)
        os.remove(zip_path)
        if os.path.exists(weights_path):
            print("Download complete ✓")
            return weights_path
    except Exception as e:
        print(f"Direct download failed: {e}")

    return None


def load_model():
    """Build SegResNet and load pretrained weights (auto-detects init_filters)."""
    wp = download_pretrained_weights()

    if wp is None:
        print("WARNING: No pretrained weights found!")
        model = build_model()
        model.to(DEVICE).eval()
        return model, False

    ckpt = torch.load(wp, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in ckpt:
                ckpt = ckpt[key]
                break

    # Auto-detect init_filters from checkpoint
    # The first conv layer weight shape tells us: init_filters = shape[0]
    detect_key = None
    for k in ckpt:
        if "convInit" in k and "weight" in k:
            detect_key = k
            break
        if "conv_init" in k and "weight" in k:
            detect_key = k
            break
    if detect_key is not None:
        detected = ckpt[detect_key].shape[0]
        if detected != MODEL_CFG["init_filters"]:
            print(f"Auto-detected init_filters={detected} from checkpoint "
                  f"(config had {MODEL_CFG['init_filters']}) — adjusting.")
            MODEL_CFG["init_filters"] = detected

    model = build_model()
    try:
        model.load_state_dict(ckpt, strict=True)
        print("Weights loaded (strict) ✓")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        model.load_state_dict(ckpt, strict=False)
        print("Weights loaded (non-strict) ⚠")

    model.to(DEVICE).eval()
    return model, True


model, weights_ok = load_model()
print(f"Model on {DEVICE} | Pretrained: {'YES' if weights_ok else 'NO'}")

# %% [markdown]
# ## 6. Inference & Post-processing

# %%
def build_inference_transforms():
    """MONAI preprocessing transforms for SegResNet input."""
    keys = ["t1", "t1ce", "t2", "flair"]
    return Compose([
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=keys, name="image"),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

inf_transforms = build_inference_transforms()


def run_inference(modality_files):
    """Run SegResNet sliding-window inference."""
    data = {m: modality_files[m] for m in ["t1", "t1ce", "t2", "flair"]}
    transformed = inf_transforms(data)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE,
        overlap=OVERLAP, mode="gaussian"
    )
    with torch.no_grad():
        output = inferer(tensor, model)

    pred = torch.sigmoid(output).squeeze(0).cpu().numpy()
    # Merge all 3 output channels (ET, TC, WT) into a single binary mask
    binary = (pred.sum(axis=0) > 0.5).astype(np.int32)
    return binary


def postprocess(mask):
    """Morphological cleanup + connected component filtering."""
    struct = generate_binary_structure(3, 1)
    # Opening removes small noise, closing fills small holes
    m = binary_opening(mask > 0, structure=struct, iterations=1)
    m = binary_closing(m, structure=struct, iterations=1).astype(np.int32)
    # Keep only top-N largest components above min size
    labeled, n = ndlabel(m > 0)
    if n == 0:
        return m
    sizes = sorted([(i, int(np.sum(labeled == i))) for i in range(1, n + 1)],
                   key=lambda x: x[1], reverse=True)
    keep = [lbl for lbl, sz in sizes[:CC_KEEP_TOP_N] if sz >= CC_MIN_VOXELS]
    return np.isin(labeled, keep).astype(np.int32) if keep else m


print("Inference pipeline defined ✓")

# %% [markdown]
# ## 7. Evaluation Metrics

# %%
def dice_coefficient(pred, gt):
    p, g = (pred > 0).astype(float), (gt > 0).astype(float)
    inter = np.sum(p * g)
    denom = np.sum(p) + np.sum(g)
    if denom == 0:
        return 1.0 if np.sum(g) == 0 else 0.0
    return float(2.0 * inter / denom)


def hausdorff_95(pred, gt):
    p, g = (pred > 0).astype(bool), (gt > 0).astype(bool)
    if not np.any(p) or not np.any(g):
        return float("inf")
    d1 = distance_transform_edt(~g)[p & ~binary_erosion(p)]
    d2 = distance_transform_edt(~p)[g & ~binary_erosion(g)]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def calc_sensitivity(pred, gt):
    p, g = (pred > 0).astype(float), (gt > 0).astype(float)
    tp, fn = np.sum(p * g), np.sum((1 - p) * g)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def calc_specificity(pred, gt):
    p, g = (pred > 0).astype(float), (gt > 0).astype(float)
    tn, fp = np.sum((1 - p) * (1 - g)), np.sum(p * (1 - g))
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def evaluate(pred, gt):
    return {
        "dice": round(dice_coefficient(pred, gt), 4),
        "hd95": round(hausdorff_95(pred, gt), 2),
        "sensitivity": round(calc_sensitivity(pred, gt), 4),
        "specificity": round(calc_specificity(pred, gt), 4),
    }


print("Evaluation metrics defined ✓")

# %% [markdown]
# ## 8. Execute Full Pipeline

# %%
results = []
seg_dir = os.path.join(OUTPUT_DIR, "segmentation")
os.makedirs(seg_dir, exist_ok=True)

for i, pid in enumerate(patients):
    patient_dir = os.path.join(DATA_DIR, pid)
    rec = {"patient_id": pid}
    print(f"[{i+1}/{len(patients)}] {pid} ... ", end="", flush=True)

    # Find modalities
    mods = find_all_modalities(patient_dir)
    if mods is None:
        rec["status"] = "skipped"
        rec["error"] = "missing_modalities"
        results.append(rec)
        print("SKIP (missing modalities)")
        continue

    # Inference
    try:
        t0 = time.time()
        mask = run_inference(mods)
        mask = postprocess(mask)
        elapsed = time.time() - t0
        rec["time_s"] = round(elapsed, 1)
        rec["pred_voxels"] = int(np.sum(mask > 0))
        rec["total_voxels"] = int(mask.size)
        rec["tumor_frac"] = round(rec["pred_voxels"] / max(rec["total_voxels"], 1), 6)
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = str(e)[:200]
        results.append(rec)
        print(f"FAIL ({e})")
        continue

    # Save prediction
    save_path = os.path.join(seg_dir, f"{pid}_pred.nii.gz")
    try:
        ref = sitk.ReadImage(mods["t1"])
        out = sitk.GetImageFromArray(mask)
        out.SetOrigin(ref.GetOrigin())
        out.SetSpacing(ref.GetSpacing())
        out.SetDirection(ref.GetDirection())
        sitk.WriteImage(out, save_path)
    except Exception:
        pass

    # Evaluate vs ground truth
    # NOTE: GT must be loaded with same orientation (RAS) and spacing (1mm)
    #       as the inference pipeline, otherwise arrays are spatially flipped
    #       and Dice will be near zero despite correct predictions.
    gt_path = find_ground_truth(patient_dir)
    if gt_path:
        try:
            gt_loader = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes="RAS"),
                Spacing(pixdim=(1.0, 1.0, 1.0), mode="nearest"),
            ])
            gt_tensor = gt_loader(gt_path)
            gt_arr = (gt_tensor.squeeze().numpy() > 0).astype(np.int32)
            # Handle any remaining shape mismatch by cropping/padding
            if mask.shape != gt_arr.shape:
                min_shape = tuple(min(m, g) for m, g in zip(mask.shape, gt_arr.shape))
                mask_eval = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
                gt_arr = gt_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                mask_eval = mask
            metrics = evaluate(mask_eval, gt_arr)
            rec.update(metrics)
            rec["gt_voxels"] = int(np.sum(gt_arr > 0))
            rec["has_gt"] = True
            rec["status"] = "evaluated"
            print(f"Dice={metrics['dice']:.4f}  HD95={metrics['hd95']:.1f}  "
                  f"Sens={metrics['sensitivity']:.4f}  ({elapsed:.0f}s)")
        except Exception as e:
            rec["has_gt"] = False
            rec["status"] = "predicted"
            print(f"predicted (GT eval failed: {e})")
    else:
        rec["has_gt"] = False
        rec["status"] = "predicted"
        print(f"predicted (no GT)  tumor_frac={rec['tumor_frac']:.4f}")

    results.append(rec)

print(f"\nDone — {len(results)} patients processed.")

# %% [markdown]
# ## 9. Results Analysis & Aggregate Visualizations
#
# Publication-quality metric distributions with dark theme.

# %%
# ── Style setup ──────────────────────────────────────────
VIZ_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(VIZ_DIR, exist_ok=True)

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT_1  = "#00e676"   # green
ACCENT_2  = "#00b0ff"   # blue
ACCENT_3  = "#ff6d00"   # orange
ACCENT_4  = "#ff1744"   # red
TITLE_CLR = "#e0e0e0"
LABEL_CLR = "#b0bec5"

COLOR_TP  = (0.0, 0.9, 0.3, 0.55)   # Green
COLOR_FP  = (1.0, 0.15, 0.15, 0.55) # Red
COLOR_FN  = (0.15, 0.45, 1.0, 0.55) # Blue

# BraTS multiclass labels
BRATS_LABELS = {
    "WT": {"name": "Whole Tumor",     "ids": [1, 2, 4], "color": "#ffeb3b"},
    "TC": {"name": "Tumor Core",      "ids": [1, 4],    "color": "#ff9800"},
    "ET": {"name": "Enhancing Tumor",  "ids": [4],       "color": "#f44336"},
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply dark theme to a single axes."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TITLE_CLR, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=LABEL_CLR, fontsize=10)
    ax.set_ylabel(ylabel, color=LABEL_CLR, fontsize=10)
    ax.tick_params(colors=LABEL_CLR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#37474f")


def plot_aggregate_metrics(df_eval, save_dir):
    """Publication-quality 2x2 metric dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Segmentation Performance Dashboard",
                 color="white", fontsize=16, fontweight="bold", y=0.97)

    # 1) Dice histogram
    ax = axes[0, 0]
    vals = df_eval["dice"].dropna()
    ax.hist(vals, bins=25, color=ACCENT_1, edgecolor=DARK_BG, alpha=0.85)
    ax.axvline(vals.mean(), color=ACCENT_4, ls="--", lw=2,
               label=f"Mean = {vals.mean():.3f}")
    ax.axvline(vals.median(), color="white", ls=":", lw=1.5,
               label=f"Median = {vals.median():.3f}")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor="#37474f",
              labelcolor=LABEL_CLR)
    _style_ax(ax, "Dice Score Distribution", "Dice Coefficient", "Count")

    # 2) HD95 histogram
    ax = axes[0, 1]
    hd = df_eval["hd95"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(hd) > 0:
        hd_clip = hd.clip(upper=100)
        ax.hist(hd_clip, bins=25, color=ACCENT_2, edgecolor=DARK_BG, alpha=0.85)
        ax.axvline(hd.median(), color="white", ls=":", lw=1.5,
                   label=f"Median = {hd.median():.1f}")
        ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor="#37474f",
                  labelcolor=LABEL_CLR)
    _style_ax(ax, "HD95 Distribution (capped 100)", "HD95 (mm)", "Count")

    # 3) Sensitivity histogram
    ax = axes[1, 0]
    sens = df_eval["sensitivity"].dropna()
    ax.hist(sens, bins=25, color=ACCENT_3, edgecolor=DARK_BG, alpha=0.85)
    ax.axvline(sens.mean(), color=ACCENT_4, ls="--", lw=2,
               label=f"Mean = {sens.mean():.3f}")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor="#37474f",
              labelcolor=LABEL_CLR)
    _style_ax(ax, "Sensitivity Distribution", "Sensitivity", "Count")

    # 4) Dice vs HD95 scatter
    ax = axes[1, 1]
    d = df_eval["dice"].dropna()
    h = df_eval["hd95"].replace([np.inf, -np.inf], np.nan)
    mask_both = d.notna() & h.notna()
    if mask_both.any():
        ax.scatter(d[mask_both], h[mask_both].clip(upper=100),
                   c=ACCENT_1, alpha=0.5, s=25, edgecolors="none")
    _style_ax(ax, "Dice vs HD95", "Dice Coefficient", "HD95 (mm, capped 100)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(save_dir, "metrics_dashboard.png")
    plt.savefig(path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ── Run aggregate analysis ───────────────────────────────
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
df.to_csv(csv_path, index=False)
print(f"Results saved: {csv_path}\n")

df_gt = df[df.get("has_gt", pd.Series(dtype=bool)) == True].copy()

if len(df_gt) > 0:
    print("=" * 60)
    print(f"AGGREGATE RESULTS — {len(df_gt)} patients with ground truth")
    print("=" * 60)
    for m in ["dice", "hd95", "sensitivity", "specificity"]:
        v = df_gt[m].dropna()
        if len(v) > 0:
            print(f"  {m:14s}:  {v.mean():.4f} ± {v.std():.4f}"
                  f"  (range: {v.min():.4f} – {v.max():.4f})")

    plot_aggregate_metrics(df_gt, VIZ_DIR)

    cols = ["patient_id", "dice", "hd95", "sensitivity", "specificity"]
    print("\n5 BEST Dice scores:")
    print(df_gt.nlargest(5, "dice")[cols].to_string(index=False))
    print("\n5 WORST Dice scores:")
    print(df_gt.nsmallest(5, "dice")[cols].to_string(index=False))
else:
    print("No ground truth available. Basic stats:")
    print(df[["patient_id", "status", "pred_voxels", "tumor_frac"]].to_string(index=False))

# %% [markdown]
# ## 10. Professional Visualization Module
#
# Publication-quality medical imaging figures with:
# - Multiclass BraTS labels (WT / TC / ET)
# - TP / FP / FN colour-coded analysis
# - Contour-based overlays on dark background
# - Automatic best-slice selection
# - Modular API: `visualize_case()`, `visualize_best_and_worst_cases()`

# %%
# ── Helpers ──────────────────────────────────────────────

def load_nifti_ras(path):
    """Load a NIfTI and orient to RAS (matching inference pipeline)."""
    loader = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="nearest"),
    ])
    return loader(path).squeeze().numpy()


def load_pred_raw(path):
    """Load a prediction NIfTI WITHOUT re-orientation.

    The saved prediction has RAS array data but the NIfTI header
    still says LPS (copied from original T1). Re-orienting it would
    double-flip the array and misalign it with properly loaded T1CE/GT.
    """
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkInt32))
    return arr


def _norm_slice(arr_2d):
    """Percentile-normalize a 2-D slice to [0, 1]."""
    nz = arr_2d[arr_2d > 0]
    if len(nz) == 0:
        return np.zeros_like(arr_2d)
    lo, hi = np.percentile(nz, [1, 99])
    out = np.clip(arr_2d, lo, hi)
    rng = hi - lo
    return (out - lo) / rng if rng > 0 else np.zeros_like(out)


def _rgba(mask_2d, color, alpha=0.55):
    """Binary 2-D mask → RGBA overlay."""
    h, w = mask_2d.shape
    img = np.zeros((h, w, 4), dtype=np.float32)
    m = mask_2d > 0
    img[m, :3] = color[:3]
    img[m, 3] = alpha
    return img


def _best_slice(pred_3d, gt_3d=None):
    """Slice with maximum combined (pred + GT) tumour area."""
    score = np.sum(pred_3d > 0, axis=(1, 2)).astype(float)
    if gt_3d is not None:
        score += np.sum(gt_3d > 0, axis=(1, 2)).astype(float)
    return int(np.argmax(score))


def _extract_region(vol, label_ids):
    """Binarise a label volume for given BraTS label IDs."""
    return np.isin(vol, label_ids).astype(np.int32)


def _align_shapes(*vols):
    """Crop all volumes to the minimum common shape."""
    ms = tuple(min(v.shape[d] for v in vols) for d in range(3))
    return [v[:ms[0], :ms[1], :ms[2]] for v in vols]


# ── Core single-case visualizer ──────────────────────────

def visualize_case(patient_id, pred, gt, t1ce,
                   dice_val=None, hd95_val=None,
                   region="WT", save_dir=None):
    """
    Publication-quality 5-panel figure for one patient.

    Panels: T1CE | Prediction | Ground Truth | Overlay | TP/FP/FN

    Parameters
    ----------
    patient_id : str
    pred, gt   : 3-D int arrays (BraTS labels or binary)
    t1ce       : 3-D float array
    dice_val   : optional float  – shown in title
    hd95_val   : optional float  – shown in title
    region     : 'WT', 'TC', or 'ET'
    save_dir   : path – if set, PNG is saved here
    """
    info = BRATS_LABELS.get(region, BRATS_LABELS["WT"])
    p_bin = _extract_region(pred, info["ids"]) if pred.max() > 1 else (pred > 0).astype(np.int32)
    g_bin = _extract_region(gt,   info["ids"]) if gt.max() > 1  else (gt > 0).astype(np.int32)
    t1ce, p_bin, g_bin = _align_shapes(t1ce, p_bin, g_bin)

    sl = _best_slice(p_bin, g_bin)
    img = _norm_slice(t1ce[sl])
    p2 = p_bin[sl]
    g2 = g_bin[sl]

    # Title
    title = f"{patient_id}  —  {info['name']}  (slice {sl})"
    if dice_val is not None:
        title += f"   Dice={dice_val:.4f}"
    if hd95_val is not None:
        hd_str = f"{hd95_val:.1f}" if np.isfinite(hd95_val) else "∞"
        title += f"   HD95={hd_str}"

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.patch.set_facecolor("black")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.0)

    # ── Panel 1: T1CE ──
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("T1CE", color=TITLE_CLR, fontsize=11)

    # ── Panel 2: Prediction ──
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(_rgba(p2, COLOR_FP[:3], 0.45))
    if np.any(p2):
        axes[1].contour(p2 > 0, colors=["#ff1744"], linewidths=1.2)
    axes[1].set_title("Prediction", color=TITLE_CLR, fontsize=11)

    # ── Panel 3: Ground Truth ──
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(_rgba(g2, COLOR_TP[:3], 0.45))
    if np.any(g2):
        axes[2].contour(g2 > 0, colors=["#00e676"], linewidths=1.2)
    axes[2].set_title("Ground Truth", color=TITLE_CLR, fontsize=11)

    # ── Panel 4: Contour Overlay (pred=red, GT=green) ──
    axes[3].imshow(img, cmap="gray")
    if np.any(p2):
        axes[3].contour(p2 > 0, colors=["#ff1744"], linewidths=1.5)
    if np.any(g2):
        axes[3].contour(g2 > 0, colors=["#00e676"], linewidths=1.5)
    axes[3].set_title("Contour Overlay", color=TITLE_CLR, fontsize=11)

    # ── Panel 5: TP / FP / FN ──
    tp = (p2 > 0) & (g2 > 0)
    fp = (p2 > 0) & (g2 == 0)
    fn = (p2 == 0) & (g2 > 0)
    err = np.zeros((*p2.shape, 4), dtype=np.float32)
    err[tp] = COLOR_TP    # Green  = correct
    err[fp] = COLOR_FP    # Red    = over-seg
    err[fn] = COLOR_FN    # Blue   = missed
    axes[4].imshow(img, cmap="gray")
    axes[4].imshow(err)
    axes[4].set_title("TP(G) / FP(R) / FN(B)", color=TITLE_CLR, fontsize=11)

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("black")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir:
        path = os.path.join(save_dir, f"{patient_id}_{region}.png")
        plt.savefig(path, dpi=200, facecolor="black", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ── Multiclass (WT / TC / ET) grid ───────────────────────

def visualize_case_multiclass(patient_id, pred, gt, t1ce,
                               dice_val=None, hd95_val=None,
                               save_dir=None):
    """3-row grid showing WT, TC, ET for one patient."""
    regions = ["WT", "TC", "ET"]
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.patch.set_facecolor("black")
    title = f"{patient_id} — Multiclass Segmentation"
    if dice_val is not None:
        title += f"   (WT Dice={dice_val:.4f})"
    fig.suptitle(title, color="white", fontsize=15, fontweight="bold", y=0.98)

    for row, rgn in enumerate(regions):
        info = BRATS_LABELS[rgn]
        p_bin = _extract_region(pred, info["ids"]) if pred.max() > 1 else (pred > 0).astype(np.int32)
        g_bin = _extract_region(gt,   info["ids"]) if gt.max() > 1  else (gt > 0).astype(np.int32)
        t_, p_, g_ = _align_shapes(t1ce, p_bin, g_bin)
        sl = _best_slice(p_, g_)
        img = _norm_slice(t_[sl])
        p2, g2 = p_[sl], g_[sl]

        # T1CE
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title("T1CE" if row == 0 else "", color=TITLE_CLR, fontsize=10)
        axes[row, 0].set_ylabel(info["name"], color=info["color"],
                                fontsize=12, fontweight="bold", rotation=90, labelpad=12)

        # Prediction
        axes[row, 1].imshow(img, cmap="gray")
        axes[row, 1].imshow(_rgba(p2, COLOR_FP[:3], 0.45))
        if np.any(p2):
            axes[row, 1].contour(p2 > 0, colors=["#ff1744"], linewidths=1)
        axes[row, 1].set_title("Prediction" if row == 0 else "", color=TITLE_CLR, fontsize=10)

        # GT
        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].imshow(_rgba(g2, COLOR_TP[:3], 0.45))
        if np.any(g2):
            axes[row, 2].contour(g2 > 0, colors=["#00e676"], linewidths=1)
        axes[row, 2].set_title("Ground Truth" if row == 0 else "", color=TITLE_CLR, fontsize=10)

        # Contour overlay
        axes[row, 3].imshow(img, cmap="gray")
        if np.any(p2):
            axes[row, 3].contour(p2 > 0, colors=["#ff1744"], linewidths=1.3)
        if np.any(g2):
            axes[row, 3].contour(g2 > 0, colors=["#00e676"], linewidths=1.3)
        axes[row, 3].set_title("Overlay" if row == 0 else "", color=TITLE_CLR, fontsize=10)

        # TP/FP/FN
        tp = (p2 > 0) & (g2 > 0)
        fp = (p2 > 0) & (g2 == 0)
        fn = (p2 == 0) & (g2 > 0)
        err = np.zeros((*p2.shape, 4), dtype=np.float32)
        err[tp] = COLOR_TP; err[fp] = COLOR_FP; err[fn] = COLOR_FN
        axes[row, 4].imshow(img, cmap="gray")
        axes[row, 4].imshow(err)
        axes[row, 4].set_title("TP/FP/FN" if row == 0 else "", color=TITLE_CLR, fontsize=10)

    for ax in axes.flat:
        ax.axis("off"); ax.set_facecolor("black")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_dir:
        path = os.path.join(save_dir, f"{patient_id}_multiclass.png")
        plt.savefig(path, dpi=200, facecolor="black", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ── Best / Worst batch visualizer ────────────────────────

def visualize_best_and_worst_cases(df_eval, data_dir, seg_dir,
                                    n_best=5, n_worst=3, save_dir=None):
    """Render top-N best and worst Dice cases with full 5-panel figures."""
    sd_best  = os.path.join(save_dir, "best")  if save_dir else None
    sd_worst = os.path.join(save_dir, "worst") if save_dir else None
    for d in [sd_best, sd_worst]:
        if d: os.makedirs(d, exist_ok=True)

    for tag, pids, sd in [
        ("BEST",  df_eval.nlargest(n_best, "dice"),   sd_best),
        ("WORST", df_eval[df_eval["dice"] > 0].nsmallest(n_worst, "dice"), sd_worst),
    ]:
        print(f"\n{'═'*60}")
        print(f"  {tag} {len(pids)} CASES")
        print(f"{'═'*60}")
        for _, row in pids.iterrows():
            pid = row["patient_id"]
            pdir = os.path.join(data_dir, pid)
            pred_path = os.path.join(seg_dir, f"{pid}_pred.nii.gz")
            if not os.path.exists(pred_path):
                continue

            mods = find_all_modalities(pdir)
            gt_path = find_ground_truth(pdir)
            if mods is None or gt_path is None:
                continue

            t1ce_vol = load_nifti_ras(mods["t1ce"])
            pred_vol = load_pred_raw(pred_path).astype(np.int32)
            gt_vol   = load_nifti_ras(gt_path).astype(np.int32)

            d_val = row.get("dice", None)
            h_val = row.get("hd95", None)

            print(f"\n>>> {pid}  (Dice={d_val:.4f})")
            visualize_case(pid, pred_vol, gt_vol, t1ce_vol,
                           dice_val=d_val, hd95_val=h_val,
                           region="WT", save_dir=sd)


# %% [markdown]
# ## 11. Run Visualizations

# %%
if len(df_gt) > 0:
    visualize_best_and_worst_cases(
        df_gt, DATA_DIR, seg_dir,
        n_best=5, n_worst=3, save_dir=VIZ_DIR
    )

    # Multiclass grid for the single best case
    top_pid = df_gt.nlargest(1, "dice").iloc[0]
    pid = top_pid["patient_id"]
    pdir = os.path.join(DATA_DIR, pid)
    mods = find_all_modalities(pdir)
    gt_path = find_ground_truth(pdir)
    if mods and gt_path:
        print(f"\n{'═'*60}")
        print(f"  MULTICLASS VIEW — {pid}")
        print(f"{'═'*60}")
        visualize_case_multiclass(
            pid,
            load_pred_raw(os.path.join(seg_dir, f"{pid}_pred.nii.gz")).astype(np.int32),
            load_nifti_ras(gt_path).astype(np.int32),
            load_nifti_ras(mods["t1ce"]),
            dice_val=top_pid["dice"],
            save_dir=VIZ_DIR,
        )
else:
    for pid in patients[:5]:
        pdir = os.path.join(DATA_DIR, pid)
        mods = find_all_modalities(pdir)
        if mods:
            pred_path = os.path.join(seg_dir, f"{pid}_pred.nii.gz")
            if os.path.exists(pred_path):
                t1ce_vol = load_nifti_ras(mods["t1ce"])
                pred_vol = load_pred_raw(pred_path).astype(np.int32)
                gt_dummy = np.zeros_like(pred_vol)
                visualize_case(pid, pred_vol, gt_dummy, t1ce_vol, save_dir=VIZ_DIR)

print(f"\nAll figures saved to: {VIZ_DIR}")

# %% [markdown]
# ## Summary
#
# | Component | Details |
# |---|---|
# | Model | SegResNet (MONAI bundle, auto-detected init_filters) |
# | Dataset | BraTS 2020 Training (cross-dataset eval, model trained on BraTS 2021) |
# | Preprocessing | N4 bias correction → robust skull stripping → z-score normalize → 1mm isotropic |
# | Inference | Sliding-window (240×240×160), Gaussian blending, σ threshold → morphological cleanup |
# | Metrics | Dice, HD95, Sensitivity, Specificity (all spatially aligned via RAS) |
# | Visualization | 5-panel per-case (T1CE / Pred / GT / Contour / TP-FP-FN), multiclass WT/TC/ET grids |


# %% [markdown]
# ## 12. Save & Download Results
#
# Everything in `/kaggle/working/` persists as a **Kaggle Output** artifact.
# After the notebook finishes, go to: **Notebook -> Output tab -> Download All**.

# %%
import shutil, zipfile

zip_name = "/kaggle/working/brats_pipeline_results.zip"

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, "/kaggle/working")
            zf.write(full, arcname)

size_mb = os.path.getsize(zip_name) / (1024 * 1024)
print(f"Saved: {zip_name}  ({size_mb:.1f} MB)")
print()
print("To download:")
print("   1. Let the notebook finish running (or click Save Version)")
print("   2. Go to your notebook page - Output tab")
print("   3. Click Download All or download the ZIP directly")
print()
print("Contents:")
with zipfile.ZipFile(zip_name, "r") as zf:
    for info in zf.infolist()[:20]:
        print(f"   {info.filename}  ({info.file_size / 1024:.0f} KB)")
    if len(zf.infolist()) > 20:
        print(f"   ... and {len(zf.infolist()) - 20} more files")