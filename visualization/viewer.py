"""
Visualization System
======================
Generates visual outputs for tumor analysis:
  1. MRI slice viewer (axial, coronal, sagittal)
  2. Tumor overlay on T1CE
  3. Tumor probability heatmap
  4. 3D tumor voxel rendering
  5. Tumor progression chart

All outputs saved to outputs/visualizations/{patient_id}/
"""
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # Non-interactive backend — works without display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _get_mid_slices(volume: np.ndarray):
    """Get axial, coronal, sagittal mid-slices from a 3D volume."""
    d, h, w = volume.shape
    return (
        volume[d // 2, :, :],   # axial
        volume[:, h // 2, :],   # coronal
        volume[:, :, w // 2],   # sagittal
    )


def save_mri_slice_viewer(volume: np.ndarray, patient_id: str,
                          save_dir: str, modality: str = "T1CE"):
    """Save a 3-plane MRI slice viewer."""
    axial, coronal, sag = _get_mid_slices(volume)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    fig.suptitle(f"{patient_id} — {modality} MRI Slice Viewer",
                 color="white", fontsize=14, fontweight="bold")

    for ax, img, label in zip(axes,
                               [axial, coronal, sag],
                               ["Axial", "Coronal", "Sagittal"]):
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_title(label, color="white", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{patient_id}_mri_slices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    return path


def save_tumor_overlay(volume: np.ndarray, mask: np.ndarray,
                       patient_id: str, save_dir: str):
    """Save MRI with tumor mask overlaid in red."""
    axial_vol, _, _ = _get_mid_slices(volume)
    axial_mask, _, _ = _get_mid_slices(mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    fig.suptitle(f"{patient_id} — Tumor Segmentation Overlay",
                 color="white", fontsize=14, fontweight="bold")

    titles = ["MRI Only", "Tumor Mask", "Overlay"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")

    axes[0].imshow(axial_vol, cmap="gray", aspect="auto")

    axes[1].imshow(axial_mask, cmap="hot", aspect="auto")

    axes[2].imshow(axial_vol, cmap="gray", aspect="auto")
    tumor_rgba = np.zeros((*axial_mask.shape, 4))
    tumor_rgba[axial_mask > 0] = [1.0, 0.2, 0.1, 0.65]  # red, semi-transparent
    axes[2].imshow(tumor_rgba, aspect="auto")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{patient_id}_tumor_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    return path


def save_heatmap(mask: np.ndarray, patient_id: str, save_dir: str):
    """Save tumor probability heatmap with smooth colormap."""
    from scipy.ndimage import gaussian_filter
    axial, coronal, sag = _get_mid_slices(mask.astype(np.float32))

    # Smooth each slice to create a probability-like heatmap
    axial_smooth = gaussian_filter(axial, sigma=3)
    coronal_smooth = gaussian_filter(coronal, sigma=3)
    sag_smooth = gaussian_filter(sag, sigma=3)

    # Custom colormap: black → blue → red → yellow
    heat_colors = ["#000000", "#0000ff", "#ff0000", "#ffff00"]
    cmap = LinearSegmentedColormap.from_list("tumor_heat", heat_colors)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    fig.suptitle(f"{patient_id} — Tumor Probability Heatmap",
                 color="white", fontsize=14, fontweight="bold")

    for ax, img, label in zip(axes,
                               [axial_smooth, coronal_smooth, sag_smooth],
                               ["Axial", "Coronal", "Sagittal"]):
        im = ax.imshow(img, cmap=cmap, vmin=0, aspect="auto")
        ax.set_title(label, color="white", fontsize=11)
        ax.axis("off")

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Probability")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{patient_id}_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    return path


def save_3d_rendering(mask: np.ndarray, patient_id: str, save_dir: str):
    """Save 3D voxel-based tumor rendering (3 orthogonal projections)."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None

    fig = plt.figure(figsize=(12, 4), facecolor="black")
    fig.suptitle(f"{patient_id} — 3D Tumor Rendering",
                 color="white", fontsize=14, fontweight="bold")

    view_pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [("Z", "Y"), ("Z", "X"), ("Y", "X")]
    subtitles = ["Axial View", "Coronal View", "Sagittal View"]

    for i, ((ax_a, ax_b), (lx, ly), title) in enumerate(
            zip(view_pairs, labels, subtitles)):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.scatter(coords[:, ax_b], coords[:, ax_a],
                   s=0.5, c="red", alpha=0.3, rasterized=True)
        ax.set_facecolor("black")
        ax.set_xlabel(lx, color="white")
        ax.set_ylabel(ly, color="white")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{patient_id}_3d_rendering.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    return path


def save_progression_chart(history: list, current_volume: float,
                           patient_id: str, save_dir: str):
    """Save a tumor volume progression bar chart over time."""
    max_points = 60
    trimmed_history = history[-max_points:] if len(history) > max_points else history
    volumes = [h.get("tumor_volume", 0) for h in trimmed_history] + [current_volume]
    offset = max(0, len(history) - len(trimmed_history))
    labels = [f"Scan {offset + i + 1}" for i in range(len(trimmed_history))] + ["Current"]

    if len(volumes) < 2:
        # Still show a single-bar chart
        pass

    fig_width = min(20, max(6, len(volumes) * 0.6))
    fig, ax = plt.subplots(figsize=(fig_width, 5),
                           facecolor="black")
    fig.suptitle(f"{patient_id} — Tumor Volume Progression",
                 color="white", fontsize=14, fontweight="bold")

    colors = ["#444488"] * len(trimmed_history) + ["#e84040"]
    bars = ax.bar(labels, volumes, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_facecolor("black")
    ax.set_ylabel("Volume (mm³)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    for bar, vol in zip(bars, volumes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{vol:.0f}", ha="center", va="bottom", color="white", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{patient_id}_progression_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    return path


def run_visualization(state: dict) -> dict:
    """
    LangGraph node: Generate all visualizations for a patient.
    """
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    preprocessed_path = state.get("preprocessed_path")
    segmentation_path = state.get("segmentation_path")
    clinical = state.get("clinical_profile", {})
    history = state.get("patient_history", [])
    errors = list(state.get("errors", []))

    viz_dir = os.path.join(output_dir, "visualizations", patient_id)
    os.makedirs(viz_dir, exist_ok=True)

    print(f"[Visualization] Processing patient: {patient_id}")

    if not MATPLOTLIB_AVAILABLE:
        msg = "matplotlib not available, skipping visualization."
        print(f"  [WARNING] {msg}")
        errors.append(msg)
        return {**state, "visualization_paths": {}, "errors": errors}

    viz_paths = {}

    try:
        # Load preprocessed volume (T1CE = channel 1)
        volume = None
        if preprocessed_path and os.path.exists(preprocessed_path):
            data = np.load(preprocessed_path)
            if data.ndim == 4:
                volume = data[1]  # T1CE channel
            else:
                volume = data

        # Load segmentation mask
        mask = None
        if segmentation_path and os.path.exists(segmentation_path):
            import SimpleITK as sitk
            mask_sitk = sitk.ReadImage(segmentation_path, sitk.sitkInt32)
            mask = sitk.GetArrayFromImage(mask_sitk)

        # 1. MRI Slice Viewer
        if volume is not None:
            path = save_mri_slice_viewer(volume, patient_id, viz_dir)
            viz_paths["mri_slices"] = path
            print(f"  Saved MRI slice viewer: {os.path.basename(path)}")

        # 2. Tumor Overlay
        if volume is not None and mask is not None:
            # Resize mask if needed
            if volume.shape != mask.shape:
                min_shape = tuple(min(s1, s2)
                                  for s1, s2 in zip(volume.shape, mask.shape))
                volume = volume[:min_shape[0], :min_shape[1], :min_shape[2]]
                mask_crop = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                mask_crop = mask
            path = save_tumor_overlay(volume, mask_crop, patient_id, viz_dir)
            viz_paths["tumor_overlay"] = path
            print(f"  Saved tumor overlay: {os.path.basename(path)}")

        # 3. Probability Heatmap
        if mask is not None:
            path = save_heatmap(mask, patient_id, viz_dir)
            viz_paths["heatmap"] = path
            print(f"  Saved heatmap: {os.path.basename(path)}")

        # 4. 3D Rendering
        if mask is not None:
            # Downsample for performance
            path = save_3d_rendering(mask, patient_id, viz_dir)
            if path:
                viz_paths["3d_rendering"] = path
                print(f"  Saved 3D rendering: {os.path.basename(path)}")

        # 5. Progression Chart
        current_volume = clinical.get("morphology", {}).get("tumor_volume", 0)
        path = save_progression_chart(history, current_volume, patient_id, viz_dir)
        viz_paths["progression_chart"] = path
        print(f"  Saved progression chart: {os.path.basename(path)}")

    except Exception as e:
        msg = f"Visualization error for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)

    print(f"  Generated {len(viz_paths)} visualizations.")
    return {**state, "visualization_paths": viz_paths, "errors": errors}
