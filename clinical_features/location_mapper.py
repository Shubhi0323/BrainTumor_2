"""
Tumor Location Mapping Node
=============================
Maps tumor regions to brain anatomy using the Harvard-Oxford cortical atlas
via nilearn. Determines which brain lobes the tumor overlaps with.
"""
import os
import numpy as np
import SimpleITK as sitk

try:
    from nilearn import datasets, image as nli_image
    import nibabel as nib
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False


# Brain lobe classification based on Harvard-Oxford atlas regions
LOBE_MAPPING = {
    "frontal": [
        "frontal pole", "superior frontal gyrus", "middle frontal gyrus",
        "inferior frontal gyrus", "precentral gyrus", "frontal medial cortex",
        "frontal orbital cortex", "frontal operculum cortex",
        "subcallosal cortex", "paracingulate gyrus",
    ],
    "temporal": [
        "temporal pole", "superior temporal gyrus", "middle temporal gyrus",
        "inferior temporal gyrus", "fusiform gyrus", "temporal fusiform cortex",
        "temporal occipital fusiform cortex", "planum polare", "planum temporale",
        "heschl's gyrus",
    ],
    "parietal": [
        "postcentral gyrus", "superior parietal lobule", "supramarginal gyrus",
        "angular gyrus", "precuneous cortex", "parietal operculum cortex",
    ],
    "occipital": [
        "lateral occipital cortex", "occipital pole", "cuneal cortex",
        "lingual gyrus", "occipital fusiform gyrus", "intracalcarine cortex",
        "supracalcarine cortex",
    ],
    "deep_structures": [
        "thalamus", "caudate", "putamen", "pallidum", "hippocampus",
        "amygdala", "accumbens", "brain-stem", "insular cortex",
    ],
}


def classify_region_to_lobe(region_name: str) -> str:
    """Map atlas region name to brain lobe."""
    region_lower = region_name.lower()
    for lobe, keywords in LOBE_MAPPING.items():
        for kw in keywords:
            if kw in region_lower:
                return lobe
    return "other"


def atlas_based_location(mask_arr: np.ndarray) -> list:
    """
    Map tumor mask to brain lobes using Harvard-Oxford atlas.
    Requires nilearn and nibabel.
    """
    try:
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
        atlas_maps = atlas.maps
        # Newer nilearn returns Nifti1Image directly; older returns a path string
        if isinstance(atlas_maps, str):
            atlas_img = nib.load(atlas_maps)
        else:
            atlas_img = atlas_maps
        atlas_data = atlas_img.get_fdata().astype(np.int32)
        atlas_labels = atlas.labels

        # Resize mask to atlas space if needed
        if mask_arr.shape != atlas_data.shape:
            # Simple nearest-neighbor resize
            from scipy.ndimage import zoom
            zoom_factors = [
                a / m for a, m in zip(atlas_data.shape, mask_arr.shape)
            ]
            mask_resized = zoom(mask_arr.astype(np.float32), zoom_factors, order=0)
            mask_resized = (mask_resized > 0.5).astype(np.int32)
        else:
            mask_resized = mask_arr

        # Find overlapping atlas regions
        tumor_regions = atlas_data[mask_resized > 0]
        unique_regions, counts = np.unique(tumor_regions, return_counts=True)

        lobe_overlap = {}
        for region_id, count in zip(unique_regions, counts):
            if region_id == 0:
                continue  # Skip background
            if region_id < len(atlas_labels):
                region_name = atlas_labels[region_id]
                lobe = classify_region_to_lobe(region_name)
                lobe_overlap[lobe] = lobe_overlap.get(lobe, 0) + int(count)

        # Sort by overlap count (descending)
        sorted_lobes = sorted(lobe_overlap.items(), key=lambda x: x[1], reverse=True)
        locations = [lobe for lobe, _ in sorted_lobes]

        return locations if locations else ["unknown"]

    except Exception as e:
        print(f"  [WARNING] Atlas-based mapping failed: {e}")
        return ["unknown"]


def heuristic_location(mask_arr: np.ndarray) -> list:
    """
    Fallback: rough anatomical mapping based on tumor centroid position
    relative to brain volume proportions.
    """
    coords = np.argwhere(mask_arr > 0)
    if len(coords) == 0:
        return ["unknown"]

    centroid = coords.mean(axis=0)
    shape = np.array(mask_arr.shape, dtype=float)

    # Normalize centroid to [0, 1] range
    norm_centroid = centroid / shape  # (z, y, x) normalized

    z_pos = norm_centroid[0]  # axial (inferior-superior)
    y_pos = norm_centroid[1]  # coronal (anterior-posterior)
    x_pos = norm_centroid[2]  # sagittal (left-right)

    locations = []

    # Anterior (frontal) vs Posterior (occipital)
    if y_pos < 0.4:
        locations.append("frontal")
    elif y_pos > 0.7:
        locations.append("occipital")

    # Superior (parietal) vs Inferior
    if z_pos > 0.6:
        locations.append("parietal")

    # Lateral (temporal)
    if x_pos < 0.3 or x_pos > 0.7:
        locations.append("temporal")

    # Deep structures
    if 0.35 < x_pos < 0.65 and 0.35 < y_pos < 0.65 and 0.3 < z_pos < 0.6:
        locations.append("deep_structures")

    return locations if locations else ["unknown"]


def map_tumor_location(state: dict) -> dict:
    """
    LangGraph node: Map tumor segmentation to brain anatomy.
    Tries atlas-based mapping first, falls back to heuristic.
    """
    patient_id = state["patient_id"]
    segmentation_path = state.get("segmentation_path")
    errors = list(state.get("errors", []))

    print(f"[Location Mapping] Processing patient: {patient_id}")

    if segmentation_path is None:
        msg = f"No segmentation mask for location mapping: {patient_id}"
        print(f"  [WARNING] {msg}")
        errors.append(msg)
        return {**state, "tumor_location": ["unknown"], "errors": errors}

    try:
        mask_sitk = sitk.ReadImage(segmentation_path, sitk.sitkInt32)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        if NILEARN_AVAILABLE:
            print("  Using atlas-based location mapping (Harvard-Oxford).")
            locations = atlas_based_location(mask_arr)
        else:
            print("  Nilearn not available. Using heuristic location mapping.")
            locations = heuristic_location(mask_arr)

        print(f"  Tumor locations: {locations}")
        return {**state, "tumor_location": locations, "errors": errors}

    except Exception as e:
        msg = f"Error in location mapping for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)
        return {**state, "tumor_location": ["unknown"], "errors": errors}
