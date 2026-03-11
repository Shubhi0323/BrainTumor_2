"""
Evaluation Metrics
====================
Research validation metrics for the full pipeline.

Segmentation:
  - Dice coefficient
  - Hausdorff distance (95th percentile)
  - Sensitivity (recall)
  - Specificity

Radiomics:
  - Intraclass Correlation Coefficient (ICC)

Similarity Retrieval:
  - Precision@K
  - Recall@K

Clinical Reasoning:
  - Expert agreement score
"""
import numpy as np


# ─── Segmentation Metrics ─────────────────────────────────────────────

def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Dice coefficient between predicted and ground truth binary masks.
    Dice = 2|A∩B| / (|A| + |B|)
    """
    pred_bin = (pred > 0).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)
    intersection = np.sum(pred_bin * gt_bin)
    denom = np.sum(pred_bin) + np.sum(gt_bin)
    if denom == 0:
        return 1.0 if np.sum(gt_bin) == 0 else 0.0
    return float(2.0 * intersection / denom)


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute 95th-percentile Hausdorff distance between two binary masks.
    Returns distance in voxels (multiply by voxel spacing for mm).
    """
    from scipy.ndimage import distance_transform_edt

    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)

    if not np.any(pred_bin) or not np.any(gt_bin):
        return float("nan")

    # Distance from pred surface to gt
    gt_dist = distance_transform_edt(~gt_bin)
    pred_surface = pred_bin & ~np.array(
        __import__("scipy").ndimage.binary_erosion(pred_bin))
    distances_pred_to_gt = gt_dist[pred_surface]

    # Distance from gt surface to pred
    pred_dist = distance_transform_edt(~pred_bin)
    gt_surface = gt_bin & ~np.array(
        __import__("scipy").ndimage.binary_erosion(gt_bin))
    distances_gt_to_pred = pred_dist[gt_surface]

    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sensitivity (True Positive Rate / Recall): TP / (TP + FN)"""
    pred_bin = (pred > 0).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)
    tp = np.sum(pred_bin * gt_bin)
    fn = np.sum((1 - pred_bin) * gt_bin)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Specificity (True Negative Rate): TN / (TN + FP)"""
    pred_bin = (pred > 0).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)
    tn = np.sum((1 - pred_bin) * (1 - gt_bin))
    fp = np.sum(pred_bin * (1 - gt_bin))
    denom = tn + fp
    return float(tn / denom) if denom > 0 else float("nan")


def compute_segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute all segmentation evaluation metrics."""
    return {
        "dice_coefficient": round(dice_coefficient(pred, gt), 4),
        "hausdorff_distance_95": round(hausdorff_distance_95(pred, gt), 4),
        "sensitivity": round(sensitivity(pred, gt), 4),
        "specificity": round(specificity(pred, gt), 4),
    }


# ─── Radiomics Metrics ────────────────────────────────────────────────

def icc(measurements_1: list, measurements_2: list,
        icc_type: str = "ICC(2,1)") -> float:
    """
    Intraclass Correlation Coefficient between two sets of measurements.
    Measures radiomics reproducibility between two extraction runs.

    icc_type: 'ICC(1,1)', 'ICC(2,1)', or 'ICC(3,1)'
    Returns ICC value in [0, 1] (higher = better reproducibility).
    """
    n = len(measurements_1)
    if n < 2:
        return float("nan")

    m1 = np.array(measurements_1, dtype=np.float64)
    m2 = np.array(measurements_2, dtype=np.float64)

    subjects = np.column_stack([m1, m2])   # shape (n, k=2)
    k = 2

    grand_mean = subjects.mean()
    subject_means = subjects.mean(axis=1)
    rater_means = subjects.mean(axis=0)

    # Sum of Squares
    SS_total = np.sum((subjects - grand_mean) ** 2)
    SS_between = k * np.sum((subject_means - grand_mean) ** 2)
    SS_within = np.sum((subjects - subject_means[:, np.newaxis]) ** 2)
    SS_error = SS_within - np.sum((rater_means - grand_mean) ** 2) * n

    # Mean Squares
    df_between = n - 1
    df_error = (n - 1) * (k - 1)

    MS_between = SS_between / df_between if df_between > 0 else 0
    MS_error = SS_error / df_error if df_error > 0 else 0

    if icc_type == "ICC(2,1)":
        icc_val = (MS_between - MS_error) / (
            MS_between + (k - 1) * MS_error + k * max(0, (MS_between - MS_error) / n)
        )
    elif icc_type == "ICC(1,1)":
        MS_within = SS_within / (n * (k - 1))
        icc_val = (MS_between - MS_within) / (MS_between + (k - 1) * MS_within)
    else:  # ICC(3,1)
        icc_val = (MS_between - MS_error) / (MS_between + (k - 1) * MS_error)

    return float(np.clip(icc_val, -1.0, 1.0))


def compute_radiomics_icc(features_run1: dict, features_run2: dict) -> dict:
    """Compute ICC for each matching radiomics feature between two runs."""
    common_keys = set(features_run1.keys()) & set(features_run2.keys())
    numeric_keys = [
        k for k in common_keys
        if isinstance(features_run1[k], (int, float))
        and isinstance(features_run2[k], (int, float))
    ]

    if not numeric_keys:
        return {"mean_icc": float("nan"), "feature_count": 0}

    icc_values = []
    for key in numeric_keys:
        # Single pair: use simplified ratio
        v1, v2 = float(features_run1[key]), float(features_run2[key])
        mean_v = (v1 + v2) / 2
        diff = abs(v1 - v2)
        # Simplified single-pair ICC proxy
        if mean_v != 0:
            icc_proxy = max(0.0, 1.0 - diff / (abs(mean_v) * 2 + 1e-8))
        else:
            icc_proxy = 1.0 if diff < 1e-8 else 0.0
        icc_values.append(icc_proxy)

    return {
        "mean_icc": round(float(np.mean(icc_values)), 4),
        "min_icc": round(float(np.min(icc_values)), 4),
        "max_icc": round(float(np.max(icc_values)), 4),
        "feature_count": len(icc_values),
    }


# ─── Similarity Retrieval Metrics ────────────────────────────────────

def precision_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    """
    Precision@K: fraction of top-K retrieved items that are relevant.
    retrieved: list of patient IDs returned by similarity engine
    relevant: list of truly similar patient IDs (ground truth)
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant)
    return float(hits / k)


def recall_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    """
    Recall@K: fraction of relevant items found in top-K results.
    """
    if not relevant:
        return float("nan")
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return float(hits / len(relevant))


def compute_retrieval_metrics(retrieved_ids: list, relevant_ids: list,
                              k: int = 5) -> dict:
    """Compute Precision@K and Recall@K for similarity retrieval."""
    return {
        "precision_at_k": round(precision_at_k(retrieved_ids, relevant_ids, k), 4),
        "recall_at_k": round(recall_at_k(retrieved_ids, relevant_ids, k), 4),
        "k": k,
        "retrieved": len(retrieved_ids),
        "relevant": len(relevant_ids),
    }


# ─── Clinical Reasoning Metrics ──────────────────────────────────────

def expert_agreement_score(ai_classification: str, expert_classification: str) -> float:
    """
    Simple expert agreement score for WHO classification.
    1.0 = exact match, 0.5 = same tumor family, 0.0 = mismatch.
    """
    if ai_classification == expert_classification:
        return 1.0

    # Same tumor family (partial credit)
    glioma_family = {"glioblastoma", "astrocytoma", "oligodendroglioma", "glioma_nos"}
    if ai_classification in glioma_family and expert_classification in glioma_family:
        return 0.5

    return 0.0
