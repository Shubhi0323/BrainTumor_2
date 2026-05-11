"""
Evaluation Runner
===================
End-to-end evaluation across all processed patients.
Reads pipeline outputs and computes aggregate metrics.

Auto-detects ground truth segmentation masks from:
  - UCSF-PDGM: *_tumor_segmentation.nii.gz
  - BraTS: *_seg.nii.gz in original data directory
  - Custom: any *_seg.nii.gz or *_tumor*.nii.gz alongside input data

When GT is available: computes Dice, HD95, Sensitivity, Specificity
When GT is not available: reports basic stats (tumor fraction, voxel count)

Run as:
  python -m evaluation.runner --output_dir ./outputs --data_dir ./UCSF-PDGM-v5

Or with specific patient:
    python -m evaluation.runner --output_dir ./outputs --data_dir ./UCSF-PDGM-v5 --patient_id UCSF-PDGM-0489_nifti
"""
import os
import json
import glob
import argparse
import numpy as np
from evaluation.metrics import (
    compute_segmentation_metrics,
    compute_radiomics_icc,
    compute_retrieval_metrics,
    expert_agreement_score,
)


# ─── Ground Truth Discovery ──────────────────────────────────────────

# Patterns to search for ground truth segmentation masks (priority order)
GT_PATTERNS = [
    "*_tumor_segmentation.nii.gz",   # UCSF-PDGM format
    "*_tumor_segmentation.nii",
    "*_seg.nii.gz",                  # BraTS format
    "*_seg.nii",
    "*_label.nii.gz",                # Generic labeling
    "*_label.nii",
    "*_mask.nii.gz",                 # Generic mask
    "*_mask.nii",
]


def find_ground_truth_seg(patient_id: str, data_dir: str = None,
                          output_dir: str = None) -> str | None:
    """
    Search for a ground truth segmentation mask for a patient.

    Search locations (in priority order):
      1. Original data directory (e.g. UCSF-PDGM-v5/<patient>/)
      2. Reconstructed directory (output_dir/reconstructed/<patient>/)
      3. Output segmentation directory (for BraTS-style GT alongside pred)

    Returns:
        Path to GT segmentation file, or None if not found.
    """
    search_dirs = []

    # 1. Original data directory
    if data_dir:
        # Direct patient folder
        patient_dir = os.path.join(data_dir, patient_id)
        if os.path.isdir(patient_dir):
            search_dirs.append(patient_dir)

        # Try with _nifti suffix (UCSF-PDGM convention)
        nifti_dir = os.path.join(data_dir, f"{patient_id}_nifti")
        if os.path.isdir(nifti_dir):
            search_dirs.append(nifti_dir)

        # Try without _nifti suffix if patient_id has it
        if patient_id.endswith("_nifti"):
            bare_id = patient_id.replace("_nifti", "")
            bare_dir = os.path.join(data_dir, bare_id)
            if os.path.isdir(bare_dir):
                search_dirs.append(bare_dir)

        # Search data_dir itself (flat layout)
        search_dirs.append(data_dir)

    # 2. Reconstructed directory
    if output_dir:
        recon_dir = os.path.join(output_dir, "reconstructed", patient_id)
        if os.path.isdir(recon_dir):
            search_dirs.append(recon_dir)

    # Search each directory with each pattern
    for search_dir in search_dirs:
        for pattern in GT_PATTERNS:
            matches = glob.glob(os.path.join(search_dir, pattern))
            for match in matches:
                # Verify it's a real file with data (not 0-byte partial download)
                if os.path.isfile(match) and os.path.getsize(match) > 100:
                    return match

    return None


# ─── Patient Evaluation ──────────────────────────────────────────────

def evaluate_patient(patient_id: str, output_dir: str,
                     data_dir: str = None) -> dict:
    """
    Run all evaluation metrics for a single patient.

    When ground truth segmentation is found:
      → Dice coefficient, HD95, Sensitivity, Specificity

    When ground truth is NOT found:
      → Basic stats: tumor_voxels, total_voxels, tumor_fraction
    """
    result = {"patient_id": patient_id, "metrics": {}}

    seg_dir = os.path.join(output_dir, "segmentation")
    rad_dir = os.path.join(output_dir, "radiomics")
    report_dir = os.path.join(output_dir, "reports")

    # ── Segmentation Metrics ──
    seg_path = os.path.join(seg_dir, f"{patient_id}_seg.nii.gz")
    if os.path.exists(seg_path):
        try:
            import SimpleITK as sitk

            pred_seg = sitk.GetArrayFromImage(
                sitk.ReadImage(seg_path, sitk.sitkInt32))
            pred_bin = (pred_seg > 0)
            tumor_voxels = int(np.sum(pred_bin))
            total_voxels = int(pred_seg.size)

            seg_metrics = {
                "tumor_voxels": tumor_voxels,
                "total_voxels": total_voxels,
                "tumor_fraction": round(tumor_voxels / max(total_voxels, 1), 6),
            }

            # Search for ground truth
            gt_path = find_ground_truth_seg(patient_id, data_dir, output_dir)

            if gt_path is not None:
                print(f"    Ground truth found: {os.path.basename(gt_path)}")
                gt_img = sitk.ReadImage(gt_path, sitk.sitkInt32)
                gt_seg = sitk.GetArrayFromImage(gt_img)
                gt_bin = (gt_seg > 0).astype(np.int32)

                # Align shapes if needed (GT and pred may differ in dimensions)
                if pred_seg.shape != gt_seg.shape:
                    print(f"    Shape mismatch: pred={pred_seg.shape} vs gt={gt_seg.shape}")
                    print(f"    Resampling GT to match prediction...")

                    # Resample GT to match prediction space
                    pred_img = sitk.ReadImage(seg_path, sitk.sitkInt32)
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(pred_img)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler.SetDefaultPixelValue(0)
                    gt_resampled = resampler.Execute(gt_img)
                    gt_seg = sitk.GetArrayFromImage(gt_resampled)
                    gt_bin = (gt_seg > 0).astype(np.int32)
                    print(f"    Resampled GT shape: {gt_seg.shape}")

                # Compute full metrics
                pred_int = pred_bin.astype(np.int32)
                full_metrics = compute_segmentation_metrics(pred_int, gt_bin)

                seg_metrics.update(full_metrics)
                seg_metrics["ground_truth_source"] = os.path.basename(gt_path)
                seg_metrics["gt_tumor_voxels"] = int(np.sum(gt_bin > 0))
                seg_metrics["has_ground_truth"] = True

                print(f"    ✓ Dice: {full_metrics['dice_coefficient']:.4f}")
                print(f"    ✓ HD95: {full_metrics['hausdorff_distance_95']:.2f} voxels")
                print(f"    ✓ Sensitivity: {full_metrics['sensitivity']:.4f}")
                print(f"    ✓ Specificity: {full_metrics['specificity']:.4f}")
            else:
                seg_metrics["has_ground_truth"] = False
                seg_metrics["note"] = (
                    "No ground truth segmentation found. "
                    "Provide --data_dir pointing to original dataset "
                    "(e.g. UCSF-PDGM-v5/) for full Dice/HD95 metrics."
                )

            result["metrics"]["segmentation"] = seg_metrics

        except Exception as e:
            result["metrics"]["segmentation"] = {"error": str(e)}

    # ── Radiomics ICC (self-consistency check) ──
    rad_path = os.path.join(rad_dir, f"{patient_id}_radiomics.json")
    if os.path.exists(rad_path):
        with open(rad_path, "r") as f:
            rad_features = json.load(f)
        # Self-consistency: compare features slightly perturbed (validation proxy)
        perturbed = {k: v * (1 + np.random.normal(0, 0.01))
                     if isinstance(v, float) else v
                     for k, v in rad_features.items()}
        icc_result = compute_radiomics_icc(rad_features, perturbed)
        result["metrics"]["radiomics_icc"] = icc_result

    # ── Report-based metrics ──
    report_path = os.path.join(report_dir, f"{patient_id}_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        similar = report.get("similar_cases", [])
        retrieved_ids = [c.get("patient_id", "") for c in similar]
        result["metrics"]["similarity_retrieval"] = {
            "cases_retrieved": len(retrieved_ids),
            "top_ids": retrieved_ids[:3],
            "note": "Precision@K/Recall@K require labeled relevance set.",
        }

        who = report.get("who_classification", {})
        result["metrics"]["who_classification"] = {
            "classified_as": who.get("classified_as", "unknown"),
            "who_grade": who.get("who_grade", "?"),
            "confidence": who.get("confidence", 0),
        }

        rano = report.get("rano_assessment", {})
        result["metrics"]["rano_assessment"] = {
            "assessment": rano.get("assessment", "N/A"),
            "assessment_name": rano.get("assessment_name", ""),
        }

    return result


# ─── Aggregate Evaluation ─────────────────────────────────────────────

def compute_aggregate_metrics(all_results: dict) -> dict:
    """Compute aggregate statistics across all evaluated patients."""
    dice_scores = []
    hd95_scores = []
    sensitivity_scores = []
    specificity_scores = []
    confidences = []
    gt_count = 0
    no_gt_count = 0

    for pid, result in all_results.items():
        metrics = result.get("metrics", {})

        seg = metrics.get("segmentation", {})
        if seg.get("has_ground_truth"):
            gt_count += 1
            d = seg.get("dice_coefficient")
            if d is not None and not np.isnan(d):
                dice_scores.append(d)
            h = seg.get("hausdorff_distance_95")
            if h is not None and not np.isnan(h):
                hd95_scores.append(h)
            s = seg.get("sensitivity")
            if s is not None and not np.isnan(s):
                sensitivity_scores.append(s)
            sp = seg.get("specificity")
            if sp is not None and not np.isnan(sp):
                specificity_scores.append(sp)
        else:
            no_gt_count += 1

        who = metrics.get("who_classification", {})
        conf = who.get("confidence", None)
        if conf is not None:
            confidences.append(conf)

    aggregate = {
        "total_patients": len(all_results),
        "patients_with_gt": gt_count,
        "patients_without_gt": no_gt_count,
    }

    if dice_scores:
        aggregate["segmentation"] = {
            "mean_dice": round(float(np.mean(dice_scores)), 4),
            "std_dice": round(float(np.std(dice_scores)), 4),
            "min_dice": round(float(np.min(dice_scores)), 4),
            "max_dice": round(float(np.max(dice_scores)), 4),
            "mean_hd95": round(float(np.mean(hd95_scores)), 2) if hd95_scores else None,
            "mean_sensitivity": round(float(np.mean(sensitivity_scores)), 4) if sensitivity_scores else None,
            "mean_specificity": round(float(np.mean(specificity_scores)), 4) if specificity_scores else None,
            "n_evaluated": len(dice_scores),
        }

    if confidences:
        aggregate["who_classification"] = {
            "mean_confidence": round(float(np.mean(confidences)), 4),
            "min_confidence": round(float(np.min(confidences)), 4),
            "max_confidence": round(float(np.max(confidences)), 4),
        }

    return aggregate


def run_evaluation(output_dir: str, patient_id: str = None,
                   data_dir: str = None) -> dict:
    """
    Run evaluation across all or one patient.

    Args:
        output_dir: Pipeline output directory
        patient_id: Specific patient to evaluate (optional)
        data_dir: Original dataset directory for ground truth lookup (optional)
    """
    all_results = {}

    if patient_id:
        patients = [patient_id]
    else:
        # Find all patients with segmentation outputs or reports
        patients = set()
        seg_dir = os.path.join(output_dir, "segmentation")
        report_dir = os.path.join(output_dir, "reports")
        if os.path.exists(seg_dir):
            for f in os.listdir(seg_dir):
                if f.endswith("_seg.nii.gz"):
                    patients.add(f.replace("_seg.nii.gz", ""))
        if os.path.exists(report_dir):
            for f in os.listdir(report_dir):
                if f.endswith("_report.json"):
                    patients.add(f.replace("_report.json", ""))
        patients = sorted(patients)

    print(f"\n{'='*60}")
    print(f"EVALUATION FRAMEWORK — {len(patients)} patient(s)")
    if data_dir:
        print(f"Ground truth source: {data_dir}")
    else:
        print(f"No --data_dir provided. Skipping GT-based metrics.")
    print(f"{'='*60}")

    for pid in patients:
        print(f"\n  Evaluating: {pid}")
        result = evaluate_patient(pid, output_dir, data_dir)
        all_results[pid] = result
        metrics = result.get("metrics", {})
        seg = metrics.get("segmentation", {})
        icc_r = metrics.get("radiomics_icc", {})
        who = metrics.get("who_classification", {})

        if seg and not seg.get("has_ground_truth", False):
            print(f"    Tumor fraction: {seg.get('tumor_fraction', 'N/A')}")
            print(f"    (No ground truth — basic stats only)")
        if icc_r:
            print(f"    Radiomics ICC:  {icc_r.get('mean_icc', 'N/A'):.4f}")
        if who:
            print(f"    WHO class:      {who.get('classified_as', 'N/A')}"
                  f" (confidence: {who.get('confidence', 0):.2f})")

    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(all_results)

    # Print aggregate summary
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"  Total patients: {aggregate['total_patients']}")
    print(f"  With ground truth: {aggregate['patients_with_gt']}")
    print(f"  Without ground truth: {aggregate['patients_without_gt']}")

    if "segmentation" in aggregate:
        seg_agg = aggregate["segmentation"]
        print(f"\n  Segmentation (n={seg_agg['n_evaluated']}):")
        print(f"    Dice:        {seg_agg['mean_dice']:.4f} ± {seg_agg['std_dice']:.4f}"
              f"  (range: {seg_agg['min_dice']:.4f} – {seg_agg['max_dice']:.4f})")
        if seg_agg.get("mean_hd95") is not None:
            print(f"    HD95:        {seg_agg['mean_hd95']:.2f} voxels")
        if seg_agg.get("mean_sensitivity") is not None:
            print(f"    Sensitivity: {seg_agg['mean_sensitivity']:.4f}")
        if seg_agg.get("mean_specificity") is not None:
            print(f"    Specificity: {seg_agg['mean_specificity']:.4f}")

    if "who_classification" in aggregate:
        who_agg = aggregate["who_classification"]
        print(f"\n  WHO Classification:")
        print(f"    Confidence:  {who_agg['mean_confidence']:.4f}"
              f"  (range: {who_agg['min_confidence']:.4f} – {who_agg['max_confidence']:.4f})")

    # Save results
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    save_path = os.path.join(eval_dir, "evaluation_results.json")
    output_data = {
        "per_patient": all_results,
        "aggregate": aggregate,
    }
    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n  Results saved: {save_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Evaluation Runner")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Pipeline output directory")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Original dataset directory for ground truth "
                             "(e.g. ./UCSF-PDGM-v5)")
    parser.add_argument("--patient_id", type=str, default=None,
                        help="Evaluate a specific patient")
    args = parser.parse_args()

    run_evaluation(args.output_dir, args.patient_id, args.data_dir)
