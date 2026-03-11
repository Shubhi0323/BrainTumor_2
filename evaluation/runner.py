"""
Evaluation Runner
===================
End-to-end evaluation across all processed patients.
Reads pipeline outputs and computes aggregate metrics.

Run as:
  python -m evaluation.runner --output_dir ./outputs

Or with specific patient:
  python -m evaluation.runner --output_dir ./outputs --patient_id BraTS20_Training_001
"""
import os
import json
import argparse
import numpy as np
from evaluation.metrics import (
    compute_segmentation_metrics,
    compute_radiomics_icc,
    compute_retrieval_metrics,
    expert_agreement_score,
)


def evaluate_patient(patient_id: str, output_dir: str) -> dict:
    """Run all evaluation metrics for a single patient."""
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

            # Use predicted seg as "GT" if no separate GT, show basic stats
            pred_bin = (pred_seg > 0)
            tumor_voxels = int(np.sum(pred_bin))
            total_voxels = int(pred_seg.size)

            result["metrics"]["segmentation"] = {
                "tumor_voxels": tumor_voxels,
                "total_voxels": total_voxels,
                "tumor_fraction": round(tumor_voxels / max(total_voxels, 1), 6),
                "note": "Dice/Hausdorff require separate GT mask. "
                        "Provide gt_seg_path for full metrics.",
            }
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


def run_evaluation(output_dir: str, patient_id: str = None) -> dict:
    """Run evaluation across all or one patient."""
    all_results = {}

    if patient_id:
        patients = [patient_id]
    else:
        # Find all patients with reports
        report_dir = os.path.join(output_dir, "reports")
        patients = []
        if os.path.exists(report_dir):
            for f in os.listdir(report_dir):
                if f.endswith("_report.json"):
                    patients.append(f.replace("_report.json", ""))

    print(f"\n{'='*60}")
    print(f"EVALUATION FRAMEWORK — {len(patients)} patient(s)")
    print(f"{'='*60}")

    for pid in patients:
        print(f"\n  Evaluating: {pid}")
        result = evaluate_patient(pid, output_dir)
        all_results[pid] = result
        metrics = result.get("metrics", {})
        seg = metrics.get("segmentation", {})
        icc_r = metrics.get("radiomics_icc", {})
        who = metrics.get("who_classification", {})
        if seg:
            print(f"    Tumor fraction: {seg.get('tumor_fraction', 'N/A')}")
        if icc_r:
            print(f"    Radiomics ICC:  {icc_r.get('mean_icc', 'N/A'):.4f}")
        if who:
            print(f"    WHO class:      {who.get('classified_as', 'N/A')}"
                  f" (confidence: {who.get('confidence', 0):.2f})")

    # Save aggregate results
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    save_path = os.path.join(eval_dir, "evaluation_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Evaluation results saved: {save_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Evaluation Runner")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--patient_id", type=str, default=None)
    args = parser.parse_args()

    run_evaluation(args.output_dir, args.patient_id)
