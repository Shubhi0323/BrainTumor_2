"""
Tumor Analysis Agent
======================
Analyzes clinical profile using WHO Classification and RANO Criteria
MCP tools. Produces a structured tumor analysis.
"""
import os
import json

from mcp_servers.who_classification import classify_tumor
from mcp_servers.rano_criteria import evaluate_response
from progression.estimator import estimate_progression, find_previous_volume


def run_tumor_analysis(state: dict) -> dict:
    """
    LangGraph node: Tumor Analysis Agent.
    Calls WHO classification and RANO evaluation tools.
    """
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    clinical_profile = state.get("clinical_profile", {})
    radiomics_features = state.get("radiomics_features", {})
    errors = list(state.get("errors", []))

    print(f"[Tumor Analysis Agent] Processing patient: {patient_id}")

    analysis = {}

    # ── 1. WHO Classification ──
    try:
        morphology = clinical_profile.get("morphology", {})
        rad_summary = clinical_profile.get("radiomics_summary", {})
        intensity = rad_summary.get("key_intensity", {})

        radiomics_patterns = {
            "intensity_std": float(intensity.get("intensity_std", 0)),
            "intensity_skewness": float(intensity.get("intensity_skewness", 0)),
        }

        who_result = classify_tumor(morphology, radiomics_patterns)
        analysis["who_classification"] = who_result
        print(f"  WHO Classification: {who_result.get('classified_as', 'unknown')} "
              f"(Grade {who_result.get('who_grade', '?')}, "
              f"confidence: {who_result.get('confidence', 0):.2f})")
    except Exception as e:
        msg = f"WHO classification failed for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)
        analysis["who_classification"] = {"error": str(e)}

    # ── 2. Tumor Progression ──
    try:
        current_volume = morphology.get("tumor_volume", 0)
        prev_volume, time_interval = find_previous_volume(patient_id, output_dir)
        progression = estimate_progression(current_volume, prev_volume, time_interval)
        analysis["progression"] = progression
        print(f"  Progression: {progression.get('progression_state', 'unknown')}")
    except Exception as e:
        msg = f"Progression estimation failed for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)
        analysis["progression"] = {"error": str(e)}

    # ── 3. RANO Assessment ──
    try:
        size_change_pct = progression.get("size_change_pct", 0) or 0
        rano_result = evaluate_response(
            tumor_size_change_pct=size_change_pct,
            contrast_enhancement="stable",  # Default — adjustable with actual imaging data
            new_lesions=False,
            clinical_condition="stable",
        )
        analysis["rano_assessment"] = rano_result
        print(f"  RANO Assessment: {rano_result.get('assessment', '?')} "
              f"({rano_result.get('assessment_name', '')})")
    except Exception as e:
        msg = f"RANO assessment failed for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)
        analysis["rano_assessment"] = {"error": str(e)}

    return {**state, "tumor_analysis": analysis, "errors": errors}
