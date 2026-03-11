"""
Report Agent
==============
Formats and saves the final structured JSON report for each patient.
Consolidates all Phase 2 analysis into a single output file.
"""
import os
import json
from datetime import datetime


def run_report_generation(state: dict) -> dict:
    """
    LangGraph node: Report Agent.
    Consolidates all analysis into a final patient report.
    """
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    errors = list(state.get("errors", []))

    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    print(f"[Report Agent] Generating report for patient: {patient_id}")

    # Extract all analysis components
    clinical = state.get("clinical_profile", {})
    analysis = state.get("tumor_analysis", {})
    similar = state.get("similar_cases", [])
    reasoning = state.get("clinical_reasoning", "")

    who = analysis.get("who_classification", {})
    rano = analysis.get("rano_assessment", {})
    progression = analysis.get("progression", {})

    # Build structured report
    report = {
        "report_metadata": {
            "patient_id": patient_id,
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "2.0",
            "phase": "Phase 2 — Agentic Clinical Intelligence",
            "has_errors": len(errors) > 0,
        },
        "tumor_summary": {
            "location": clinical.get("tumor_location", []),
            "primary_location": clinical.get("primary_location", "unknown"),
            "volume_severity": clinical.get("volume_severity", "unknown"),
            "morphology": clinical.get("morphology", {}),
            "inferred_symptoms": clinical.get("inferred_symptoms", []),
        },
        "who_classification": {
            "classified_as": who.get("classified_as", "unknown"),
            "who_grade": who.get("who_grade", "unknown"),
            "full_name": who.get("full_name", ""),
            "confidence": who.get("confidence", 0),
            "description": who.get("description", ""),
            "reasoning": who.get("reasoning", []),
            "prognosis": who.get("prognosis", ""),
            "standard_treatment": who.get("standard_treatment", ""),
            "differential_diagnosis": who.get("differential", []),
        },
        "rano_assessment": {
            "assessment": rano.get("assessment", "N/A"),
            "assessment_name": rano.get("assessment_name", "N/A"),
            "reasoning": rano.get("reasoning", []),
        },
        "tumor_progression": {
            "state": progression.get("progression_state", "unknown"),
            "growth_rate": progression.get("growth_rate"),
            "size_change_pct": progression.get("size_change_pct"),
            "reasoning": progression.get("reasoning", ""),
        },
        "similar_cases": [
            {
                "patient_id": c.get("patient_id", ""),
                "tumor_location": c.get("tumor_location", []),
                "volume_severity": c.get("volume_severity", ""),
                "similarity_score": c.get("similarity", c.get("distance")),
            }
            for c in similar[:5]
        ],
        "ai_clinical_reasoning": reasoning,
        "pipeline_errors": errors,
    }

    # Save report
    save_path = os.path.join(report_dir, f"{patient_id}_report.json")
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Report saved: {save_path}")
    print(f"  WHO: {report['who_classification']['classified_as']} "
          f"(Grade {report['who_classification']['who_grade']})")
    print(f"  RANO: {report['rano_assessment']['assessment']}")
    print(f"  Progression: {report['tumor_progression']['state']}")
    print(f"  Similar cases: {len(report['similar_cases'])}")

    return {**state, "report_path": save_path, "errors": errors}
