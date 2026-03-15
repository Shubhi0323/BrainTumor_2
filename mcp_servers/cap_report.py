"""
CAP Structured Reporting MCP Server
======================================
Implements CAP (College of American Pathologists) structured report
template for brain tumor MRI studies.

Auto-generates all 9 CAP report sections from pipeline state.

Can run as:
  - Standalone MCP server (stdio mode)
  - Direct function call (in-process)
"""
import json
from datetime import datetime


def generate_cap_report(state: dict) -> dict:
    """
    Generate a structured CAP report from the full pipeline state.

    Returns a dict with all 9 CAP sections.
    """
    patient_id = state.get("patient_id", "unknown")
    clinical = state.get("clinical_profile", {})
    analysis = state.get("tumor_analysis", {})
    radiomics = state.get("radiomics_features", {})
    similar = state.get("similar_cases", [])
    reasoning = state.get("clinical_reasoning", "")
    corrections = state.get("physician_corrections", {})
    history = state.get("patient_history", [])

    who = analysis.get("who_classification", {})
    rano = analysis.get("rano_assessment", {})
    progression = analysis.get("progression", {})
    morph = clinical.get("morphology", {})

    # ── Section 1: Patient Information ──
    scan_history_count = len(history)
    section_patient = {
        "patient_id": patient_id,
        "prior_scans": scan_history_count,
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "report_time": datetime.now().strftime("%H:%M:%S"),
        "institution": "Agentic Clinical Brain Tumor Intelligence Platform",
        "pipeline_version": "3.0",
    }

    # ── Section 2: MRI Study Information ──
    section_study = {
        "modalities": ["T1", "T1CE", "T2", "FLAIR"],
        "segmentation_method": "DynUNet (MONAI) / Ground-truth fallback",
        "preprocessing": [
            "N4 Bias Field Correction",
            "Skull Stripping",
            "Z-score Intensity Normalization",
            "1mm Isotropic Resampling",
        ],
    }

    # ── Section 3: Tumor Characteristics ──
    section_tumor = {
        "location": clinical.get("tumor_location", []),
        "primary_location": clinical.get("primary_location", "unknown"),
        "volume_mm3": morph.get("tumor_volume", 0),
        "volume_severity": clinical.get("volume_severity", "unknown"),
        "max_diameter_mm": morph.get("max_diameter", 0),
        "sphericity": morph.get("sphericity", 0),
        "surface_area_mm2": morph.get("surface_area", 0),
    }

    # ── Section 4: Radiomics Summary ──
    rad_summary = clinical.get("radiomics_summary", {})
    section_radiomics = {
        "total_features_extracted": rad_summary.get("num_features", len(radiomics)),
        "shape_features": rad_summary.get("key_shape", {}),
        "intensity_features": rad_summary.get("key_intensity", {}),
        "texture_features_count": len(rad_summary.get("key_texture", {})),
    }

    # ── Section 5: RANO Classification ──
    section_rano = {
        "assessment": rano.get("assessment", "N/A"),
        "assessment_name": rano.get("assessment_name", "Not assessed"),
        "reasoning": rano.get("reasoning", []),
        "physician_override": rano.get("physician_override", False),
        "progression_state": progression.get("progression_state", "unknown"),
        "growth_rate_mm3_per_day": progression.get("growth_rate"),
        "size_change_pct": progression.get("size_change_pct"),
    }

    # ── Section 6: WHO Classification ──
    section_who = {
        "classified_as": who.get("classified_as", "unknown"),
        "who_grade": who.get("who_grade", "unknown"),
        "full_name": who.get("full_name", ""),
        "confidence": who.get("confidence", 0),
        "description": who.get("description", ""),
        "prognosis": who.get("prognosis", ""),
        "standard_treatment": who.get("standard_treatment", ""),
        "differential_diagnosis": who.get("differential", []),
        "physician_override": who.get("physician_override", False),
    }

    # ── Section 7: Similar Tumor Cases ──
    section_similar = {
        "retrieval_method": (
            "Weaviate vector similarity" if similar else "NumPy cosine fallback"
        ),
        "cases_retrieved": len(similar),
        "top_cases": [
            {
                "patient_id": c.get("patient_id", ""),
                "location": c.get("tumor_location", []),
                "severity": c.get("volume_severity", ""),
                "similarity_score": c.get("similarity", c.get("distance")),
            }
            for c in similar[:5]
        ],
    }

    # ── Section 8: Clinical Interpretation ──
    section_interpretation = {
        "ai_clinical_reasoning": reasoning,
        "inferred_symptoms": clinical.get("inferred_symptoms", []),
        "reasoning_engine": (
            "Llama 3 (Ollama)" if "DIAGNOSIS ASSESSMENT" not in reasoning else "Rule-based fallback"
        ),
    }

    # ── Section 9: Physician Notes ──
    section_physician = {
        "review_status": corrections.get("action", "not_reviewed"),
        "physician": corrections.get("physician", "N/A"),
        "notes": corrections.get("physician_notes", ""),
        "segmentation_edited": corrections.get("segmentation_edited", False),
        "review_timestamp": corrections.get("timestamp", ""),
    }

    # ── Assemble Full Report ──
    cap_report = {
        "cap_report_type": "CAP Structured MRI Brain Tumor Report",
        "section_1_patient_information": section_patient,
        "section_2_mri_study_information": section_study,
        "section_3_tumor_characteristics": section_tumor,
        "section_4_radiomics_summary": section_radiomics,
        "section_5_rano_classification": section_rano,
        "section_6_who_classification": section_who,
        "section_7_similar_tumor_cases": section_similar,
        "section_8_clinical_interpretation": section_interpretation,
        "section_9_physician_notes": section_physician,
    }

    return cap_report


def run_cap_reporting(state: dict) -> dict:
    """
    LangGraph node: Generate and save the CAP structured report.
    """
    import os
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    errors = list(state.get("errors", []))

    cap_dir = os.path.join(output_dir, "reports", "cap")
    os.makedirs(cap_dir, exist_ok=True)

    print(f"[CAP Report] Generating for patient: {patient_id}")

    try:
        cap_report = generate_cap_report(state)
        save_path = os.path.join(cap_dir, f"{patient_id}_cap_report.json")
        with open(save_path, "w") as f:
            json.dump(cap_report, f, indent=2, default=str)
        print(f"  CAP report saved: {save_path}")
        print(f"  WHO: {cap_report['section_6_who_classification']['classified_as']}")
        print(f"  RANO: {cap_report['section_5_rano_classification']['assessment']}")
        return {**state, "cap_report": cap_report, "cap_report_path": save_path, "errors": errors}
    except Exception as e:
        msg = f"CAP report generation failed for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)
        return {**state, "cap_report": {}, "errors": errors}


# ─── MCP Server mode ──────────────────────────────────────────────────

def run_mcp_server():
    """Run as standalone MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("ERROR: 'mcp' package not installed. pip install mcp[cli]")
        return

    mcp = FastMCP("CAP Brain Tumor Report Generator")

    @mcp.tool()
    def generate_cap_structured_report(state_json: str) -> str:
        """
        Generate a full CAP structured report from a patient pipeline state JSON string.
        Returns the report as a JSON string with all 9 CAP sections.
        """
        state = json.loads(state_json)
        report = generate_cap_report(state)
        return json.dumps(report, indent=2, default=str)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
