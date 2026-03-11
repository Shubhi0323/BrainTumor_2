"""
Human-in-the-Loop (HITL) Validation
=======================================
Physician review workflow for AI-generated tumor analysis.

Workflow:
  1. AI generates tumor segmentation + analysis
  2. Physician reviews summary (CLI or auto-approve)
  3. Physician may override WHO classification, RANO, or add notes
  4. Corrections are stored in ChromaDB patient memory
  5. Validated output is passed to the reporting pipeline

Use --skip_hitl flag to auto-approve (for batch processing).
"""
import os
import json
from datetime import datetime
from memory.patient_memory import store_doctor_feedback


def _print_summary(state: dict):
    """Print a formatted summary of AI outputs for physician review."""
    pid = state.get("patient_id", "unknown")
    clinical = state.get("clinical_profile", {})
    analysis = state.get("tumor_analysis", {})
    reasoning = state.get("clinical_reasoning", "")

    who = analysis.get("who_classification", {})
    rano = analysis.get("rano_assessment", {})
    progression = analysis.get("progression", {})

    print(f"\n{'─'*65}")
    print(f"  PHYSICIAN REVIEW — Patient: {pid}")
    print(f"{'─'*65}")

    # Tumor summary
    print(f"\n  📍 Location    : {', '.join(clinical.get('tumor_location', ['unknown']))}")
    morph = clinical.get("morphology", {})
    print(f"  📏 Volume      : {morph.get('tumor_volume', 0):.0f} mm³"
          f" ({clinical.get('volume_severity', 'unknown')})")
    print(f"  ⬤  Sphericity  : {morph.get('sphericity', 0):.3f}")
    print(f"  📐 Diameter    : {morph.get('max_diameter', 0):.1f} mm")

    # AI Classification
    print(f"\n  🔬 WHO Class   : {who.get('classified_as', 'unknown')}"
          f" Grade {who.get('who_grade', '?')}"
          f" (confidence: {who.get('confidence', 0):.0%})")
    print(f"  📊 RANO        : {rano.get('assessment', 'N/A')}"
          f" — {rano.get('assessment_name', '')}")
    print(f"  📈 Progression : {progression.get('progression_state', 'unknown')}")

    # Symptoms
    symptoms = clinical.get("inferred_symptoms", [])
    print(f"\n  🩺 Symptoms    : {', '.join(symptoms[:4]) if symptoms else 'none'}")

    # Reasoning excerpt
    if reasoning:
        excerpt = reasoning.strip().split("\n")[0][:120]
        print(f"\n  🤖 AI Reasoning: {excerpt}...")

    # Similar cases
    similar = state.get("similar_cases", [])
    if similar:
        print(f"\n  🔍 Top Similar Cases:")
        for c in similar[:3]:
            score = c.get("similarity", c.get("distance", "?"))
            print(f"     • {c.get('patient_id', '?')} "
                  f"({c.get('volume_severity', '?')}, "
                  f"score: {score})")

    print(f"\n{'─'*65}")


def _interactive_review(state: dict) -> dict:
    """
    Interactive CLI physician review session.
    Returns corrections dict.
    """
    analysis = state.get("tumor_analysis", {})
    who = analysis.get("who_classification", {})
    rano = analysis.get("rano_assessment", {})

    corrections = {
        "action": "approved",
        "who_override": None,
        "rano_override": None,
        "segmentation_edited": False,
        "physician_notes": "",
        "physician": "physician",
        "timestamp": datetime.now().isoformat(),
    }

    print("\n  [1] Approve AI analysis (press Enter)")
    print("  [2] Override WHO classification")
    print("  [3] Override RANO assessment")
    print("  [4] Flag segmentation as edited")
    print("  [5] Add physician notes")
    print("  [6] Reject — mark for re-analysis")

    try:
        choice = input("\n  Your choice [1-6, default=1]: ").strip() or "1"

        if choice == "2":
            who_options = ["glioblastoma", "astrocytoma", "oligodendroglioma",
                           "meningioma", "glioma_nos"]
            print(f"  Options: {who_options}")
            override = input("  Enter WHO classification: ").strip()
            if override in who_options:
                corrections["who_override"] = override
                corrections["action"] = "who_corrected"

        elif choice == "3":
            print("  Options: CR, PR, SD, PD")
            override = input("  Enter RANO assessment: ").strip().upper()
            if override in ("CR", "PR", "SD", "PD"):
                corrections["rano_override"] = override
                corrections["action"] = "rano_corrected"

        elif choice == "4":
            corrections["segmentation_edited"] = True
            corrections["action"] = "segmentation_edited"

        elif choice == "5":
            notes = input("  Enter physician notes: ").strip()
            corrections["physician_notes"] = notes
            corrections["action"] = "notes_added"

        elif choice == "6":
            corrections["action"] = "rejected"

        if corrections.get("who_override") is None and corrections.get("rano_override") is None:
            notes = input("  Add notes (optional, press Enter to skip): ").strip()
            if notes:
                corrections["physician_notes"] = notes

    except (EOFError, KeyboardInterrupt):
        print("\n  [HITL] Interactive mode skipped (non-interactive environment).")
        corrections["action"] = "auto_approved"

    return corrections


def run_hitl_validation(state: dict) -> dict:
    """
    LangGraph node: Human-in-the-Loop physician validation.
    Shows AI analysis summary and captures physician input.
    Applies corrections to state if physician overrides.
    """
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    skip_hitl = state.get("skip_hitl", False)
    errors = list(state.get("errors", []))

    print(f"[HITL Validation] Patient: {patient_id}")

    # Print summary for physician
    _print_summary(state)

    if skip_hitl:
        print("  [AUTO-APPROVED] --skip_hitl flag set.")
        corrections = {
            "action": "auto_approved",
            "physician": "system",
            "timestamp": datetime.now().isoformat(),
            "physician_notes": "",
        }
    else:
        corrections = _interactive_review(state)

    print(f"  Action: {corrections.get('action', 'approved')}")

    # Store feedback in memory
    store_doctor_feedback(patient_id, corrections, output_dir)

    # Apply overrides to state
    analysis = dict(state.get("tumor_analysis", {}))

    if corrections.get("who_override"):
        who = dict(analysis.get("who_classification", {}))
        who["classified_as"] = corrections["who_override"]
        who["physician_override"] = True
        analysis["who_classification"] = who
        print(f"  WHO override applied: {corrections['who_override']}")

    if corrections.get("rano_override"):
        rano = dict(analysis.get("rano_assessment", {}))
        rano["assessment"] = corrections["rano_override"]
        rano["physician_override"] = True
        analysis["rano_assessment"] = rano
        print(f"  RANO override applied: {corrections['rano_override']}")

    return {
        **state,
        "tumor_analysis": analysis,
        "physician_corrections": corrections,
        "errors": errors,
    }
