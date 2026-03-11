"""
Medical Imaging Pipeline — Entry Point
=========================================
Phase 1: MRI preprocessing → segmentation → radiomics → clinical profiling
Phase 2: Embeddings → similarity → WHO/RANO analysis → AI reasoning → reports
Phase 3: Memory → HITL validation → CAP report → visualization → evaluation

Dataset formats:
  nifti: Standard BraTS NIfTI format (patient folders)
  h5:    HDF5 slice format (volume_X_slice_Y.h5)
"""
import os
import sys
import json
import argparse


def _load_phase1_state(patient_id, base_dir, output_dir):
    """Load existing Phase 1 outputs into a state dict."""
    clinical_path = os.path.join(output_dir, "clinical_features",
                                 f"{patient_id}_clinical.json")
    radiomics_path = os.path.join(output_dir, "radiomics",
                                  f"{patient_id}_radiomics.json")
    seg_path = os.path.join(output_dir, "segmentation",
                            f"{patient_id}_seg.nii.gz")
    prep_path = os.path.join(output_dir, "preprocessed",
                             f"{patient_id}_preprocessed.npy")

    clinical, radiomics = {}, {}
    if os.path.exists(clinical_path):
        with open(clinical_path) as f:
            clinical = json.load(f)
    if os.path.exists(radiomics_path):
        with open(radiomics_path) as f:
            radiomics = json.load(f)

    return {
        "patient_id": patient_id,
        "base_dir": base_dir,
        "output_dir": output_dir,
        "preprocessed_path": prep_path if os.path.exists(prep_path) else None,
        "segmentation_path": seg_path if os.path.exists(seg_path) else None,
        "radiomics_features": radiomics,
        "tumor_location": clinical.get("tumor_location", []),
        "clinical_profile": clinical,
        "errors": [],
    }


def _phase1(patient_id, base_dir, output_dir):
    """Run Phase 1 for a single patient."""
    from pipeline.graph import build_pipeline
    print(f"\n{'='*65}")
    print(f"[Phase 1] {patient_id}")
    print(f"{'='*65}")
    state = {
        "patient_id": patient_id, "base_dir": base_dir,
        "output_dir": output_dir,
        "preprocessed_path": None, "segmentation_path": None,
        "radiomics_features": {}, "tumor_location": [],
        "clinical_profile": {}, "errors": [],
    }
    app = build_pipeline()
    return app.invoke(state)


def _phase2(state, skip_hitl=False):
    """Run Phase 2 for a single patient."""
    from agents.orchestrator import build_phase2_pipeline
    print(f"\n[Phase 2] {state['patient_id']}")
    state = {
        **state,
        "embedding": None, "embedding_path": None,
        "similar_cases": [], "tumor_analysis": {},
        "clinical_reasoning": None, "report_path": None,
        "patient_history": [], "physician_corrections": {},
        "cap_report": {}, "cap_report_path": None,
        "visualization_paths": {}, "skip_hitl": skip_hitl,
    }
    app = build_phase2_pipeline()
    return app.invoke(state)


def _phase3(state, skip_hitl=False):
    """Run Phase 3 for a single patient."""
    from agents.orchestrator import build_phase3_pipeline
    print(f"\n[Phase 3] {state['patient_id']}")
    if "skip_hitl" not in state:
        state["skip_hitl"] = skip_hitl
    # Ensure Phase 3 keys exist
    for key, default in [("patient_history", []),
                         ("physician_corrections", {}),
                         ("cap_report", {}), ("cap_report_path", None),
                         ("visualization_paths", {}),
                         ("embedding", None), ("similar_cases", []),
                         ("tumor_analysis", {}), ("clinical_reasoning", None),
                         ("report_path", None), ("embedding_path", None)]:
        if key not in state:
            state[key] = default
    app = build_phase3_pipeline()
    return app.invoke(state)


def _phase23(state, skip_hitl=False):
    """Run Phase 2 + 3 combined."""
    from agents.orchestrator import build_full_pipeline
    print(f"\n[Phase 2+3] {state['patient_id']}")
    state = {
        **state,
        "embedding": None, "embedding_path": None,
        "similar_cases": [], "tumor_analysis": {},
        "clinical_reasoning": None, "report_path": None,
        "patient_history": [], "physician_corrections": {},
        "cap_report": {}, "cap_report_path": None,
        "visualization_paths": {}, "skip_hitl": skip_hitl,
    }
    app = build_full_pipeline()
    return app.invoke(state)


def process_patient(patient_id, base_dir, output_dir, phase, skip_hitl):
    """Full pipeline for a single patient."""
    if phase == "1":
        return _phase1(patient_id, base_dir, output_dir)
    elif phase == "2":
        state = _load_phase1_state(patient_id, base_dir, output_dir)
        return _phase2(state, skip_hitl)
    elif phase == "3":
        state = _load_phase1_state(patient_id, base_dir, output_dir)
        # Try to load Phase 2 report
        report_path = os.path.join(output_dir, "reports",
                                   f"{patient_id}_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            state["tumor_analysis"] = {
                "who_classification": report.get("who_classification", {}),
                "rano_assessment": report.get("rano_assessment", {}),
                "progression": report.get("tumor_progression", {}),
            }
            state["clinical_reasoning"] = report.get("ai_clinical_reasoning", "")
            state["similar_cases"] = report.get("similar_cases", [])
        return _phase3(state, skip_hitl)
    elif phase in ("23", "both"):
        state = _load_phase1_state(patient_id, base_dir, output_dir)
        return _phase23(state, skip_hitl)
    elif phase == "all":
        state = _phase1(patient_id, base_dir, output_dir)
        return _phase23(state, skip_hitl)
    else:
        print(f"[ERROR] Unknown phase: {phase}")
        return {}


def process_nifti(data_dir, output_dir, phase, skip_hitl, patient_id=None):
    if patient_id:
        process_patient(patient_id, os.path.join(data_dir, patient_id),
                        output_dir, phase, skip_hitl)
    else:
        for folder in sorted(os.listdir(data_dir)):
            full_path = os.path.join(data_dir, folder)
            if os.path.isdir(full_path):
                process_patient(folder, full_path, output_dir, phase, skip_hitl)


def process_h5(data_dir, output_dir, phase, skip_hitl, max_patients=None):
    from utils.h5_adapter import discover_volumes, reconstruct_volume, save_reconstructed_volume
    volumes = discover_volumes(data_dir)
    vol_ids = sorted(volumes.keys())[:max_patients] if max_patients else sorted(volumes.keys())
    for vol_id in vol_ids:
        pid = f"volume_{vol_id}"
        try:
            img, msk, _ = reconstruct_volume(volumes[vol_id])
            patient_dir, _ = save_reconstructed_volume(img, msk, pid, output_dir)
            process_patient(pid, patient_dir, output_dir, phase, skip_hitl)
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agentic Clinical Brain Tumor Intelligence Platform"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Dataset path (NIfTI folders or H5 slices)")
    parser.add_argument("--output_dir", default="./outputs",
                        help="Output directory")
    parser.add_argument("--patient_id", default=None,
                        help="Single patient ID (nifti mode)")
    parser.add_argument("--format", choices=["nifti", "h5"], default="h5",
                        help="Dataset format")
    parser.add_argument("--phase",
                        choices=["1", "2", "3", "23", "all"],
                        default="all",
                        help="Pipeline phase: 1=imaging, 2=agents, 3=clinical, "
                             "23=agents+clinical, all=end-to-end")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Max patients (H5 mode)")
    parser.add_argument("--skip_hitl", action="store_true",
                        help="Auto-approve HITL validation (batch mode)")
    parser.add_argument("--ollama_url", default="http://localhost:11434",
                        help="Ollama API URL for Llama 3")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation framework after pipeline")
    args = parser.parse_args()

    os.environ["OLLAMA_URL"] = args.ollama_url
    os.makedirs(args.output_dir, exist_ok=True)

    if args.format == "h5":
        process_h5(args.data_dir, args.output_dir, args.phase,
                   args.skip_hitl, args.max_patients)
    else:
        process_nifti(args.data_dir, args.output_dir, args.phase,
                      args.skip_hitl, args.patient_id)

    # Optional evaluation pass
    if args.evaluate:
        from evaluation.runner import run_evaluation
        run_evaluation(args.output_dir, args.patient_id)
