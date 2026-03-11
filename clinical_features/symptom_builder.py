"""
Clinical Feature / Symptom Builder Node
========================================
Consolidates all extracted features (radiomics, location, morphology)
into a final clinical profile with inferred symptoms.
"""
import os
import json
import numpy as np


# Rule-based symptom mapping: tumor location → expected symptoms
SYMPTOM_RULES = {
    "frontal": [
        "headache",
        "personality changes",
        "cognitive decline",
        "motor weakness",
        "speech difficulty (Broca's area)",
    ],
    "temporal": [
        "seizures",
        "memory impairment",
        "auditory hallucinations",
        "language comprehension difficulty (Wernicke's area)",
    ],
    "parietal": [
        "sensory loss",
        "spatial disorientation",
        "difficulty with reading/writing",
        "neglect syndrome",
    ],
    "occipital": [
        "visual disturbances",
        "visual field deficits",
        "hallucinations (visual)",
    ],
    "deep_structures": [
        "movement disorders",
        "hormonal imbalance",
        "hydrocephalus risk",
        "altered consciousness",
    ],
    "unknown": [
        "non-localizing headache",
        "increased intracranial pressure",
    ],
}

# Tumor volume severity thresholds (in voxels, assuming 1mm³ spacing)
VOLUME_SEVERITY = {
    "small": (0, 5000),         # < 5 cm³
    "medium": (5000, 30000),    # 5-30 cm³
    "large": (30000, 100000),   # 30-100 cm³
    "very_large": (100000, float("inf")),  # > 100 cm³
}


def classify_volume_severity(volume_voxels: float) -> str:
    """Classify tumor volume into severity categories."""
    for severity, (low, high) in VOLUME_SEVERITY.items():
        if low <= volume_voxels < high:
            return severity
    return "unknown"


def infer_symptoms(locations: list) -> list:
    """Infer possible symptoms based on tumor location(s)."""
    symptoms = []
    for loc in locations:
        loc_lower = loc.lower()
        if loc_lower in SYMPTOM_RULES:
            for symptom in SYMPTOM_RULES[loc_lower]:
                if symptom not in symptoms:
                    symptoms.append(symptom)
    return symptoms


def compute_morphology_features(radiomics: dict) -> dict:
    """Extract key morphology descriptors from radiomics features."""
    morphology = {}

    # Try PyRadiomics keys first, then manual keys
    volume_keys = [
        "original_shape_VoxelVolume",
        "original_shape_MeshVolume",
        "shape_volume_voxels",
        "shape_volume_mm3",
    ]
    for key in volume_keys:
        if key in radiomics:
            morphology["tumor_volume"] = radiomics[key]
            break

    surface_keys = [
        "original_shape_SurfaceArea",
        "shape_surface_area",
    ]
    for key in surface_keys:
        if key in radiomics:
            morphology["surface_area"] = radiomics[key]
            break

    sphericity_keys = [
        "original_shape_Sphericity",
        "shape_sphericity",
    ]
    for key in sphericity_keys:
        if key in radiomics:
            morphology["sphericity"] = radiomics[key]
            break

    diameter_keys = [
        "original_shape_Maximum3DDiameter",
        "original_shape_Maximum2DDiameterSlice",
        "shape_max_diameter",
    ]
    for key in diameter_keys:
        if key in radiomics:
            morphology["max_diameter"] = radiomics[key]
            break

    return morphology


def build_clinical_profile(state: dict) -> dict:
    """
    LangGraph node: Build the final clinical tumor profile.
    Consolidates location, radiomics, morphology, and inferred symptoms.
    """
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    tumor_location = state.get("tumor_location", ["unknown"])
    radiomics_features = state.get("radiomics_features", {})
    errors = list(state.get("errors", []))

    clinical_dir = os.path.join(output_dir, "clinical_features")
    os.makedirs(clinical_dir, exist_ok=True)

    print(f"[Clinical Builder] Processing patient: {patient_id}")

    # 1. Compute morphology features
    morphology = compute_morphology_features(radiomics_features)

    # 2. Classify volume severity
    volume = morphology.get("tumor_volume", 0)
    severity = classify_volume_severity(volume)

    # 3. Infer symptoms from location
    symptoms = infer_symptoms(tumor_location)

    # 4. Build clinical profile
    clinical_profile = {
        "patient_id": patient_id,
        "tumor_location": tumor_location,
        "primary_location": tumor_location[0] if tumor_location else "unknown",
        "morphology": morphology,
        "volume_severity": severity,
        "inferred_symptoms": symptoms,
        "radiomics_summary": {
            "num_features": len(radiomics_features),
            "key_intensity": {
                k: v for k, v in radiomics_features.items()
                if "intensity" in k.lower() or "firstorder" in k.lower()
            },
            "key_shape": {
                k: v for k, v in radiomics_features.items()
                if "shape" in k.lower()
            },
            "key_texture": {
                k: v for k, v in radiomics_features.items()
                if "glcm" in k.lower()
            },
        },
        "has_errors": len(errors) > 0,
        "pipeline_errors": errors,
    }

    # 5. Save clinical profile
    save_path = os.path.join(clinical_dir, f"{patient_id}_clinical.json")
    with open(save_path, "w") as f:
        json.dump(clinical_profile, f, indent=2)

    print(f"  Location: {tumor_location}")
    print(f"  Severity: {severity}")
    print(f"  Symptoms: {symptoms}")
    print(f"  Saved clinical profile: {save_path}")

    return {**state, "clinical_profile": clinical_profile, "errors": errors}
