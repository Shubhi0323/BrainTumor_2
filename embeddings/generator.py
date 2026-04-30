"""
Embedding Generation Node
===========================
Converts clinical tumor profiles into dense vector embeddings
using BioClinicalBERT (emilyalsentzer/Bio_ClinicalBERT).

Each patient's clinical features (radiomics, morphology, location,
symptoms) are serialized into a clinical text description, then
encoded into a 768-dimensional embedding vector.
"""
import os
import json
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
_tokenizer = None
_model = None


def _load_model():
    """Lazy-load BioClinicalBERT model and tokenizer."""
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"  Loading {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Prefer safetensors to avoid torch.load-based checkpoint loading on
        # restricted runtimes with older torch versions.
        _model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.to("cuda")
            print("  BioClinicalBERT loaded on GPU")
    return _tokenizer, _model


def clinical_profile_to_text(profile: dict) -> str:
    """
    Serialize a clinical profile dict into a natural-language
    clinical description suitable for BERT encoding.
    """
    parts = []

    pid = profile.get("patient_id", "unknown")
    parts.append(f"Patient {pid} brain tumor clinical summary.")

    # Location
    locations = profile.get("tumor_location", [])
    if locations:
        parts.append(f"Tumor located in {', '.join(locations)} region.")

    # Severity
    severity = profile.get("volume_severity", "unknown")
    parts.append(f"Tumor volume classified as {severity}.")

    # Morphology
    morph = profile.get("morphology", {})
    if morph.get("tumor_volume"):
        parts.append(f"Tumor volume: {morph['tumor_volume']:.1f} mm³.")
    if morph.get("max_diameter"):
        parts.append(f"Maximum diameter: {morph['max_diameter']:.1f} mm.")
    if morph.get("sphericity"):
        parts.append(f"Sphericity: {morph['sphericity']:.3f}.")
    if morph.get("surface_area"):
        parts.append(f"Surface area: {morph['surface_area']:.1f} mm².")

    # Symptoms
    symptoms = profile.get("inferred_symptoms", [])
    if symptoms:
        parts.append(f"Inferred symptoms: {', '.join(symptoms)}.")

    # Key radiomics
    rad_summary = profile.get("radiomics_summary", {})
    intensity = rad_summary.get("key_intensity", {})
    if intensity:
        # Pick a few key metrics
        for key in ["intensity_mean", "intensity_std", "intensity_skewness"]:
            if key in intensity:
                parts.append(f"{key.replace('_', ' ').title()}: {intensity[key]:.4f}.")

    texture = rad_summary.get("key_texture", {})
    if texture:
        count = len(texture)
        parts.append(f"Texture features extracted: {count} GLCM metrics.")

    return " ".join(parts)


@torch.no_grad()
def generate_embedding(text: str) -> np.ndarray:
    """Generate a 768-dim embedding from clinical text using BioClinicalBERT."""
    tokenizer, model = _load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    # Use [CLS] token embedding as the sentence representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return cls_embedding.cpu().numpy()


def generate_embedding_fallback(profile: dict) -> np.ndarray:
    """
    Fallback embedding when transformers is unavailable.
    Creates a deterministic feature vector from numerical clinical features.
    """
    features = []

    morph = profile.get("morphology", {})
    features.append(morph.get("tumor_volume", 0.0))
    features.append(morph.get("max_diameter", 0.0))
    features.append(morph.get("sphericity", 0.0))
    features.append(morph.get("surface_area", 0.0))

    # Volume severity as ordinal
    sev_map = {"small": 1, "medium": 2, "large": 3, "very_large": 4}
    features.append(sev_map.get(profile.get("volume_severity", ""), 0))

    # Location as one-hot
    lobe_names = ["frontal", "temporal", "parietal", "occipital", "deep_structures"]
    locations = [loc.lower() for loc in profile.get("tumor_location", [])]
    for lobe in lobe_names:
        features.append(1.0 if lobe in locations else 0.0)

    # Key radiomics intensity features
    rad = profile.get("radiomics_summary", {}).get("key_intensity", {})
    for key in ["intensity_mean", "intensity_variance", "intensity_std",
                "intensity_skewness", "intensity_kurtosis"]:
        features.append(float(rad.get(key, 0.0)))

    # Pad or truncate to 768 dims for compatibility
    vec = np.array(features, dtype=np.float32)
    embedding = np.zeros(768, dtype=np.float32)
    embedding[:len(vec)] = vec
    return embedding


def run_embedding_generation(state: dict) -> dict:
    """
    LangGraph node: Generate embedding for a patient's clinical profile.
    """
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    clinical_profile = state.get("clinical_profile", {})
    errors = list(state.get("errors", []))

    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    print(f"[Embedding] Processing patient: {patient_id}")

    if not clinical_profile:
        msg = f"No clinical profile for {patient_id}, skipping embedding."
        print(f"  [WARNING] {msg}")
        errors.append(msg)
        return {**state, "embedding": None, "errors": errors}

    try:
        if TRANSFORMERS_AVAILABLE:
            print("  Using BioClinicalBERT for embedding generation.")
            text = clinical_profile_to_text(clinical_profile)
            print(f"  Clinical text ({len(text)} chars): {text[:120]}...")
            try:
                embedding = generate_embedding(text)
            except Exception as model_err:
                # Graceful fallback when hosted runtimes block torch.load-based
                # model loading due security/version constraints.
                print(f"  [WARNING] Transformer embedding unavailable: {model_err}")
                print("  Falling back to deterministic clinical feature vector.")
                embedding = generate_embedding_fallback(clinical_profile)
                errors.append(f"Embedding model fallback used for {patient_id}")
        else:
            print("  Transformers not available. Using fallback feature vector.")
            embedding = generate_embedding_fallback(clinical_profile)

        save_path = os.path.join(emb_dir, f"{patient_id}_embedding.npy")
        np.save(save_path, embedding)
        print(f"  Embedding shape: {embedding.shape} | Saved to: {save_path}")

        return {**state, "embedding": embedding.tolist(), "embedding_path": save_path, "errors": errors}

    except Exception as e:
        msg = f"Error generating embedding for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)
        return {**state, "embedding": None, "errors": errors}
