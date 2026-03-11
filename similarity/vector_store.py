"""
Vector Similarity Engine
==========================
Stores patient embeddings and retrieves similar tumor cases.
Primary: Weaviate vector database.
Fallback: NumPy brute-force cosine similarity on local .npy files.
"""
import os
import json
import numpy as np

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import MetadataQuery
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

COLLECTION_NAME = "TumorCase"


# ─── Weaviate Backend ────────────────────────────────────────────────

def _get_weaviate_client(url: str = None):
    """Connect to Weaviate (embedded or remote)."""
    if url:
        client = weaviate.connect_to_custom(
            http_host=url.split("://")[-1].split(":")[0],
            http_port=int(url.split(":")[-1]) if ":" in url.split("://")[-1] else 8080,
            http_secure=url.startswith("https"),
            grpc_host=url.split("://")[-1].split(":")[0],
            grpc_port=50051,
            grpc_secure=False,
        )
    else:
        client = weaviate.connect_to_embedded()
    return client


def ensure_collection(client):
    """Create the TumorCase collection if it doesn't exist."""
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="patient_id", data_type=DataType.TEXT),
                Property(name="tumor_location", data_type=DataType.TEXT_ARRAY),
                Property(name="volume_severity", data_type=DataType.TEXT),
                Property(name="inferred_symptoms", data_type=DataType.TEXT_ARRAY),
                Property(name="tumor_volume", data_type=DataType.NUMBER),
                Property(name="max_diameter", data_type=DataType.NUMBER),
                Property(name="sphericity", data_type=DataType.NUMBER),
                Property(name="primary_location", data_type=DataType.TEXT),
            ],
        )
    return client.collections.get(COLLECTION_NAME)


def upsert_patient_weaviate(client, patient_id: str, embedding: list,
                            clinical_profile: dict):
    """Store or update a patient's embedding + metadata in Weaviate."""
    collection = ensure_collection(client)

    morph = clinical_profile.get("morphology", {})
    properties = {
        "patient_id": patient_id,
        "tumor_location": clinical_profile.get("tumor_location", []),
        "volume_severity": clinical_profile.get("volume_severity", "unknown"),
        "inferred_symptoms": clinical_profile.get("inferred_symptoms", []),
        "tumor_volume": float(morph.get("tumor_volume", 0)),
        "max_diameter": float(morph.get("max_diameter", 0)),
        "sphericity": float(morph.get("sphericity", 0)),
        "primary_location": clinical_profile.get("primary_location", "unknown"),
    }

    # Delete existing entry for this patient (upsert)
    existing = collection.query.fetch_objects(
        filters=weaviate.classes.query.Filter.by_property("patient_id").equal(patient_id),
        limit=1,
    )
    for obj in existing.objects:
        collection.data.delete_by_id(obj.uuid)

    collection.data.insert(properties=properties, vector=embedding)


def query_similar_weaviate(client, embedding: list, top_k: int = 5,
                           exclude_patient: str = None) -> list:
    """Query top-K similar tumor cases from Weaviate."""
    collection = ensure_collection(client)

    results = collection.query.near_vector(
        near_vector=embedding,
        limit=top_k + (1 if exclude_patient else 0),
        return_metadata=MetadataQuery(distance=True),
    )

    similar_cases = []
    for obj in results.objects:
        pid = obj.properties.get("patient_id", "")
        if exclude_patient and pid == exclude_patient:
            continue
        similar_cases.append({
            "patient_id": pid,
            "tumor_location": obj.properties.get("tumor_location", []),
            "volume_severity": obj.properties.get("volume_severity", ""),
            "inferred_symptoms": obj.properties.get("inferred_symptoms", []),
            "tumor_volume": obj.properties.get("tumor_volume", 0),
            "primary_location": obj.properties.get("primary_location", ""),
            "distance": obj.metadata.distance if obj.metadata else None,
        })

    return similar_cases[:top_k]


# ─── NumPy Fallback Backend ──────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def query_similar_numpy(query_embedding: np.ndarray, embeddings_dir: str,
                        clinical_dir: str, top_k: int = 5,
                        exclude_patient: str = None) -> list:
    """
    Brute-force similarity search using saved .npy embedding files.
    """
    similarities = []

    for fname in os.listdir(embeddings_dir):
        if not fname.endswith("_embedding.npy"):
            continue
        pid = fname.replace("_embedding.npy", "")
        if exclude_patient and pid == exclude_patient:
            continue

        emb_path = os.path.join(embeddings_dir, fname)
        stored_emb = np.load(emb_path)
        sim = _cosine_similarity(query_embedding, stored_emb)

        # Load clinical profile if available
        clinical_path = os.path.join(clinical_dir, f"{pid}_clinical.json")
        profile = {}
        if os.path.exists(clinical_path):
            with open(clinical_path, "r") as f:
                profile = json.load(f)

        similarities.append({
            "patient_id": pid,
            "similarity": sim,
            "tumor_location": profile.get("tumor_location", []),
            "volume_severity": profile.get("volume_severity", ""),
            "inferred_symptoms": profile.get("inferred_symptoms", []),
            "tumor_volume": profile.get("morphology", {}).get("tumor_volume", 0),
            "primary_location": profile.get("primary_location", ""),
        })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


# ─── Unified Interface ───────────────────────────────────────────────

def upsert_patient(patient_id: str, embedding, clinical_profile: dict,
                   weaviate_url: str = None):
    """Store patient embedding (Weaviate or local file — file is always saved by generator)."""
    if WEAVIATE_AVAILABLE:
        try:
            client = _get_weaviate_client(weaviate_url)
            emb_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            upsert_patient_weaviate(client, patient_id, emb_list, clinical_profile)
            client.close()
            print(f"  Stored in Weaviate: {patient_id}")
            return True
        except Exception as e:
            print(f"  [WARNING] Weaviate upsert failed: {e}")
    return False


def query_similar(embedding, output_dir: str, top_k: int = 5,
                  exclude_patient: str = None, weaviate_url: str = None) -> list:
    """Retrieve top-K similar cases (Weaviate or numpy fallback)."""
    if WEAVIATE_AVAILABLE:
        try:
            client = _get_weaviate_client(weaviate_url)
            emb_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            results = query_similar_weaviate(client, emb_list, top_k, exclude_patient)
            client.close()
            if results:
                return results
        except Exception as e:
            print(f"  [WARNING] Weaviate query failed, using numpy fallback: {e}")

    # Numpy fallback
    emb_dir = os.path.join(output_dir, "embeddings")
    clinical_dir = os.path.join(output_dir, "clinical_features")
    query_emb = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
    return query_similar_numpy(query_emb, emb_dir, clinical_dir, top_k, exclude_patient)


def run_similarity_retrieval(state: dict) -> dict:
    """
    LangGraph node: Store embedding and retrieve similar tumor cases.
    """
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    embedding = state.get("embedding")
    clinical_profile = state.get("clinical_profile", {})
    errors = list(state.get("errors", []))

    print(f"[Similarity] Processing patient: {patient_id}")

    if embedding is None:
        msg = f"No embedding for {patient_id}, skipping similarity retrieval."
        print(f"  [WARNING] {msg}")
        errors.append(msg)
        return {**state, "similar_cases": [], "errors": errors}

    try:
        # Store this patient
        upsert_patient(patient_id, embedding, clinical_profile)

        # Query for similar cases
        similar = query_similar(
            embedding, output_dir, top_k=5, exclude_patient=patient_id
        )
        print(f"  Found {len(similar)} similar cases.")
        for i, case in enumerate(similar):
            score = case.get("similarity", case.get("distance", "N/A"))
            print(f"    {i+1}. {case['patient_id']} (score: {score})")

        return {**state, "similar_cases": similar, "errors": errors}

    except Exception as e:
        msg = f"Error in similarity retrieval for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)
        return {**state, "similar_cases": [], "errors": errors}
