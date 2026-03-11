"""
Patient Memory System
========================
Persistent patient history storage using ChromaDB.
Stores scan history, tumor measurements, radiomics features,
AI interpretations, and doctor feedback per patient.

Falls back to flat JSON files if ChromaDB is unavailable.
"""
import os
import json
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "chromadb")


# ─── ChromaDB Backend ─────────────────────────────────────────────────

def _get_client(db_path: str = None):
    """Get or create a ChromaDB persistent client."""
    path = db_path or DB_PATH
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def _get_collections(client):
    """Get or create the collections."""
    scans = client.get_or_create_collection(
        name="patient_scans",
        metadata={"hnsw:space": "cosine"},
    )
    feedback = client.get_or_create_collection(
        name="doctor_feedback",
    )
    return scans, feedback


def store_patient_scan_chroma(
    client, patient_id: str, scan_data: dict, embedding: list = None
):
    """Store scan data in ChromaDB."""
    scans, _ = _get_collections(client)

    # Build document text for semantic search
    doc = json.dumps({
        "patient_id": patient_id,
        "tumor_location": scan_data.get("tumor_location", []),
        "volume_severity": scan_data.get("volume_severity", ""),
        "inferred_symptoms": scan_data.get("inferred_symptoms", []),
        "who_classification": scan_data.get("who_classification", ""),
        "rano_assessment": scan_data.get("rano_assessment", ""),
        "scan_date": scan_data.get("scan_date", datetime.now().isoformat()),
    })

    metadata = {
        "patient_id": patient_id,
        "scan_date": scan_data.get("scan_date", datetime.now().isoformat()),
        "tumor_volume": float(scan_data.get("tumor_volume", 0)),
        "who_classification": str(scan_data.get("who_classification", "")),
        "rano_assessment": str(scan_data.get("rano_assessment", "")),
        "volume_severity": str(scan_data.get("volume_severity", "")),
    }

    scan_id = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if embedding and len(embedding) > 0:
        scans.upsert(
            ids=[scan_id],
            documents=[doc],
            embeddings=[embedding[:768]],
            metadatas=[metadata],
        )
    else:
        scans.upsert(
            ids=[scan_id],
            documents=[doc],
            metadatas=[metadata],
        )

    return scan_id


def retrieve_patient_history_chroma(client, patient_id: str) -> list:
    """Retrieve all scans for a patient from ChromaDB."""
    scans, _ = _get_collections(client)

    results = scans.get(
        where={"patient_id": patient_id},
        include=["documents", "metadatas"],
    )

    history = []
    for doc, meta in zip(results.get("documents", []),
                         results.get("metadatas", [])):
        try:
            entry = json.loads(doc)
            entry.update(meta)
            history.append(entry)
        except Exception:
            history.append(meta)

    history.sort(key=lambda x: x.get("scan_date", ""), reverse=True)
    return history


def store_doctor_feedback_chroma(client, patient_id: str, feedback: dict):
    """Store physician feedback/corrections in ChromaDB."""
    _, fb_collection = _get_collections(client)

    fb_id = f"{patient_id}_fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata = {
        "patient_id": patient_id,
        "feedback_date": datetime.now().isoformat(),
        "physician": feedback.get("physician", "unknown"),
        "action": feedback.get("action", ""),
    }

    fb_collection.upsert(
        ids=[fb_id],
        documents=[json.dumps(feedback)],
        metadatas=[metadata],
    )
    return fb_id


def get_doctor_feedback_chroma(client, patient_id: str) -> list:
    """Retrieve all doctor feedback for a patient."""
    _, fb_collection = _get_collections(client)

    results = fb_collection.get(
        where={"patient_id": patient_id},
        include=["documents", "metadatas"],
    )

    feedbacks = []
    for doc, meta in zip(results.get("documents", []),
                         results.get("metadatas", [])):
        try:
            entry = json.loads(doc)
            entry["_meta"] = meta
            feedbacks.append(entry)
        except Exception:
            feedbacks.append(meta)

    return feedbacks


# ─── JSON Fallback Backend ────────────────────────────────────────────

def _json_db_path(output_dir: str) -> str:
    path = os.path.join(output_dir, "memory")
    os.makedirs(path, exist_ok=True)
    return path


def store_patient_scan_json(output_dir: str, patient_id: str, scan_data: dict):
    """JSON fallback: append scan to patient history file."""
    db_path = _json_db_path(output_dir)
    history_file = os.path.join(db_path, f"{patient_id}_history.json")

    history = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

    scan_data["stored_at"] = datetime.now().isoformat()
    history.append(scan_data)

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2, default=str)


def retrieve_patient_history_json(output_dir: str, patient_id: str) -> list:
    """JSON fallback: read patient history file."""
    db_path = _json_db_path(output_dir)
    history_file = os.path.join(db_path, f"{patient_id}_history.json")

    if not os.path.exists(history_file):
        return []

    with open(history_file, "r") as f:
        return json.load(f)


def store_doctor_feedback_json(output_dir: str, patient_id: str, feedback: dict):
    """JSON fallback: store doctor feedback."""
    db_path = _json_db_path(output_dir)
    feedback_file = os.path.join(db_path, f"{patient_id}_feedback.json")

    feedbacks = []
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            feedbacks = json.load(f)

    feedback["stored_at"] = datetime.now().isoformat()
    feedbacks.append(feedback)

    with open(feedback_file, "w") as f:
        json.dump(feedbacks, f, indent=2, default=str)


# ─── Unified Interface ────────────────────────────────────────────────

def store_patient_scan(patient_id: str, scan_data: dict,
                       output_dir: str, embedding: list = None):
    """Store patient scan data in memory (ChromaDB or JSON fallback)."""
    if CHROMADB_AVAILABLE:
        try:
            client = _get_client(os.path.join(output_dir, "chromadb"))
            store_patient_scan_chroma(client, patient_id, scan_data, embedding)
            print(f"  Stored in ChromaDB: {patient_id}")
            return
        except Exception as e:
            print(f"  [WARNING] ChromaDB store failed: {e}")
    store_patient_scan_json(output_dir, patient_id, scan_data)
    print(f"  Stored in JSON memory: {patient_id}")


def retrieve_patient_history(patient_id: str, output_dir: str) -> list:
    """Retrieve patient scan history (ChromaDB or JSON fallback)."""
    if CHROMADB_AVAILABLE:
        try:
            client = _get_client(os.path.join(output_dir, "chromadb"))
            return retrieve_patient_history_chroma(client, patient_id)
        except Exception as e:
            print(f"  [WARNING] ChromaDB retrieval failed: {e}")
    return retrieve_patient_history_json(output_dir, patient_id)


def store_doctor_feedback(patient_id: str, feedback: dict, output_dir: str):
    """Store physician feedback (ChromaDB or JSON fallback)."""
    if CHROMADB_AVAILABLE:
        try:
            client = _get_client(os.path.join(output_dir, "chromadb"))
            store_doctor_feedback_chroma(client, patient_id, feedback)
            return
        except Exception as e:
            print(f"  [WARNING] ChromaDB feedback store failed: {e}")
    store_doctor_feedback_json(output_dir, patient_id, feedback)


def run_patient_memory(state: dict) -> dict:
    """
    LangGraph node: Store current patient scan data in memory.
    Retrieves and attaches prior history to the state.
    """
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    clinical = state.get("clinical_profile", {})
    analysis = state.get("tumor_analysis", {})
    embedding = state.get("embedding")
    errors = list(state.get("errors", []))

    print(f"[Patient Memory] Processing patient: {patient_id}")

    # Build scan data snapshot
    who = analysis.get("who_classification", {})
    rano = analysis.get("rano_assessment", {})
    scan_data = {
        "scan_date": datetime.now().isoformat(),
        "tumor_location": clinical.get("tumor_location", []),
        "volume_severity": clinical.get("volume_severity", ""),
        "tumor_volume": clinical.get("morphology", {}).get("tumor_volume", 0),
        "inferred_symptoms": clinical.get("inferred_symptoms", []),
        "who_classification": who.get("classified_as", "unknown"),
        "rano_assessment": rano.get("assessment", "N/A"),
    }

    # Retrieve prior history first
    history = retrieve_patient_history(patient_id, output_dir)
    print(f"  Found {len(history)} prior scan(s) in memory.")

    # Store current scan
    store_patient_scan(patient_id, scan_data, output_dir, embedding)

    return {**state, "patient_history": history, "errors": errors}
