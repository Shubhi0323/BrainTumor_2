"""
Similarity Retrieval Agent
============================
Generates patient embedding and retrieves similar tumor cases
from the vector store.
"""
from embeddings.generator import run_embedding_generation
from similarity.vector_store import run_similarity_retrieval


def run_similarity_agent(state: dict) -> dict:
    """
    LangGraph node: Similarity Retrieval Agent.
    1. Generates embedding for the current patient.
    2. Stores it in the vector database.
    3. Retrieves top-K similar cases.
    """
    patient_id = state["patient_id"]
    print(f"[Similarity Agent] Processing patient: {patient_id}")

    # Step 1: Generate embedding
    state = run_embedding_generation(state)

    # Step 2: Store + retrieve similar cases
    state = run_similarity_retrieval(state)

    return state
