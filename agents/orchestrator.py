"""
Phase 2 + 3 Agent Orchestrator
================================
LangGraph StateGraph for all Phase 2 and Phase 3 agents.

Phase 2 agents:
  tumor_analysis → similarity → clinical_reasoning → report

Phase 3 agents (appended after Phase 2):
  patient_memory → hitl_validation → cap_report → visualize

State flows end-to-end from Phase 1 output through all agents.
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Phase 2 agents
from agents.tumor_analysis import run_tumor_analysis
from agents.similarity_agent import run_similarity_agent
from agents.clinical_reasoning import run_clinical_reasoning
from agents.report_agent import run_report_generation

# Phase 3 agents
from memory.patient_memory import run_patient_memory
from validation.hitl import run_hitl_validation
from mcp_servers.cap_report import run_cap_reporting
from visualization.viewer import run_visualization


class Phase2State(TypedDict):
    # ── From Phase 1 ──
    patient_id: str
    base_dir: str
    output_dir: str
    preprocessed_path: Optional[str]
    segmentation_path: Optional[str]
    radiomics_features: Dict[str, Any]
    tumor_location: List[str]
    clinical_profile: Dict[str, Any]
    errors: List[str]

    # ── Phase 2 ──
    embedding: Optional[list]
    embedding_path: Optional[str]
    similar_cases: List[Dict[str, Any]]
    tumor_analysis: Dict[str, Any]
    clinical_reasoning: Optional[str]
    report_path: Optional[str]

    # ── Phase 3 ──
    patient_history: List[Dict[str, Any]]
    physician_corrections: Dict[str, Any]
    cap_report: Dict[str, Any]
    cap_report_path: Optional[str]
    visualization_paths: Dict[str, str]
    skip_hitl: bool


def build_phase2_pipeline():
    """Phase 2 only pipeline (4 agents)."""
    workflow = StateGraph(Phase2State)

    workflow.add_node("tumor_analysis", run_tumor_analysis)
    workflow.add_node("similarity", run_similarity_agent)
    workflow.add_node("clinical_reasoning", run_clinical_reasoning)
    workflow.add_node("report", run_report_generation)

    workflow.add_edge(START, "tumor_analysis")
    workflow.add_edge("tumor_analysis", "similarity")
    workflow.add_edge("similarity", "clinical_reasoning")
    workflow.add_edge("clinical_reasoning", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def build_phase3_pipeline():
    """Phase 3 only pipeline (4 agents, expects Phase 2 state)."""
    workflow = StateGraph(Phase2State)

    workflow.add_node("patient_memory", run_patient_memory)
    workflow.add_node("hitl_validation", run_hitl_validation)
    workflow.add_node("cap_report", run_cap_reporting)
    workflow.add_node("visualize", run_visualization)

    workflow.add_edge(START, "patient_memory")
    workflow.add_edge("patient_memory", "hitl_validation")
    workflow.add_edge("hitl_validation", "cap_report")
    workflow.add_edge("cap_report", "visualize")
    workflow.add_edge("visualize", END)

    return workflow.compile()


def build_full_pipeline():
    """Full Phase 2 + 3 pipeline (8 agents end-to-end)."""
    workflow = StateGraph(Phase2State)

    # Phase 2
    workflow.add_node("tumor_analysis", run_tumor_analysis)
    workflow.add_node("similarity", run_similarity_agent)
    workflow.add_node("clinical_reasoning", run_clinical_reasoning)
    workflow.add_node("report", run_report_generation)

    # Phase 3
    workflow.add_node("patient_memory", run_patient_memory)
    workflow.add_node("hitl_validation", run_hitl_validation)
    workflow.add_node("cap_report", run_cap_reporting)
    workflow.add_node("visualize", run_visualization)

    # Flow
    workflow.add_edge(START, "tumor_analysis")
    workflow.add_edge("tumor_analysis", "similarity")
    workflow.add_edge("similarity", "clinical_reasoning")
    workflow.add_edge("clinical_reasoning", "report")
    workflow.add_edge("report", "patient_memory")
    workflow.add_edge("patient_memory", "hitl_validation")
    workflow.add_edge("hitl_validation", "cap_report")
    workflow.add_edge("cap_report", "visualize")
    workflow.add_edge("visualize", END)

    return workflow.compile()
