"""
Clinical Reasoning Agent
==========================
Uses Llama 3 (via Ollama) to synthesize all tumor analysis data
into a coherent clinical interpretation.

Falls back to a rule-based summary if Ollama/Llama 3 is unavailable.
"""
import json

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


def _build_prompt(state: dict) -> str:
    """Build a clinical reasoning prompt from patient state."""
    patient_id = state.get("patient_id", "unknown")
    clinical = state.get("clinical_profile", {})
    analysis = state.get("tumor_analysis", {})
    similar = state.get("similar_cases", [])

    who = analysis.get("who_classification", {})
    rano = analysis.get("rano_assessment", {})
    progression = analysis.get("progression", {})

    prompt = f"""You are a neuro-oncology AI assistant. Based on the following tumor data for patient {patient_id}, provide a concise clinical reasoning summary.

## Patient Tumor Data

**Location**: {clinical.get('tumor_location', ['unknown'])}
**Volume Severity**: {clinical.get('volume_severity', 'unknown')}
**Morphology**: {json.dumps(clinical.get('morphology', {}), indent=2)}
**Inferred Symptoms**: {clinical.get('inferred_symptoms', [])}

## WHO Classification
**Classified as**: {who.get('classified_as', 'unknown')} (Grade {who.get('who_grade', '?')})
**Confidence**: {who.get('confidence', 0):.2f}
**Description**: {who.get('description', 'N/A')}
**Reasoning**: {who.get('reasoning', [])}

## RANO Assessment
**Assessment**: {rano.get('assessment', '?')} — {rano.get('assessment_name', '')}
**Reasoning**: {rano.get('reasoning', [])}

## Tumor Progression
**State**: {progression.get('progression_state', 'unknown')}
**Growth Rate**: {progression.get('growth_rate', 'N/A')} mm³/day
**Explanation**: {progression.get('reasoning', 'No follow-up data')}

## Similar Cases Found: {len(similar)}
"""
    for i, case in enumerate(similar[:3]):
        prompt += f"  {i+1}. Patient {case.get('patient_id', '?')} — "
        prompt += f"Location: {case.get('tumor_location', [])}, "
        prompt += f"Severity: {case.get('volume_severity', '?')}\n"

    prompt += """
## Instructions
Provide a structured clinical reasoning summary with:
1. **Diagnosis Assessment**: Your assessment based on WHO classification
2. **Treatment Considerations**: Based on the tumor type and severity
3. **Prognosis Indicators**: Based on morphology and progression
4. **Symptom Correlation**: How inferred symptoms align with tumor location
5. **Recommendation**: Next steps for clinical management

Keep the response concise, structured, and medically focused. Use professional clinical language."""

    return prompt


def call_ollama(prompt: str, model: str = DEFAULT_MODEL,
                url: str = DEFAULT_OLLAMA_URL) -> str:
    """Call Ollama API for LLM inference."""
    endpoint = f"{url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }

    response = httpx.post(endpoint, json=payload, timeout=120.0)
    response.raise_for_status()
    return response.json().get("response", "")


def rule_based_reasoning(state: dict) -> str:
    """Fallback rule-based clinical reasoning when LLM is unavailable."""
    clinical = state.get("clinical_profile", {})
    analysis = state.get("tumor_analysis", {})

    who = analysis.get("who_classification", {})
    rano = analysis.get("rano_assessment", {})
    progression = analysis.get("progression", {})

    parts = []

    # Diagnosis
    classified_as = who.get("classified_as", "unknown")
    grade = who.get("who_grade", "?")
    confidence = who.get("confidence", 0)
    parts.append(
        f"DIAGNOSIS ASSESSMENT: Tumor classified as {classified_as} "
        f"(WHO Grade {grade}) with {confidence:.0%} confidence. "
        f"{who.get('description', '')}"
    )

    # Treatment
    treatment = who.get("standard_treatment", "Consult neuro-oncology team.")
    parts.append(f"TREATMENT CONSIDERATIONS: {treatment}")

    # Prognosis
    prognosis = who.get("prognosis", "Prognosis depends on molecular markers.")
    severity = clinical.get("volume_severity", "unknown")
    parts.append(
        f"PROGNOSIS INDICATORS: {prognosis} "
        f"Current volume classified as {severity}."
    )

    # Progression
    prog_state = progression.get("progression_state", "unknown")
    parts.append(f"PROGRESSION: {progression.get('reasoning', 'No follow-up data.')}")

    # RANO
    rano_name = rano.get("assessment_name", "Not assessed")
    parts.append(f"RANO ASSESSMENT: {rano_name}. {'; '.join(rano.get('reasoning', []))}")

    # Symptoms
    symptoms = clinical.get("inferred_symptoms", [])
    locations = clinical.get("tumor_location", [])
    parts.append(
        f"SYMPTOM CORRELATION: Tumor in {', '.join(locations)} region(s) "
        f"correlates with: {', '.join(symptoms) if symptoms else 'no specific symptoms'}."
    )

    # Recommendation
    if classified_as == "glioblastoma":
        rec = "Urgent multidisciplinary team review recommended. Consider maximal safe resection."
    elif severity in ("large", "very_large"):
        rec = "Recommend surgical evaluation and comprehensive imaging follow-up."
    else:
        rec = "Recommend continued monitoring with serial MRI at 3-month intervals."
    parts.append(f"RECOMMENDATION: {rec}")

    return "\n\n".join(parts)


def run_clinical_reasoning(state: dict) -> dict:
    """
    LangGraph node: Clinical Reasoning Agent.
    Uses Llama 3 to produce a clinical interpretation, with rule-based fallback.
    """
    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    print(f"[Clinical Reasoning Agent] Processing patient: {patient_id}")

    reasoning_text = None

    # Try LLM-based reasoning first
    if HTTPX_AVAILABLE:
        try:
            prompt = _build_prompt(state)
            print(f"  Calling Llama 3 via Ollama...")
            reasoning_text = call_ollama(prompt)
            print(f"  LLM reasoning generated ({len(reasoning_text)} chars)")
        except Exception as e:
            print(f"  [WARNING] Ollama/Llama 3 unavailable: {e}")
            print(f"  Falling back to rule-based reasoning.")

    # Fallback
    if not reasoning_text:
        reasoning_text = rule_based_reasoning(state)
        print(f"  Rule-based reasoning generated ({len(reasoning_text)} chars)")

    return {**state, "clinical_reasoning": reasoning_text, "errors": errors}
