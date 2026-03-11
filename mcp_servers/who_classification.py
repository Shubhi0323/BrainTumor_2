"""
WHO CNS Tumor Classification MCP Server
==========================================
Implements WHO CNS5 (2021) tumor classification guidelines as an MCP tool.
Classifies brain tumors based on morphology, radiomics patterns, and
growth characteristics.

Can run as:
  - Standalone MCP server (stdio transport)
  - Direct function call (for in-process use)
"""

# ─── WHO CNS5 Classification Knowledge Base ──────────────────────────

WHO_CLASSIFICATIONS = {
    "glioblastoma": {
        "who_grade": "IV",
        "full_name": "Glioblastoma, IDH-wildtype",
        "description": (
            "Highly malignant diffuse astrocytic glioma. "
            "Characterized by microvascular proliferation and/or necrosis. "
            "Most common primary malignant brain tumor in adults."
        ),
        "typical_features": {
            "volume": "large",
            "enhancement": "ring-enhancing with central necrosis",
            "growth_rate": "rapid",
            "sphericity": "irregular (low sphericity)",
            "typical_location": ["frontal", "temporal", "parietal"],
        },
        "morphology_indicators": {
            "min_volume": 15000,
            "max_sphericity": 0.6,
            "intensity_heterogeneity": "high",
        },
        "prognosis": "Poor. Median survival 14-16 months with standard treatment.",
        "standard_treatment": "Maximal safe resection, radiotherapy, temozolomide.",
    },
    "astrocytoma": {
        "who_grade": "II-III",
        "full_name": "Astrocytoma, IDH-mutant",
        "description": (
            "Diffuse astrocytic glioma with IDH mutation. "
            "Grade II (low-grade) or Grade III (anaplastic). "
            "Better prognosis than glioblastoma."
        ),
        "typical_features": {
            "volume": "small to medium",
            "enhancement": "minimal or no enhancement (grade II), variable (grade III)",
            "growth_rate": "slow to moderate",
            "sphericity": "moderate",
            "typical_location": ["frontal", "temporal"],
        },
        "morphology_indicators": {
            "min_volume": 3000,
            "max_volume": 40000,
            "min_sphericity": 0.4,
        },
        "prognosis": "Variable. Grade II median survival 7-10 years. Grade III: 3-5 years.",
        "standard_treatment": "Surgery when feasible, radiation, chemotherapy for higher grades.",
    },
    "oligodendroglioma": {
        "who_grade": "II-III",
        "full_name": "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted",
        "description": (
            "Diffuse glioma with IDH mutation and 1p/19q codeletion. "
            "Characteristically demonstrates calcifications on imaging. "
            "Generally better prognosis among diffuse gliomas."
        ),
        "typical_features": {
            "volume": "small to medium",
            "enhancement": "variable, often cortical involvement",
            "growth_rate": "slow",
            "sphericity": "moderate to high",
            "typical_location": ["frontal"],
        },
        "morphology_indicators": {
            "min_volume": 2000,
            "max_volume": 35000,
            "min_sphericity": 0.5,
        },
        "prognosis": "Favorable. Grade II median survival >10 years.",
        "standard_treatment": "Surgery, PCV chemotherapy, radiation.",
    },
    "meningioma": {
        "who_grade": "I-III",
        "full_name": "Meningioma",
        "description": (
            "Extra-axial tumor arising from meningothelial cells. "
            "Most common primary intracranial tumor. "
            "Majority are WHO Grade I (benign)."
        ),
        "typical_features": {
            "volume": "variable",
            "enhancement": "homogeneous, intense enhancement",
            "growth_rate": "slow",
            "sphericity": "high (well-circumscribed)",
            "typical_location": ["parietal", "frontal"],
        },
        "morphology_indicators": {
            "min_sphericity": 0.65,
        },
        "prognosis": "Excellent for grade I. Grade II/III have higher recurrence.",
        "standard_treatment": "Surgical resection. Radiation for incompletely resected or higher grade.",
    },
    "glioma_nos": {
        "who_grade": "II-IV",
        "full_name": "Glioma, not otherwise specified",
        "description": (
            "Diffuse glioma that cannot be further classified due to "
            "insufficient molecular data. Grading based on histology."
        ),
        "typical_features": {
            "volume": "variable",
            "enhancement": "variable",
            "growth_rate": "variable",
            "sphericity": "variable",
            "typical_location": ["frontal", "temporal", "parietal", "occipital"],
        },
        "morphology_indicators": {},
        "prognosis": "Depends on histological grade.",
        "standard_treatment": "Surgery, radiation and/or chemotherapy based on grade.",
    },
}


def classify_tumor(morphology: dict, radiomics_patterns: dict = None,
                   growth_characteristics: dict = None) -> dict:
    """
    Classify a tumor using WHO CNS5 guidelines based on morphological
    and radiomics features.

    Args:
        morphology: dict with tumor_volume, max_diameter, sphericity, surface_area
        radiomics_patterns: dict with intensity/texture features (optional)
        growth_characteristics: dict with growth_rate, enhancement info (optional)

    Returns:
        dict with classification, confidence, reasoning
    """
    radiomics_patterns = radiomics_patterns or {}
    growth_characteristics = growth_characteristics or {}

    volume = morphology.get("tumor_volume", 0)
    sphericity = morphology.get("sphericity", 0.5)
    max_diameter = morphology.get("max_diameter", 0)

    scores = {}
    reasoning = {}

    for tumor_type, info in WHO_CLASSIFICATIONS.items():
        score = 0.0
        reasons = []
        indicators = info["morphology_indicators"]

        # Volume scoring
        min_vol = indicators.get("min_volume", 0)
        max_vol = indicators.get("max_volume", float("inf"))
        if min_vol <= volume <= max_vol:
            score += 2.0
            reasons.append(f"Volume {volume:.0f} mm³ matches {tumor_type} range")
        elif volume > 0:
            # Partial score if close
            if volume > min_vol * 0.5:
                score += 0.5

        # Sphericity scoring
        min_sph = indicators.get("min_sphericity", 0)
        max_sph = indicators.get("max_sphericity", 1.0)
        if min_sph <= sphericity <= max_sph:
            score += 2.0
            reasons.append(f"Sphericity {sphericity:.3f} consistent with {tumor_type}")
        elif sphericity > 0:
            score += 0.5

        # Size scoring (large tumors favor glioblastoma)
        if tumor_type == "glioblastoma" and volume > 30000:
            score += 2.0
            reasons.append("Large volume strongly suggests high-grade lesion")
        elif tumor_type == "meningioma" and sphericity > 0.7:
            score += 2.0
            reasons.append("High sphericity suggests well-circumscribed extra-axial mass")

        # Growth rate (if available)
        growth_rate = growth_characteristics.get("growth_rate", None)
        if growth_rate is not None:
            typical_growth = info["typical_features"]["growth_rate"]
            if "rapid" in typical_growth and growth_rate > 0.5:
                score += 1.5
                reasons.append("Rapid growth rate matches profile")
            elif "slow" in typical_growth and growth_rate < 0.1:
                score += 1.5
                reasons.append("Slow growth rate matches profile")

        # Intensity heterogeneity (from radiomics)
        intensity_std = radiomics_patterns.get("intensity_std", 0)
        intensity_skew = radiomics_patterns.get("intensity_skewness", 0)
        if tumor_type == "glioblastoma" and intensity_std > 0.5:
            score += 1.0
            reasons.append("High intensity heterogeneity consistent with GBM")
        elif tumor_type in ["astrocytoma", "oligodendroglioma"] and intensity_std < 0.5:
            score += 1.0
            reasons.append("Low intensity heterogeneity consistent with lower-grade glioma")

        scores[tumor_type] = score
        reasoning[tumor_type] = reasons

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type = ranked[0][0]
    top_score = ranked[0][1]
    max_possible = 8.5

    confidence = min(top_score / max_possible, 1.0)

    # Build result
    classification = WHO_CLASSIFICATIONS[top_type].copy()
    classification["classified_as"] = top_type
    classification["confidence"] = round(confidence, 3)
    classification["reasoning"] = reasoning[top_type]
    classification["differential"] = [
        {"type": t, "score": round(s, 2), "reasoning": reasoning[t]}
        for t, s in ranked[1:3]
        if s > 0
    ]

    return classification


# ─── MCP Server (optional standalone mode) ────────────────────────────

def run_mcp_server():
    """Run as a standalone MCP server via stdio transport."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("ERROR: 'mcp' package not installed. pip install mcp[cli]")
        return

    mcp = FastMCP("WHO Tumor Classification")

    @mcp.tool()
    def who_classify_tumor(
        tumor_volume: float = 0,
        sphericity: float = 0.5,
        max_diameter: float = 0,
        surface_area: float = 0,
        intensity_std: float = 0,
        intensity_skewness: float = 0,
        growth_rate: float = None,
    ) -> dict:
        """
        Classify a brain tumor using WHO CNS5 guidelines.
        Provide morphological measurements and radiomics features.
        Returns classification, confidence, and reasoning.
        """
        morphology = {
            "tumor_volume": tumor_volume,
            "sphericity": sphericity,
            "max_diameter": max_diameter,
            "surface_area": surface_area,
        }
        radiomics = {
            "intensity_std": intensity_std,
            "intensity_skewness": intensity_skewness,
        }
        growth = {"growth_rate": growth_rate} if growth_rate is not None else {}
        return classify_tumor(morphology, radiomics, growth)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
