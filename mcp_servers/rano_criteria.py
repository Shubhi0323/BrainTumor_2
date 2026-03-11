"""
RANO Criteria MCP Server
==========================
Implements RANO (Response Assessment in Neuro-Oncology) criteria
for evaluating tumor treatment response.

Can run as:
  - Standalone MCP server (stdio transport)
  - Direct function call (for in-process use)

RANO Response Categories:
  CR — Complete Response
  PR — Partial Response
  SD — Stable Disease
  PD — Progressive Disease
"""

# ─── RANO Criteria Knowledge Base ─────────────────────────────────────

RANO_CRITERIA = {
    "CR": {
        "name": "Complete Response",
        "description": (
            "Complete disappearance of all enhancing measurable and "
            "non-measurable disease sustained for at least 4 weeks. "
            "No new lesions. Stable or improved non-enhancing FLAIR/T2 lesions. "
            "Patient off corticosteroids or on physiologic replacement. "
            "Clinically stable or improved."
        ),
        "requirements": {
            "enhancing_tumor": "Complete disappearance",
            "non_enhancing_tumor": "Stable or decreased",
            "new_lesions": "None",
            "corticosteroids": "None or physiologic replacement",
            "clinical_status": "Stable or improved",
        },
    },
    "PR": {
        "name": "Partial Response",
        "description": (
            "≥50% decrease in the sum of products of perpendicular diameters "
            "of all measurable enhancing lesions sustained for at least 4 weeks. "
            "No progression of non-measurable disease. No new lesions. "
            "Stable or reduced corticosteroid dose. Clinically stable or improved."
        ),
        "requirements": {
            "enhancing_tumor": "≥50% decrease",
            "non_enhancing_tumor": "Stable or decreased",
            "new_lesions": "None",
            "corticosteroids": "Stable or decreased",
            "clinical_status": "Stable or improved",
        },
    },
    "SD": {
        "name": "Stable Disease",
        "description": (
            "Does not qualify for complete response, partial response, "
            "or progressive disease. Stable non-enhancing FLAIR/T2 lesions. "
            "Clinically stable."
        ),
        "requirements": {
            "enhancing_tumor": "<50% decrease to <25% increase",
            "non_enhancing_tumor": "Stable",
            "new_lesions": "None",
            "corticosteroids": "Stable or decreased",
            "clinical_status": "Stable",
        },
    },
    "PD": {
        "name": "Progressive Disease",
        "description": (
            "≥25% increase in the sum of products of perpendicular diameters "
            "of enhancing lesions. Or significant increase in non-enhancing "
            "FLAIR/T2 lesions. Or any new lesions. Or clinical deterioration "
            "not attributable to other causes."
        ),
        "requirements": {
            "enhancing_tumor": "≥25% increase",
            "non_enhancing_tumor": "Significant increase",
            "new_lesions": "Present",
            "corticosteroids": "N/A (determination independent)",
            "clinical_status": "Deteriorated",
        },
    },
}


def evaluate_response(
    tumor_size_change_pct: float = 0.0,
    contrast_enhancement: str = "stable",
    new_lesions: bool = False,
    clinical_condition: str = "stable",
    non_enhancing_change: str = "stable",
    corticosteroid_change: str = "stable",
) -> dict:
    """
    Evaluate tumor treatment response using RANO criteria.

    Args:
        tumor_size_change_pct: Percentage change in tumor size
            (negative = shrinkage, positive = growth). E.g. -60 means 60% shrinkage.
        contrast_enhancement: One of 'absent', 'decreased', 'stable', 'increased'
        new_lesions: Whether new lesions are detected
        clinical_condition: One of 'improved', 'stable', 'deteriorated'
        non_enhancing_change: One of 'decreased', 'stable', 'increased'
        corticosteroid_change: One of 'decreased', 'stable', 'increased', 'none'

    Returns:
        dict with assessment, criteria details, and reasoning
    """
    reasoning = []
    assessment = None

    # ─── Progressive Disease (check first, most critical) ───
    if (
        tumor_size_change_pct >= 25
        or new_lesions
        or clinical_condition == "deteriorated"
        or non_enhancing_change == "increased"
    ):
        assessment = "PD"
        if tumor_size_change_pct >= 25:
            reasoning.append(
                f"Tumor size increased by {tumor_size_change_pct:.1f}% (≥25% threshold for PD)")
        if new_lesions:
            reasoning.append("New lesions detected — automatic PD criterion")
        if clinical_condition == "deteriorated":
            reasoning.append("Clinical condition deteriorated")
        if non_enhancing_change == "increased":
            reasoning.append("Non-enhancing tumor/FLAIR signal increased")

    # ─── Complete Response ───
    elif (
        tumor_size_change_pct <= -99
        and contrast_enhancement in ("absent", "decreased")
        and not new_lesions
        and clinical_condition in ("improved", "stable")
        and non_enhancing_change in ("decreased", "stable")
    ):
        assessment = "CR"
        reasoning.append("Complete disappearance of enhancing disease")
        reasoning.append("No new lesions, clinical condition stable/improved")

    # ─── Partial Response ───
    elif (
        tumor_size_change_pct <= -50
        and not new_lesions
        and clinical_condition in ("improved", "stable")
    ):
        assessment = "PR"
        reasoning.append(
            f"Tumor size decreased by {abs(tumor_size_change_pct):.1f}% (≥50% decrease)")
        reasoning.append("No new lesions, clinical condition stable/improved")

    # ─── Stable Disease (default) ───
    else:
        assessment = "SD"
        reasoning.append(
            f"Tumor size change: {tumor_size_change_pct:+.1f}% "
            f"(between -50% and +25%)")
        reasoning.append("Does not meet criteria for CR, PR, or PD")

    # Build result
    criteria_info = RANO_CRITERIA[assessment].copy()
    result = {
        "assessment": assessment,
        "assessment_name": criteria_info["name"],
        "description": criteria_info["description"],
        "reasoning": reasoning,
        "input_summary": {
            "tumor_size_change_pct": tumor_size_change_pct,
            "contrast_enhancement": contrast_enhancement,
            "new_lesions": new_lesions,
            "clinical_condition": clinical_condition,
            "non_enhancing_change": non_enhancing_change,
            "corticosteroid_change": corticosteroid_change,
        },
        "criteria_requirements": criteria_info["requirements"],
    }

    return result


# ─── MCP Server (optional standalone mode) ────────────────────────────

def run_mcp_server():
    """Run as a standalone MCP server via stdio transport."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("ERROR: 'mcp' package not installed. pip install mcp[cli]")
        return

    mcp = FastMCP("RANO Criteria Evaluator")

    @mcp.tool()
    def rano_evaluate_response(
        tumor_size_change_pct: float = 0.0,
        contrast_enhancement: str = "stable",
        new_lesions: bool = False,
        clinical_condition: str = "stable",
        non_enhancing_change: str = "stable",
        corticosteroid_change: str = "stable",
    ) -> dict:
        """
        Evaluate tumor treatment response using RANO criteria.
        Provide tumor size change percentage (negative = shrinkage),
        enhancement status, lesion info, and clinical condition.
        Returns assessment (CR/PR/SD/PD) with reasoning.
        """
        return evaluate_response(
            tumor_size_change_pct=tumor_size_change_pct,
            contrast_enhancement=contrast_enhancement,
            new_lesions=new_lesions,
            clinical_condition=clinical_condition,
            non_enhancing_change=non_enhancing_change,
            corticosteroid_change=corticosteroid_change,
        )

    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
