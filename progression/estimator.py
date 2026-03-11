"""
Tumor Progression Estimator
==============================
Estimates tumor progression state by comparing current and previous
scan volumes over a time interval.

growth_rate = (current_volume − previous_volume) / time_interval

Progression states:
  - tumor_progression : growth_rate > threshold
  - tumor_stability   : growth_rate near zero
  - tumor_regression  : growth_rate < -threshold
"""
import os
import json
import glob


# Thresholds (in mm³ per day)
PROGRESSION_THRESHOLD = 50.0    # >50 mm³/day → progression
REGRESSION_THRESHOLD = -50.0    # <-50 mm³/day → regression


def estimate_progression(
    current_volume: float,
    previous_volume: float = None,
    time_interval_days: float = None,
) -> dict:
    """
    Estimate tumor progression state.

    Args:
        current_volume: Current tumor volume in mm³
        previous_volume: Previous tumor volume in mm³ (from earlier scan)
        time_interval_days: Days between scans

    Returns:
        dict with growth_rate, progression_state, size_change_pct, etc.
    """
    result = {
        "current_volume": current_volume,
        "previous_volume": previous_volume,
        "time_interval_days": time_interval_days,
    }

    if previous_volume is None or time_interval_days is None:
        result["progression_state"] = "unknown"
        result["growth_rate"] = None
        result["size_change_pct"] = None
        result["reasoning"] = "No follow-up scan data available for comparison."
        return result

    if time_interval_days <= 0:
        result["progression_state"] = "unknown"
        result["growth_rate"] = None
        result["size_change_pct"] = None
        result["reasoning"] = "Invalid time interval."
        return result

    volume_change = current_volume - previous_volume
    growth_rate = volume_change / time_interval_days

    if previous_volume > 0:
        size_change_pct = (volume_change / previous_volume) * 100
    else:
        size_change_pct = 100.0 if current_volume > 0 else 0.0

    # Classify
    if growth_rate > PROGRESSION_THRESHOLD:
        state = "tumor_progression"
        reasoning = (
            f"Tumor volume increased by {volume_change:.0f} mm³ "
            f"({size_change_pct:+.1f}%) over {time_interval_days:.0f} days. "
            f"Growth rate: {growth_rate:.1f} mm³/day — indicates progression."
        )
    elif growth_rate < REGRESSION_THRESHOLD:
        state = "tumor_regression"
        reasoning = (
            f"Tumor volume decreased by {abs(volume_change):.0f} mm³ "
            f"({size_change_pct:+.1f}%) over {time_interval_days:.0f} days. "
            f"Regression rate: {abs(growth_rate):.1f} mm³/day — indicates regression."
        )
    else:
        state = "tumor_stability"
        reasoning = (
            f"Tumor volume changed by {volume_change:.0f} mm³ "
            f"({size_change_pct:+.1f}%) over {time_interval_days:.0f} days. "
            f"Growth rate: {growth_rate:.1f} mm³/day — within stability range."
        )

    result["growth_rate"] = round(growth_rate, 2)
    result["size_change_pct"] = round(size_change_pct, 2)
    result["progression_state"] = state
    result["reasoning"] = reasoning

    return result


def find_previous_volume(patient_id: str, output_dir: str) -> tuple:
    """
    Look for a previous clinical profile to extract prior volume.
    Returns (previous_volume, time_interval_days) or (None, None).
    """
    clinical_dir = os.path.join(output_dir, "clinical_features")
    history_path = os.path.join(clinical_dir, f"{patient_id}_history.json")

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        if len(history) >= 2:
            prev = history[-2]
            return prev.get("tumor_volume"), prev.get("time_interval_days", 90)

    return None, None
