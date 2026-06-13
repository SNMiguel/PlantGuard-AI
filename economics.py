"""
Economic exposure estimator for PlantGuard-AI.

Turns a diagnosis (crop + severity) plus a field size into a concrete,
deterministic "$ at risk if untreated" range. The numbers are transparent
estimates — typical US gross revenue per acre multiplied by severity-based
yield-loss ranges — NOT precise financial advice. We compute them in code
(not via the LLM) so the figures are reproducible and can't be hallucinated.

Public API:
    from economics import estimate
    e = estimate("Tomato", "severe", acres=20)
    print(e["headline"])   # "$160,000 – $256,000 at risk if untreated"
"""

# Rough US gross revenue per acre (USD). Ballpark, for relative scale only.
CROP_REVENUE_PER_ACRE = {
    "Tomato": 8000,
    "Potato": 3000,
    "Corn": 800,
    "Apple": 7000,
    "Grape": 6000,
    "Pepper": 6000,
    "Peach": 5000,
    "Strawberry": 25000,
    "Cherry": 6000,
    "Orange": 4000,
    "Squash": 4000,
    "Blueberry": 9000,
    "Raspberry": 12000,
    "Soybean": 600,
    "_default": 3000,
}

# Potential yield-loss range (fraction) if the disease is left UNTREATED,
# keyed by the agent's severity rating.
SEVERITY_YIELD_LOSS = {
    "none": (0.0, 0.0),
    "low": (0.02, 0.05),
    "moderate": (0.10, 0.20),
    "high": (0.25, 0.40),
    "severe": (0.40, 0.80),
    "unknown": (0.10, 0.25),
}


def _revenue_per_acre(crop):
    """Look up gross revenue/acre, normalizing crop names like 'Pepper, bell'."""
    if not crop:
        return CROP_REVENUE_PER_ACRE["_default"]
    key = crop.split()[0].strip(",").capitalize()   # 'Corn (maize)' -> 'Corn'
    return CROP_REVENUE_PER_ACRE.get(key, CROP_REVENUE_PER_ACRE["_default"])


def _money(x):
    return f"${x:,.0f}"


def estimate(crop, severity, acres):
    """Estimate economic exposure for a diagnosis.

    Args:
        crop: crop name (e.g. "Tomato").
        severity: one of none/low/moderate/high/severe/unknown.
        acres: field size in acres (number).

    Returns:
        dict with the raw numbers plus display-ready strings, or None if there's
        nothing to estimate (no acreage, or a healthy/none-severity crop).
    """
    try:
        acres = float(acres)
    except (TypeError, ValueError):
        return None
    if acres <= 0:
        return None

    severity = (severity or "unknown").lower()
    loss_lo, loss_hi = SEVERITY_YIELD_LOSS.get(severity, SEVERITY_YIELD_LOSS["unknown"])
    if loss_hi == 0:
        return None  # healthy / no exposure

    rev = _revenue_per_acre(crop)
    total_value = rev * acres
    dollars_lo = total_value * loss_lo
    dollars_hi = total_value * loss_hi

    return {
        "crop": crop,
        "severity": severity,
        "acres": acres,
        "revenue_per_acre": rev,
        "total_crop_value": total_value,
        "yield_loss_low_pct": round(loss_lo * 100),
        "yield_loss_high_pct": round(loss_hi * 100),
        "dollars_low": dollars_lo,
        "dollars_high": dollars_hi,
        "headline": f"{_money(dollars_lo)} – {_money(dollars_hi)} at risk if untreated",
        "subline": (f"~{round(loss_lo*100)}–{round(loss_hi*100)}% potential yield loss "
                    f"on {acres:g} acres of {crop} "
                    f"(≈{_money(total_value)} total crop value)"),
        "disclaimer": ("Estimate only — typical US gross revenue/acre × severity-based "
                       "loss range. Timely treatment can avoid most of this exposure."),
    }


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    for crop, sev, ac in [("Tomato", "severe", 20), ("Potato", "moderate", 50),
                          ("Apple", "high", 8), ("Tomato", "none", 10)]:
        e = estimate(crop, sev, ac)
        if e:
            print(f"{crop:8} {sev:9} {ac:>4} ac -> {e['headline']}")
            print(f"             {e['subline']}")
        else:
            print(f"{crop:8} {sev:9} {ac:>4} ac -> (no exposure)")
