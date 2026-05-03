# fertilizer.py

"""
Backend utilities for:
1. Soil fertility scoring from current soil/weather inputs
2. Crop-specific fertilizer recommendations from NPK gaps

Note:
- `compute_soil_fertility_score(...)` is used at inference time by the API.
- The old `df.apply(...)` snippet is only useful in preprocessing notebooks/scripts
  if you want to add fertility columns to a full dataset.
"""


FERTILITY_WEIGHTS = {
    "ph": 25,
    "N": 20,
    "P": 15,
    "K": 15,
    "temperature": 15,
    "humidity": 10,
}

FERTILITY_IDEAL_RANGES = {
    "ph": (6.0, 7.5),
    "N": (40, 120),
    "P": (20, 80),
    "K": (20, 120),
    "temperature": (20, 32),
    "humidity": (50, 80),
}


def _score_in_range(value, low, high):
    """Returns 1.0 inside range and decays smoothly outside."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, 1 - (low - value) / max(abs(low), 1))
    return max(0.0, 1 - (value - high) / max(abs(high), 1))


def compute_soil_fertility_score(row):
    """
    Compute a 0-100 soil fertility score and a health label.

    Expected keys:
    ph, N, P, K, temperature, humidity

    Works for:
    - a single API input dict
    - a pandas Series during dataset-wide `df.apply(...)`
    """
    score = 0.0
    breakdown = {}

    for feature, weight in FERTILITY_WEIGHTS.items():
        low, high = FERTILITY_IDEAL_RANGES[feature]
        feature_score = _score_in_range(float(row[feature]), low, high)
        weighted_score = feature_score * weight
        score += weighted_score
        breakdown[feature] = {
            "value": float(row[feature]),
            "ideal_range": [low, high],
            "weighted_score": round(weighted_score, 2),
        }

    score = round(score, 2)

    if score >= 80:
        label = "Excellent"
    elif score >= 65:
        label = "Good"
    elif score >= 50:
        label = "Moderate"
    elif score >= 35:
        label = "Poor"
    else:
        label = "Critical"

    return {
        "score": score,
        "label": label,
        "breakdown": breakdown,
    }


# --- Ideal NPK requirements (kg/hectare) ---
CROP_NPK_REQUIREMENTS = {
    "rice": {"N": 120, "P": 60, "K": 60},
    "wheat": {"N": 120, "P": 60, "K": 40},
    "maize": {"N": 150, "P": 75, "K": 75},
    "chickpea": {"N": 20, "P": 50, "K": 30},
    "kidneybeans": {"N": 25, "P": 60, "K": 30},
    "pigeonpeas": {"N": 20, "P": 50, "K": 20},
    "mothbeans": {"N": 20, "P": 40, "K": 20},
    "mungbean": {"N": 25, "P": 50, "K": 25},
    "blackgram": {"N": 25, "P": 50, "K": 25},
    "lentil": {"N": 20, "P": 40, "K": 20},
    "cotton": {"N": 150, "P": 60, "K": 60},
    "jute": {"N": 60, "P": 30, "K": 30},
    "coffee": {"N": 100, "P": 60, "K": 100},
    "coconut": {"N": 100, "P": 40, "K": 200},
    "mango": {"N": 100, "P": 50, "K": 100},
    "banana": {"N": 200, "P": 60, "K": 300},
    "grapes": {"N": 100, "P": 50, "K": 100},
    "apple": {"N": 70, "P": 35, "K": 70},
    "orange": {"N": 80, "P": 40, "K": 80},
    "papaya": {"N": 200, "P": 200, "K": 250},
    "watermelon": {"N": 100, "P": 50, "K": 100},
    "muskmelon": {"N": 80, "P": 40, "K": 80},
    "pomegranate": {"N": 100, "P": 50, "K": 100},
}

# --- Fertilizer nutrient composition ---
FERTILIZER_NUTRIENTS = {
    "Urea": {"N": 0.46, "P": 0, "K": 0},
    "DAP": {"N": 0.18, "P": 0.46, "K": 0},
    "MOP": {"N": 0, "P": 0, "K": 0.60},
    "SSP": {"N": 0, "P": 0.16, "K": 0},
}

# --- Approx fertilizer price (INR/kg) ---
FERTILIZER_PRICES = {
    "Urea": 7,
    "DAP": 27,
    "MOP": 17,
    "SSP": 8,
}


def calculate_fertilizer(crop_name, current_N, current_P, current_K):
    """
    Calculate fertilizer recommendation based on soil vs crop needs.
    """
    crop_key = crop_name.lower().strip().replace(" ", "")

    if crop_key not in CROP_NPK_REQUIREMENTS:
        return {"error": f"Crop '{crop_name}' not found"}

    ideal = CROP_NPK_REQUIREMENTS[crop_key]

    N_def = max(0, ideal["N"] - current_N)
    P_def = max(0, ideal["P"] - current_P)
    K_def = max(0, ideal["K"] - current_K)

    N_exc = max(0, current_N - ideal["N"])
    P_exc = max(0, current_P - ideal["P"])
    K_exc = max(0, current_K - ideal["K"])

    recommendations = []
    total_cost = 0

    if N_def > 0:
        qty = round(N_def / FERTILIZER_NUTRIENTS["Urea"]["N"], 1)
        cost = round(qty * FERTILIZER_PRICES["Urea"])
        total_cost += cost
        recommendations.append(
            {
                "fertilizer": "Urea",
                "quantity_kg_per_hectare": qty,
                "nutrient": f"{N_def} kg Nitrogen",
                "cost_inr": cost,
                "note": "Apply in 2 splits",
            }
        )

    if P_def > 0:
        qty = round(P_def / FERTILIZER_NUTRIENTS["DAP"]["P"], 1)
        cost = round(qty * FERTILIZER_PRICES["DAP"])
        total_cost += cost
        recommendations.append(
            {
                "fertilizer": "DAP",
                "quantity_kg_per_hectare": qty,
                "nutrient": f"{P_def} kg Phosphorus",
                "cost_inr": cost,
                "note": "Apply at sowing",
            }
        )

    if K_def > 0:
        qty = round(K_def / FERTILIZER_NUTRIENTS["MOP"]["K"], 1)
        cost = round(qty * FERTILIZER_PRICES["MOP"])
        total_cost += cost
        recommendations.append(
            {
                "fertilizer": "MOP",
                "quantity_kg_per_hectare": qty,
                "nutrient": f"{K_def} kg Potassium",
                "cost_inr": cost,
                "note": "Apply at sowing",
            }
        )

    warnings = []
    if N_exc > 20:
        warnings.append(f"Excess Nitrogen: {N_exc} kg/ha. Avoid Urea.")
    if P_exc > 10:
        warnings.append(f"Excess Phosphorus: {P_exc} kg/ha. Avoid DAP.")
    if K_exc > 20:
        warnings.append(f"Excess Potassium: {K_exc} kg/ha. Avoid MOP.")

    if not recommendations:
        recommendations.append(
            {"message": "Soil nutrients are sufficient. No major fertilizer needed."}
        )

    return {
        "crop": crop_name,
        "ideal_npk": ideal,
        "current_npk": {"N": current_N, "P": current_P, "K": current_K},
        "deficit": {"N": N_def, "P": P_def, "K": K_def},
        "recommendations": recommendations,
        "warnings": warnings,
        "total_cost_inr_per_hectare": total_cost,
    }
