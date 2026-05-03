YIELD_TABLE = {
    "rice": {"min": 15, "max": 30, "avg": 22},
    "wheat": {"min": 12, "max": 25, "avg": 18},
    "maize": {"min": 20, "max": 40, "avg": 28},
    "chickpea": {"min": 6, "max": 12, "avg": 9},
    "kidneybeans": {"min": 8, "max": 15, "avg": 11},
    "pigeonpeas": {"min": 5, "max": 10, "avg": 7},
    "mothbeans": {"min": 4, "max": 8, "avg": 6},
    "mungbean": {"min": 5, "max": 10, "avg": 7},
    "blackgram": {"min": 4, "max": 9, "avg": 6},
    "lentil": {"min": 6, "max": 12, "avg": 9},
    "pomegranate": {"min": 80, "max": 150, "avg": 110},
    "banana": {"min": 200, "max": 350, "avg": 280},
    "mango": {"min": 50, "max": 120, "avg": 80},
    "grapes": {"min": 100, "max": 200, "avg": 150},
    "watermelon": {"min": 100, "max": 200, "avg": 150},
    "muskmelon": {"min": 80, "max": 160, "avg": 120},
    "apple": {"min": 60, "max": 120, "avg": 90},
    "orange": {"min": 80, "max": 150, "avg": 110},
    "papaya": {"min": 150, "max": 300, "avg": 220},
    "coconut": {"min": 40, "max": 80, "avg": 60},
    "cotton": {"min": 8, "max": 18, "avg": 12},
    "jute": {"min": 20, "max": 35, "avg": 28},
    "coffee": {"min": 5, "max": 12, "avg": 8},
}


def estimate_yield(crop, n, rainfall, ph, temperature):
    crop_key = crop.strip().lower().replace(" ", "")
    if crop_key not in YIELD_TABLE:
        return {"crop": crop, "estimated_yield": "N/A", "unit": "quintals/acre"}

    base = YIELD_TABLE[crop_key]
    modifier = 1.0

    # Simple rule-based adjustments
    if 60 <= n <= 120:
        modifier += 0.10
    elif n < 20:
        modifier -= 0.20

    if 100 <= rainfall <= 200:
        modifier += 0.10
    elif rainfall < 50:
        modifier -= 0.25
    elif rainfall > 300:
        modifier -= 0.15

    if 5.5 <= ph <= 7.5:
        modifier += 0.10
    elif ph < 4.5 or ph > 8.5:
        modifier -= 0.20

    if 20 <= temperature <= 30:
        modifier += 0.05
    elif temperature < 10 or temperature > 40:
        modifier -= 0.20

    estimated = round(base["avg"] * modifier, 1)
    estimated = max(base["min"], min(base["max"], estimated))

    return {
        "crop": crop_key,
        "estimated_yield": estimated,
        "range": f'{base["min"]}-{base["max"]} quintals/acre',
        "unit": "quintals/acre",
    }


if __name__ == "__main__":
    crop = input("Crop name: ")
    n = float(input("N: "))
    rainfall = float(input("Rainfall: "))
    ph = float(input("pH: "))
    temperature = float(input("Temperature: "))

    print(estimate_yield(crop, n, rainfall, ph, temperature))
