import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _score_in_range(value, low, high):
    """Returns 1.0 inside range, decays smoothly outside."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, 1 - (low - value) / max(low, 1))
    return max(0.0, 1 - (value - high) / max(high, 1))


def compute_soil_fertility_score(row):
    # Weights from notes (total 100)
    weights = {"ph": 25, "N": 20, "P": 15, "K": 15, "temperature": 15, "humidity": 10}
    # Simple ideal ranges
    ideal = {
        "ph": (6.0, 7.5),
        "N": (40, 120),
        "P": (20, 80),
        "K": (20, 120),
        "temperature": (20, 32),
        "humidity": (50, 80),
    }

    score = 0.0
    for col, w in weights.items():
        low, high = ideal[col]
        score += _score_in_range(row[col], low, high) * w

    if score >= 80:
        label = "excellent"
    elif score >= 65:
        label = "good"
    elif score >= 50:
        label = "moderate"
    elif score >= 35:
        label = "poor"
    else:
        label = "critical"
    return round(score, 2), label


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.fillna(df.mean(numeric_only=True))

    # Encode season and crop label
    season_le = LabelEncoder()
    crop_le = LabelEncoder()
    df["season_encoded"] = season_le.fit_transform(df["season"])
    df["crop_label_encoded"] = crop_le.fit_transform(df["label"])

    # Soil fertility score
    fertility = df.apply(compute_soil_fertility_score, axis=1, result_type="expand")
    fertility.columns = ["soil_fertility_score", "soil_fertility_label"]
    df = pd.concat([df, fertility], axis=1)

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "season_encoded"]
    X = df[feature_cols]
    y = df["crop_label_encoded"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(season_le, "models/le_season.pkl")
    joblib.dump(crop_le, "models/le_crop.pkl")

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    df.to_csv("data/processed/crop_with_fertility.csv", index=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Change path as needed
    load_and_preprocess("C:/Users/Rashmi S/Desktop/PROJECT/Crop-Recommendation-System/backend/data/processed/crop_india_merged.csv")
