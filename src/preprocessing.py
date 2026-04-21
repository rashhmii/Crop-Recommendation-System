import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_preprocess(filepath):

    df = pd.read_csv(filepath)
    print(f"Loaded: {df.shape}")

    # Clean
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Feature Engineering
    df['N_P_ratio']           = df['N'] / (df['P'] + 1)
    df['N_K_ratio']           = df['N'] / (df['K'] + 1)
    df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
    df['rainfall_humidity']   = df['rainfall'] * df['humidity'] / 100

    # Encode target
    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['label'])

    os.makedirs('../models', exist_ok=True)
    joblib.dump(le, '../models/le_crop.pkl')
    print(f"Crops: {list(le.classes_)}")

    # Features — season is NOT included, only for display later
    feature_cols = [
        'N', 'P', 'K',
        'temperature', 'humidity', 'ph', 'rainfall',
        'N_P_ratio', 'N_K_ratio',
        'temp_humidity_index', 'rainfall_humidity'
    ]

    X = df[feature_cols]
    y = df['crop_encoded']

    # Scale
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    joblib.dump(scaler, '../models/scaler.pkl')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save
    os.makedirs('../data/processed', exist_ok=True)
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    X_test.to_csv('../data/processed/X_test.csv',   index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv',   index=False)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, le, scaler, feature_cols


if __name__ == "__main__":
    load_and_preprocess('../data/processed/crop_india_merged.csv')