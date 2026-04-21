import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import joblib
import os
import time

def train_all_models(X_train, y_train):

    os.makedirs('../models', exist_ok=True)

    # --- Define 4 Base Models ---
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        ),
        'svm': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            verbosity=0
        )
    }

    # --- Train and Save Each Model ---
    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - start, 2)
        joblib.dump(model, f'../models/{name}.pkl')
        trained[name] = model
        print(f"  Done in {elapsed}s → saved {name}.pkl")

    # --- Build Voting Ensemble from all 4 ---
    print("\nBuilding Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr',  trained['logistic_regression']),
            ('svm', trained['svm']),
            ('rf',  trained['random_forest']),
            ('xgb', trained['xgboost'])
        ],
        voting='soft',    # uses predicted probabilities
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, '../models/ensemble.pkl')
    print("  Ensemble saved → ensemble.pkl")

    trained['ensemble'] = ensemble

    print("\n✅ All models trained and saved!")
    return trained


if __name__ == "__main__":
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()

    print(f"X_train: {X_train.shape} | Classes: {y_train.nunique()}")
    train_all_models(X_train, y_train)