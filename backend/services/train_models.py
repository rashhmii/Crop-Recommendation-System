import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_all_models(X_train, y_train):
    os.makedirs("models", exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, solver="lbfgs", multi_class="multinomial", random_state=42
        ),
        "svm": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=1000, max_depth=None, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name}.pkl")
        trained[name] = model

    # Main model: soft-voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("lr", trained["logistic_regression"]),
            ("svm", trained["svm"]),
            ("rf", trained["random_forest"]),
            ("xgb", trained["xgboost"]),
        ],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, "models/ensemble.pkl")
    trained["ensemble"] = ensemble
    return trained


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    train_all_models(X_train, y_train)
