import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def run_kfold(model, X, y, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    model_names = ["logistic_regression", "svm", "random_forest", "xgboost", "ensemble"]
    rows = []

    for name in model_names:
        model = joblib.load(f"models/{name}.pkl")
        y_pred = model.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_pred))
        kfold_mean, kfold_std = run_kfold(model, X_train, y_train, folds=5)
        rows.append(
            {
                "model": name,
                "test_accuracy": round(test_acc, 4),
                "kfold_5_mean": round(kfold_mean, 4),
                "kfold_5_std": round(kfold_std, 4),
            }
        )

    summary = pd.DataFrame(rows).sort_values(by="test_accuracy", ascending=False)
    print("\nAccuracies :")
    print(summary.to_string(index=False))
