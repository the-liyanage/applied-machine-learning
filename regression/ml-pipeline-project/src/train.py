import json
import yaml
import joblib 
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_config(path: str) -> dict:
    """Load YAML config - all params from config, not hardcoded"""
    with open(path) as f:
        return yaml.safe_load(f)
    

def load_data(test_size: float, random_state: int):
    """Load and split data. In real projects: replce with your data source."""
    print("[1/4] Loading data...")
    X, y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split (
        X,y,
        test_size = test_size,
        random_state = random_state
    )

    print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_params: dict):
    """Train the model. Seperated function = easy to test & swap"""
    print("[2/4] Training model...")
    model = RandomForestClassifier(**model_params)       # unpack the dict RandomForestClassifier(n_es..., max_dep..)
    model.fit(X_train, y_train)
    print(f"    Trained {model_params['n_estimators']} trees")
    return model

def evaluate_model(model, X_test, y_test) -> dict:
    """ Evaluate and return metrics"""
    print("[3/4] Evaluating model...")
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred),4),
        "n_test_samples":len(y_test),
    }
    print(f"    Accuracy: {metrics['accuracy']:.2%}")
    return metrics

def save_artifacts(model, metrics:dict, model_path: str, metrics_path: str):
    """ Save model and metrics - the key production step"""
    print("[4/4] Saving artifacts...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Model saved: {model_path}")
    print(f"   Metrics saved: {metrics_path} ")



def main():
    print("--- ML Training Pipeline Starting---")
    config = load_config("configs/config.yaml")

    X_train, X_test, y_train, y_test = load_data(
        test_size = config["data"]["test_size"],
        random_state=config["data"]["random_state"]

    )

    model = train_model(X_train, y_train, config["model"]["params"])
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(
        model, 
        metrics,
        config["output"]["model_path"],
        config["output"]["metrics_path"]
    )

    print("---Pipeline Complete ---")

if __name__ == "__main__":
    main()