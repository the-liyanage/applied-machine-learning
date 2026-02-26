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


def main():
    print("--- ML Training Pipeline Starting---")
    config = load_config("configs/config.yaml")

    X_train, X_test, y_train, y_test = load_data(
        test_size = config["data"]["test_size"],
        random_state=config["data"]["random_state"]

    )

    print("---Pipeline Complete ---")

if __name__ == "__main__":
    main()