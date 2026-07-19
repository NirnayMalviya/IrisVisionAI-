"""
persistence.py
----------------
Saves and loads the best-performing trained model (plus its scaler) to
disk with joblib, versioned by a timestamp so multiple training runs
don't silently overwrite each other's artifacts.
"""

import os
import glob
import time
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def save_best_model(model, scaler, model_name: str, accuracy: float, use_scaling: bool):
    """
    Persists the given model (+ scaler, if any) to the models/ directory,
    versioned with a timestamp. Returns the saved artifact's file path.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = model_name.replace(" ", "_").lower()
    filename = f"{safe_name}_{timestamp}.joblib"
    path = os.path.join(MODELS_DIR, filename)

    payload = {
        "model": model,
        "scaler": scaler,
        "model_name": model_name,
        "accuracy": accuracy,
        "use_scaling": use_scaling,
        "timestamp": timestamp,
    }
    joblib.dump(payload, path)
    return path


def list_saved_models():
    """Returns saved model artifact paths, most recent first."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
    return sorted(paths, key=os.path.getmtime, reverse=True)


def load_model(path: str):
    """Loads a previously saved model payload dict from disk."""
    return joblib.load(path)
