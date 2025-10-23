import os
import pandas as pd
import joblib

def load_policy_data(dataset_path=None, models_path=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ---------------- Default paths ---------------- #
    if dataset_path is None:
        dataset_path = os.path.join(BASE_DIR, "..", "data", "schemes.csv")
    if models_path is None:
        models_path = os.path.join(BASE_DIR, "models")
    
    dataset_path = os.path.abspath(dataset_path)
    models_path = os.path.abspath(models_path)

    # ---------------- Load CSV ---------------- #
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CSV file not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    # Ensure required columns exist
    for col in ["policy_id", "title", "full_text", "state", "year", "category", "status", "region"]:
        if col not in df.columns:
            df[col] = None  # Fill missing columns with None
    
    # ---------------- Load models ---------------- #
    vectorizer_file = os.path.join(models_path, "policy_vectorizer.pkl")
    tfidf_matrix_file = os.path.join(models_path, "policy_tfidf_matrix.pkl")
    
    if not os.path.exists(vectorizer_file) or not os.path.exists(tfidf_matrix_file):
        raise FileNotFoundError("Vectorizer or TF-IDF matrix not found in models folder")
    
    vectorizer = joblib.load(vectorizer_file)
    tfidf_data = joblib.load(tfidf_matrix_file)
    tfidf_matrix = tfidf_data["matrix"]

    return df, vectorizer, tfidf_matrix
