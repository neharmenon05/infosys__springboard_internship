import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import textwrap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "policy_vectorizer.pkl")
MATRIX_PATH = os.path.join(BASE_DIR, "models", "policy_tfidf_matrix.pkl")

vectorizer = joblib.load(MODEL_PATH)
data = joblib.load(MATRIX_PATH)
tfidf_matrix = data["matrix"]
df = data["df"]

def search_policies(query: str, top_k: int = 5):
    """Return top_k policies for a query using TF-IDF cosine similarity."""
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "policy_id": row["policy_id"],
            "title": row["title"],
            "state": row["state"],
            "category": row["category"],
            "year": row["year"],
            "summary": textwrap.shorten(row["full_text"], width=250, placeholder="..."),
            "score": round(sims[idx], 3)
        })
    return results

def get_policy_by_id(policy_id):
    """Fetch policy details by ID."""
    row = df[df['policy_id'] == policy_id].iloc[0]
    return row.to_dict()
