import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# === Paths ===
CSV_PATH = "data/schemes.csv"
MODEL_DIR = "app/models"

# === Load the CSV ===
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded {len(df)} records from {CSV_PATH}")

# === Check and clean full_text column ===
if 'full_text' not in df.columns:
    raise ValueError("CSV must contain a 'full_text' column for vectorization.")

# Replace NaN or non-string values with empty strings
df['full_text'] = df['full_text'].fillna("").astype(str)

# Optional: drop rows with completely empty text
empty_rows = df['full_text'].str.strip() == ""
if empty_rows.any():
    print(f"[WARN] Found {empty_rows.sum()} empty documents. They will be ignored.")
    df = df[~empty_rows]

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['full_text'].values)

# === Save models ===
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "policy_vectorizer.pkl"))
joblib.dump({"matrix": tfidf_matrix, "df": df}, os.path.join(MODEL_DIR, "policy_tfidf_matrix.pkl"))

print("[INFO] TF-IDF vectorizer and matrix saved successfully!")
print(f"[INFO] Final dataset size after cleaning: {len(df)} rows")
