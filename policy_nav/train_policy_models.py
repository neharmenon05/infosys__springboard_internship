import os
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_PATH = "kaggle_dataset"
OUTPUT_CSV = "data/schemes.csv"

def extract_state_from_path(path):
    return os.path.basename(os.path.dirname(path)).replace("-", " ").title()

def extract_year(text):
    match = re.search(r'20\d{2}', text)
    return int(match.group(0)) if match else 2020

def assign_category(text):
    text = text.lower()
    if "education" in text or "student" in text or "fee" in text:
        return "Education"
    elif "women" in text or "girl" in text:
        return "Women Welfare"
    elif "health" in text or "hospital" in text:
        return "Health"
    elif "agriculture" in text or "farmer" in text:
        return "Agriculture"
    elif "transport" in text or "vehicle" in text:
        return "Transport"
    else:
        return "Other"

records = []
for root, dirs, files in os.walk(DATASET_PATH):
    for filename in files:
        if filename.endswith(".txt"):
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                lines = content.split("\n")
                title = lines[0] if lines else "Unknown Title"
                state = extract_state_from_path(filepath)
                year = extract_year(content)
                category = assign_category(content)
                records.append({
                    "policy_id": len(records),
                    "title": title,
                    "full_text": content,
                    "state": state,
                    "year": year,
                    "category": category,
                    "status": "Active",
                    "region": state
                })

df = pd.DataFrame(records)
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Saved CSV to {OUTPUT_CSV}")

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['full_text'].values)

os.makedirs("app/models", exist_ok=True)
joblib.dump(vectorizer, "app/models/policy_vectorizer.pkl")
joblib.dump({"matrix": tfidf_matrix, "df": df}, "app/models/policy_tfidf_matrix.pkl")
print("[INFO] TF-IDF vectorizer and matrix saved successfully!")
