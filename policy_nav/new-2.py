# generate_wordcloud.py

import pandas as pd
from wordcloud import WordCloud
import os

# -------- Load your data -------- #
DATA_PATH = "app/models/policy_tfidf_matrix.pkl"  # or wherever your df is stored

try:
    import joblib
    data = joblib.load(DATA_PATH)
    df = data.get("df", pd.DataFrame())
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()

# -------- Prepare text for wordcloud -------- #
if "full_text" in df.columns:
    # Combine all policy texts
    text = " ".join(df["full_text"].dropna().astype(str).tolist())
else:
    text = ""

# -------- Generate WordCloud -------- #
if text:
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=100,
        colormap="Blues"
    ).generate(text)
    
    # Make sure static folder exists
    output_dir = "app/static"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the wordcloud image
    output_path = os.path.join(output_dir, "wordcloud.png")
    wc.to_file(output_path)
    print(f"WordCloud saved at {output_path}")
else:
    print("No text data available to generate WordCloud.")
