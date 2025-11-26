from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from quantum_model import predict_score
from io import StringIO

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import os

# ---------- Load Model + Data ----------
MODEL_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

try:
    vectorizer = joblib.load(MODEL_PATH)
    data = joblib.load(MATRIX_PATH)
    tfidf_matrix = data["matrix"]
    df = data["df"]
    print("Model and data loaded successfully.")
except Exception as e:
    print("Error loading model or data:", e)
    vectorizer = None
    tfidf_matrix = None
    df = pd.DataFrame()

# ---------- FastAPI App Setup ----------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------- Search Function ----------
def search_policies(query: str, top_k: int = 5, domain: str = None, region: str = None):
    if vectorizer is None or tfidf_matrix is None or df.empty:
        return []

    filtered_df = df.copy()
    if domain and "domain" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["domain"].str.lower() == domain.lower()]
    if region:
        filtered_df = filtered_df[filtered_df["region"].str.lower() == region.lower()]
    if filtered_df.empty:
        return []

    filtered_indices = filtered_df.index.tolist()
    try:
        filtered_matrix = tfidf_matrix[filtered_indices]
    except Exception as e:
        print("Error slicing TF-IDF matrix:", e)
        return []

    try:
        query_vec = vectorizer.transform([query.lower()])
        sims = cosine_similarity(query_vec, filtered_matrix).flatten()
    except Exception as e:
        print("Error computing similarity:", e)
        return []

    results = []
    for i, row in filtered_df.iterrows():
        try:
            results.append({
                "title": str(row.get("title", "Untitled")),
                "policy_id": str(row.get("policy_id", "Unknown")),
                "region": str(row.get("region", "Unknown")),
                "year": int(row.get("year", 0)),
                "status": str(row.get("status", "Unknown")),
                "summary": textwrap.shorten(str(row.get("full_text", "")), width=250, placeholder="..."),
                "score": float(round(float(sims[i]), 3))
            })
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results

# ---------- Quantum Search ----------
def quantum_search(query: str, top_k: int = 5, region: str = None):
    if vectorizer is None or tfidf_matrix is None or df.empty:
        return []

    filtered_df = df.copy()
    if region:
        filtered_df = filtered_df[filtered_df["region"].str.lower() == region.lower()]
    if filtered_df.empty:
        return []

    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = []
    for i, row in filtered_df.iterrows():
        try:
            policy_vec = vectorizer.transform([str(row["full_text"]).lower()]).toarray()[0]
            quantum_score = predict_score(policy_vec)
            results.append({
                "title": row.get("title", "Untitled"),
                "policy_id": row.get("policy_id", "Unknown"),
                "region": row.get("region", "Unknown"),
                "year": int(row.get("year", 0)),
                "status": row.get("status", "Unknown"),
                "summary": textwrap.shorten(str(row.get("full_text", "")), width=250, placeholder="..."),
                "score": round(float(sims[i]), 3),
                "quantum_score": round(float(quantum_score), 3)
            })
        except Exception as e:
            print(f"Error in quantum scoring: {e}")

    results = sorted(results, key=lambda x: x["quantum_score"], reverse=True)[:top_k]
    return results

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/search_education", response_class=HTMLResponse)
async def search_education(request: Request, query: str = Form(...), region: str = Form("")):
    results = search_policies(query, domain="education", region=region)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

@app.post("/search_poverty", response_class=HTMLResponse)
async def search_poverty(request: Request, query: str = Form(...), region: str = Form("")):
    results = search_policies(query, domain="poverty", region=region)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

@app.post("/search_quantum", response_class=HTMLResponse)
async def search_quantum_route(request: Request, query: str = Form(...), region: str = Form("")):
    results = quantum_search(query, region=region)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

@app.post("/export_csv")
async def export_csv(query: str = Form(...)):
    results = search_policies(query, top_k=20)
    if not results:
        return HTMLResponse(content="No results to export.", status_code=404)

    df_export = pd.DataFrame(results)
    stream = StringIO()
    df_export.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=policy_results.csv"}
    )

@app.post("/voice_search", response_class=HTMLResponse)
async def voice_search(request: Request, query: str = Form(...)):
    results = search_policies(query, top_k=5)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "query": query
    })

@app.get("/debug", response_class=HTMLResponse)
async def debug(request: Request):
    dummy_results = [{
        "title": "Sample Policy",
        "policy_id": "P001",
        "region": "India",
        "year": 2023,
        "status": "Active",
        "summary": "This is a sample policy summary for testing.",
        "score": 0.95
    }]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": dummy_results,
        "query": "test"
    })

@app.get("/test_search", response_class=HTMLResponse)
async def test_search(request: Request):
    query = "digital education access in rural India"
    region = "India"
    results = search_policies(query, domain="education", region=region)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

# ---------- Run App ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)