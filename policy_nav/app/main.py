from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

from .nlp_search import search_policies, get_policy_by_id
from .data_loader import load_policy_data

from sklearn.feature_extraction.text import TfidfVectorizer
from .quantum_ner_model import quantum_ner_text

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

df = pd.read_csv("data/schemes.csv")

# Ensure all text fields are strings
df['title'] = df['title'].fillna('').astype(str)
df['full_text'] = df['full_text'].fillna('').astype(str)

# Combine text for vectorizer
policy_texts = (df['title'] + ": " + df['full_text']).tolist()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer().fit(policy_texts)

# -------- Load vectorizer + TF-IDF matrix -------- #
VECTOR_PATH = "app/models/policy_vectorizer.pkl"
MATRIX_PATH = "app/models/policy_tfidf_matrix.pkl"

try:
    vectorizer = joblib.load(VECTOR_PATH)
    data = joblib.load(MATRIX_PATH)
    tfidf_matrix = data["matrix"]
    df = data["df"]
except Exception as e:
    print(f"Error loading model or data: {e}")
    vectorizer = None
    tfidf_matrix = None
    df = pd.DataFrame()

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ---------------- Load additional data ---------------- #
try:
    df, vectorizer, tfidf_matrix = load_policy_data()
except Exception as e:
    print(f"Error loading data: {e}")
    df = None


# ---------------- Helper functions ---------------- #
def safe_value_counts(column_name):
    if df is not None and column_name in df.columns:
        return df[column_name].value_counts().to_dict()
    return {}

def safe_unique(column_name):
    if df is not None and column_name in df.columns:
        return sorted(df[column_name].dropna().unique())
    return []

# ---------------- Home Page ---------------- #
from fastapi.responses import RedirectResponse

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # 🔒 Require login before showing the dashboard
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    if df is None:
        return HTMLResponse(content="<h1>Data not loaded!</h1>", status_code=500)

    total_policies = len(df)
    categories = safe_value_counts("category")
    states = safe_value_counts("state")
    years = safe_value_counts("year")
    recently_added = (
        df.sort_values("year", ascending=False).head(5).to_dict("records")
        if "full_text" in df.columns
        else []
    )

    # Ensure wordcloud_words is always a list
    wordcloud_words = []
    if "full_text" in df.columns:
        try:
            wordcloud_words = [
                {"text": word, "weight": int(count)}
                for word, count in (
                    df["full_text"]
                    .str.split(expand=True)
                    .stack()
                    .value_counts()
                    .head(50)
                    .items()
                )
            ]
        except Exception:
            wordcloud_words = []

    # 👇 Include username in template context
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": request.session.get("user"),
        "total_policies": total_policies,
        "categories": categories,
        "states": states,
        "years": years,
        "recently_added": recently_added,
        "wordcloud_words": wordcloud_words,
        "results": None
    })

# ---------------- Search ---------------- #
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = search_policies(query, top_k=10) if query else []

    context = {
        "request": request,
        "results": results,
        "query": query,
        "total_policies": len(df) if df is not None else 0,
        "categories": safe_value_counts("category"),
        "states": safe_value_counts("state"),
        "years": safe_value_counts("year"),
        "recently_added": df.tail(5).to_dict("records") if df is not None else [],
        "wordcloud_words": [],
    }

    if df is not None and "full_text" in df.columns:
        context["wordcloud_words"] = [
            {"text": word, "weight": int(count)}
            for word, count in df["full_text"].str.split(expand=True).stack().value_counts().head(50).items()
        ]

    return templates.TemplateResponse("index.html", context)

# ---------------- Categories ---------------- #
@app.get("/categories", response_class=HTMLResponse)
async def categories_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    categories = safe_value_counts("category")
    return templates.TemplateResponse("categories.html", {
        "request": request,
        "categories": categories
    })

# ---------------- Explore ---------------- #
@app.get("/explore", response_class=HTMLResponse)
async def explore_page(request: Request, category: Optional[str] = None):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    filtered_results = []
    if df is not None and category:
        filtered_results = df[df['category'] == category].to_dict('records')

    return templates.TemplateResponse("explore.html", {
        "request": request,
        "states": safe_unique("state"),
        "categories": safe_unique("category"),
        "results": filtered_results,
        "selected_category": category
    })

@app.post("/explore_search", response_class=HTMLResponse)
async def explore_search(
    request: Request,
    query: Optional[str] = Form(None),
    state: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    year: Optional[str] = Form(None)
):
    results = search_policies(query, top_k=20) if query else df.to_dict("records") if df is not None else []

    if df is not None:
        if state:
            results = [r for r in results if r.get("state") == state]
        if category:
            results = [r for r in results if r.get("category") == category]
        if year:
            try:
                year_int = int(year)
                results = [r for r in results if r.get("year") == year_int]
            except ValueError:
                pass

    return templates.TemplateResponse("explore.html", {
        "request": request,
        "results": results,
        "query": query or "",
        "states": safe_unique("state"),
        "categories": safe_unique("category"),
        "selected_state": state or "",
        "selected_category": category or "",
        "selected_year": year or ""
    })

# ---------------- Policy Detail ---------------- #

@app.get("/policy/{policy_id}", response_class=HTMLResponse)
async def policy_detail(request: Request, policy_id: str):
    if df is None or df.empty:
        return HTMLResponse(content="<h1>Data not loaded!</h1>", status_code=500)

    # Fetch the policy
    policy_row = df[df["policy_id"].astype(str) == str(policy_id)]
    if policy_row.empty:
        return HTMLResponse(content="<h1>Policy not found!</h1>", status_code=404)

    policy = policy_row.iloc[0].to_dict()

    # AI summary
    policy['ai_summary'] = textwrap.shorten(policy.get("full_text", ""), width=200, placeholder="...")

    # Keyword histogram (top 10 words)
    words = policy.get("full_text", "").split()
    keyword_hist = pd.Series(words).value_counts().head(10).to_dict()

    # Target audience (static example)
    audience_chart = {"Public": 60, "Private": 40}

    # Similar policies
    similar_policies = df[(df['category'] == policy.get('category')) &
                          (df['policy_id'].astype(str) != str(policy_id))].head(3).to_dict('records')

    return templates.TemplateResponse("policy_detail.html", {
        "request": request,
        "policy": policy,
        "summary": policy.get("full_text", "")[:500],
        "keyword_chart": keyword_hist,
        "audience_chart": audience_chart,
        "similar_policies": similar_policies
    })

@app.get("/quantum_ner", response_class=HTMLResponse)
async def quantum_ner_page(request: Request):
    return templates.TemplateResponse("quantum_ner.html", {"request": request})

@app.post("/quantum_ner", response_class=HTMLResponse)
async def quantum_ner_post(request: Request, query: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if not query:
        return HTMLResponse("<h1>Query is required!</h1>", 400)

    entities, scores = quantum_ner_text(query, vectorizer)

    query_vec = vectorizer.transform([query]).toarray()[0]
    similarities = []
    for i, text in enumerate(policy_texts):
        sim = np.dot(query_vec, vectorizer.transform([text]).toarray()[0])
        if sim > 0.05:
            similarities.append((i, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)

    answers = []
    for idx, _ in similarities[:5]:
        policy = df.iloc[idx]
        answers.append({
            "policy_id": policy['policy_id'],
            "title": policy['title'],
            "full_text": policy['full_text'],
            "state": policy['state'],
            "year": policy['year'],
            "category": policy['category'],
            "status": policy['status'],
            "region": policy['region']
        })

    return templates.TemplateResponse("quantum_ner.html", {
        "request": request,
        "policy_text": query,
        "entities": entities,
        "scores": scores,
        "answers": answers
    })


# ---------------- About ---------------- #
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("about.html", {"request": request})


import json, os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app.add_middleware(SessionMiddleware, secret_key="supersecretkey")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

USERS_FILE = "app/data/users.json"

# ------------- Helper functions -------------
def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump([], f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# ------------- AUTH ROUTES -------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    users = load_users()
    for user in users:
        if user["username"] == username and user["password"] == password:
            request.session["user"] = username
            return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})

@app.post("/register", response_class=HTMLResponse)
async def register_post(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    users = load_users()
    if any(u["username"] == username for u in users):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already exists"})
    users.append({"username": username, "email": email, "password": password})
    save_users(users)
    return RedirectResponse("/login", status_code=303)

@app.get("/forgot_password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request, "message": None, "error": None})

@app.post("/forgot_password", response_class=HTMLResponse)
async def forgot_password_post(request: Request, email: str = Form(...), new_password: str = Form(...)):
    users = load_users()
    for user in users:
        if user["email"] == email:
            user["password"] = new_password
            save_users(users)
            return templates.TemplateResponse("forgot_password.html", {"request": request, "message": "Password updated successfully!", "error": None})
    return templates.TemplateResponse("forgot_password.html", {"request": request, "error": "Email not found", "message": None})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)
