### TASK 2

**DATE**
1) Creating FASTAPI connection [9/7/2025]
2) Connecting it to HTML [10/7/2025]

**Submodule:** `task_2/fastapi_policy/` — REST API for policy search and navigation using FastAPI

This module implements a **FastAPI-based backend service** that exposes endpoints for querying and managing policy data. It is the second step in the internship project, extending the preprocessing work from Task 1 into a deployable API layer.

## Key Features
- **FastAPI Backend** — High-performance Python web framework for APIs  
- **Policy Endpoints** — Query, filter, and retrieve policy information  
- **Template Integration** — Basic HTML templates for rendering responses  
- **Interactive Docs** — Auto-generated Swagger UI and ReDoc at runtime  
- **Modular Design** — Extendable for additional routes or ML integration  

---

## Folder Structure
```
fastapi_policy/
│
├── main.py          # FastAPI application entry point
├── templates/       # HTML templates for rendering responses
├── __pycache__/     # Compiled Python cache files
├── About.md         # Module description
└── README.md
```

---

## Setup Guide

### 1. Create and Activate Virtual Environment
```bash
cd task_2/fastapi_policy
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn jinja2
```

### 3. Run the Application
```bash
uvicorn main:app --reload
```

### 4. Access in Browser
- API root: `http://127.0.0.1:8000/`  
- Swagger UI: `http://127.0.0.1:8000/docs`  
- ReDoc: `http://127.0.0.1:8000/redoc`  

---

## Example Endpoints
- `GET /` — Root endpoint with welcome message  
- `GET /policies` — Retrieve list of policies (extendable with filters)  
- `GET /policies/{id}` — Retrieve details of a specific policy  

---

## Tech Stack
| Component   | Description                          |
|-------------|--------------------------------------|
| FastAPI     | Backend framework for REST APIs      |
| Uvicorn     | ASGI server for running FastAPI apps |
| Jinja2      | Template rendering engine            |
| Python 3.x  | Core programming language            |

---

## Future Enhancements
- Integrate trained ML models from Task 1 for intelligent policy classification  
- Add authentication and role-based access control  
- Connect to a persistent database (PostgreSQL / MongoDB)  
- Extend endpoints for advanced search and analytics  

## Credits
- Infosys Springboard Internship Program  
- FastAPI and Python open-source community  
