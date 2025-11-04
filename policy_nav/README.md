# Policy Navigation Assistant (Policy_Nav Module)

**Submodule:** `policy_nav/` — AI-powered policy classification and navigation system using machine learning models and structured datasets

This module implements a machine learning–driven assistant that helps analyze, classify, and navigate government or institutional policies. It integrates **custom-trained models**, **structured datasets**, and **interactive querying** to support policy research and exploration.

---

## Key Features
- **Policy Classification** — Train and evaluate ML models on policy datasets  
- **Dataset Integration** — Includes curated Kaggle datasets for experimentation  
- **Custom Training Scripts** — Build and fine-tune models for policy categorization  
- **Data Preprocessing** — Scripts for cleaning and preparing policy text  
- **Extensible Design** — Easily adapt to new policy domains or datasets  

---

## Folder Structure
```
policy_nav/
│
├── app/                     # Application logic and supporting modules
├── data/                    # Policy datasets and processed files
├── kaggle_dataset/          # External dataset integration
├── train_policy_models.py   # Script to train ML models
├── new.py                   # Experimental script
├── new-2.py                 # Alternate experimental script
├── requirements.txt         # Python dependencies
└── README.md

---

## Setup Guide

**1. Create and Activate Virtual Environment**
```bash
cd policy_nav
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Train Policy Models**
```bash
python train_policy_models.py
```

This script:
- Loads datasets from `data/` or `kaggle_dataset/`  
- Preprocesses and vectorizes policy text  
- Trains classification models for policy categorization  
- Saves trained models for later use  


## Dataset Format
The datasets are expected to follow a structured CSV format with fields such as:

```
policy_id,title,full_text,state,year,category,status,region
1,Education Reform Act,Full text of the policy...,Delhi,2021,Education,Active,North
```

## Tech Stack
| Component         | Description                                |
|-------------------|--------------------------------------------|
| Python            | Core programming language                  |
| scikit-learn      | Machine learning model training            |
| Pandas / NumPy    | Data preprocessing and manipulation        |
| Kaggle Dataset    | Source of policy-related training data     |



## Example Use Cases
- Classify policies into categories such as Education, Poverty, or Healthcare  
- Train models on new datasets for domain-specific policy analysis  
- Explore regional or temporal trends in policy data  

---

## Credits
- Infosys Springboard Internship Program  
- Kaggle Datasets  
- scikit-learn and Python ML ecosystem   
