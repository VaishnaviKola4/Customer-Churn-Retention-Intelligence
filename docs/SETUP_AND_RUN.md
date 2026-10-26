# Installation, Configuration, and Deployment Guide

This document provides complete instructions for configuring, deploying, and maintaining **RetainPulse AI** across local development, Docker, and enterprise cloud production environments.

---

## 🛠️ Local Environment Setup

### 1. System Requirements
- Python 3.10, 3.11, or 3.12
- 4GB+ System RAM
- Operating Systems: macOS, Linux (Ubuntu 20.04+), Windows 10/11

### 2. Virtual Environment Initialization
```bash
# Clone the repository
git clone https://github.com/your-username/customer-churn-retention-intelligence.git
cd "customer-churn-retention-intelligence"

# Create a clean virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 3. Dependency Installation
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration & Data Pipeline Execution

### Step 1: Generate or Load Data
To generate a realistic synthetic dataset simulating 3,000 SaaS customer accounts:
```bash
python src/data/generator.py
```
Output: `data/customer_churn_data.csv`

### Step 2: Fit Transformers & Train Machine Learning Models
```bash
# Set Python path to project root
export PYTHONPATH=.   # Linux/macOS
$env:PYTHONPATH="."   # Windows PowerShell

# Train predictive classification models
python -m src.models.churn_predictor
```
This trains:
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `LogisticRegression`
Outputs: `src/models/churn_models.joblib` and `src/models/model_metadata.json`

### Step 3: Train Behavioral Clustering Engine
```bash
python -m src.clustering.segmentation
```
This fits K-Means and 2D PCA transformations and saves `src/models/segmenter.joblib`.

### Step 4: Run Automated Tests
Verify all 8 unit and integration tests pass:
```bash
pytest tests/ -v
```

---

## 🖥️ Running the Applications

### Launch the Streamlit Interactive Dashboard
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Open your browser at: `http://localhost:8501`

### Launch the Production FastAPI Microservice
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

---

## 🐳 Docker Deployment

To package and deploy RetainPulse AI as a lightweight containerized service:

### 1. Build the Docker Image
```bash
docker build -t retainpulse-ai:latest .
```

### 2. Run the Container
```bash
docker run -d -p 8000:8000 -p 8501:8501 --name retainpulse retainpulse-ai:latest
```

---

## 🛡️ Monitoring & Retraining Schedule

In production SaaS environments, customer behavior shifts over time (concept drift). We recommend the following operational rhythm:
1. **Weekly Batch Scoring**: Ingest active telemetry every Sunday night via `pipeline.process_dataset()`.
2. **Monthly Model Recalibration**: Retrain classification weights against newly realized 90-day churn outcomes.
3. **Drift Monitoring**: Track the stability of population feature distributions using Population Stability Index (PSI).

