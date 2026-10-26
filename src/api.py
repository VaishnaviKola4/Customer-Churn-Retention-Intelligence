"""
Customer Churn & Retention Intelligence - FastAPI Production Service
Provides high-throughput REST endpoints for:
- Batch customer churn scoring
- Individual customer retention playbooks & risk explanation
- Behavioral segmentation assignment
- Causal intervention metrics
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import uvicorn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ChurnRetentionPipeline
from src.clustering.segmentation import SEGMENT_PROFILES

app = FastAPI(
    title="RetainPulse AI Churn & Retention API",
    description="Production-ready predictive churn scoring, causal diagnostics, and retention playbook API.",
    version="1.0.0"
)

pipeline = ChurnRetentionPipeline()


@app.on_event("startup")
def startup_event():
    """Load pre-trained artifacts into memory upon API initialization."""
    success = pipeline.load_artifacts()
    if not success:
        print("[Warning] Trained artifacts not found. Initiating training...")
        from src.models.churn_predictor import train_churn_pipeline
        from src.clustering.segmentation import train_and_save_segmenter
        train_churn_pipeline()
        train_and_save_segmenter()
        pipeline.load_artifacts()


class CustomerInput(BaseModel):
    customer_id: str = "CUST-9999"
    subscription_plan: str = "Professional"
    contract_type: str = "Monthly"
    billing_method: str = "Credit Card"
    auto_renew: int = 1
    tenure_months: int = 8
    monthly_charges: float = 149.0
    total_charges: float = 1192.0
    team_size: int = 10
    monthly_active_days: float = 18.0
    usage_trend_30d: float = 0.85
    days_since_last_login: int = 6
    feature_adoption_rate: float = 0.55
    support_tickets_count: int = 2
    avg_ticket_resolution_hours: float = 18.5
    payment_failures_past_6m: int = 0
    recent_price_increase: int = 0
    nps_score: int = 7
    customer_feedback: Optional[str] = "Good features but resolution on some tickets was slow."


class ChurnPredictionResponse(BaseModel):
    customer_id: str
    predicted_churn_prob: float
    risk_tier: str
    cluster_id: int
    segment_name: str
    annual_account_value: float
    revenue_at_risk: float


class DeepDiveResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    segment_name: str
    local_explanations: List[Dict[str, Any]]
    retention_strategy: Dict[str, Any]


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RetainPulse AI API", "version": "1.0.0"}


@app.post("/predict", response_model=List[ChurnPredictionResponse])
def predict_churn_batch(customers: List[CustomerInput]):
    """Score a batch of customers for churn risk and assign behavioral cohorts."""
    try:
        df = pd.DataFrame([c.dict() for c in customers])
        scored = pipeline.process_dataset(df)
        
        results = []
        for _, row in scored.iterrows():
            results.append(ChurnPredictionResponse(
                customer_id=row["customer_id"],
                predicted_churn_prob=float(row["predicted_churn_prob"]),
                risk_tier=row["risk_tier"],
                cluster_id=int(row["cluster_id"]),
                segment_name=row["segment_name"],
                annual_account_value=float(row["annual_value"]),
                revenue_at_risk=float(row["revenue_at_risk"])
            ))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=DeepDiveResponse)
def get_retention_recommendations(customer: CustomerInput):
    """Generate personalized root-cause breakdown and targeted retention action items."""
    try:
        df = pd.DataFrame([customer.dict()])
        scored = pipeline.process_dataset(df)
        deep_dive = pipeline.get_customer_deep_dive(customer.customer_id, scored)

        return DeepDiveResponse(
            customer_id=customer.customer_id,
            churn_probability=float(scored.iloc[0]["predicted_churn_prob"]),
            risk_tier=scored.iloc[0]["risk_tier"],
            segment_name=scored.iloc[0]["segment_name"],
            local_explanations=deep_dive["local_explanations"],
            retention_strategy=deep_dive["retention_strategy"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments")
def list_segments():
    """Retrieve defined behavioral segmentation taxonomy and strategic focuses."""
    return SEGMENT_PROFILES


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

