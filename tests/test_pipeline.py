"""
Unit and integration tests for the Customer Churn & Retention Intelligence system.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import generate_synthetic_data
from src.data.preprocessor import DataPreprocessor
from src.models.churn_predictor import ChurnPredictorModel
from src.clustering.segmentation import CustomerSegmenter
from src.causal.causal_analyzer import CausalInferenceEngine, CustomerFeedbackAnalyzer
from src.retention.recommendation_engine import RetentionRecommendationEngine
from src.pipeline import ChurnRetentionPipeline


def test_synthetic_data_generation():
    df = generate_synthetic_data(n_samples=100, random_seed=42)
    assert len(df) == 100
    assert "churn" in df.columns
    assert "monthly_charges" in df.columns
    assert "customer_feedback" in df.columns
    assert df["churn"].nunique() == 2


def test_feature_engineering_and_preprocessor():
    df = generate_synthetic_data(n_samples=80, random_seed=42)
    prep = DataPreprocessor()
    X = prep.fit_transform(df)
    assert X.shape[0] == 80
    assert X.shape[1] > 15
    assert not np.isnan(X).any()


def test_churn_predictor_training():
    df = generate_synthetic_data(n_samples=150, random_seed=42)
    prep = DataPreprocessor()
    X = prep.fit_transform(df)
    y = df["churn"].values

    model = ChurnPredictorModel()
    metrics = model.train_and_evaluate(X, X, y, y, prep.get_feature_names())
    
    assert "random_forest" in metrics
    assert "gradient_boosting" in metrics
    assert "logistic_regression" in metrics
    
    probs = model.predict_churn_probability(X)
    assert len(probs) == 150
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_clustering_segmenter():
    df = generate_synthetic_data(n_samples=100, random_seed=42)
    prep = DataPreprocessor()
    X = prep.fit_transform(df)

    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    segmenter.fit(X, prep.get_feature_names())
    labels = segmenter.predict(X)

    assert len(labels) == 100
    assert len(np.unique(labels)) <= 4

    pca_coords = segmenter.get_pca_projection(X)
    assert pca_coords.shape == (100, 2)


def test_causal_inference_engine():
    df = generate_synthetic_data(n_samples=200, random_seed=42)
    engine = CausalInferenceEngine(df)
    effects = engine.run_all_causal_interventions()
    
    assert len(effects) >= 3
    for eff in effects:
        assert "ate" in eff
        assert "relative_change_pct" in eff


def test_feedback_nlp_analyzer():
    analyzer = CustomerFeedbackAnalyzer()
    res1 = analyzer.categorize_feedback("The platform is outstanding and customer support is fantastic!")
    assert res1["sentiment"] == "Positive"

    res2 = analyzer.categorize_feedback("Frequent crashes and annoying bugs in dashboard export.")
    assert res2["sentiment"] == "Negative"
    assert "Bugs" in res2["topic"]


def test_retention_recommendation_engine():
    engine = RetentionRecommendationEngine()
    dummy = {
        "customer_id": "CUST-9999",
        "monthly_charges": 199.0,
        "contract_type": "Monthly",
        "support_tickets_count": 4,
        "customer_feedback": "Support is slow",
        "auto_renew": 0
    }
    rec = engine.generate_strategy(dummy, churn_prob=0.82, cluster_id=0, primary_causes=["support_intensity"])
    assert rec["risk_tier"] == "Severe Emergency"
    assert rec["expected_saved_revenue"] > 0
    assert len(rec["targeted_actions"]) > 0
    assert len(rec["incentive_offers"]) > 0


def test_end_to_end_pipeline():
    pipeline = ChurnRetentionPipeline()
    success = pipeline.load_artifacts()
    assert success is True

    df = generate_synthetic_data(n_samples=25, random_seed=42)
    scored = pipeline.process_dataset(df)

    assert "predicted_churn_prob" in scored.columns
    assert "risk_tier" in scored.columns
    assert "segment_name" in scored.columns
    assert "revenue_at_risk" in scored.columns

