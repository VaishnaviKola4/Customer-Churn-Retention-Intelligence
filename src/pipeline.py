"""
Customer Churn & Retention Intelligence - Unified Pipeline Orchestrator
Coordinates data ingestion, preprocessing, predictive modeling, clustering,
causal analysis, and recommendation generation into a single callable API.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.data.preprocessor import DataPreprocessor
from src.models.churn_predictor import ChurnPredictorModel
from src.clustering.segmentation import CustomerSegmenter, SEGMENT_PROFILES
from src.causal.causal_analyzer import CausalInferenceEngine, CustomerFeedbackAnalyzer
from src.retention.recommendation_engine import RetentionRecommendationEngine


class ChurnRetentionPipeline:
    """Unified engine driving all churn intelligence modules."""

    def __init__(self, models_dir: str = "src/models"):
        self.models_dir = models_dir
        self.preprocessor: Optional[DataPreprocessor] = None
        self.model_engine: Optional[ChurnPredictorModel] = None
        self.segmenter: Optional[CustomerSegmenter] = None
        self.recommendation_engine = RetentionRecommendationEngine()
        self.feedback_analyzer = CustomerFeedbackAnalyzer()

    def load_artifacts(self) -> bool:
        """Load fitted models, preprocessor, and segmenter from disk."""
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
        seg_path = os.path.join(self.models_dir, "segmenter.joblib")

        if not os.path.exists(prep_path) or not os.path.exists(seg_path):
            return False

        self.preprocessor = DataPreprocessor.load(prep_path)
        self.model_engine = ChurnPredictorModel.load(self.models_dir)
        self.segmenter = CustomerSegmenter.load(seg_path)
        return True

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run end-to-end inference over an entire customer dataset:
        1. Feature transformation
        2. Churn probability scoring
        3. Behavioral cluster assignment
        4. NLP topic tagging
        5. Value-at-risk computation
        """
        if self.preprocessor is None or self.model_engine is None or self.segmenter is None:
            if not self.load_artifacts():
                raise RuntimeError("Models and preprocessors not initialized. Train pipeline first.")

        df_out = df.copy()
        X = self.preprocessor.transform(df_out)

        # 1. Churn Predictions
        churn_probs = self.model_engine.predict_churn_probability(X)
        df_out["predicted_churn_prob"] = np.round(churn_probs, 4)
        df_out["predicted_churn"] = (churn_probs >= 0.50).astype(int)
        
        # Risk Tiers
        df_out["risk_tier"] = [
            self.recommendation_engine.evaluate_customer_risk_tier(p) for p in churn_probs
        ]

        # 2. Clusters
        cluster_labels = self.segmenter.predict(X)
        df_out["cluster_id"] = cluster_labels
        df_out["segment_name"] = [
            SEGMENT_PROFILES.get(c, {}).get("name", f"Segment {c}") for c in cluster_labels
        ]

        # 2D PCA for visual mapping
        pca_coords = self.segmenter.get_pca_projection(X)
        df_out["pca_x"] = np.round(pca_coords[:, 0], 3)
        df_out["pca_y"] = np.round(pca_coords[:, 1], 3)

        # 3. Financial Exposure
        df_out["annual_value"] = np.round(df_out["monthly_charges"] * 12.0, 2)
        df_out["revenue_at_risk"] = np.round(df_out["annual_value"] * df_out["predicted_churn_prob"], 2)

        # 4. Sentiment Topic
        if "customer_feedback" in df_out.columns:
            topics = []
            sentiments = []
            for text in df_out["customer_feedback"]:
                res = self.feedback_analyzer.categorize_feedback(str(text))
                topics.append(res["topic"])
                sentiments.append(res["sentiment"])
            df_out["feedback_topic"] = topics
            df_out["feedback_sentiment"] = sentiments

        return df_out

    def get_customer_deep_dive(self, customer_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Retrieve full 360-degree retention intelligence profile for a single customer."""
        matches = df[df["customer_id"] == customer_id]
        if matches.empty:
            raise ValueError(f"Customer ID {customer_id} not found.")

        row = matches.iloc[0].to_dict()
        churn_prob = float(row.get("predicted_churn_prob", 0.5))
        cluster_id = int(row.get("cluster_id", 0))

        # Local explanation
        X_single = self.preprocessor.transform(pd.DataFrame([row]))
        local_explanations = self.model_engine.explain_instance(X_single[0], top_n=5)

        # Recommendations
        primary_causes = [exp["feature"] for exp in local_explanations if exp["direction"] == "Risk Driver"]
        strategy = self.recommendation_engine.generate_strategy(
            customer_record=row,
            churn_prob=churn_prob,
            cluster_id=cluster_id,
            primary_causes=primary_causes
        )

        return {
            "customer_profile": row,
            "local_explanations": local_explanations,
            "retention_strategy": strategy,
            "cluster_info": SEGMENT_PROFILES.get(cluster_id, {})
        }

