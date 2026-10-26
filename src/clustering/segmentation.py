"""
Customer Churn & Retention Intelligence - Customer Segmentation & Clustering
Performs unsupervised clustering (K-Means and PCA decomposition) to segment customers
into distinct behavioral and risk cohorts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import json
import os

SEGMENT_PROFILES = {
    0: {
        "name": "Disengaged High-Touch Users",
        "description": "High ticket volume with steep usage drops. Customers struggling with support bottlenecks and reliability.",
        "risk_level": "Critical Risk",
        "strategic_focus": "Support escalation, SLA turnaround guarantees, executive sponsor check-in."
    },
    1: {
        "name": "Healthy Power Accounts",
        "description": "High active days, strong feature adoption, annual contracts, and high NPS. Solid retention anchors.",
        "risk_level": "Low Risk / Loyal",
        "strategic_focus": "Account expansion, advocacy, upsell to enterprise tiers."
    },
    2: {
        "name": "Passive Under-Utilizers",
        "description": "Infrequent logins, low feature adoption, high charge-per-active-day ratio. High risk of silent churn.",
        "risk_level": "High Risk (Silent)",
        "strategic_focus": "Automated workflow onboarding, targeted feature walk-throughs, usage nudges."
    },
    3: {
        "name": "Price & Billing Sensitive",
        "description": "Monthly contract, recent price increase sensitivity, or recurring payment failures.",
        "risk_level": "Moderate-to-High Risk",
        "strategic_focus": "Annual billing switch discount, automated billing update prompts, grace period extension."
    }
}


class CustomerSegmenter:
    """Clustering engine to segment customer base into actionable behavioral cohorts."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.pca = PCA(n_components=2, random_state=random_state)
        self.feature_columns: List[str] = []
        self.cluster_profiles: Dict[int, Dict[str, Any]] = SEGMENT_PROFILES
        self.is_fitted: bool = False

    def fit(self, features_matrix: np.ndarray, feature_names: List[str]) -> "CustomerSegmenter":
        """Fit K-Means clustering and 2D PCA representation."""
        self.feature_columns = feature_names
        self.kmeans.fit(features_matrix)
        self.pca.fit(features_matrix)
        self.is_fitted = True
        return self

    def predict(self, features_matrix: np.ndarray) -> np.ndarray:
        """Assign clusters to observations."""
        return self.kmeans.predict(features_matrix)

    def get_pca_projection(self, features_matrix: np.ndarray) -> np.ndarray:
        """Project features onto 2D space for intuitive scatter visualization."""
        return self.pca.transform(features_matrix)

    def get_cluster_summary(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        """Generate statistical summary of each cluster from the raw dataframe."""
        df_temp = df.copy()
        df_temp["cluster"] = cluster_labels
        df_temp["segment_name"] = df_temp["cluster"].map(lambda c: SEGMENT_PROFILES.get(c, {}).get("name", f"Cluster {c}"))
        df_temp["risk_level"] = df_temp["cluster"].map(lambda c: SEGMENT_PROFILES.get(c, {}).get("risk_level", "Unknown"))

        agg_dict = {
            "customer_id": "count",
            "churn": "mean",
            "monthly_charges": "mean",
            "monthly_active_days": "mean",
            "feature_adoption_rate": "mean",
            "days_since_last_login": "mean",
            "support_tickets_count": "mean",
            "nps_score": "mean"
        }
        
        # Only aggregate existing columns
        agg_cols = {k: v for k, v in agg_dict.items() if k in df_temp.columns}
        summary = df_temp.groupby(["cluster", "segment_name", "risk_level"]).agg(agg_cols).reset_index()
        summary = summary.rename(columns={"customer_id": "customer_count", "churn": "churn_rate"})
        return summary

    def save(self, filepath: str = "src/models/segmenter.joblib") -> None:
        """Save clustering model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "src/models/segmenter.joblib") -> "CustomerSegmenter":
        return joblib.load(filepath)


def train_and_save_segmenter(data_path: str = "data/customer_churn_data.csv") -> CustomerSegmenter:
    from src.data.preprocessor import DataPreprocessor
    df = pd.read_csv(data_path)
    prep = DataPreprocessor.load("src/models/preprocessor.joblib")
    X = prep.transform(df)

    segmenter = CustomerSegmenter(n_clusters=4)
    segmenter.fit(X, prep.get_feature_names())
    segmenter.save("src/models/segmenter.joblib")
    return segmenter


if __name__ == "__main__":
    train_and_save_segmenter()
    segmenter = CustomerSegmenter.load("src/models/segmenter.joblib")
    df = pd.read_csv("data/customer_churn_data.csv")
    from src.data.preprocessor import DataPreprocessor
    prep = DataPreprocessor.load("src/models/preprocessor.joblib")
    X = prep.transform(df)
    labels = segmenter.predict(X)
    summary = segmenter.get_cluster_summary(df, labels)
    print("Cluster Analysis Summary:")
    print(summary.to_string())

