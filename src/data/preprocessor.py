"""
Customer Churn & Retention Intelligence - Data Preprocessing & Feature Engineering
Handles data cleaning, normalization, categorical encoding, feature engineering,
and train/test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

NUMERICAL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "team_size",
    "monthly_active_days",
    "usage_trend_30d",
    "days_since_last_login",
    "feature_adoption_rate",
    "support_tickets_count",
    "avg_ticket_resolution_hours",
    "payment_failures_past_6m",
    "nps_score",
    "auto_renew",
    "recent_price_increase",
    # Engineered features
    "inactivity_ratio",
    "support_intensity",
    "charge_per_active_day",
    "engagement_composite"
]

CATEGORICAL_FEATURES = [
    "subscription_plan",
    "contract_type",
    "billing_method"
]


class DataPreprocessor:
    """End-to-end preprocessor for customer subscription data."""
    
    def __init__(self):
        self.column_transformer = None
        self.feature_names = []
        self.scaler = None
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive meaningful behavioral and financial ratios."""
        df_out = df.copy()

        # Inactivity ratio: days since last login relative to 30 days
        df_out["inactivity_ratio"] = np.clip(df_out["days_since_last_login"] / 30.0, 0.0, 3.0)

        # Support intensity: tickets count multiplied by resolution lag
        df_out["support_intensity"] = df_out["support_tickets_count"] * (df_out["avg_ticket_resolution_hours"] / 24.0)

        # Charge per active day (unit cost efficiency for customer)
        safe_active_days = np.maximum(df_out["monthly_active_days"], 1.0)
        df_out["charge_per_active_day"] = df_out["monthly_charges"] / safe_active_days

        # Engagement composite index: (active days / 30) * adoption_rate * usage_trend
        df_out["engagement_composite"] = (
            (df_out["monthly_active_days"] / 30.0) 
            * df_out["feature_adoption_rate"] 
            * np.clip(df_out["usage_trend_30d"], 0.2, 2.0)
        )

        return df_out

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit scaler and encoders on training data."""
        df_eng = self.engineer_features(df)
        
        # Define preprocessor
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), [c for c in NUMERICAL_FEATURES if c in df_eng.columns]),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES)
            ],
            remainder="drop"
        )
        self.column_transformer.fit(df_eng)

        # Extract generated feature names
        num_cols = [c for c in NUMERICAL_FEATURES if c in df_eng.columns]
        cat_encoder = self.column_transformer.named_transformers_["cat"]
        cat_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names = num_cols + cat_cols
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataset using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
        df_eng = self.engineer_features(df)
        return self.column_transformer.transform(df_eng)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def save(self, filepath: str = "src/models/preprocessor.joblib") -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str = "src/models/preprocessor.joblib") -> "DataPreprocessor":
        return joblib.load(filepath)


def prepare_datasets(
    csv_path: str = "data/customer_churn_data.csv",
    test_size: float = 0.20,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, DataPreprocessor]:
    """Load data, engineer features, split into train/test, and return preprocessed arrays."""
    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=df["churn"]
    )
    
    preprocessor = DataPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    y_train = train_df["churn"].values
    y_test = test_df["churn"].values

    return X_train, X_test, y_train, y_test, train_df, test_df, preprocessor


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, tr_df, te_df, prep = prepare_datasets()
    prep.save("src/models/preprocessor.joblib")
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print(f"Features: {prep.get_feature_names()}")

