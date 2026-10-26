"""
Customer Churn & Retention Intelligence - Predictive Churn Models
Trains multiple classification models (Random Forest, Gradient Boosting, Logistic Regression),
evaluates them with ROC-AUC, PR-AUC, F1-Score, and provides an ensemble predictor
with individual feature importance explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
import joblib
import json
import os

from src.data.preprocessor import DataPreprocessor, prepare_datasets


class ChurnPredictorModel:
    """Multi-algorithm churn model trainer, evaluator, and prediction engine."""

    def __init__(self):
        self.models: Dict[str, Any] = {
            "random_forest": RandomForestClassifier(
                n_estimators=150, max_depth=8, min_samples_split=6, random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
            ),
            "logistic_regression": LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            )
        }
        self.best_model_name: str = "gradient_boosting"
        self.feature_names: List[str] = []
        self.evaluation_results: Dict[str, Dict[str, float]] = {}
        self.is_trained: bool = False

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train all models, evaluate on test set, and select best performer."""
        self.feature_names = feature_names
        best_score = -1.0

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "pr_auc": float(average_precision_score(y_test, y_proba))
            }
            self.evaluation_results[name] = metrics

            # Select model with highest ROC-AUC
            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                self.best_model_name = name

        self.is_trained = True
        return self.evaluation_results

    def get_best_model(self):
        """Return the top-performing trained model."""
        return self.models[self.best_model_name]

    def predict_churn_probability(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Predict probability of churn."""
        if not self.is_trained:
            raise ValueError("Models are not trained yet.")
        name = model_name or self.best_model_name
        return self.models[name].predict_proba(X)[:, 1]

    def get_feature_importances(self, top_n: int = 12) -> List[Dict[str, Any]]:
        """Extract global feature importances from the best tree-based model."""
        model = self.get_best_model()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return []

        sorted_indices = np.argsort(importances)[::-1]
        results = []
        for idx in sorted_indices[:top_n]:
            results.append({
                "feature": self.feature_names[idx],
                "importance": float(importances[idx])
            })
        return results

    def explain_instance(self, single_features: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Local explanation approximation based on feature deviation from mean
        weighted by model coefficients or tree importance.
        """
        model = self.get_best_model()
        if hasattr(model, "feature_importances_"):
            weights = model.feature_importances_
        else:
            weights = np.abs(model.coef_[0])
        
        # Absolute impact = feature value (normalized) * importance weight
        impacts = single_features * weights
        sorted_indices = np.argsort(np.abs(impacts))[::-1]
        
        explanations = []
        for idx in sorted_indices[:top_n]:
            explanations.append({
                "feature": self.feature_names[idx],
                "direction": "Risk Driver" if impacts[idx] > 0 else "Retention Anchor",
                "impact_score": float(np.abs(impacts[idx]))
            })
        return explanations

    def save(self, model_dir: str = "src/models") -> None:
        """Serialize models and evaluation metrics."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.models, os.path.join(model_dir, "churn_models.joblib"))
        
        metadata = {
            "best_model_name": self.best_model_name,
            "feature_names": self.feature_names,
            "evaluation_results": self.evaluation_results
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, model_dir: str = "src/models") -> "ChurnPredictorModel":
        """Load trained models and metadata."""
        instance = cls()
        models_path = os.path.join(model_dir, "churn_models.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        
        if os.path.exists(models_path) and os.path.exists(metadata_path):
            instance.models = joblib.load(models_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            instance.best_model_name = metadata.get("best_model_name", "gradient_boosting")
            instance.feature_names = metadata.get("feature_names", [])
            instance.evaluation_results = metadata.get("evaluation_results", {})
            instance.is_trained = True
        return instance


def train_churn_pipeline() -> Tuple[ChurnPredictorModel, Dict[str, Any]]:
    """Complete pipeline to load data, fit preprocessor, train models, and save artifacts."""
    X_train, X_test, y_train, y_test, train_df, test_df, preprocessor = prepare_datasets()
    preprocessor.save("src/models/preprocessor.joblib")

    model_engine = ChurnPredictorModel()
    metrics = model_engine.train_and_evaluate(
        X_train, X_test, y_train, y_test, preprocessor.get_feature_names()
    )
    model_engine.save("src/models")
    return model_engine, metrics


if __name__ == "__main__":
    engine, results = train_churn_pipeline()
    print("Model Training & Evaluation Results:")
    for model_name, score_dict in results.items():
        print(f"\n[{model_name.upper()}]")
        for k, v in score_dict.items():
            print(f"  {k}: {v:.4f}")
    print(f"\nBest Performing Model: {engine.best_model_name}")
    print("Top Feature Importances:")
    for fi in engine.get_feature_importances(6):
        print(f"  {fi['feature']}: {fi['importance']:.4f}")

