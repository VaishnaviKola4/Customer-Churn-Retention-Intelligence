"""
Customer Churn & Retention Intelligence - Causal Analysis & Feedback NLP Engine
Determines underlying root causes of churn through:
1. Causal Impact & Treatment Effects (Average Treatment Effect using Inverse Propensity Weighting / Doubly Robust concepts)
2. NLP sentiment and thematic pain point categorization on customer feedback strings.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class CausalInferenceEngine:
    """
    Performs causal effect estimations on customer interventions and pain points.
    Computes Average Treatment Effect (ATE) for actionable levers:
    - Annual vs Monthly Contract Commitment
    - Price Increase Impact
    - Active Support SLA Resolution (< 24 hrs vs > 24 hrs)
    - Proactive Auto-Renewal Configuration
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def estimate_treatment_effect(
        self,
        treatment_col: str,
        treatment_val: Any = 1,
        control_val: Any = 0,
        outcome_col: str = "churn"
    ) -> Dict[str, Any]:
        """
        Estimate Average Treatment Effect (ATE) using Propensity Score Weighting (IPW).
        ATE = E[Y | T=1] - E[Y | T=0] adjusted for confounding variables.
        """
        df = self.data.copy()
        
        # Confounders
        confounder_cols = [
            "tenure_months", "monthly_charges", "team_size",
            "monthly_active_days", "feature_adoption_rate", "nps_score"
        ]
        confounder_cols = [c for c in confounder_cols if c in df.columns and c != treatment_col]
        
        # Binary treatment indicator
        df["treatment_bin"] = (df[treatment_col] == treatment_val).astype(int)
        
        # Check support
        t_counts = df["treatment_bin"].value_counts()
        if len(t_counts) < 2 or (df["treatment_bin"] == 1).sum() < 20 or (df["treatment_bin"] == 0).sum() < 20:
            return {
                "treatment": treatment_col,
                "ate": 0.0,
                "relative_risk_reduction_pct": 0.0,
                "interpretation": "Insufficient sample support for causal estimation."
            }

        # 1. Fit Propensity Model: P(T=1 | X)
        propensity_model = LogisticRegression(max_iter=500, random_state=42)
        propensity_model.fit(df[confounder_cols], df["treatment_bin"])
        propensity_scores = propensity_model.predict_proba(df[confounder_cols])[:, 1]
        
        # Clip propensity to avoid extreme weights
        propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
        
        # 2. Inverse Propensity Weights
        weights_t1 = df["treatment_bin"] / propensity_scores
        weights_t0 = (1 - df["treatment_bin"]) / (1 - propensity_scores)
        
        # 3. Weighted Average Outcomes
        y = df[outcome_col].values
        mu_1 = np.sum(weights_t1 * y) / np.sum(weights_t1)
        mu_0 = np.sum(weights_t0 * y) / np.sum(weights_t0)
        
        ate = float(mu_1 - mu_0)
        baseline_rate = float(mu_0) if mu_0 > 0 else 0.001
        rel_change = (ate / baseline_rate) * 100.0

        return {
            "treatment": treatment_col,
            "treated_churn_rate_adj": round(float(mu_1), 4),
            "control_churn_rate_adj": round(float(mu_0), 4),
            "ate": round(ate, 4),
            "relative_change_pct": round(rel_change, 2),
            "confounders_adjusted": confounder_cols
        }

    def run_all_causal_interventions(self) -> List[Dict[str, Any]]:
        """Run standard suite of causal hypotheses on the subscription dataset."""
        results = []
        
        # 1. Contract Type: Monthly vs Long-term
        df_temp = self.data.copy()
        df_temp["is_monthly_contract"] = (df_temp["contract_type"] == "Monthly").astype(int)
        sub_engine = CausalInferenceEngine(df_temp)
        res1 = sub_engine.estimate_treatment_effect("is_monthly_contract", 1, 0)
        res1["hypothesis"] = "Impact of Monthly Contract vs Long-Term (Annual/2-Year) on Churn"
        res1["recommendation"] = "Incentivize annual contract commitments via 15-20% discounts to lock in churn reduction."
        results.append(res1)

        # 2. Price Increase
        res2 = self.estimate_treatment_effect("recent_price_increase", 1, 0)
        res2["hypothesis"] = "Impact of Recent Price Increase on Churn"
        res2["recommendation"] = "Grandfather long-term active tiers and bundle newly launched features with fee updates."
        results.append(res2)

        # 3. Support Resolution Lag (> 24 hours vs <= 24 hours)
        df_temp["slow_support_resolution"] = (df_temp["avg_ticket_resolution_hours"] > 24.0).astype(int)
        sub_engine3 = CausalInferenceEngine(df_temp)
        res3 = sub_engine3.estimate_treatment_effect("slow_support_resolution", 1, 0)
        res3["hypothesis"] = "Impact of Slow Support Resolution (>24 hrs) on Churn"
        res3["recommendation"] = "Implement SLA alerting and escalate at-risk customer tickets directly to senior engineering."
        results.append(res3)

        # 4. Auto-renew Enabled
        res4 = self.estimate_treatment_effect("auto_renew", 1, 0)
        res4["hypothesis"] = "Impact of Auto-Renewal Enrollment on Churn"
        res4["recommendation"] = "Introduce 1-click renewal authorization with card-updater integrations."
        results.append(res4)

        return results


class CustomerFeedbackAnalyzer:
    """Natural Language Processing and Sentiment Categorizer for unstructured feedback."""

    KEYWORDS_TOPICS = {
        "Pricing & Cost": ["price", "expensive", "cost", "hike", "tier", "subscription", "roi", "dollar", "bill"],
        "Customer Support": ["support", "ticket", "response", "hours", "unhelpful", "agent", "resolution", "wait"],
        "Product Bugs & Stability": ["crash", "lag", "bug", "broken", "timeout", "issue", "performance", "error", "sync"],
        "Missing Features & Integrations": ["missing", "integration", "slack", "crm", "workflow", "feature", "export", "report"],
        "High Satisfaction & Value": ["outstanding", "seamless", "productivity", "fantastic", "helpful", "great", "love"]
    }

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")

    def categorize_feedback(self, text: str) -> Dict[str, Any]:
        """Categorize customer feedback into pain points and calculate sentiment tone."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {"topic": "General", "sentiment": "Neutral", "score": 0.0}

        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        
        # Topic matching
        scores = {}
        for topic, keywords in self.KEYWORDS_TOPICS.items():
            count = sum(1 for kw in keywords if kw in cleaned)
            scores[topic] = count
            
        best_topic = max(scores, key=scores.get)
        if scores[best_topic] == 0:
            best_topic = "General Feedback"

        # Simple polarity heuristic
        neg_indicators = ["hard", "expensive", "unhelpful", "deteriorated", "crash", "lag", "broken", "bug", "disrupting", "missing", "fails"]
        pos_indicators = ["outstanding", "seamless", "productivity", "intuitive", "reliable", "fantastic", "helpful", "top notch"]

        neg_count = sum(1 for w in neg_indicators if w in cleaned)
        pos_count = sum(1 for w in pos_indicators if w in cleaned)

        if pos_count > neg_count:
            sentiment = "Positive"
            sentiment_score = 0.8
        elif neg_count > pos_count:
            sentiment = "Negative"
            sentiment_score = -0.8
        else:
            sentiment = "Neutral"
            sentiment_score = 0.0

        return {
            "topic": best_topic,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score
        }

    def analyze_feedback_corpus(self, df: pd.DataFrame, text_col: str = "customer_feedback") -> pd.DataFrame:
        """Process an entire column of feedback and summarize topic distributions."""
        if text_col not in df.columns:
            return pd.DataFrame()

        records = [self.categorize_feedback(txt) for txt in df[text_col]]
        topics_df = pd.DataFrame(records)
        
        summary = (
            topics_df.groupby(["topic", "sentiment"])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )
        return summary


if __name__ == "__main__":
    df = pd.read_csv("data/customer_churn_data.csv")
    causal_engine = CausalInferenceEngine(df)
    causal_results = causal_engine.run_all_causal_interventions()
    
    print("Causal Intervention Effects (ATE):")
    for cr in causal_results:
        print(f"\n{cr['hypothesis']}:")
        print(f"  ATE: {cr['ate']:.4f} (Rel Change: {cr['relative_change_pct']}%)")
        print(f"  Treated Churn: {cr['treated_churn_rate_adj']:.2%}, Control Churn: {cr['control_churn_rate_adj']:.2%}")
        print(f"  Recommendation: {cr['recommendation']}")

    nlp = CustomerFeedbackAnalyzer()
    feedback_summary = nlp.analyze_feedback_corpus(df)
    print("\nFeedback Pain Points Topic Distribution:")
    print(feedback_summary.head(10).to_string())

