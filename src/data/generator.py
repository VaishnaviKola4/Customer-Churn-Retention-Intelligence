"""
Customer Churn & Retention Intelligence - Data Generation Module
Generates realistic, rich synthetic SaaS subscription data with behavioral patterns,
touchpoints, NPS, billing, feedback text, and ground-truth churn indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import os

FEEDBACK_TEMPLATES = {
    "pricing": [
        "The recent price hike makes it hard to justify renewal for our team size.",
        "Subscription costs have increased significantly compared to alternatives.",
        "Good tool but too expensive for small businesses with tiered pricing.",
        "ROI is unclear given the premium tier pricing.",
        "Competitors offer similar functionality at half the monthly cost."
    ],
    "support": [
        "Customer support tickets take over 48 hours to receive any response.",
        "Unhelpful support responses when dealing with mission-critical outages.",
        "Support has deteriorated drastically over the past three quarters.",
        "Frustrated with tier-1 support closing tickets before resolving bugs.",
        "Long resolution times on simple billing inquiries."
    ],
    "bugs": [
        "Encountering frequent session crashes and UI lag during peak hours.",
        "Integration sync fails repeatedly causing data inconsistencies.",
        "Recent product update broke our automated weekly reporting pipeline.",
        "Persistent bugs in dashboard exports remain unfixed for months.",
        "Performance degradation and API timeouts are disrupting our workflow."
    ],
    "missing_features": [
        "Lacks key integrations with modern CRM and Slack workflows.",
        "Missing advanced custom reporting and role-based permissions.",
        "Need better mobile app access and offline synchronization.",
        "Analytics capabilities are too basic for enterprise level reporting.",
        "Competitor product has much stronger automation workflow triggers."
    ],
    "satisfied": [
        "Outstanding platform! Onboarding was seamless and support is top notch.",
        "Our team productivity increased by 35% within the first two months.",
        "Very intuitive user experience, reliable uptime, and clear documentation.",
        "Fantastic ROI and steady stream of helpful new product releases.",
        "Customer success manager has been proactive and exceptionally helpful."
    ]
}


def generate_synthetic_data(n_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic subscription customer dataset with behavioral, engagement,
    financial, feedback, and churn attributes.
    """
    np.random.seed(random_seed)

    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    
    # Contract and plan details
    subscription_plans = np.random.choice(["Starter", "Professional", "Enterprise"], size=n_samples, p=[0.45, 0.35, 0.20])
    contract_types = np.random.choice(["Monthly", "Annual", "Two-Year"], size=n_samples, p=[0.55, 0.35, 0.10])
    billing_methods = np.random.choice(["Credit Card", "ACH / Direct Debit", "Invoice / Wire"], size=n_samples, p=[0.60, 0.25, 0.15])
    auto_renew = np.random.choice([1, 0], size=n_samples, p=[0.78, 0.22])

    # Tenures in months
    tenure_months = np.random.gamma(shape=2.5, scale=6.0, size=n_samples).astype(int) + 1
    tenure_months = np.clip(tenure_months, 1, 60)

    # Base pricing based on plan
    monthly_charges = []
    for plan in subscription_plans:
        if plan == "Starter":
            monthly_charges.append(np.random.normal(49.0, 8.0))
        elif plan == "Professional":
            monthly_charges.append(np.random.normal(149.0, 20.0))
        else: # Enterprise
            monthly_charges.append(np.random.normal(499.0, 75.0))
    monthly_charges = np.clip(np.array(monthly_charges).round(2), 25.0, 800.0)
    total_charges = (monthly_charges * tenure_months * np.random.uniform(0.92, 1.05, n_samples)).round(2)

    # Product Usage and Behavioral Metrics
    # Active users per account
    team_sizes = []
    for plan in subscription_plans:
        if plan == "Starter":
            team_sizes.append(np.random.randint(1, 6))
        elif plan == "Professional":
            team_sizes.append(np.random.randint(5, 25))
        else:
            team_sizes.append(np.random.randint(20, 120))
    team_sizes = np.array(team_sizes)

    # Monthly active days out of 30
    monthly_active_days = np.random.beta(a=5, b=2, size=n_samples) * 30
    monthly_active_days = np.clip(monthly_active_days.round(1), 1.0, 30.0)

    # Usage trend (30-day change ratio: < 1 means dropping activity, > 1 means increasing)
    usage_trend_30d = np.random.normal(0.98, 0.28, size=n_samples)
    usage_trend_30d = np.clip(usage_trend_30d.round(3), 0.1, 2.5)

    # Days since last login
    days_since_last_login = np.random.exponential(scale=5.0, size=n_samples).astype(int)
    # Long inactive bump for struggling users
    long_inactive_idx = np.random.choice(n_samples, size=int(n_samples * 0.12), replace=False)
    days_since_last_login[long_inactive_idx] += np.random.randint(15, 45, size=len(long_inactive_idx))
    days_since_last_login = np.clip(days_since_last_login, 0, 75)

    # Core feature adoption rate (0.0 - 1.0)
    feature_adoption_rate = np.random.beta(a=3, b=2, size=n_samples).round(3)

    # Support and ticket metrics
    support_tickets_count = np.random.poisson(lam=2.2, size=n_samples)
    avg_ticket_resolution_hours = np.random.gamma(shape=3.0, scale=8.0, size=n_samples).round(1)
    
    # Billing incidents (payment failures, disputes)
    payment_failures_past_6m = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.72, 0.18, 0.07, 0.03])
    recent_price_increase = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])

    # Net Promoter Score (-1 to 10 scale or 0-10)
    nps_score = np.random.choice(np.arange(0, 11), size=n_samples, p=[0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.12, 0.16, 0.14, 0.10, 0.08])

    # Customer Feedback Text generation based on underlying dissatisfaction drivers
    feedback_texts = []
    primary_pain_points = []

    for i in range(n_samples):
        # Determine likely driver
        scores = {
            "pricing": (recent_price_increase[i] * 2.5) + (1.5 if monthly_charges[i] > 180 else 0.5),
            "support": (support_tickets_count[i] * 0.8) + (avg_ticket_resolution_hours[i] / 12.0),
            "bugs": 2.0 if (usage_trend_30d[i] < 0.75 and feature_adoption_rate[i] > 0.4) else 0.6,
            "missing_features": 2.2 if feature_adoption_rate[i] < 0.35 else 0.8,
            "satisfied": (nps_score[i] / 2.5) + (1.5 if usage_trend_30d[i] > 1.05 else 0.2)
        }
        
        # Softmax selection
        keys = list(scores.keys())
        vals = np.array([scores[k] for k in keys])
        probs = np.exp(vals - np.max(vals))
        probs = probs / probs.sum()
        selected_category = np.random.choice(keys, p=probs)
        
        feedback_texts.append(np.random.choice(FEEDBACK_TEMPLATES[selected_category]))
        primary_pain_points.append(selected_category if selected_category != "satisfied" else "None")

    # Latent Churn Propensity Calculation (Realistic non-linear causal relationship)
    # Centered intercept to yield ~22-26% baseline churn rate
    churn_logits = (
        0.80
        - 0.040 * tenure_months
        + 0.95 * (contract_types == "Monthly").astype(int)
        - 0.70 * (contract_types == "Two-Year").astype(int)
        - 1.10 * auto_renew
        + 0.045 * days_since_last_login
        - 1.65 * (usage_trend_30d - 1.0)
        - 0.05 * monthly_active_days
        - 1.40 * feature_adoption_rate
        + 0.28 * support_tickets_count
        + 0.02 * avg_ticket_resolution_hours
        + 0.55 * payment_failures_past_6m
        + 0.40 * recent_price_increase
        - 0.16 * nps_score
        + 0.0012 * monthly_charges
    )

    # Convert logits to probability
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logits))
    # Add minor stochastic noise
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.04, size=n_samples), 0.01, 0.99)
    churned = (churn_prob > 0.50).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "subscription_plan": subscription_plans,
        "contract_type": contract_types,
        "billing_method": billing_methods,
        "auto_renew": auto_renew,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "team_size": team_sizes,
        "monthly_active_days": monthly_active_days,
        "usage_trend_30d": usage_trend_30d,
        "days_since_last_login": days_since_last_login,
        "feature_adoption_rate": feature_adoption_rate,
        "support_tickets_count": support_tickets_count,
        "avg_ticket_resolution_hours": avg_ticket_resolution_hours,
        "payment_failures_past_6m": payment_failures_past_6m,
        "recent_price_increase": recent_price_increase,
        "nps_score": nps_score,
        "primary_pain_point": primary_pain_points,
        "customer_feedback": feedback_texts,
        "churn_probability_ground_truth": churn_prob.round(4),
        "churn": churned
    })

    return df


def save_dataset(df: pd.DataFrame, filepath: str = "data/customer_churn_data.csv") -> None:
    """Save dataframe to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[DataGen] Saved {len(df)} records to {filepath}")


if __name__ == "__main__":
    data = generate_synthetic_data(n_samples=3000, random_seed=42)
    save_dataset(data, "data/customer_churn_data.csv")
    print(f"Dataset generated with churn rate: {data['churn'].mean():.2%}")
