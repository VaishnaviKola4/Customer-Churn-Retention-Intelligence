# REST API & Integration Guide

The **RetainPulse AI** FastAPI microservice provides production endpoints for real-time customer churn scoring, segmentation, and retention playbook generation.

- **Base URL:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## 📌 Endpoint Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status check |
| `GET` | `/segments` | List defined behavioral clusters and descriptions |
| `POST`| `/predict` | Batch churn probability scoring & cohort assignment |
| `POST`| `/recommend` | Individual customer root-cause deep dive & playbook |

---

## 🔍 Detailed Endpoint Documentation

### 1. Health Check
`GET /health`

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "service": "RetainPulse AI API",
  "version": "1.0.0"
}
```

---

### 2. List Behavioral Segments
`GET /segments`

Returns the behavioral cluster taxonomy with risk levels and strategic focuses.

**Response (`200 OK`):**
```json
{
  "0": {
    "name": "Disengaged High-Touch Users",
    "description": "High ticket volume with steep usage drops...",
    "risk_level": "Critical Risk",
    "strategic_focus": "Support escalation, SLA turnaround guarantees..."
  },
  "1": {
    "name": "Healthy Power Accounts",
    "risk_level": "Low Risk / Loyal",
    "strategic_focus": "Account expansion, advocacy, upsell to enterprise tiers."
  }
}
```

---

### 3. Batch Churn Scoring
`POST /predict`

Scores an array of customer accounts, generating calibrated churn probabilities, revenue at risk, and segment assignments.

**Request Body:**
```json
[
  {
    "customer_id": "CUST-10492",
    "subscription_plan": "Professional",
    "contract_type": "Monthly",
    "billing_method": "Credit Card",
    "auto_renew": 0,
    "tenure_months": 8,
    "monthly_charges": 149.0,
    "total_charges": 1192.0,
    "team_size": 12,
    "monthly_active_days": 14.5,
    "usage_trend_30d": 0.62,
    "days_since_last_login": 19,
    "feature_adoption_rate": 0.38,
    "support_tickets_count": 4,
    "avg_ticket_resolution_hours": 32.5,
    "payment_failures_past_6m": 1,
    "recent_price_increase": 1,
    "nps_score": 4,
    "customer_feedback": "The recent price hike makes it hard to justify renewal, and support ticket response is lagging."
  }
]
```

**Response (`200 OK`):**
```json
[
  {
    "customer_id": "CUST-10492",
    "predicted_churn_prob": 0.8124,
    "risk_tier": "Severe Emergency",
    "cluster_id": 0,
    "segment_name": "Disengaged High-Touch Users",
    "annual_account_value": 1788.0,
    "revenue_at_risk": 1452.57
  }
]
```

---

### 4. Customer Retention Playbook & Deep Dive
`POST /recommend`

Returns individual feature risk contributions and tailored action items.

**Request Example:**
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "CUST-10492",
       "subscription_plan": "Professional",
       "contract_type": "Monthly",
       "billing_method": "Credit Card",
       "auto_renew": 0,
       "tenure_months": 8,
       "monthly_charges": 149.0,
       "total_charges": 1192.0,
       "team_size": 12,
       "monthly_active_days": 14.5,
       "usage_trend_30d": 0.62,
       "days_since_last_login": 19,
       "feature_adoption_rate": 0.38,
       "support_tickets_count": 4,
       "avg_ticket_resolution_hours": 32.5,
       "payment_failures_past_6m": 1,
       "recent_price_increase": 1,
       "nps_score": 4,
       "customer_feedback": "Price hike makes it hard to justify renewal, support response is lagging."
     }'
```

**Response (`200 OK`):**
```json
{
  "customer_id": "CUST-10492",
  "churn_probability": 0.8124,
  "risk_tier": "Severe Emergency",
  "segment_name": "Disengaged High-Touch Users",
  "local_explanations": [
    {
      "feature": "contract_type_Monthly",
      "direction": "Risk Driver",
      "impact_score": 5.54
    },
    {
      "feature": "auto_renew",
      "direction": "Risk Driver",
      "impact_score": 2.85
    },
    {
      "feature": "usage_trend_30d",
      "direction": "Risk Driver",
      "impact_score": 2.59
    }
  ],
  "retention_strategy": {
    "annual_account_value": 1788.0,
    "revenue_at_risk": 1452.57,
    "expected_saved_revenue": 653.66,
    "total_incentive_cost": 516.7,
    "expected_roi_ratio": 0.26,
    "targeted_actions": [
      {
        "channel": "Email & In-App Banner",
        "timing": "Immediate (Triggered on Session)",
        "subject": "Exclusive Partner Upgrade: Lock in 20% savings for the year",
        "action_item": "Present tailored annual billing modal with 1-click upgrade confirmation."
      },
      {
        "channel": "White-Glove Success Outreach",
        "timing": "Within 4 Hours",
        "subject": "Executive Support Escalation & Architecture Review",
        "action_item": "Assign Dedicated Customer Success Engineer to audit open tickets and provide phone resolution."
      }
    ],
    "incentive_offers": [
      {
        "type": "Contract Lock-In Discount",
        "offer": "20% Discount on Annual Upgrade",
        "projected_cost": 357.6,
        "expected_mrr_secured": 1430.4,
        "details": "Converts month-to-month volatility into a 12-month committed ARR stream."
      }
    ]
  }
}
```

