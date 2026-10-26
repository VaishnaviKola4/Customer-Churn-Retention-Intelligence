"""
Customer Churn & Retention Intelligence - Personalized Recommendation Engine
Generates tailored retention strategies, communication templates, financial incentives,
and product playbooks based on individual customer churn probability, segment, and root causes.
"""

from typing import Dict, List, Any
import numpy as np


class RetentionRecommendationEngine:
    """Decision engine generating personalized retention playbooks and expected ROI."""

    def __init__(self):
        pass

    def evaluate_customer_risk_tier(self, churn_prob: float) -> str:
        """Classify customer risk tier."""
        if churn_prob >= 0.70:
            return "Severe Emergency"
        elif churn_prob >= 0.45:
            return "High Risk"
        elif churn_prob >= 0.20:
            return "Moderate Warning"
        else:
            return "Healthy Account"

    def generate_strategy(
        self,
        customer_record: Dict[str, Any],
        churn_prob: float,
        cluster_id: int,
        primary_causes: List[str]
    ) -> Dict[str, Any]:
        """
        Synthesize multi-channel retention actions:
        - Targeted Communication
        - Incentive & Financial Offer
        - Product & Feature Recommendation
        - Customer Success Workflow & Priority
        - Projected Retention ROI & Saved MRR
        """
        risk_tier = self.evaluate_customer_risk_tier(churn_prob)
        monthly_charge = float(customer_record.get("monthly_charges", 99.0))
        plan = customer_record.get("subscription_plan", "Professional")
        contract = customer_record.get("contract_type", "Monthly")
        feedback = customer_record.get("customer_feedback", "")
        nps = customer_record.get("nps_score", 7)
        support_tickets = customer_record.get("support_tickets_count", 1)

        actions: List[Dict[str, str]] = []
        incentives: List[Dict[str, Any]] = []
        product_playbooks: List[str] = []

        # 1. Root Cause & Segment Tailored Actions
        if contract == "Monthly" and churn_prob > 0.35:
            incentives.append({
                "type": "Contract Lock-In Discount",
                "offer": "20% Discount on Annual Upgrade",
                "projected_cost": round(monthly_charge * 12 * 0.20, 2),
                "expected_mrr_secured": round(monthly_charge * 12 * 0.80, 2),
                "details": "Converts month-to-month volatility into a 12-month committed ARR stream."
            })
            actions.append({
                "channel": "Email & In-App Banner",
                "timing": "Immediate (Triggered on Session)",
                "subject": "Exclusive Partner Upgrade: Lock in 20% savings for the year",
                "action_item": "Present tailored annual billing modal with 1-click upgrade confirmation."
            })

        if support_tickets >= 3 or "support" in feedback.lower():
            actions.append({
                "channel": "White-Glove Success Outreach",
                "timing": "Within 4 Hours",
                "subject": "Executive Support Escalation & Architecture Review",
                "action_item": f"Assign Dedicated Customer Success Engineer to audit open tickets and provide phone resolution."
            })
            incentives.append({
                "type": "SLA & Support Guarantee",
                "offer": "Free Priority 1-Hour SLA Support for 90 Days",
                "projected_cost": 0.0,
                "expected_mrr_secured": round(monthly_charge * 6, 2),
                "details": "Restores trust by bypassing tier-1 support queue."
            })

        if customer_record.get("days_since_last_login", 0) > 14 or customer_record.get("usage_trend_30d", 1.0) < 0.70:
            product_playbooks.append("Automated Re-Engagement Drip: Showcase under-utilized workflow automations.")
            product_playbooks.append("Complimentary 1-on-1 Workflow Audit with Product Specialist.")
            actions.append({
                "channel": "Personalized Video Check-in",
                "timing": "Within 24 Hours",
                "subject": "How to unlock 3x more speed in your current workflow",
                "action_item": "Send customized Loom demo highlighting automated reporting features."
            })

        if customer_record.get("payment_failures_past_6m", 0) > 0 or customer_record.get("auto_renew", 1) == 0:
            actions.append({
                "channel": "Automated Billing Modal",
                "timing": "Next Login Event",
                "subject": "Ensure uninterrupted service with seamless automated billing",
                "action_item": "Offer card updater incentive (e.g. $25 credit on next invoice for setting auto-pay)."
            })
            incentives.append({
                "type": "Payment Method Incentive",
                "offer": "$25 Bill Credit for Auto-Pay Enrollment",
                "projected_cost": 25.0,
                "expected_mrr_secured": round(monthly_charge * 12, 2),
                "details": "Eliminates involuntary dunning churn."
            })

        if "price" in feedback.lower() or customer_record.get("recent_price_increase", 0) == 1:
            incentives.append({
                "type": "Grandfathered Pricing Protection",
                "offer": "6-Month Price Freeze at Legacy Rate",
                "projected_cost": round(monthly_charge * 0.15 * 6, 2),
                "expected_mrr_secured": round(monthly_charge * 6, 2),
                "details": "Appeases budget constraints while maintaining active customer relationship."
            })

        # Default fallback for healthy accounts
        if not actions:
            actions.append({
                "channel": "CSM Quarterly Business Review",
                "timing": "Within 30 Days",
                "subject": "Quarterly Value Realization & Roadmap Preview",
                "action_item": "Share product roadmap, collect feature requests, and assess enterprise expansion potential."
            })

        # Expected Value and ROI calculation
        annual_account_value = monthly_charge * 12.0
        risk_adjusted_value_at_risk = round(annual_account_value * churn_prob, 2)
        total_incentive_cost = sum([inc.get("projected_cost", 0.0) for inc in incentives])
        
        # Assume retention campaign achieves 45% recovery rate on at-risk accounts
        projected_retention_success_prob = 0.45 if churn_prob > 0.4 else 0.75
        expected_saved_revenue = round(risk_adjusted_value_at_risk * projected_retention_success_prob, 2)
        roi_ratio = round((expected_saved_revenue - total_incentive_cost) / max(total_incentive_cost, 1.0), 2)

        return {
            "customer_id": customer_record.get("customer_id", "N/A"),
            "risk_tier": risk_tier,
            "churn_probability": round(churn_prob, 4),
            "annual_account_value": round(annual_account_value, 2),
            "revenue_at_risk": risk_adjusted_value_at_risk,
            "expected_saved_revenue": expected_saved_revenue,
            "total_incentive_cost": total_incentive_cost,
            "expected_roi_ratio": roi_ratio,
            "targeted_actions": actions,
            "incentive_offers": incentives,
            "product_recommendations": product_playbooks if product_playbooks else ["Standard feature adoption checklist"]
        }


if __name__ == "__main__":
    engine = RetentionRecommendationEngine()
    dummy_cust = {
        "customer_id": "CUST-10492",
        "monthly_charges": 149.0,
        "subscription_plan": "Professional",
        "contract_type": "Monthly",
        "customer_feedback": "The price hike makes it hard to justify renewal, and support ticket response is lagging.",
        "days_since_last_login": 18,
        "usage_trend_30d": 0.62,
        "support_tickets_count": 4,
        "auto_renew": 0,
        "nps_score": 4
    }
    recommendation = engine.generate_strategy(dummy_cust, churn_prob=0.74, cluster_id=0, primary_causes=["Pricing", "Support Lag"])
    print("Recommendation Sample:")
    import pprint
    pprint.pprint(recommendation)

