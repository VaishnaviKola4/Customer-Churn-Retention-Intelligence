"""
Customer Churn & Retention Intelligence - Interactive Streamlit Web Application
Provides a dashboard with:
1. Executive Intelligence Overview (KPIs, ARR At Risk, Churn Probability Distribution)
2. Behavioral Clustering & Segmentation Map (Interactive 2D PCA & Cluster Profile Cards)
3. Causal Impact & Pain Points Analysis (ATE treatment cards & Feedback NLP breakdown)
4. At-Risk Customer Directory & Deep Dive (Individual Root Cause Explanation & Actionable Playbook)
5. Interactive Strategy Simulator ("What-If" Analysis & ROI Calculator)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ChurnRetentionPipeline
from src.clustering.segmentation import SEGMENT_PROFILES
from src.causal.causal_analyzer import CausalInferenceEngine, CustomerFeedbackAnalyzer

# Page configuration
st.set_page_config(
    page_title="RetainPulse AI | Churn & Retention Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tag-severe {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .tag-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .tag-moderate {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .tag-healthy {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    pipeline = ChurnRetentionPipeline()
    pipeline.load_artifacts()
    return pipeline


@st.cache_data
def load_and_score_data():
    pipeline = load_pipeline()
    data_path = "data/customer_churn_data.csv"
    if not os.path.exists(data_path):
        from src.data.generator import generate_synthetic_data, save_dataset
        df_gen = generate_synthetic_data(n_samples=3000)
        save_dataset(df_gen, data_path)
    df = pd.read_csv(data_path)
    scored_df = pipeline.process_dataset(df)
    return scored_df


# Header Section
st.title("⚡ RetainPulse AI")
st.caption("AI-Powered Customer Churn Prediction, Causal Diagnostics & Algorithmic Retention Strategies")

# Sidebar navigation & filters
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/combo-chart.png", width=64)
    st.title("Navigation & Filters")
    
    app_mode = st.radio(
        "Module Selection:",
        [
            "📊 Executive Dashboard",
            "🧩 Behavioral Segments",
            "🔍 Causal Impact & NLP",
            "🎯 Customer Deep Dive & Actions",
            "🧪 What-If Retention Simulator"
        ]
    )

    st.markdown("---")
    st.subheader("Cohort Filters")
    
    raw_df = load_and_score_data()
    selected_plans = st.multiselect(
        "Subscription Tier:",
        options=list(raw_df["subscription_plan"].unique()),
        default=list(raw_df["subscription_plan"].unique())
    )
    
    selected_contracts = st.multiselect(
        "Contract Duration:",
        options=list(raw_df["contract_type"].unique()),
        default=list(raw_df["contract_type"].unique())
    )
    
    risk_threshold = st.slider("Churn Probability Threshold:", 0.0, 1.0, 0.40, 0.05)

# Filter data
filtered_df = raw_df[
    (raw_df["subscription_plan"].isin(selected_plans)) &
    (raw_df["contract_type"].isin(selected_contracts))
]

pipeline = load_pipeline()


# MODULE 1: EXECUTIVE DASHBOARD
if app_mode == "📊 Executive Dashboard":
    st.markdown("### 📈 Executive Health & Revenue Exposure")
    
    total_customers = len(filtered_df)
    churn_rate = (filtered_df["predicted_churn_prob"] >= 0.50).mean()
    total_arr = (filtered_df["annual_value"]).sum()
    arr_at_risk = (filtered_df["revenue_at_risk"]).sum()
    severe_risk_count = (filtered_df["predicted_churn_prob"] >= 0.70).sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Monitored Accounts</div>
            <div class="metric-val">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ef4444;">
            <div class="metric-lbl">Predicted Churn Rate</div>
            <div class="metric-val">{churn_rate:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f59e0b;">
            <div class="metric-lbl">Total Active ARR</div>
            <div class="metric-val">${total_arr:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #dc2626;">
            <div class="metric-lbl">ARR Exposure at Risk</div>
            <div class="metric-val">${arr_at_risk:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Churn Probability Distribution")
        fig_hist = px.histogram(
            filtered_df,
            x="predicted_churn_prob",
            nbins=30,
            color="risk_tier",
            color_discrete_map={
                "Severe Emergency": "#dc2626",
                "High Risk": "#f97316",
                "Moderate Warning": "#eab308",
                "Healthy Account": "#16a34a"
            },
            title="Customer Distribution across Risk Tiers"
        )
        fig_hist.update_layout(xaxis_title="Predicted Churn Probability", yaxis_title="Customer Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col2:
        st.subheader("Revenue at Risk by Plan Tier")
        plan_rev = filtered_df.groupby("subscription_plan").agg(
            total_rev=("annual_value", "sum"),
            risk_rev=("revenue_at_risk", "sum")
        ).reset_index()

        fig_bar = go.Figure(data=[
            go.Bar(name='Safe ARR', x=plan_rev['subscription_plan'], y=plan_rev['total_rev'] - plan_rev['risk_rev'], marker_color='#22c55e'),
            go.Bar(name='ARR at Risk', x=plan_rev['subscription_plan'], y=plan_rev['risk_rev'], marker_color='#ef4444')
        ])
        fig_bar.update_layout(barmode='stack', title="Active vs At-Risk ARR by Plan")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Global feature importances
    st.markdown("---")
    st.subheader("Key Macro Drivers Influencing Churn Likelihood")
    importances = pipeline.model_engine.get_feature_importances(8)
    imp_df = pd.DataFrame(importances)
    fig_imp = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Reds",
        title="Top Algorithmic Predictors of Churn Propensity"
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)


# MODULE 2: BEHAVIORAL SEGMENTS
elif app_mode == "🧩 Behavioral Segments":
    st.markdown("### 🧩 Unsupervised Behavioral Customer Segmentation")
    st.write(
        "K-Means clustering with 2D Principal Component Analysis (PCA) projections reveal natural groupings in usage, support load, and billing habits."
    )

    col_seg1, col_seg2 = st.columns([3, 2])
    with col_seg1:
        fig_pca = px.scatter(
            filtered_df,
            x="pca_x",
            y="pca_y",
            color="segment_name",
            hover_data=["customer_id", "subscription_plan", "predicted_churn_prob", "monthly_charges"],
            title="2D PCA Behavioral Projection of Customer Cohorts",
            opacity=0.75
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_seg2:
        st.subheader("Cohort Profile Summary")
        for cid, profile in SEGMENT_PROFILES.items():
            count = (filtered_df["cluster_id"] == cid).sum()
            avg_p = filtered_df[filtered_df["cluster_id"] == cid]["predicted_churn_prob"].mean()
            st.markdown(f"""
            <div style="background:#f1f5f9; padding:12px; border-radius:8px; margin-bottom:12px;">
                <div style="font-weight:700; color:#1e293b; font-size:16px;">{profile['name']}</div>
                <div style="font-size:12px; color:#64748b; margin-top:2px;">Accounts: <b>{count:,}</b> | Avg Churn: <b>{avg_p:.1%}</b></div>
                <div style="font-size:13px; color:#334155; margin-top:6px;">{profile['description']}</div>
                <div style="font-size:12px; color:#2563eb; font-weight:600; margin-top:4px;">🎯 Strategic Focus: {profile['strategic_focus']}</div>
            </div>
            """, unsafe_allow_html=True)


# MODULE 3: CAUSAL IMPACT & NLP
elif app_mode == "🔍 Causal Impact & NLP":
    st.markdown("### 🔍 Causal Inference & Customer Feedback Diagnostics")
    st.write("Understand **why** churn occurs by isolating causal levers using Inverse Propensity Weighting (IPW).")

    causal_engine = CausalInferenceEngine(filtered_df)
    effects = causal_engine.run_all_causal_interventions()

    st.subheader("Causal Interventions: Average Treatment Effect (ATE)")
    c_cols = st.columns(len(effects))
    for idx, eff in enumerate(effects):
        with c_cols[idx]:
            delta_color = "inverse" if eff["ate"] > 0 else "normal"
            st.metric(
                label=eff["treatment"].replace("_", " ").title(),
                value=f"{eff['ate']:+.2%}",
                delta=f"{eff['relative_change_pct']:+.1f}% Rel Churn",
                delta_color=delta_color
            )
            st.caption(eff["recommendation"])

    st.markdown("---")
    st.subheader("Customer Feedback & Voice-of-Customer NLP Breakdown")
    feedback_summary = pipeline.feedback_analyzer.analyze_feedback_corpus(filtered_df)
    
    col_nlp1, col_nlp2 = st.columns(2)
    with col_nlp1:
        fig_pie = px.sunburst(
            feedback_summary,
            path=['sentiment', 'topic'],
            values='count',
            color='sentiment',
            color_discrete_map={'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#94a3b8'},
            title="Pain Point Topics & Sentiment Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_nlp2:
        st.write("**Top Recent Verbatim Pain Points:**")
        sample_complaints = filtered_df[filtered_df["feedback_sentiment"] == "Negative"][["customer_id", "feedback_topic", "customer_feedback"]].head(6)
        st.dataframe(sample_complaints, use_container_width=True, hide_index=True)


# MODULE 4: CUSTOMER DEEP DIVE & ACTIONS
elif app_mode == "🎯 Customer Deep Dive & Actions":
    st.markdown("### 🎯 Account Retention Playbook & Action Generator")
    
    at_risk_pool = filtered_df[filtered_df["predicted_churn_prob"] >= risk_threshold].sort_values(
        by="revenue_at_risk", ascending=False
    )

    st.write(f"Displaying **{len(at_risk_pool):,}** accounts flagged with churn probability ≥ {risk_threshold:.0%}.")

    selected_cid = st.selectbox(
        "Select Customer Account for Retention Diagnostic:",
        options=at_risk_pool["customer_id"].tolist() if not at_risk_pool.empty else filtered_df["customer_id"].head(50).tolist()
    )

    if selected_cid:
        deep_dive = pipeline.get_customer_deep_dive(selected_cid, filtered_df)
        cust = deep_dive["customer_profile"]
        strategy = deep_dive["retention_strategy"]
        explanations = deep_dive["local_explanations"]

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown(f"**Customer ID:** `{cust['customer_id']}`")
            st.markdown(f"**Subscription Plan:** `{cust['subscription_plan']}`")
            st.markdown(f"**Contract:** `{cust['contract_type']}`")
            st.markdown(f"**Tenure:** `{cust['tenure_months']} months`")
        with col_p2:
            st.markdown(f"**Monthly Billing:** `${cust['monthly_charges']:.2f}`")
            st.markdown(f"**Annual Value:** `${strategy['annual_account_value']:.2f}`")
            st.markdown(f"**Days Since Login:** `{cust['days_since_last_login']} days`")
            st.markdown(f"**Support Tickets:** `{cust['support_tickets_count']}`")
        with col_p3:
            st.markdown(f"**Churn Likelihood:** `{strategy['churn_probability']:.1%}`")
            st.markdown(f"**Risk Level:** `{strategy['risk_tier']}`")
            st.markdown(f"**Annual Revenue At Risk:** `${strategy['revenue_at_risk']:,.2f}`")
            st.markdown(f"**Expected Saved ARR:** `${strategy['expected_saved_revenue']:,.2f}`")

        st.markdown("---")
        st.subheader("Root Cause Feature Explanations")
        exp_df = pd.DataFrame(explanations)
        fig_local = px.bar(
            exp_df,
            x="impact_score",
            y="feature",
            color="direction",
            color_discrete_map={"Risk Driver": "#ef4444", "Retention Anchor": "#22c55e"},
            orientation="h",
            title="Local Feature Contribution to Account Risk"
        )
        st.plotly_chart(fig_local, use_container_width=True)

        st.subheader("Actionable Retention Blueprint")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            st.write("##### ✉️ Targeted Interventions & Outreach")
            for act in strategy["targeted_actions"]:
                st.info(f"**[{act['channel']}]** ({act['timing']})\n\n**Subject:** {act['subject']}\n\n👉 *{act['action_item']}*")

        with col_act2:
            st.write("##### 🎁 Financial Incentives & Retention Offers")
            for inc in strategy["incentive_offers"]:
                st.success(f"**Offer:** {inc['offer']}\n\n**Type:** {inc['type']} | **Cost:** ${inc['projected_cost']:.2f}\n\n💡 *{inc['details']}*")


# MODULE 5: WHAT-IF RETENTION SIMULATOR
elif app_mode == "🧪 What-If Retention Simulator":
    st.markdown("### 🧪 What-If Retention Campaign Simulator & ROI Estimator")
    st.write(
        "Simulate business outcomes of strategic interventions (e.g. converting monthly accounts to annual, cutting support turnaround, auto-pay drive)."
    )

    col_sim_in, col_sim_res = st.columns([1, 2])
    with col_sim_in:
        st.subheader("Campaign Parameters")
        target_discount = st.slider("Annual Switch Discount Offer (%):", 5, 30, 20)
        support_sla_improvement = st.slider("Support SLA Latency Reduction (%):", 10, 60, 35)
        autopay_adoption_target = st.slider("Autopay Conversion Goal (% of non-autopay):", 10, 80, 40)
        
        sim_sample_size = len(filtered_df)
        monthly_users_count = (filtered_df["contract_type"] == "Monthly").sum()
        current_risk_arr = filtered_df["revenue_at_risk"].sum()

    with col_sim_res:
        st.subheader("Projected Campaign Financial Impact")
        
        # Algorithmic projection based on causal ATE coefficients
        # Annual switch saves ~22% absolute churn on monthly users
        est_annual_converts = int(monthly_users_count * 0.28)
        arr_protected_contracts = est_annual_converts * (filtered_df["monthly_charges"].mean() * 12) * 0.2212
        
        # SLA reduction saves ~7.6% churn on high support accounts
        arr_protected_support = (filtered_df["support_tickets_count"] >= 3).sum() * (filtered_df["monthly_charges"].mean() * 12) * 0.0769
        
        total_protected_arr = arr_protected_contracts + arr_protected_support
        program_cost = est_annual_converts * (filtered_df["monthly_charges"].mean() * 12) * (target_discount / 100.0)
        net_saved = max(total_protected_arr - program_cost, 0.0)
        campaign_roi = (net_saved / max(program_cost, 1.0)) * 100.0

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        sim_c1.metric("ARR Protected", f"${total_protected_arr:,.0f}")
        sim_c2.metric("Program Incentive Cost", f"${program_cost:,.0f}")
        sim_c3.metric("Projected Net ROI", f"{campaign_roi:,.1f}%")

        # Waterfall comparison
        fig_waterfall = go.Figure(go.Waterfall(
            name="Impact",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Baseline Risk ARR", "Contract Upgrades", "Support SLAs", "Remaining Risk ARR"],
            textposition="outside",
            y=[current_risk_arr, -arr_protected_contracts, -arr_protected_support, current_risk_arr - total_protected_arr],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title="Simulated ARR Risk Reduction Waterfall", showlegend=False)
        st.plotly_chart(fig_waterfall, use_container_width=True)

