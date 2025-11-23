# streamlit_app.py
"""
B2B Lead Scoring & CLV Dashboard
Usage: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from io import BytesIO

st.set_page_config(page_title="B2B Lead Scoring & CLV Dashboard", layout="wide")

# ------------------------
# Configuration / Helpers
# ------------------------
DEFAULT_CSV_PATH = "b2b_synthetic_dataset.csv"  # default dataset included in environment
MODEL_DIRS = ["models", "b2b_lead_scoring_project", "/mnt/data"]  # places to look for pickles

@st.cache_data
def load_models():
    clf = None
    reg = None
    # try several possible locations for pickles
    for d in MODEL_DIRS:
        try_clf = os.path.join(d, "lead_scoring_clf.pkl")
        try_reg = os.path.join(d, "models/clv_regressor.pkl")
        if os.path.exists(try_clf):
            try:
                with open(try_clf, "rb") as f:
                    clf = pickle.load(f)
            except Exception:
                clf = None
        if os.path.exists(try_reg):
            try:
                with open(try_reg, "rb") as f:
                    reg = pickle.load(f)
            except Exception:
                reg = None
    return clf, reg

def heuristic_lead_score(df):
    """
    Compute a normalized heuristic lead score (0-1) from available fields.
    This runs if no model pickle is available.
    """
    df = df.copy()
    # ensure columns exist
    for c in ['website_visits_30d','email_clicks_30d','demo_requested','days_since_last_activity','historical_purchase_count','company_size','deal_stage']:
        if c not in df.columns:
            df[c] = 0

    # numeric contributions
    v = df['website_visits_30d'].astype(float).fillna(0)
    clicks = df['email_clicks_30d'].astype(float).fillna(0)
    demo = df['demo_requested'].astype(float).fillna(0)
    hist = df['historical_purchase_count'].astype(float).fillna(0)
    recency = df['days_since_last_activity'].astype(float).fillna(999)

    # basic scoring: visits + clicks*2 + demo*10 + historical*5 - recency*0.02
    raw = v + 2*clicks + 10*demo + 5*hist - 0.02*recency

    # add categorical boosts
    size_map = {'Small':1.0, 'Medium':1.8, 'Large':3.5}
    stage_map = {'Lead':1.0, 'MQL':1.3, 'SQL':1.7, 'Opportunity':2.0, 'Closed Won':2.5, 'Closed Lost':0.5}
    size_factor = df['company_size'].map(size_map).fillna(1.0)
    stage_factor = df['deal_stage'].map(stage_map).fillna(1.0)

    raw = raw * size_factor * stage_factor

    # normalize to 0..1
    minr = raw.min()
    maxr = raw.max()
    if maxr - minr == 0:
        score = pd.Series(0.0, index=df.index)
    else:
        score = (raw - minr) / (maxr - minr)
    return score.clip(0,1)

def heuristic_clv(df):
    """
    Heuristic CLV estimation based on company size, historical purchases and revenue.
    """
    df = df.copy()
    if 'annual_revenue_musd' not in df.columns:
        df['annual_revenue_musd'] = 0.0
    if 'company_size' not in df.columns:
        df['company_size'] = 'Small'
    if 'historical_purchase_count' not in df.columns:
        df['historical_purchase_count'] = 0

    size_multiplier = df['company_size'].map({'Small':1.0,'Medium':3.0,'Large':8.0}).fillna(1.0)
    base = 1000 + 2000 * df['historical_purchase_count'].astype(float) + 5000 * (df.get('converted', 0).astype(float))
    clv = (base * size_multiplier) + df['annual_revenue_musd'].astype(float) * 500
    # add noise and floor
    clv = clv + np.random.normal(0, 1500, size=len(clv))
    return clv.clip(lower=0)

def prepare_input(df):
    """Return a copy with minimal cleaning (drop unnamed columns)"""
    df = df.copy()
    # drop unnamed index columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def to_csv_bytes(df):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# ------------------------
# UI - Sidebar
# ------------------------
st.sidebar.header("Inputs & Models")
uploaded = st.sidebar.file_uploader("Upload leads CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox("Use default synthetic dataset", value=(uploaded is None))
st.sidebar.markdown("**Model options**")
clf, reg = load_models()
st.sidebar.write(f"Classifier loaded: {'Yes' if clf is not None else 'No'}")
st.sidebar.write(f"Regressor loaded: {'Yes' if reg is not None else 'No'}")

threshold = st.sidebar.slider("Lead score threshold to mark HOT leads", 0.0, 1.0, 0.70, 0.01)
top_k = st.sidebar.number_input("Top K leads to show", min_value=5, max_value=500, value=50, step=5)

# ------------------------
# Load dataset
# ------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Loaded uploaded CSV")
else:
    if os.path.exists(DEFAULT_CSV_PATH) and use_default:
        df = pd.read_csv(DEFAULT_CSV_PATH)
        st.sidebar.info(f"Loaded default CSV: {DEFAULT_CSV_PATH}")
    else:
        st.sidebar.warning("No CSV uploaded and default not available; upload a CSV to proceed.")
        st.stop()

df = prepare_input(df)
st.title("B2B Lead Scoring & CLV Dashboard (Synthetic Demo)")

# ------------------------
# Run predictions
# ------------------------
st.markdown("### Dataset preview")
st.dataframe(df.head(10))

st.markdown("### Run predictions")
if st.button("Score leads and predict CLV"):
    df_in = df.copy()

    # decide whether to use models or heuristics
    use_model = (clf is not None and reg is not None)
    if use_model:
        st.success("Using trained models for predictions.")
        # Attempt to align columns expected by model: try/except to be robust
        try:
            X = df_in.drop(columns=[c for c in ['lead_id','true_conversion_prob','converted','CLV_usd'] if c in df_in.columns], errors='ignore')
            pred_proba = clf.predict_proba(X)[:,1]
            pred_clv = reg.predict(X)
        except Exception as e:
            st.warning("Model prediction failed due to input mismatch or missing features. Falling back to heuristics. Error: " + str(e))
            pred_proba = heuristic_lead_score(df_in)
            pred_clv = heuristic_clv(df_in)
    else:
        st.info("No trained pickles found — using heuristic scoring and CLV.")
        pred_proba = heuristic_lead_score(df_in)
        pred_clv = heuristic_clv(df_in)

    df_in['predicted_lead_score'] = np.round(pred_proba, 4)
    df_in['predicted_CLV_usd'] = np.round(pred_clv, 2)
    df_in['hot_lead'] = df_in['predicted_lead_score'] >= threshold

    st.success("Predictions completed — see results below.")

    # Show top K by lead score
    st.markdown(f"### Top {top_k} leads by predicted lead score")
    st.dataframe(df_in.sort_values('predicted_lead_score', ascending=False).head(int(top_k)))

    # Summary metrics
    avg_score = df_in['predicted_lead_score'].mean()
    avg_clv = df_in['predicted_CLV_usd'].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Average predicted lead score", f"{avg_score:.3f}")
    col2.metric("Average predicted CLV (USD)", f"${avg_clv:,.0f}")
    col3.metric("Hot leads (>= threshold)", f"{df_in['hot_lead'].sum()} / {len(df_in)}")

    # Segment breakdowns
    st.markdown("### Segment summaries")
    left, right = st.columns(2)
    with left:
        if 'industry' in df_in.columns:
            st.subheader("Avg lead score by industry")
            industry_table = df_in.groupby('industry')['predicted_lead_score'].mean().sort_values(ascending=False)
            st.table(industry_table)
    with right:
        if 'region' in df_in.columns:
            st.subheader("Avg predicted CLV by region")
            region_table = df_in.groupby('region')['predicted_CLV_usd'].mean().sort_values(ascending=False)
            st.table(region_table)

    # Feature importance if clf available
    st.markdown("### Model insights")
    if clf is not None:
        try:
            feat_imp = clf.named_steps['clf'].feature_importances_
            pre = clf.named_steps['pre']
            # get feature names if onehot encoder available
            num_feats = list(pre.transformers_[0][2])
            cat_feats = pre.transformers_[1][1].get_feature_names_out(pre.transformers_[1][2])
            feat_names = num_feats + list(cat_feats)
            fi_df = pd.DataFrame({'feature': feat_names, 'importance': feat_imp}).sort_values('importance', ascending=False).head(20)
            st.table(fi_df)
        except Exception as e:
            st.write("Could not extract feature importances automatically:", e)
    else:
        st.write("No model available — feature importances unavailable (use heuristics).")

    # Download scored file
    csv_bytes = to_csv_bytes(df_in)
    st.download_button("Download scored leads CSV", csv_bytes, file_name="scored_leads.csv", mime="text/csv")

    # Expose dataframe for further exploration
    st.markdown("### Full scored dataset (first 200 rows)")
    st.dataframe(df_in.head(200))

    # Allow simple filter by hot leads
    if st.checkbox("Show only HOT leads"):
        st.dataframe(df_in[df_in['hot_lead']].sort_values('predicted_lead_score', ascending=False).head(200))

else:
    st.info("Click the button 'Score leads and predict CLV' to compute lead scores and CLV for the dataset.")

# ------------------------
# Footer / notes
# ------------------------
st.markdown("---")
st.markdown("**Notes & next steps:**")
st.markdown("""
- To use trained models, place `lead_scoring_clf.pkl` and `clv_regressor.pkl` in a folder named `models/` in the repo root, or in `/mnt/data/b2b_lead_scoring_project/` (if working in this environment).
- For per-lead explainability, add SHAP in the pipeline and show `shap.force_plot` or `shap.waterfall_plot` for selected rows (can be added later).
- For production: replace pickle models with a model server or a job that re-trains periodically, and secure the model artifacts.
""")
