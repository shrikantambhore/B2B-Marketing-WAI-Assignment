# streamlit_app.py
"""
B2B Lead Scoring & CLV Dashboard (Auto-train models inside Streamlit)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="B2B Lead Scoring & CLV Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ==============================
# Helper functions
# ==============================

def prepare_input(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def to_csv_bytes(df):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# ------------------------------
# HEURISTICS (FALLBACK)
# ------------------------------
def heuristic_lead_score(df):
    df = df.copy()
    for c in ['website_visits_30d','email_clicks_30d','demo_requested',
              'days_since_last_activity','historical_purchase_count',
              'company_size','deal_stage']:
        if c not in df.columns:
            df[c] = 0

    v = df['website_visits_30d'].astype(float).fillna(0)
    clicks = df['email_clicks_30d'].astype(float).fillna(0)
    demo = df['demo_requested'].astype(float).fillna(0)
    hist = df['historical_purchase_count'].astype(float).fillna(0)
    rec = df['days_since_last_activity'].astype(float).fillna(999)

    raw = v + 2*clicks + 10*demo + 5*hist - 0.02*rec

    size_map = {'Small':1.0,'Medium':1.8,'Large':3.5}
    stage_map = {'Lead':1.0,'MQL':1.3,'SQL':1.7,'Opportunity':2.0,'Closed Won':2.5,'Closed Lost':0.5}

    raw *= df['company_size'].map(size_map).fillna(1.0)
    raw *= df['deal_stage'].map(stage_map).fillna(1.0)

    if raw.max() == raw.min():
        return pd.Series(0.5, index=df.index)
    return ((raw - raw.min()) / (raw.max() - raw.min())).clip(0,1)

def heuristic_clv(df):
    df = df.copy()
    for c in ['annual_revenue_musd','historical_purchase_count','company_size']:
        if c not in df.columns:
            df[c] = 0

    size_mult = df['company_size'].map({'Small':1,'Medium':3,'Large':8}).fillna(1)
    base = 1000 + 2000 * df['historical_purchase_count'].astype(float)
    clv = base * size_mult + df['annual_revenue_musd'] * 500
    clv += np.random.normal(0, 1500, len(df))
    return clv.clip(lower=0)

# ------------------------------
# AUTO TRAIN MODELS (CACHED)
# ------------------------------

@st.cache_resource(show_spinner=True)
def train_models(df):

    feature_cols = [
        'company_size','industry','annual_revenue_musd','employee_count',
        'job_title','website_visits_30d','email_opens_30d','email_clicks_30d',
        'demo_requested','days_since_last_activity','deal_stage','lead_source',
        'region','historical_purchase_count','months_since_first_contact'
    ]

    # Ensure required columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = "Unknown" if df.dtypes.get(c, "object") == "object" else 0

    if 'converted' not in df.columns or 'CLV_usd' not in df.columns:
        raise ValueError("Dataset must contain 'converted' and 'CLV_usd' columns.")

    X = df[feature_cols]
    y_cls = df['converted']
    y_reg = df['CLV_usd']

    numeric = [
        'annual_revenue_musd','employee_count','website_visits_30d','email_opens_30d',
        'email_clicks_30d','demo_requested','days_since_last_activity',
        'historical_purchase_count','months_since_first_contact'
    ]
    categorical = ['company_size','industry','job_title','deal_stage','lead_source','region']

    # FIX: OneHotEncoder sparse parameter depending on sklearn version
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', encoder, categorical)
    ], remainder='drop')

    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ])
    clf.fit(X_train, y_train_cls)

    reg = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        ))
    ])
    reg.fit(X_train, df.loc[X_train.index, 'CLV_usd'])

    return clf, reg


# ==============================
# Sidebar UI
# ==============================

st.sidebar.header("Inputs & Models")

uploaded = st.sidebar.file_uploader("Upload leads CSV", type=["csv"])
use_default = st.sidebar.checkbox("Use default dataset", value=True)

threshold = st.sidebar.slider("Lead score threshold", 0.0, 1.0, 0.70, 0.01)
top_k = st.sidebar.number_input("Top K leads to show", 5, 500, 50, 5)

DEFAULT_CSV = "b2b_synthetic_dataset.csv"


# ==============================
# Load dataset
# ==============================

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Uploaded CSV loaded.")
else:
    if use_default and os.path.exists(DEFAULT_CSV):
        df = pd.read_csv(DEFAULT_CSV)
        st.sidebar.info(f"Loaded default CSV: {DEFAULT_CSV}")
    else:
        st.sidebar.error("No CSV available. Upload a CSV to continue.")
        st.stop()

df = prepare_input(df)


st.title("B2B Lead Scoring & CLV Dashboard (Auto-Trained Models)")


# ==============================
# Train models
# ==============================

try:
    clf, reg = train_models(df)
    st.sidebar.success("Model trained successfully!")
except Exception as e:
    clf, reg = None, None
    st.sidebar.error(f"Model training failed: {e}")
    st.sidebar.warning("Using heuristic scoring.")
    st.info("Model training failed — falling back to heuristics.")


# ==============================
# Main UI — predictions
# ==============================

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

if st.button("Score leads and predict CLV"):

    df_out = df.copy()

    if clf is not None:
        try:
            X = df_out.drop(columns=[
                c for c in ['lead_id','converted','CLV_usd','true_conversion_prob']
                if c in df_out.columns
            ], errors="ignore")

            df_out['predicted_lead_score'] = clf.predict_proba(X)[:,1]
            df_out['predicted_CLV_usd'] = reg.predict(X)

        except Exception as e:
            st.warning(f"Model prediction failed ({e}). Using heuristics.")
            df_out['predicted_lead_score'] = heuristic_lead_score(df_out)
            df_out['predicted_CLV_usd'] = heuristic_clv(df_out)
    else:
        df_out['predicted_lead_score'] = heuristic_lead_score(df_out)
        df_out['predicted_CLV_usd'] = heuristic_clv(df_out)

    df_out['hot_lead'] = df_out['predicted_lead_score'] >= threshold

    st.success("Predictions completed.")

    st.subheader(f"Top {top_k} Leads")
    st.dataframe(df_out.sort_values("predicted_lead_score", ascending=False).head(int(top_k)))

    st.subheader("Download Scored CSV")
    st.download_button("Download", to_csv_bytes(df_out),
                       "scored_leads.csv", mime="text/csv")

else:
    st.info("Click the button to run predictions.")

# Footer
st.markdown("---")
st.caption("Auto-trained ML model (no pickle compatibility issues).")
