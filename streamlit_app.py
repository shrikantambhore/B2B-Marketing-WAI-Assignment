# streamlit_app.py
"""
B2B Lead Scoring & CLV Dashboard (Auto-train models inside Streamlit)
Usage: streamlit run streamlit_app.py
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

st.set_page_config(page_title="B2B Lead Scoring & CLV Dashboard", layout="wide", initial_sidebar_state="expanded")

# ------------------------
# Configuration
# ------------------------
# Default dataset path (if you commit CSV to repo root or include in Streamlit)
DEFAULT_CSV_PATH = "b2b_synthetic_dataset.csv"  # please ensure this file is in repo root if you want default

# ------------------------
# Helper functions
# ------------------------
def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with minimal cleaning (drop unnamed columns)."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", regex=True)]
    return df

def to_csv_bytes(df: pd.DataFrame) -> BytesIO:
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# Heuristics (fallback)
def heuristic_lead_score(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    for c in ['website_visits_30d','email_clicks_30d','demo_requested','days_since_last_activity','historical_purchase_count','company_size','deal_stage']:
        if c not in df.columns:
            df[c] = 0
    v = df['website_visits_30d'].astype(float).fillna(0)
    clicks = df['email_clicks_30d'].astype(float).fillna(0)
    demo = df['demo_requested'].astype(float).fillna(0)
    hist = df['historical_purchase_count'].astype(float).fillna(0)
    recency = df['days_since_last_activity'].astype(float).fillna(999)
    raw = v + 2*clicks + 10*demo + 5*hist - 0.02*recency
    size_map = {'Small':1.0, 'Medium':1.8, 'Large':3.5}
    stage_map = {'Lead':1.0, 'MQL':1.3, 'SQL':1.7, 'Opportunity':2.0, 'Closed Won':2.5, 'Closed Lost':0.5}
    size_factor = df['company_size'].map(size_map).fillna(1.0)
    stage_factor = df['deal_stage'].map(stage_map).fillna(1.0)
    raw = raw * size_factor * stage_factor
    minr = raw.min()
    maxr = raw.max()
    if maxr - minr == 0:
        score = pd.Series(0.0, index=df.index)
    else:
        score = (raw - minr) / (maxr - minr)
    return score.clip(0,1)

def heuristic_clv(df: pd.DataFrame) -> pd.Series:
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
    clv = clv + np.random.normal(0, 1500, size=len(clv))
    return clv.clip(lower=0)

# ------------------------
# AUTO-TRAIN MODELS (cached resource)
# ------------------------
@st.cache_resource(show_spinner=True)
def train_models_on_the_fly(df: pd.DataFrame):
    """
    Train classifier + regressor inside the Streamlit environment.
    This runs once per session (cached).
    """
    # Features expected
    feature_cols = [
        'company_size','industry','annual_revenue_musd','employee_count'
