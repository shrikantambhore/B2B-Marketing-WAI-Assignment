# streamlit_app.py
"""
Robust B2B Lead Scoring & CLV Dashboard
- Loads CSV from upload, repo-root, or the uploaded path: /mnt/data/b2b_synthetic_dataset.csv
- Train models on-demand (button)
- Score leads on-demand (button)
- Always render dashboard UI; fallback to heuristics
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

st.set_page_config(page_title="B2B Lead Scoring & CLV Dashboard", layout="wide")

# ------------------------------
# CONFIG: path to the CSV you uploaded earlier
# ------------------------------
UPLOADED_CSV_PATH = "/mnt/data/b2b_synthetic_dataset.csv"   # <--- use this path if you want the uploaded CSV to be used
REPO_CSV_NAME = "b2b_synthetic_dataset.csv"                 # repo-root fallback

# ------------------------------
# HELPERS
# ------------------------------
def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", regex=True)]
    return df

def to_csv_bytes(df: pd.DataFrame) -> BytesIO:
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

def heuristic_lead_score(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    for c in ['website_visits_30d','email_clicks_30d','demo_requested',
              'days_since_last_activity','historical_purchase_count','company_size','deal_stage']:
        if c not in df.columns:
            df[c] = 0
    v = df['website_visits_30d'].astype(float).fillna(0)
    clicks = df['email_clicks_30d'].astype(float).fillna(0)
    demo = df['demo_requested'].astype(float).fillna(0)
    hist = df['historical_purchase_count'].astype(float).fillna(0)
    rec = df['days_since_last_activity'].astype(float).fillna(999)
    raw = v + 2*clicks + 10*demo + 5*hist - 0.02*rec
    raw = raw * df['company_size'].map({'Small':1.0,'Medium':1.8,'Large':3.5}).fillna(1.0)
    raw = raw * df['deal_stage'].map({'Lead':1.0,'MQL':1.3,'SQL':1.7,'Opportunity':2.0,'Closed Won':2.5,'Closed Lost':0.5}).fillna(1.0)
    if raw.max() == raw.min():
        return pd.Series(0.0, index=df.index)
    return ((raw - raw.min())/(raw.max()-raw.min())).clip(0,1)

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

# ------------------------------
# MODEL TRAIN / PREDICT helpers
# ------------------------------
def make_preprocessor():
    numeric = [
        'annual_revenue_musd','employee_count','website_visits_30d','email_opens_30d',
        'email_clicks_30d','demo_requested','days_since_last_activity',
        'historical_purchase_count','months_since_first_contact'
    ]
    categorical = ['company_size','industry','job_title','deal_stage','lead_source','region']
    # support both sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    pre = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', ohe, categorical)
    ], remainder='drop')
    return pre

def train_models(df: pd.DataFrame):
    # ensure features exist
    feature_cols = [
        'company_size','industry','annual_revenue_musd','employee_count','job_title',
        'website_visits_30d','email_opens_30d','email_clicks_30d','demo_requested',
        'days_since_last_activity','deal_stage','lead_source','region',
        'historical_purchase_count','months_since_first_contact'
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0 if c in ['annual_revenue_musd','employee_count','website_visits_30d','email_opens_30d','email_clicks_30d','demo_requested','days_since_last_activity','historical_purchase_count','months_since_first_contact'] else "Unknown"

    if 'converted' not in df.columns or 'CLV_usd' not in df.columns:
        raise ValueError("CSV must contain 'converted' and 'CLV_usd' to train models.")

    X = df[feature_cols]
    y_cls = df['converted']
    pre = make_preprocessor()

    # classifier
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    clf = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))])
    clf.fit(X_train, y_train_cls)

    # regressor (train on same X_train indices)
    reg = Pipeline([('pre', pre), ('reg', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))])
    reg.fit(X_train, df.loc[X_train.index, 'CLV_usd'])

    return clf, reg

def predict_with_models(df: pd.DataFrame, clf, reg):
    """Return predicted_proba array and predicted_clv; wrap errors"""
    X = df.copy()
    X = X.drop(columns=[c for c in ['lead_id','true_conversion_prob','converted','CLV_usd'] if c in X.columns], errors='ignore')
    pred_proba = None
    pred_clv = None
    try:
        pred_proba = clf.predict_proba(X)[:,1]
    except Exception as e:
        raise RuntimeError(f"Classifier prediction error: {e}")
    try:
        pred_clv = reg.predict(X)
    except Exception as e:
        raise RuntimeError(f"Regressor prediction error: {e}")
    return pred_proba, pred_clv

# ------------------------------
# SIDEBAR: inputs & actions
# ------------------------------
st.sidebar.header("Inputs & Models")
uploaded = st.sidebar.file_uploader("Upload leads CSV (optional)", type=["csv"])
use_uploaded_path = st.sidebar.checkbox("Use uploaded-file path (/mnt/data...) if present", value=True)
train_now = st.sidebar.button("Train models now (on filtered dataset)")
score_now = st.sidebar.button("Score leads (use models or heuristics)")

threshold = st.sidebar.slider("Lead score threshold to mark HOT leads", 0.0, 1.0, 0.70, 0.01)
top_k = st.sidebar.number_input("Top K leads to show", min_value=5, max_value=500, value=50, step=5)

# ------------------------------
# LOAD DATA (robust)
# ------------------------------
data_load_errors = []
df = None

# 1) If user uploaded through widget, prefer that
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Loaded CSV from upload widget.")
    except Exception as e:
        data_load_errors.append(f"Upload read error: {e}")

# 2) If user asked to use the uploaded path and file exists on disk (the path we have)
if df is None and use_uploaded_path:
    if os.path.exists(UPLOADED_CSV_PATH):
        try:
            df = pd.read_csv(UPLOADED_CSV_PATH)
            st.sidebar.success(f"Loaded CSV from path: {UPLOADED_CSV_PATH}")
        except Exception as e:
            data_load_errors.append(f"Failed loading {UPLOADED_CSV_PATH}: {e}")

# 3) Try repo-root CSV
if df is None and os.path.exists(REPO_CSV_NAME):
    try:
        df = pd.read_csv(REPO_CSV_NAME)
        st.sidebar.success(f"Loaded CSV from repo root: {REPO_CSV_NAME}")
    except Exception as e:
        data_load_errors.append(f"Failed loading repo CSV {REPO_CSV_NAME}: {e}")

# 4) Try default path (alternate) if present
if df is None and os.path.exists(DEFAULT_CSV_PATH := UPLOADED_CSV_PATH):
    try:
        df = pd.read_csv(DEFAULT_CSV_PATH)
        st.sidebar.success(f"Loaded CSV from default path: {DEFAULT_CSV_PATH}")
    except Exception as e:
        data_load_errors.append(f"Failed loading default path: {e}")

# If still None, create a tiny placeholder so UI shows (but disable scoring)
if df is None:
    st.sidebar.error("No CSV loaded. To enable full dashboard, upload a CSV or commit one to the repo.")
    if data_load_errors:
        st.sidebar.info("Load attempts and errors (first few):")
        for err in data_load_errors[:5]:
            st.sidebar.write("-", err)
    # create a small blank DF so UI components render
    df = pd.DataFrame({
        'lead_id': [],
        'company_size': [],
        'industry': [],
        'annual_revenue_musd': [],
    })
    models_available = False
else:
    df = prepare_input(df)
    models_available = False

# Make a copy for filtering
df_filtered = df.copy()

# ------------------------------
# Dashboard Filters (works even with placeholder)
# ------------------------------
st.title("B2B Lead Scoring & CLV Dashboard (Robust)")
st.markdown("Use filters below, then train models or score leads.")

# show preview
st.subheader("Dataset preview")
st.dataframe(df.head(10))

# filters
st.markdown("---")
st.subheader("Filters")
c1, c2, c3 = st.columns(3)
with c1:
    industries = sorted(df['industry'].dropna().unique()) if 'industry' in df.columns else []
    selected_industries = st.multiselect("Industry", options=industries, default=industries if industries else [])
    company_sizes = sorted(df['company_size'].dropna().unique()) if 'company_size' in df.columns else []
    selected_sizes = st.multiselect("Company size", options=company_sizes, default=company_sizes if company_sizes else [])
with c2:
    regions = sorted(df['region'].dropna().unique()) if 'region' in df.columns else []
    selected_regions = st.multiselect("Region", options=regions, default=regions if regions else [])
    min_visits = st.slider("Min website visits (30d)", 0, int(df['website_visits_30d'].max()) if 'website_visits_30d' in df.columns else 50, 0)
with c3:
    demo_only = st.checkbox("Only demo requested", value=False)
    lead_source_filter = st.multiselect("Lead source", options=sorted(df['lead_source'].dropna().unique()) if 'lead_source' in df.columns else [], default=[])

# apply filters safely
if 'industry' in df.columns and selected_industries:
    df_filtered = df_filtered[df_filtered['industry'].isin(selected_industries)]
if 'company_size' in df.columns and selected_sizes:
    df_filtered = df_filtered[df_filtered['company_size'].isin(selected_sizes)]
if 'region' in df.columns and selected_regions:
    df_filtered = df_filtered[df_filtered['region'].isin(selected_regions)]
if 'website_visits_30d' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['website_visits_30d'] >= min_visits]
if demo_only and 'demo_requested' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['demo_requested'] == 1]
if lead_source_filter and 'lead_source' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['lead_source'].isin(lead_source_filter)]

st.markdown(f"Filtered leads: **{len(df_filtered)}** rows")

# ------------------------------
# Train models on-demand
# ------------------------------
train_error = None
clf = None
reg = None

if train_now:
    try:
        clf, reg = train_models(df_filtered)
        models_available = True
        st.sidebar.success("Models trained on the filtered data.")
    except Exception as e:
        train_error = str(e)
        st.sidebar.error(f"Training failed: {train_error}")
        st.sidebar.info("If training fails, the app will use the heuristic fallback for scoring.")
        models_available = False

# ------------------------------
# Score leads (models if available otherwise heuristics)
# ------------------------------
if score_now:
    df_in = df_filtered.copy()
    if clf is not None and reg is not None:
        try:
            pred_proba, pred_clv = predict_with_models(df_in, clf, reg)
            df_in['predicted_lead_score'] = np.round(pred_proba, 4)
            df_in['predicted_CLV_usd'] = np.round(pred_clv, 2)
            st.success("Predictions produced by trained models.")
        except Exception as e:
            st.warning(f"Model prediction error: {e}. Falling back to heuristics.")
            df_in['predicted_lead_score'] = heuristic_lead_score(df_in)
            df_in['predicted_CLV_usd'] = heuristic_clv(df_in)
    else:
        st.info("No trained models available — using heuristic scoring and CLV.")
        df_in['predicted_lead_score'] = heuristic_lead_score(df_in)
        df_in['predicted_CLV_usd'] = heuristic_clv(df_in)

    df_in['hot_lead'] = df_in['predicted_lead_score'] >= threshold

    # Top K
    st.markdown("---")
    st.subheader(f"Top {top_k} leads by predicted lead score")
    st.dataframe(df_in.sort_values('predicted_lead_score', ascending=False).head(int(top_k)))

    # Metrics
    avg_score = df_in['predicted_lead_score'].mean()
    avg_clv = df_in['predicted_CLV_usd'].mean()
    a,b,c = st.columns(3)
    a.metric("Average predicted lead score", f"{avg_score:.3f}")
    b.metric("Average predicted CLV (USD)", f"${avg_clv:,.0f}")
    c.metric("Hot leads (>= threshold)", f"{df_in['hot_lead'].sum()} / {len(df_in)}")

    # Segment summaries
    st.markdown("### Segment summaries")
    s1, s2 = st.columns(2)
    with s1:
        if 'industry' in df_in.columns:
            industry_table = df_in.groupby('industry')['predicted_lead_score'].mean().sort_values(ascending=False)
            st.table(industry_table)
    with s2:
        if 'region' in df_in.columns:
            region_table = df_in.groupby('region')['predicted_CLV_usd'].mean().sort_values(ascending=False)
            st.table(region_table)

    # Feature importance (best-effort)
    st.markdown("### Model insights")
    if clf is not None:
        try:
            pre = clf.named_steps.get('pre', None)
            pipe_clf = clf.named_steps.get('clf', clf) if hasattr(clf, 'named_steps') else clf
            importances = pipe_clf.feature_importances_ if hasattr(pipe_clf, 'feature_importances_') else None
            if importances is not None and pre is not None:
                try:
                    num_feats = list(pre.transformers_[0][2])
                    cat_feats = pre.transformers_[1][1].get_feature_names_out(pre.transformers_[1][2])
                    feat_names = num_feats + list(cat_feats)
                    fi_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(20)
                    st.table(fi_df)
                except Exception as e:
                    st.write("Feature importance extraction failed:", e)
            else:
                st.write("Model present but feature importances not available.")
        except Exception as e:
            st.write("Could not extract feature importances automatically:", e)
    else:
        st.write("No trained model present — feature importances unavailable (use heuristics).")

    # Download
    st.download_button("Download scored leads CSV", to_csv_bytes(df_in), file_name="scored_leads.csv", mime="text/csv")

    # Show scored sample
    st.markdown("### Scored dataset (first 200 rows)")
    st.dataframe(df_in.head(200))

else:
    st.info("Use the 'Train models now' button to train a model on the filtered dataset (optional). Then use 'Score leads' to run predictions (or just use heuristics).")

# Footer
st.markdown("---")
st.caption("Note: If you want models pre-saved as pickles in the repo 'models/' folder, we can change the app to load them. Current app trains on-demand and falls back to heuristics if needed.")
