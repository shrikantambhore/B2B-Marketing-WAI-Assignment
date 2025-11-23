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
        'company_size','industry','annual_revenue_musd','employee_count','job_title',
        'website_visits_30d','email_opens_30d','email_clicks_30d','demo_requested',
        'days_since_last_activity','deal_stage','lead_source','region',
        'historical_purchase_count','months_since_first_contact'
    ]

    # Ensure columns exist; if missing fill defaults
    for c in feature_cols:
        if c not in df.columns:
            if c in ['annual_revenue_musd','employee_count','website_visits_30d','email_opens_30d',
                     'email_clicks_30d','demo_requested','days_since_last_activity','historical_purchase_count','months_since_first_contact']:
                df[c] = 0
            else:
                df[c] = "Unknown"

    X = df[feature_cols]
    # Ensure target columns exist
    if 'converted' not in df.columns or 'CLV_usd' not in df.columns:
        raise ValueError("Training requires 'converted' and 'CLV_usd' columns in the CSV.")

    y_cls = df['converted']
    y_reg = df['CLV_usd']

    categorical = ['company_size','industry','job_title','deal_stage','lead_source','region']
    numeric = [
        'annual_revenue_musd','employee_count','website_visits_30d','email_opens_30d','email_clicks_30d',
        'demo_requested','days_since_last_activity','historical_purchase_count','months_since_first_contact'
    ]

try:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', encoder, categorical),
])



    # Train-test split (stratify classification)
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    # Classifier pipeline
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    ])
    clf.fit(X_train, y_train_cls)

    # Regressor pipeline - train regressor on CLV values for same X_train
    reg = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))
    ])
    reg.fit(X_train, df.loc[X_train.index, 'CLV_usd'])

    return clf, reg

# ------------------------
# Sidebar UI
# ------------------------
st.sidebar.header("Inputs & Models")
uploaded = st.sidebar.file_uploader("Upload leads CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox("Use default synthetic dataset", value=True)
st.sidebar.markdown("Model options: models will be trained automatically in this environment (no pickles required).")

threshold = st.sidebar.slider("Lead score threshold to mark HOT leads", 0.0, 1.0, 0.70, 0.01)
top_k = st.sidebar.number_input("Top K leads to show", min_value=5, max_value=500, value=50, step=5)

# ------------------------
# Load dataset
# ------------------------
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Loaded uploaded CSV")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    # try repo-local default CSV (if committed to repo)
    if os.path.exists(DEFAULT_CSV_PATH) and use_default:
        try:
            df = pd.read_csv(DEFAULT_CSV_PATH)
            st.sidebar.info(f"Loaded default CSV: {DEFAULT_CSV_PATH}")
        except Exception as e:
            st.sidebar.error(f"Failed to load default CSV ({DEFAULT_CSV_PATH}): {e}")
            st.stop()
    else:
        st.sidebar.warning("No CSV uploaded and default not available; upload a CSV to proceed.")
        st.stop()

df = prepare_input(df)
st.title("B2B Lead Scoring & CLV Dashboard (Auto-train)")

# ------------------------
# Train models (auto)
# ------------------------
clf = None
reg = None
training_msg = None
try:
    clf, reg = train_models_on_the_fly(df)
    st.sidebar.success("Models trained successfully in-session.")
    training_msg = "Models trained in this session (no pickles used)."
except Exception as e:
    # If training fails, we'll fallback to heuristics
    st.sidebar.warning(f"Model training failed: {e}. App will use heuristic scoring.")
    clf, reg = None, None
    training_msg = "Training failed; using heuristics."

# ------------------------
# Main UI & Predictions
# ------------------------
st.markdown("### Dataset preview")
st.dataframe(df.head(10))

st.markdown("### Run predictions")
if st.button("Score leads and predict CLV"):
    df_in = df.copy()

    use_model = (clf is not None and reg is not None)
    if use_model:
        st.success("Using trained models for predictions.")
        # prepare X by dropping label columns if present
        X = df_in.drop(columns=[c for c in ['lead_id','true_conversion_prob','converted','CLV_usd'] if c in df_in.columns], errors='ignore')
        try:
            pred_proba = clf.predict_proba(X)[:, 1]
            pred_clv = reg.predict(X)
        except Exception as e:
            # if any error occurs, fallback to heuristics
            st.warning("Model prediction failed due to input mismatch. Falling back to heuristics. Error: " + str(e))
            pred_proba = heuristic_lead_score(df_in)
            pred_clv = heuristic_clv(df_in)
    else:
        st.info("No trained models available — using heuristic scoring and CLV.")
        pred_proba = heuristic_lead_score(df_in)
        pred_clv = heuristic_clv(df_in)

    df_in['predicted_lead_score'] = np.round(pred_proba, 4)
    df_in['predicted_CLV_usd'] = np.round(pred_clv, 2)
    df_in['hot_lead'] = df_in['predicted_lead_score'] >= threshold

    st.success("Predictions completed — see results below.")

    # Top K leads
    st.markdown(f"### Top {top_k} leads by predicted lead score")
    st.dataframe(df_in.sort_values('predicted_lead_score', ascending=False).head(int(top_k)))

    # Metrics
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

    # Feature importance (if model present)
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

    # Download scored CSV
    csv_bytes = to_csv_bytes(df_in)
    st.download_button("Download scored leads CSV", csv_bytes, file_name="scored_leads.csv", mime="text/csv")

    st.markdown("### Full scored dataset (first 200 rows)")
    st.dataframe(df_in.head(200))

    if st.checkbox("Show only HOT leads"):
        st.dataframe(df_in[df_in['hot_lead']].sort_values('predicted_lead_score', ascending=False).head(200))

else:
    st.info("Click the button 'Score leads and predict CLV' to compute lead scores and CLV for the dataset.")

# Footer / notes
st.markdown("---")
st.markdown("**Notes & next steps:**")
st.markdown("""
- Models are trained inside the Streamlit environment at startup to avoid pickle compatibility issues.
- If you prefer to use pre-trained pickles from a models/ folder, remove the auto-train block and load pickles instead.
- For production: use a model server, schedule re-training, or export to a portable format (ONNX) for stable deployments.
""")
