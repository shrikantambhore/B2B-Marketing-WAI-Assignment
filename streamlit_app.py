# streamlit_app.py
"""
B2B Lead Scoring & CLV Dashboard (Optimized model loading)
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import requests
from io import BytesIO, StringIO
from pathlib import Path
from typing import Tuple

st.set_page_config(page_title="B2B Lead Scoring & CLV Dashboard", layout="wide")

# ------------------------
# Configuration
# ------------------------
# Default dataset path (you uploaded this earlier in the environment)
DEFAULT_CSV_PATH = "/mnt/data/b2b_synthetic_dataset.csv"

# Model filenames expected in a models/ folder in the repo root
MODEL_FILENAMES = {
    "clf": "lead_scoring_clf.pkl",
    "reg": "clv_regressor.pkl"
}

# ------------------------
# Helpers: model loading
# ------------------------
@st.cache_data(show_spinner=False)
def load_pickle_from_local(path: str):
    """Load a pickle from a local file path if it exists, else return None."""
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.debug(f"Local pickle load failed for {path}: {e}")
    return None

@st.cache_data(show_spinner=False)
def download_file(url: str, timeout=20) -> bytes:
    """Download a file and return bytes, or raise."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content

@st.cache_data(show_spinner=False)
def fetch_model_from_github(owner_repo: str, branch: str, filename: str) -> bytes:
    """
    Download a file from GitHub raw URL.
    owner_repo: "owner/repo"
    branch: branch name (e.g., "main")
    filename: path within repo (e.g., "models/lead_scoring_clf.pkl")
    Returns bytes if successful, else raises.
    """
    raw_url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{filename}"
    return download_file(raw_url)

@st.cache_data(show_spinner=False)
def load_models(github_repo: str = "", github_branch: str = "main") -> Tuple[object, object, dict]:
    """
    Try loading classifier and regressor in this order:
      1) Local models/ folder in repo (models/<filename>)
      2) Attempt to fetch from GitHub raw (if github_repo provided)
      3) Return (None, None) if both fail.

    Returns (clf, reg, info) where info contains paths/urls tried (for debugging).
    """
    info = {"tried_local": [], "tried_github": [], "loaded": {"clf": False, "reg": False}}
    clf = None
    reg = None

    # 1) Try repo-local models folder
    repo_local_models = Path("models")
    for key, fname in MODEL_FILENAMES.items():
        p = repo_local_models / fname
        info["tried_local"].append(str(p))
        m = load_pickle_from_local(str(p))
        if m is not None:
            if key == "clf":
                clf = m
                info["loaded"]["clf"] = True
            else:
                reg = m
                info["loaded"]["reg"] = True

    # 2) Try /mnt/data (useful in ephemeral environments or when uploading)
    for key, fname in MODEL_FILENAMES.items():
        if (key == "clf" and clf is not None) or (key == "reg" and reg is not None):
            continue
        p = Path("/mnt/data") / fname
        info["tried_local"].append(str(p))
        m = load_pickle_from_local(str(p))
        if m is not None:
            if key == "clf":
                clf = m
                info["loaded"]["clf"] = True
            else:
                reg = m
                info["loaded"]["reg"] = True

    # 3) If github repo provided, try raw.githubusercontent.com
    if github_repo:
        for key, fname in MODEL_FILENAMES.items():
            if (key == "clf" and clf is not None) or (key == "reg" and reg is not None):
                continue
            target = f"models/{fname}"
            info["tried_github"].append(target)
            try:
                b = fetch_model_from_github(github_repo, github_branch, target)
                # load bytes into pickle
                obj = pickle.loads(b)
                if key == "clf":
                    clf = obj
                    info["loaded"]["clf"] = True
                else:
                    reg = obj
                    info["loaded"]["reg"] = True
            except Exception as e:
                # store failure message (do not raise)
                info.setdefault("errors", []).append({"target": target, "error": str(e)})

    return clf, reg, info

# ------------------------
# Heuristics (fallback)
# ------------------------
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

def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", regex=True)]
    return df

def to_csv_bytes(df: pd.DataFrame) -> BytesIO:
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# ------------------------
# Sidebar UI: allow user to specify GitHub repo/branch
# ------------------------
st.sidebar.header("Inputs & Models")
uploaded = st.sidebar.file_uploader("Upload leads CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox("Use default synthetic dataset", value=True)

st.sidebar.markdown("**Model loading from GitHub (optional)**")
gh_repo = st.sidebar.text_input("GitHub repo (owner/repo)", value="")  # e.g. "yourname/yourrepo"
gh_branch = st.sidebar.text_input("Branch", value="main")

# show where models will be looked for
st.sidebar.write("Model filenames expected in `models/` folder:")
st.sidebar.write(", ".join(MODEL_FILENAMES.values()))

# load models (attempt local first, then GitHub raw if provided)
clf, reg, load_info = load_models(github_repo=gh_repo.strip(), github_branch=gh_branch.strip())
st.sidebar.write(f"Classifier loaded: {'Yes' if clf is not None else 'No'}")
st.sidebar.write(f"Regressor loaded: {'Yes' if reg is not None else 'No'}")

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
    # try default CSV path (useful during dev or on Streamlit server if included in repo)
    if os.path.exists(DEFAULT_CSV_PATH) and use_default:
        df = pd.read_csv(DEFAULT_CSV_PATH)
        st.sidebar.info(f"Loaded default CSV: {DEFAULT_CSV_PATH}")
    else:
        # also check repo-local CSV (useful when CSV is committed to repo root)
        if os.path.exists("b2b_synthetic_dataset.csv") and use_default:
            df = pd.read_csv("b2b_synthetic_dataset.csv")
            st.sidebar.info("Loaded default CSV from repo root: b2b_synthetic_dataset.csv")
        else:
            st.sidebar.warning("No CSV uploaded and default not available; upload a CSV to proceed.")
            st.stop()

df = prepare_input(df)
st.title("B2B Lead Scoring & CLV Dashboard (Optimized)")

# ------------------------
# Run predictions
# ------------------------
st.markdown("### Dataset preview")
st.dataframe(df.head(10))

st.markdown("### Run predictions")
if st.button("Score leads and predict CLV"):
    df_in = df.copy()

    use_model = (clf is not None and reg is not None)
    if use_model:
        st.success("Using trained models for predictions.")
        # attempt to build X consistent with training expectations:
        # drop obvious label columns if present
        X = df_in.drop(columns=[c for c in ['lead_id','true_conversion_prob','converted','CLV_usd'] if c in df_in.columns], errors='ignore')
        try:
            pred_proba = clf.predict_proba(X)[:, 1]
            pred_clv = reg.predict(X)
        except Exception as e:
            st.warning("Model prediction failed due to input mismatch. Falling back to heuristics. Error: " + str(e))
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
        st.write("No model available — feature importances unavailable (use heuristics).")

    # Download scored file
    csv_bytes = to_csv_bytes(df_in)
    st.download_button("Download scored leads CSV", csv_bytes, file_name="scored_leads.csv", mime="text/csv")

    # show first 200 rows and hot leads option
    st.markdown("### Full scored dataset (first 200 rows)")
    st.dataframe(df_in.head(200))

    if st.checkbox("Show only HOT leads"):
        st.dataframe(df_in[df_in['hot_lead']].sort_values('predicted_lead_score', ascending=False).head(200))

else:
    st.info("Click the button 'Score leads and predict CLV' to compute lead scores and CLV for the dataset.")

# Footer
st.markdown("---")
st.markdown("**Notes & next steps:**")
st.markdown("""
- Place trained pickles in `models/` folder at the repo root (or provide GitHub repo in the sidebar).
- If using GitHub raw fetch, set `GitHub repo` to `owner/repo` and `Branch` to the branch name.
- For production, consider serving models from a model store or secured artifact storage instead of raw pickles.
""")
