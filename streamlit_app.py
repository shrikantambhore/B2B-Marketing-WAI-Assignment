
"""Streamlit app for B2B Lead Scoring and CLV prediction (demo)
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle, os

MODEL_DIR = "models"

st.set_page_config(page_title="B2B Lead Scoring Dashboard", layout="wide")
st.title("B2B Lead Scoring & CLV Dashboard (Synthetic Demo)")

st.sidebar.header("Upload / Inputs")
uploaded = st.sidebar.file_uploader("Upload leads CSV (optional)", type=["csv"])
if uploaded is not None:
    df_input = pd.read_csv(uploaded)
else:
    st.sidebar.write("Using synthetic sample dataset included with the project.")
    df_input = pd.read_csv("b2b_synthetic_dataset.csv")

st.write("Dataset preview:")
st.dataframe(df_input.head(10))

st.write("This demo expects pre-trained models in the /models folder (lead_scoring_clf.pkl and clv_regressor.pkl).")
st.write("If models are not present, you can still inspect the dataset or upload your own predictions file.")
