
# B2B Lead Scoring & CLV - Demo Repo

This GitHub-ready repo contains:
- `b2b_synthetic_dataset.csv` : Synthetic dataset (5,000 rows) for demo and model training.
- `streamlit_app.py` : Minimal Streamlit app to preview dataset and run scoring (requires models).
- `model.py` : Helper functions to load models and score leads.
- `models/` : Place pre-trained model pickles here (`lead_scoring_clf.pkl`, `clv_regressor.pkl`).
- `requirements.txt` : Python dependencies.

## How to use
1. Clone the repo.
2. Place trained model pickles into `models/` (optional for demo).
3. `pip install -r requirements.txt`
4. `streamlit run streamlit_app.py`
