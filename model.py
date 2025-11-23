
import pickle as os, 
import pandas as pd

def load_models(model_dir="models"):
    clf_path = os.path.join(model_dir, "models/lead_scoring_clf.pkl")
    reg_path = os.path.join(model_dir, "models/clv_regressor.pkl")
    models = {}
    if os.path.exists(clf_path):
        models['clf'] = pickle.load(open(clf_path, "rb"))
    if os.path.exists(reg_path):
        models['reg'] = pickle.load(open(reg_path, "rb"))
    return models

def score_leads(df, models):
    X = df.copy()
    if 'clf' in models:
        X['predicted_lead_score'] = models['clf'].predict_proba(X)[:,1]
    if 'reg' in models:
        X['predicted_CLV_usd'] = models['reg'].predict(X)
    return X
