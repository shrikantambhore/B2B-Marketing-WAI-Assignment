# model.py
import os
import pickle
from typing import Dict, Optional
import pandas as pd

def _safe_load_pickle(path: str) -> Optional[object]:
    """Return loaded object or None on failure (and do not raise)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # In a real app you might log this exception instead of printing
        print(f"[model.py] Failed to load pickle {path}: {e}")
        return None

def load_models(model_dir: str = "models") -> Dict[str, object]:
    """
    Load models from a models directory.
    Default expects:
      models/lead_scoring_clf.pkl
      models/clv_regressor.pkl

    Returns a dict which may contain keys 'clf' and/or 'reg'.
    """
    models = {}
    # canonical paths to try (repo-local, then /mnt/data fallback)
    clf_candidates = [
        os.path.join(model_dir, "lead_scoring_clf.pkl"),
        os.path.join("/mnt/data", "lead_scoring_clf.pkl"),
        os.path.join("/mnt/data", model_dir, "lead_scoring_clf.pkl"),
    ]
    reg_candidates = [
        os.path.join(model_dir, "clv_regressor.pkl"),
        os.path.join("/mnt/data", "clv_regressor.pkl"),
        os.path.join("/mnt/data", model_dir, "clv_regressor.pkl"),
    ]

    # try classifier
    for p in clf_candidates:
        obj = _safe_load_pickle(p)
        if obj is not None:
            models["clf"] = obj
            models["_clf_path"] = p
            break

    # try regressor
    for p in reg_candidates:
        obj = _safe_load_pickle(p)
        if obj is not None:
            models["reg"] = obj
            models["_reg_path"] = p
            break

    return models

def _prepare_X_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop likely target/auxiliary columns so the features match training X.
    You may customize if your training pipeline expects specific columns.
    """
    df = df.copy()
    drop_candidates = ["lead_id", "true_conversion_prob", "converted", "CLV_usd"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")
    # Also drop Unnamed columns from CSVs
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def score_leads(df: pd.DataFrame, models: Dict[str, object]) -> pd.DataFrame:
    """
    Score leads using loaded models (or raise informative errors).
    - df: pandas DataFrame (raw)
    - models: dict from load_models(), may contain 'clf' and/or 'reg'

    Returns a DataFrame with added columns:
      - predicted_lead_score (float in [0,1]) when classifier present
      - predicted_CLV_usd (float) when regressor present
    """
    df_out = df.copy()
    X = _prepare_X_for_model(df_out)

    # CLASSIFIER
    if "clf" in models and models["clf"] is not None:
        clf = models["clf"]
        try:
            # if pipeline, ensure we pass raw X (pipeline contains preprocessing)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)[:, 1]
            else:
                # fallback: if only predict exists, use predict (0/1) as score
                proba = clf.predict(X).astype(float)
            df_out["predicted_lead_score"] = pd.Series(proba, index=df_out.index)
        except Exception as e:
            print(f"[model.py] Classifier prediction failed: {e}")
            # do not raise; omitting column indicates failure
    else:
        # No classifier loaded — do nothing (upstream app should fallback to heuristics)
        pass

    # REGRESSOR
    if "reg" in models and models["reg"] is not None:
        reg = models["reg"]
        try:
            pred_clv = reg.predict(X)
            df_out["predicted_CLV_usd"] = pd.Series(pred_clv, index=df_out.index)
        except Exception as e:
            print(f"[model.py] Regressor prediction failed: {e}")
    else:
        # No regressor loaded
        pass

    return df_out
