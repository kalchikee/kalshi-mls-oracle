#!/usr/bin/env python3
"""
MLS Oracle v4.2 -- Training Pipeline
Loads enriched training data from data/training_data.csv (built by enrich_features.py),
trains a multinomial logistic regression (H/D/A), and exports model artifacts.

If training_data.csv doesn't exist, falls back to fetching live from ESPN and
building features on the fly (legacy v4.1 behavior).

Artifacts exported to data/model/:
  coefficients.json   -- three coefficient vectors (home/draw/away) + intercepts
  scaler.json         -- mean/std for each feature (StandardScaler params)
  metadata.json       -- feature names, training seasons, walk-forward CV results
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1"

TRAINING_SEASONS = list(range(2019, 2026))  # 2019-2025 inclusive
K_FACTOR           = 20
HOME_ADVANTAGE_ELO = 70
GOAL_CAP           = 4
DEFAULT_ELO        = 1500
OFFSEASON_REGRESS  = 0.75

ALTITUDE_VENUES = {"COL": 5282, "RSL": 4327}

CITY_COORDS = {
    "ATL": (33.7554, -84.4009), "ATX": (30.3870, -97.7191), "CHI": (41.8623, -87.6167),
    "CIN": (39.1116, -84.5232), "CLB": (39.9675, -82.9913), "COL": (39.8061, -104.9776),
    "DAL": (32.9120, -97.0601), "DC":  (38.8682, -77.0122), "HOU": (29.7527, -95.4101),
    "KC":  (39.1219, -94.8232), "LA":  (34.0144, -118.2872), "LAFC": (34.0126, -118.2842),
    "MIA": (25.9580, -80.2389), "MIN": (44.9732, -93.1675), "MONT": (45.4678, -73.6748),
    "MTL": (45.4678, -73.6748),
    "NE":  (42.0914, -71.2643), "NSH": (36.1306, -86.7717), "NY":  (40.7315, -74.0685),
    "RBNY": (40.7315, -74.0685),
    "NYC": (40.7505, -73.9934), "ORL": (28.5392, -81.3890), "PHI": (39.9012, -75.1674),
    "POR": (45.5215, -122.6916), "RSL": (40.5830, -111.8927), "SJ":  (37.3519, -121.9250),
    "SEA": (47.5952, -122.3316), "SKC": (38.8895, -94.8234), "STL": (38.6322, -90.1987),
    "TOR": (43.6333, -79.4189), "VAN": (49.2767, -123.1108), "SD":  (32.7573, -117.1664),
}

# Feature columns -- matches enrich_features.py output
FEATURE_NAMES = [
    "elo_diff",
    "ppg_diff",
    "goals_for_diff",
    "goals_against_diff",
    "form_5g_diff",
    "rest_days_diff",
    "home_advantage_diff",
    "xg_for_diff",
    "xg_against_diff",
    "possession_diff",
    "draw_tendency_diff",
    "travel_distance",
    "midweek_flag",
    "altitude_flag",
    "is_home",
    # Poisson priors (computed analytically from rolling xG)
    "poisson_home",
    "poisson_draw",
    "poisson_away",
]

TRAINING_DATA_CSV = "data/training_data.csv"
MATCH_CACHE_PATH  = "data/matches_cache.json"


# ---------------------------------------------------------------------------
# CSV-based training (primary path)
# ---------------------------------------------------------------------------
def load_csv_dataset():
    """Load training data from enriched CSV."""
    df = pd.read_csv(TRAINING_DATA_CSV)
    print(f"Loaded {len(df)} samples from {TRAINING_DATA_CSV}")

    # Validate all feature columns exist
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in CSV: {missing}")

    seasons = sorted(int(s) for s in df["season"].unique())
    print(f"Seasons: {seasons}")
    for s in seasons:
        n = (df["season"] == s).sum()
        print(f"  Season {s}: {n} samples")

    return df, seasons


def walk_forward_cv_csv(df, seasons):
    """Walk-forward CV using enriched CSV data."""
    results = []
    for fold_idx in range(1, len(seasons)):
        train_seasons = seasons[:fold_idx]
        test_season = seasons[fold_idx]

        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]

        X_train = train_df[FEATURE_NAMES].fillna(0).values.astype(np.float32)
        y_train = train_df["outcome"].values
        X_test = test_df[FEATURE_NAMES].fillna(0).values.astype(np.float32)
        y_test = test_df["outcome"].values

        if len(X_train) < 50 or len(X_test) < 10:
            print(f"  CV Fold {fold_idx}: skipping (train={len(X_train)}, test={len(X_test)})")
            continue

        print(f"\n  CV Fold {fold_idx}: train={list(train_seasons)}, test={test_season}")

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        model = LogisticRegression(C=0.8, solver="lbfgs", max_iter=500, random_state=42)
        model.fit(X_tr_scaled, y_train)

        preds = model.predict_proba(X_te_scaled)
        classes = list(model.classes_)

        ll = log_loss(y_test, preds)
        acc = (model.predict(X_te_scaled) == y_test).mean()

        # Brier score (3-way)
        outcome_map = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}
        brier = np.mean([
            sum((preds[i][classes.index(c)] - outcome_map[y_test[i]][j]) ** 2
                for j, c in enumerate(["H", "D", "A"]))
            for i in range(len(y_test))
        ])

        print(f"    n_test={len(X_test)}, acc={acc:.3f}, log_loss={ll:.4f}, brier={brier:.4f}")
        results.append({
            "fold": fold_idx,
            "train_seasons": list(train_seasons),
            "test_season": int(test_season),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": round(float(acc), 4),
            "log_loss": round(float(ll), 4),
            "brier_score": round(float(brier), 4),
        })

    return results


def train_final_model_csv(df):
    """Train final model on all CSV data."""
    X = df[FEATURE_NAMES].fillna(0).values.astype(np.float32)
    y = df["outcome"].values

    print(f"\nTraining final model on {len(X)} samples, {len(FEATURE_NAMES)} features...")
    if len(X) < 50:
        raise RuntimeError("Insufficient training data")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=0.8, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    print(f"  Classes: {model.classes_}")
    print(f"  Train accuracy: {(model.predict(X_scaled) == y).mean():.4f}")

    return model, scaler


# ---------------------------------------------------------------------------
# Export artifacts
# ---------------------------------------------------------------------------
def export_artifacts(model, scaler, cv_results, seasons, n_matches):
    os.makedirs("data/model", exist_ok=True)

    classes = list(model.classes_)
    coef_map = {cls: model.coef_[i].tolist() for i, cls in enumerate(classes)}
    intercept_map = {cls: float(model.intercept_[i]) for i, cls in enumerate(classes)}

    # coefficients.json -- structure expected by metaModel.ts
    coef_json = {
        "home":  {"coefficients": coef_map.get("H", coef_map.get(classes[0])), "intercept": intercept_map.get("H", 0.0)},
        "draw":  {"coefficients": coef_map.get("D", coef_map.get(classes[1])), "intercept": intercept_map.get("D", 0.0)},
        "away":  {"coefficients": coef_map.get("A", coef_map.get(classes[2])), "intercept": intercept_map.get("A", 0.0)},
        "classes": classes,
        "feature_names": FEATURE_NAMES,
    }
    with open("data/model/coefficients.json", "w") as f:
        json.dump(coef_json, f, indent=2)

    # scaler.json
    scaler_json = {
        "mean": scaler.mean_.tolist(),
        "std":  scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open("data/model/scaler.json", "w") as f:
        json.dump(scaler_json, f, indent=2)

    # metadata.json
    avg_acc = sum(r["accuracy"] for r in cv_results) / len(cv_results) if cv_results else None
    avg_ll  = sum(r["log_loss"] for r in cv_results) / len(cv_results) if cv_results else None
    avg_bs  = sum(r["brier_score"] for r in cv_results) / len(cv_results) if cv_results else None

    meta = {
        "version": "4.2.0",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "training_seasons": [int(s) for s in seasons],
        "n_matches": n_matches,
        "model_type": "MultinomialLogisticRegression",
        "regularization_C": 0.8,
        "solver": "lbfgs",
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "walk_forward_cv": cv_results,
        "avg_cv_accuracy": round(avg_acc, 4) if avg_acc else None,
        "avg_cv_log_loss": round(avg_ll, 4) if avg_ll else None,
        "avg_cv_brier": round(avg_bs, 4) if avg_bs else None,
    }
    with open("data/model/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifacts saved to data/model/")
    print(f"  coefficients.json -- {len(FEATURE_NAMES)} features x {len(classes)} classes")
    print(f"  scaler.json       -- StandardScaler params")
    print(f"  metadata.json     -- CV results, provenance")
    if avg_acc:
        print(f"\nWalk-forward CV summary:")
        print(f"  Mean accuracy:   {avg_acc:.4f}")
        print(f"  Mean log-loss:   {avg_ll:.4f}")
        print(f"  Mean Brier:      {avg_bs:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MLS Oracle v4.2 -- Training Pipeline")
    print("=" * 60)

    if os.path.exists(TRAINING_DATA_CSV):
        # Primary path: load from enriched CSV (built by enrich_features.py)
        print(f"\nUsing enriched training data from {TRAINING_DATA_CSV}")
        df, seasons = load_csv_dataset()

        # Walk-forward CV
        print("\nRunning walk-forward cross-validation...")
        cv_results = walk_forward_cv_csv(df, seasons)

        # Train final model
        model, scaler = train_final_model_csv(df)

        # Export
        export_artifacts(model, scaler, cv_results, seasons, len(df))
    else:
        print(f"\nNo {TRAINING_DATA_CSV} found.")
        print("Run enrich_features.py first to build the training dataset:")
        print("  python python/enrich_features.py")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
