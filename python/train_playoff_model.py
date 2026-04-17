#!/usr/bin/env python3
"""MLS Playoff Model Trainer."""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

PLAYOFF_CSV = DATA_DIR / "playoff_data.csv"
FEATURE_NAMES = ["elo_diff"]


def main():
    print("MLS Playoff Model Trainer"); print("=" * 40)
    if not PLAYOFF_CSV.exists():
        print("Run fetch_playoff_data.py first."); sys.exit(1)

    df = pd.read_csv(PLAYOFF_CSV)
    print(f"Loaded {len(df)} playoff games")
    feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
    seasons = sorted(df["season"].unique())
    print(f"Seasons: {seasons}\n")
    lr_accs = []

    for i, ts in enumerate(seasons[1:], 1):
        train = df[df["season"].isin(seasons[:i])]; test = df[df["season"]==ts]
        if len(train) < 8 or len(test) < 4: continue
        X_tr = train[feat_cols].fillna(0).values; y_tr = train["label"].values
        X_te = test[feat_cols].fillna(0).values;  y_te = test["label"].values
        sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        lr = LogisticRegression(C=1.0, max_iter=1000); lr.fit(X_tr_s, y_tr)
        lr_p = np.clip(lr.predict_proba(X_te_s)[:, 1], 0.01, 0.99)
        acc = accuracy_score(y_te, lr_p >= 0.5); lr_accs.append(acc)
        print(f"  {ts}: n={len(test)}, LR={acc:.3f}")

    if lr_accs: print(f"\nAvg LR accuracy: {np.mean(lr_accs):.4f}")

    X_all = df[feat_cols].fillna(0).values; y_all = df["label"].values
    sc_f = StandardScaler(); X_all_s = sc_f.fit_transform(X_all)
    lr_f = LogisticRegression(C=1.0, max_iter=1000); lr_f.fit(X_all_s, y_all)
    (MODEL_DIR / "playoff_coefficients.json").write_text(json.dumps({
        "intercept": float(lr_f.intercept_[0]), "coefficients": lr_f.coef_[0].tolist(), "feature_names": feat_cols,
    }, indent=2))
    (MODEL_DIR / "playoff_metadata.json").write_text(json.dumps({
        "sport": "MLS", "cv_accuracy_lr": float(np.mean(lr_accs)) if lr_accs else None,
        "playoff_seasons": [int(s) for s in seasons],
    }, indent=2))
    print(f"Saved to {MODEL_DIR}/playoff_*.json")

if __name__ == "__main__":
    main()
