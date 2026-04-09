#!/usr/bin/env python3
"""
MLS Oracle v4.1 — Backtest Report
Loads trained model artifacts and runs a detailed walk-forward backtest,
printing per-season and aggregate performance metrics.

Usage:
  python python/backtest.py                  # uses data from train_model.py run
  python python/backtest.py --seasons 2023   # evaluate only 2023
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, confusion_matrix

# Reuse helpers from train_model
sys.path.insert(0, os.path.dirname(__file__))
from train_model import (
    TRAINING_SEASONS, FEATURE_NAMES, EloEngine, TeamStats,
    build_feature_vector, fetch_all_matches, is_cup_congestion,
    haversine_miles, CITY_COORDS,
)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
def load_artifacts():
    base = "data/model"
    with open(f"{base}/coefficients.json") as f:
        coef_data = json.load(f)
    with open(f"{base}/scaler.json") as f:
        scaler_data = json.load(f)
    with open(f"{base}/metadata.json") as f:
        meta = json.load(f)
    return coef_data, scaler_data, meta

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def predict_proba(features: list, coef_data: dict, scaler_data: dict):
    """Manual softmax prediction matching metaModel.ts logic."""
    x = np.array(features, dtype=np.float64)
    mean = np.array(scaler_data["mean"])
    std  = np.array(scaler_data["std"])
    x_scaled = (x - mean) / np.maximum(std, 1e-8)

    logit_h = float(np.dot(coef_data["home"]["coefficients"], x_scaled) + coef_data["home"]["intercept"])
    logit_d = float(np.dot(coef_data["draw"]["coefficients"], x_scaled) + coef_data["draw"]["intercept"])
    logit_a = float(np.dot(coef_data["away"]["coefficients"], x_scaled) + coef_data["away"]["intercept"])

    probs = softmax([logit_h, logit_d, logit_a])
    return {"H": probs[0], "D": probs[1], "A": probs[2]}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def brier_score_3way(probs_list, outcomes):
    """3-way Brier score: mean of sum of (p - actual)^2 across H/D/A."""
    outcome_map = {"H": [1,0,0], "D": [0,1,0], "A": [0,0,1]}
    scores = []
    for probs, outcome in zip(probs_list, outcomes):
        actual = outcome_map[outcome]
        score = (probs["H"] - actual[0])**2 + (probs["D"] - actual[1])**2 + (probs["A"] - actual[2])**2
        scores.append(score)
    return sum(scores) / len(scores)

def accuracy(probs_list, outcomes):
    correct = sum(1 for p, o in zip(probs_list, outcomes) if max(p, key=p.get) == o)
    return correct / len(outcomes)

def high_conviction_accuracy(probs_list, outcomes, threshold=0.60):
    """Accuracy on picks where top probability >= threshold."""
    hc = [(p, o) for p, o in zip(probs_list, outcomes) if max(p.values()) >= threshold]
    if not hc:
        return None, 0
    correct = sum(1 for p, o in hc if max(p, key=p.get) == o)
    return correct / len(hc), len(hc)

def roi_simulation(probs_list, outcomes, kelly_fraction=0.25):
    """
    Simulate flat-stake ROI using model edge over implied fair odds.
    Assumes fair Poisson odds (no vig). Real betting would use market odds.
    """
    bankroll = 1000.0
    bet_size = 10.0
    wins = losses = 0

    for probs, outcome in zip(probs_list, outcomes):
        pick = max(probs, key=probs.get)
        if probs[pick] < 0.55:
            continue  # skip marginal picks

        # Fair decimal odds = 1 / probability
        fair_odds = 1.0 / probs[pick]
        # Apply typical 5% vig: offered_odds = fair_odds * 0.95
        offered_odds = fair_odds * 0.95

        if pick == outcome:
            bankroll += bet_size * (offered_odds - 1)
            wins += 1
        else:
            bankroll -= bet_size
            losses += 1

    n_bets = wins + losses
    roi = (bankroll - 1000.0) / (n_bets * bet_size) * 100 if n_bets > 0 else 0
    return roi, wins, losses

# ---------------------------------------------------------------------------
# Build evaluation dataset for specific seasons
# ---------------------------------------------------------------------------
def evaluate_seasons(target_seasons: list, all_matches_by_season: dict,
                     coef_data: dict, scaler_data: dict):
    """
    For each target season, train on all prior seasons, then predict on target.
    Returns flat lists of predictions and outcomes.
    """
    all_probs = []
    all_outcomes = []
    season_results = {}

    for season in target_seasons:
        # Prior seasons for Elo/stats warm-up
        prior_seasons = [s for s in TRAINING_SEASONS if s < season]
        if not prior_seasons:
            print(f"  No prior seasons for {season} — skipping")
            continue

        print(f"\n  Evaluating season {season} (trained on {prior_seasons})...")

        # Warm up Elo on prior seasons
        elo = EloEngine()
        stats = TeamStats()
        matchweek_counter = defaultdict(int)
        last_match_dates = {}

        for ps in sorted(prior_seasons):
            stats.reset_season()
            matchweek_counter.clear()
            for m in all_matches_by_season.get(ps, []):
                h, a = m["home_team"], m["away_team"]
                stats.update(m)
                elo.update(h, a, m["home_score"], m["away_score"], ps)
                matchweek_counter[h] += 1
                matchweek_counter[a] += 1
                last_match_dates[h] = m["date"]
                last_match_dates[a] = m["date"]

        # Now evaluate target season
        stats.reset_season()
        matchweek_counter.clear()

        season_probs = []
        season_outcomes = []
        skipped = 0

        for m in all_matches_by_season.get(season, []):
            h, a = m["home_team"], m["away_team"]

            fv = build_feature_vector(m, elo, stats, matchweek_counter, last_match_dates)

            stats.update(m)
            elo.update(h, a, m["home_score"], m["away_score"], season)
            matchweek_counter[h] += 1
            matchweek_counter[a] += 1
            last_match_dates[h] = m["date"]
            last_match_dates[a] = m["date"]

            if fv is None:
                skipped += 1
                continue

            probs = predict_proba(fv, coef_data, scaler_data)
            season_probs.append(probs)
            season_outcomes.append(m["outcome"])

        if not season_probs:
            print(f"    No evaluable matches (all skipped={skipped})")
            continue

        acc = accuracy(season_probs, season_outcomes)
        bs  = brier_score_3way(season_probs, season_outcomes)
        hc_acc, hc_n = high_conviction_accuracy(season_probs, season_outcomes)
        roi, wins, losses = roi_simulation(season_probs, season_outcomes)

        season_results[season] = {
            "n_matches": len(season_probs),
            "skipped": skipped,
            "accuracy": round(acc, 4),
            "brier": round(bs, 4),
            "hc_accuracy": round(hc_acc, 4) if hc_acc else None,
            "hc_n": hc_n,
            "roi_pct": round(roi, 2),
            "roi_wins": wins,
            "roi_losses": losses,
        }

        print(f"    n={len(season_probs)} | acc={acc:.3f} | brier={bs:.4f} | hc={hc_acc:.3f}({hc_n}) | roi={roi:+.1f}%")

        # Confusion matrix
        preds = [max(p, key=p.get) for p in season_probs]
        cm = confusion_matrix(season_outcomes, preds, labels=["H", "D", "A"])
        print(f"    Confusion matrix (rows=actual, cols=pred, order=H/D/A):")
        print(f"      H: {cm[0]}")
        print(f"      D: {cm[1]}")
        print(f"      A: {cm[2]}")

        all_probs.extend(season_probs)
        all_outcomes.extend(season_outcomes)

    return all_probs, all_outcomes, season_results

# ---------------------------------------------------------------------------
# Print summary report
# ---------------------------------------------------------------------------
def print_report(all_probs, all_outcomes, season_results, meta):
    print("\n" + "=" * 60)
    print("MLS Oracle v4.1 — Backtest Report")
    print("=" * 60)
    print(f"Model trained: {meta.get('trained_at', 'unknown')}")
    print(f"Training seasons: {meta.get('training_seasons')}")
    print(f"Total training matches: {meta.get('n_matches')}")
    print()

    print("Per-Season Results:")
    print(f"  {'Season':>8} {'N':>6} {'Acc':>6} {'Brier':>7} {'HC%':>7} {'HC N':>6} {'ROI%':>7}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*7}")
    for season, r in sorted(season_results.items()):
        hc_str = f"{r['hc_accuracy']:.3f}" if r["hc_accuracy"] else "  N/A"
        print(f"  {season:>8} {r['n_matches']:>6} {r['accuracy']:>6.3f} "
              f"{r['brier']:>7.4f} {hc_str:>7} {r['hc_n']:>6} {r['roi_pct']:>+7.1f}%")

    if all_probs:
        print()
        print("Aggregate (all seasons):")
        agg_acc = accuracy(all_probs, all_outcomes)
        agg_bs  = brier_score_3way(all_probs, all_outcomes)
        agg_hc_acc, agg_hc_n = high_conviction_accuracy(all_probs, all_outcomes)
        agg_roi, agg_w, agg_l = roi_simulation(all_probs, all_outcomes)

        print(f"  Total matches:       {len(all_probs)}")
        print(f"  Accuracy:            {agg_acc:.4f}  ({agg_acc*100:.1f}%)")
        print(f"  Brier score:         {agg_bs:.4f}  (lower = better, baseline ~0.63)")
        print(f"  HC accuracy (≥60%):  {agg_hc_acc:.4f}  n={agg_hc_n}" if agg_hc_acc else "  HC: insufficient picks")
        print(f"  Simulated ROI:       {agg_roi:+.2f}%  (W={agg_w}, L={agg_l})")

        # Baseline: always predict home win (most common outcome in soccer)
        home_rate = all_outcomes.count("H") / len(all_outcomes)
        draw_rate = all_outcomes.count("D") / len(all_outcomes)
        away_rate = all_outcomes.count("A") / len(all_outcomes)
        print()
        print(f"Outcome distribution:")
        print(f"  Home wins:  {home_rate:.3f} ({home_rate*100:.1f}%)")
        print(f"  Draws:      {draw_rate:.3f} ({draw_rate*100:.1f}%)")
        print(f"  Away wins:  {away_rate:.3f} ({away_rate*100:.1f}%)")

        # Walk-forward CV from metadata
        cv = meta.get("walk_forward_cv", [])
        if cv:
            print()
            print("Walk-forward CV (from training):")
            for fold in cv:
                print(f"  Fold {fold['fold']}: test={fold['test_season']} "
                      f"acc={fold['accuracy']:.3f} ll={fold['log_loss']:.4f} brier={fold['brier_score']:.4f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLS Oracle v4.1 Backtest")
    parser.add_argument("--seasons", nargs="+", type=int,
                        help="Seasons to evaluate (default: all training seasons except first)")
    args = parser.parse_args()

    # Load model artifacts
    if not os.path.exists("data/model/coefficients.json"):
        print("ERROR: Model artifacts not found. Run python/train_model.py first.")
        sys.exit(1)

    print("Loading model artifacts...")
    coef_data, scaler_data, meta = load_artifacts()
    print(f"  Model version: {meta.get('version')}")
    print(f"  Features: {meta.get('n_features')}")

    # Fetch data
    target_seasons = args.seasons if args.seasons else TRAINING_SEASONS[1:]
    all_needed_seasons = sorted(set(TRAINING_SEASONS) | set(target_seasons))

    print(f"\nFetching historical matches (seasons {min(all_needed_seasons)}–{max(all_needed_seasons)})...")
    matches = fetch_all_matches(all_needed_seasons)

    from collections import defaultdict as ddict
    matches_by_season = ddict(list)
    for m in matches:
        matches_by_season[m["season"]].append(m)

    # Evaluate
    print(f"\nEvaluating seasons: {target_seasons}")
    all_probs, all_outcomes, season_results = evaluate_seasons(
        target_seasons, matches_by_season, coef_data, scaler_data
    )

    print_report(all_probs, all_outcomes, season_results, meta)

if __name__ == "__main__":
    main()
