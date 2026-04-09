#!/usr/bin/env python3
"""
MLS Oracle v4.1 — Training Pipeline
Fetches 2019–2024 MLS historical data from ESPN, builds rolling feature vectors,
trains a multinomial logistic regression (H/D/A), and exports model artifacts.

Artifacts exported to data/model/:
  coefficients.json   — three coefficient vectors (home/draw/away) + intercepts
  scaler.json         — mean/std for each feature (StandardScaler params)
  metadata.json       — feature names, training seasons, walk-forward CV results
"""

import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants (must match src/features/eloEngine.ts and src/features/featureEngine.ts)
# ---------------------------------------------------------------------------
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1"
ASA_BASE  = "https://app.americansocceranalysis.com/api/v1/mls"

TRAINING_SEASONS = list(range(2019, 2025))  # 2019–2024 inclusive
K_FACTOR           = 20
HOME_ADVANTAGE_ELO = 70
GOAL_CAP           = 4
EXPANSION_PENALTY  = 150
DEFAULT_ELO        = 1500
OFFSEASON_REGRESS  = 0.75   # 75% current, 25% mean

ALTITUDE_VENUES = {"COL": 5282, "RSL": 4327}
TURF_VENUES     = {"SEA", "NE", "VAN"}

CITY_COORDS = {
    "ATL": (33.7554, -84.4009), "ATX": (30.3870, -97.7191), "CHI": (41.8623, -87.6167),
    "CIN": (39.1116, -84.5232), "CLB": (39.9675, -82.9913), "COL": (39.8061, -104.9776),
    "DAL": (32.9120, -97.0601), "DC":  (38.8682, -77.0122), "HOU": (29.7527, -95.4101),
    "KC":  (39.1219, -94.8232), "LA":  (34.0144, -118.2872),"LAFC":(34.0126, -118.2842),
    "MIA": (25.9580, -80.2389), "MIN": (44.9732, -93.1675), "MONT":(45.4678, -73.6748),
    "NE":  (42.0914, -71.2643), "NSH": (36.1306, -86.7717), "NY":  (40.7315, -74.0685),
    "NYC": (40.7505, -73.9934), "ORL": (28.5392, -81.3890), "PHI": (39.9012, -75.1674),
    "POR": (45.5215, -122.6916),"RSL": (40.5830, -111.8927),"SJ":  (37.3519, -121.9250),
    "SEA": (47.5952, -122.3316),"SKC": (38.8895, -94.8234), "STL": (38.6322, -90.1987),
    "TOR": (43.6333, -79.4189), "VAN": (49.2767, -123.1108),"SD":  (32.7573, -117.1664),
}

FEATURE_NAMES = [
    "elo_diff", "xg_for_diff", "xg_against_diff", "xg_diff", "xpts_diff", "ppg_diff",
    "possession_diff", "pass_pct_diff", "form_5g_xpts_diff", "overperformance_diff",
    "draw_tendency_diff", "is_home", "home_advantage_diff",
    "altitude_flag", "altitude_penalty", "turf_flag",
    "travel_distance_diff", "cross_country_flag", "rest_days_diff", "midweek_flag",
    "cup_congestion", "dp_impact_diff", "dp_available_diff", "roster_salary_diff",
    "conference_diff", "expansion_flag", "playoff_position_diff", "manager_tenure_diff",
    "vegas_home_prob", "vegas_draw_prob", "vegas_away_prob",
    # Poisson priors (computed analytically from rolling xG)
    "poisson_home", "poisson_draw", "poisson_away",
]

# ---------------------------------------------------------------------------
# Haversine distance (miles)
# ---------------------------------------------------------------------------
def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ---------------------------------------------------------------------------
# ESPN API helpers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MLS-Oracle-Trainer/4.1"})

def espn_get(url: str, params: dict = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}

def parse_scoreboard_event(event: dict) -> Optional[dict]:
    """Parse a scoreboard event (from /scoreboard endpoint) into a match record."""
    try:
        comp = event["competitions"][0]

        # Only completed matches
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed", False):
            return None

        competitors = comp["competitors"]
        home_comp = next((c for c in competitors if c["homeAway"] == "home"), None)
        away_comp = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home_comp or not away_comp:
            return None

        home_score = int(home_comp.get("score", 0) or 0)
        away_score = int(away_comp.get("score", 0) or 0)
        home_team  = home_comp.get("team", {}).get("abbreviation", "")
        away_team  = away_comp.get("team", {}).get("abbreviation", "")

        if not home_team or not away_team:
            return None

        # Date from event (UTC ISO → YYYY-MM-DD)
        date = event.get("date", "")[:10]

        def get_stat(comp_data: dict, name: str) -> float:
            for s in comp_data.get("statistics", []):
                if s.get("name", "") == name:
                    try:
                        return float(str(s.get("displayValue", "0")).replace("%", ""))
                    except (ValueError, TypeError):
                        pass
            return 0.0

        home_sot  = get_stat(home_comp, "shotsOnTarget")
        away_sot  = get_stat(away_comp, "shotsOnTarget")
        home_tot  = get_stat(home_comp, "totalShots")
        away_tot  = get_stat(away_comp, "totalShots")
        home_poss = get_stat(home_comp, "possessionPct")
        away_poss = get_stat(away_comp, "possessionPct")

        # xG proxy: SOT * 0.30 + (total - SOT) * 0.05
        home_xg = home_sot * 0.30 + max(0, home_tot - home_sot) * 0.05
        away_xg = away_sot * 0.30 + max(0, away_tot - away_sot) * 0.05

        if home_score > away_score:
            outcome = "H"
        elif home_score < away_score:
            outcome = "A"
        else:
            outcome = "D"

        return {
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "home_poss": home_poss,
            "away_poss": away_poss,
            "home_pass_pct": 0.0,  # not available in scoreboard stats
            "away_pass_pct": 0.0,
            "outcome": outcome,
            "event_id": event.get("id", ""),
        }
    except Exception:
        return None

def fetch_scoreboard_date(date: datetime) -> list:
    """Fetch all MLS matches from the scoreboard on a given date."""
    date_str = date.strftime("%Y%m%d")
    url = f"{ESPN_BASE}/scoreboard"
    data = espn_get(url, params={"dates": date_str})
    results = []
    for ev in data.get("events", []):
        parsed = parse_scoreboard_event(ev)
        if parsed:
            results.append(parsed)
    return results

# ---------------------------------------------------------------------------
# Data fetching — all seasons via scoreboard
# ---------------------------------------------------------------------------
def fetch_all_matches(seasons: list) -> list:
    """
    Fetch MLS matches by iterating through each day of each season's date range
    using the /scoreboard endpoint (the team schedule endpoint returns 0 events
    for historical seasons — scoreboard is the correct endpoint for historical data).
    """
    all_matches: dict = {}  # event_id -> match (deduplicated)

    for season in seasons:
        # MLS season runs late Feb through early Dec
        season_start = datetime(season, 2, 15)
        season_end   = datetime(season, 12, 10)

        print(f"\nFetching season {season} ({season_start.date()} → {season_end.date()})...")
        current = season_start
        match_days = 0
        season_count = 0

        while current <= season_end:
            # Only fetch on likely match days: Fri/Sat/Sun + Tue/Wed (saves ~42% of requests)
            weekday = current.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
            if weekday in (1, 2, 4, 5, 6):  # Tue, Wed, Fri, Sat, Sun
                try:
                    matches = fetch_scoreboard_date(current)
                    if matches:
                        match_days += 1
                        for m in matches:
                            if m["event_id"] not in all_matches:
                                m["season"] = season
                                all_matches[m["event_id"]] = m
                                season_count += 1
                    time.sleep(0.05)
                except Exception as e:
                    print(f"  Warning: {current.date()} failed: {e}")
            current += timedelta(days=1)

        print(f"  {season_count} matches found across {match_days} match days")

    matches = sorted(all_matches.values(), key=lambda m: m["date"])
    print(f"\nTotal unique matches: {len(matches)}")
    return matches

# ---------------------------------------------------------------------------
# Elo engine (mirrors eloEngine.ts)
# ---------------------------------------------------------------------------
class EloEngine:
    def __init__(self):
        self.ratings: dict = defaultdict(lambda: DEFAULT_ELO)
        self.seasons_played: dict = defaultdict(int)

    def expected_score(self, home_elo: float, away_elo: float) -> float:
        return 1.0 / (1.0 + 10 ** ((away_elo - home_elo - HOME_ADVANTAGE_ELO) / 400))

    def mov_multiplier(self, margin: int) -> float:
        return math.log(1 + min(abs(margin), GOAL_CAP))

    def update(self, home_team: str, away_team: str, home_score: int, away_score: int, season: int):
        # Offseason regression if new season
        for team in [home_team, away_team]:
            if self.seasons_played[team] < season and self.ratings[team] != DEFAULT_ELO:
                current = self.ratings[team]
                self.ratings[team] = current * OFFSEASON_REGRESS + DEFAULT_ELO * (1 - OFFSEASON_REGRESS)
                self.seasons_played[team] = season

        home_elo = self.ratings[home_team]
        away_elo = self.ratings[away_team]
        exp = self.expected_score(home_elo, away_elo)

        margin = abs(home_score - away_score)
        mov = self.mov_multiplier(margin)

        if home_score > away_score:
            actual = 1.0
        elif home_score == away_score:
            actual = 0.5
        else:
            actual = 0.0

        delta = K_FACTOR * mov * (actual - exp)
        self.ratings[home_team] += delta
        self.ratings[away_team] -= delta

    def get_diff(self, home_team: str, away_team: str) -> float:
        return self.ratings[home_team] - self.ratings[away_team]

# ---------------------------------------------------------------------------
# Rolling team stats
# ---------------------------------------------------------------------------
class TeamStats:
    """Tracks rolling season stats per team (xG, xPts, possession, form)."""

    def __init__(self):
        # Per-team accumulators
        self.xg_for:     dict = defaultdict(list)
        self.xg_against: dict = defaultdict(list)
        self.poss:        dict = defaultdict(list)
        self.pass_pct:    dict = defaultdict(list)
        self.goals_for:   dict = defaultdict(list)
        self.goals_against: dict = defaultdict(list)
        self.results:     dict = defaultdict(list)  # list of (xg_for, xg_against) for xpts
        self.last_match_date: dict = {}

    def update(self, match: dict):
        h, a = match["home_team"], match["away_team"]
        hxg, axg = match["home_xg"], match["away_xg"]
        hg, ag   = match["home_score"], match["away_score"]
        hp, ap   = match["home_poss"], match["away_poss"]
        hpa, apa = match["home_pass_pct"], match["away_pass_pct"]

        for team, xg_f, xg_a, g_f, g_a, poss, pass_p in [
            (h, hxg, axg, hg, ag, hp, hpa),
            (a, axg, hxg, ag, hg, ap, apa),
        ]:
            self.xg_for[team].append(xg_f)
            self.xg_against[team].append(xg_a)
            self.goals_for[team].append(g_f)
            self.goals_against[team].append(g_a)
            if poss > 0:
                self.poss[team].append(poss)
            if pass_p > 0:
                self.pass_pct[team].append(pass_p)
            self.results[team].append((xg_f, xg_a))
            self.last_match_date[team] = match["date"]

    def _xpts(self, xg_f: float, xg_a: float) -> float:
        """Expected points from one match via Poisson."""
        win = draw = 0.0
        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, max(xg_f, 0.01)) * poisson.pmf(a, max(xg_a, 0.01))
                if h > a: win += p
                elif h == a: draw += p
        return win * 3 + draw * 1

    def rolling_avg(self, vals: list, n: int = 6) -> float:
        if not vals:
            return 0.0
        return sum(vals[-n:]) / len(vals[-n:])

    def get_stats(self, team: str) -> dict:
        n_matches = len(self.xg_for[team])
        xg_f_avg  = self.rolling_avg(self.xg_for[team])
        xg_a_avg  = self.rolling_avg(self.xg_against[team])
        g_f_avg   = self.rolling_avg(self.goals_for[team])
        g_a_avg   = self.rolling_avg(self.goals_against[team])

        # xPts from last 6 matches
        recent = self.results[team][-6:]
        xpts_recent = sum(self._xpts(f, a) for f, a in recent)

        # Actual points per game (ppg) from last 10
        pts = []
        for gf, ga in zip(self.goals_for[team][-10:], self.goals_against[team][-10:]):
            if gf > ga: pts.append(3)
            elif gf == ga: pts.append(1)
            else: pts.append(0)
        ppg = sum(pts) / len(pts) if pts else 1.5

        # Draw tendency: fraction of last 10 that were draws
        draw_results = [1 if gf == ga else 0
                        for gf, ga in zip(self.goals_for[team][-10:], self.goals_against[team][-10:])]
        draw_tendency = sum(draw_results) / len(draw_results) if draw_results else 0.25

        # Overperformance: goals - xG (last 6)
        over = sum(g - x for g, x in zip(self.goals_for[team][-6:], self.xg_for[team][-6:]))

        return {
            "xg_for":       xg_f_avg,
            "xg_against":   xg_a_avg,
            "xg_diff":      xg_f_avg - xg_a_avg,
            "xpts":         xpts_recent,
            "ppg":          ppg,
            "possession":   self.rolling_avg(self.poss[team]),
            "pass_pct":     self.rolling_avg(self.pass_pct[team]),
            "draw_tendency": draw_tendency,
            "overperformance": over,
            "n_matches":    n_matches,
        }

    def reset_season(self):
        """Call between seasons to clear rolling stats."""
        for d in [self.xg_for, self.xg_against, self.poss, self.pass_pct,
                  self.goals_for, self.goals_against, self.results]:
            d.clear()

# ---------------------------------------------------------------------------
# Poisson 3-way probability (analytical)
# ---------------------------------------------------------------------------
def poisson_3way(lambda_home: float, lambda_away: float, max_goals: int = 8) -> tuple:
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, max(lambda_home, 0.01)) * poisson.pmf(a, max(lambda_away, 0.01))
            if h > a: home_win += p
            elif h == a: draw += p
            else: away_win += p
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total

# ---------------------------------------------------------------------------
# Blend config (mirrors featureEngine.ts)
# ---------------------------------------------------------------------------
def get_blend_weight(matchweek: int) -> float:
    if matchweek <= 2:  return 0.80
    if matchweek <= 4:  return 0.60
    if matchweek <= 7:  return 0.40
    if matchweek <= 12: return 0.25
    return 0.10

def blend(prior: float, current: float, mw: int) -> float:
    w = get_blend_weight(mw)
    return w * prior + (1 - w) * current

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def is_cup_congestion(date_str: str) -> int:
    """August = Leagues Cup."""
    try:
        return 1 if datetime.strptime(date_str, "%Y-%m-%d").month == 8 else 0
    except:
        return 0

def is_midweek(date_str: str) -> int:
    try:
        return 1 if datetime.strptime(date_str, "%Y-%m-%d").weekday() in [1, 2, 3] else 0
    except:
        return 0

def get_rest_days(team: str, date_str: str, last_match_dates: dict) -> int:
    if team not in last_match_dates:
        return 7
    last = last_match_dates[team]
    try:
        delta = datetime.strptime(date_str, "%Y-%m-%d") - datetime.strptime(last, "%Y-%m-%d")
        return min(delta.days, 14)
    except:
        return 7

def build_feature_vector(match: dict, elo: EloEngine, stats: TeamStats,
                          matchweek_counter: dict, last_match_dates: dict) -> Optional[list]:
    """Build a feature vector for a match BEFORE updating stats/elo with its result."""
    h, a = match["home_team"], match["away_team"]

    h_stats = stats.get_stats(h)
    a_stats = stats.get_stats(a)

    # Skip if either team has <3 matches of data in this season
    if h_stats["n_matches"] < 3 or a_stats["n_matches"] < 3:
        return None

    mw_h = matchweek_counter.get(h, 1)
    mw_a = matchweek_counter.get(a, 1)
    mw   = max(mw_h, mw_a)

    # Prior stats (league average priors)
    prior_xg   = 1.4
    prior_xpts = 1.5
    prior_ppg  = 1.5
    prior_poss = 50.0
    prior_pass = 75.0

    # Blended stats
    def bld(current_val: float, prior_val: float, mw_: int) -> float:
        return blend(prior_val, current_val, mw_)

    hxgf = bld(h_stats["xg_for"],    prior_xg,   mw_h)
    hxga = bld(h_stats["xg_against"], prior_xg,  mw_h)
    axgf = bld(a_stats["xg_for"],    prior_xg,   mw_a)
    axga = bld(a_stats["xg_against"], prior_xg,  mw_a)

    # Altitude & turf
    home_alt = ALTITUDE_VENUES.get(h, 0)
    away_alt = ALTITUDE_VENUES.get(a, 0)
    altitude_flag    = 1 if home_alt > 0 else 0
    altitude_penalty = home_alt / 100_000

    turf_flag = 1 if h in TURF_VENUES else 0

    # Travel distance
    h_coords = CITY_COORDS.get(h)
    a_coords = CITY_COORDS.get(a)
    travel_dist = 0.0
    if h_coords and a_coords:
        travel_dist = haversine_miles(a_coords[0], a_coords[1], h_coords[0], h_coords[1])
    cross_country = 1 if travel_dist > 1500 else 0

    # Rest days
    rest_h = get_rest_days(h, match["date"], last_match_dates)
    rest_a = get_rest_days(a, match["date"], last_match_dates)
    rest_diff = rest_h - rest_a

    # Poisson priors (lambda = blended xG)
    lambda_home = max(hxgf * 0.9 + axga * 0.1, 0.1)  # simplified lambda estimate
    lambda_away = max(axgf * 0.9 + hxga * 0.1, 0.1)
    pois_h, pois_d, pois_a = poisson_3way(lambda_home, lambda_away)

    elo_diff = elo.get_diff(h, a)

    features = [
        elo_diff,                                   # elo_diff
        hxgf - axgf,                                # xg_for_diff
        hxga - axga,                                # xg_against_diff
        (hxgf - hxga) - (axgf - axga),             # xg_diff
        bld(h_stats["xpts"], prior_xpts, mw_h) - bld(a_stats["xpts"], prior_xpts, mw_a),  # xpts_diff
        bld(h_stats["ppg"], prior_ppg, mw_h) - bld(a_stats["ppg"], prior_ppg, mw_a),      # ppg_diff
        bld(h_stats["possession"], prior_poss, mw_h) - bld(a_stats["possession"], prior_poss, mw_a),  # possession_diff
        bld(h_stats["pass_pct"], prior_pass, mw_h) - bld(a_stats["pass_pct"], prior_pass, mw_a),      # pass_pct_diff
        h_stats["xpts"] - a_stats["xpts"],          # form_5g_xpts_diff (recent form, no blend)
        h_stats["overperformance"] - a_stats["overperformance"],  # overperformance_diff
        h_stats["draw_tendency"] - a_stats["draw_tendency"],      # draw_tendency_diff
        1.0,                                        # is_home (always 1 for home team perspective)
        HOME_ADVANTAGE_ELO / 400,                   # home_advantage_diff (normalized)
        float(altitude_flag),                       # altitude_flag
        altitude_penalty,                           # altitude_penalty
        float(turf_flag),                           # turf_flag
        travel_dist / 3000.0,                       # travel_distance_diff (normalized)
        float(cross_country),                       # cross_country_flag
        float(rest_diff),                           # rest_days_diff
        float(is_midweek(match["date"])),           # midweek_flag
        float(is_cup_congestion(match["date"])),    # cup_congestion
        0.0,                                        # dp_impact_diff (not available from ESPN)
        0.0,                                        # dp_available_diff
        0.0,                                        # roster_salary_diff
        0.0,                                        # conference_diff
        0.0,                                        # expansion_flag
        0.0,                                        # playoff_position_diff
        0.0,                                        # manager_tenure_diff
        pois_h,                                     # vegas_home_prob (Poisson prior as proxy)
        pois_d,                                     # vegas_draw_prob
        pois_a,                                     # vegas_away_prob
        pois_h,                                     # poisson_home
        pois_d,                                     # poisson_draw
        pois_a,                                     # poisson_away
    ]

    assert len(features) == len(FEATURE_NAMES), f"Feature count mismatch: {len(features)} vs {len(FEATURE_NAMES)}"
    return features

# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------
def walk_forward_cv(matches_by_season: dict, seasons: list) -> list:
    """Train on seasons before N, evaluate on season N. Returns per-fold results."""
    results = []
    # Need at least 2 seasons to do any CV
    for fold_idx in range(1, len(seasons)):
        train_seasons = seasons[:fold_idx]
        test_season   = seasons[fold_idx]

        print(f"\n  CV Fold {fold_idx}: train={train_seasons}, test={test_season}")

        # Build training data
        X_train, y_train = build_dataset(matches_by_season, train_seasons)
        X_test,  y_test  = build_dataset(matches_by_season, [test_season])

        if len(X_train) < 50 or len(X_test) < 10:
            print(f"    Skipping — insufficient data (train={len(X_train)}, test={len(X_test)})")
            continue

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        model = LogisticRegression(multi_class="multinomial", C=0.8, solver="lbfgs",
                                   max_iter=500, random_state=42)
        model.fit(X_tr_scaled, y_train)

        preds = model.predict_proba(X_te_scaled)
        classes = list(model.classes_)
        # Reorder probs to [H, D, A]
        h_idx = classes.index("H") if "H" in classes else 0
        d_idx = classes.index("D") if "D" in classes else 1
        a_idx = classes.index("A") if "A" in classes else 2

        ll = log_loss(y_test, preds)
        acc = (model.predict(X_te_scaled) == y_test).mean()

        # Brier score (3-way)
        outcome_map = {"H": [1,0,0], "D": [0,1,0], "A": [0,0,1]}
        brier = np.mean([
            sum((preds[i][classes.index(c)] - outcome_map[y_test[i]][j])**2
                for j, c in enumerate(["H","D","A"]))
            for i in range(len(y_test))
        ])

        print(f"    n_test={len(X_test)}, acc={acc:.3f}, log_loss={ll:.4f}, brier={brier:.4f}")
        results.append({
            "fold": fold_idx,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
        })

    return results

def build_dataset(matches_by_season: dict, seasons: list):
    """Build (X, y) arrays for given seasons using rolling stats."""
    elo   = EloEngine()
    stats = TeamStats()
    matchweek_counter: dict = defaultdict(int)
    last_match_dates: dict  = {}

    X, y = [], []

    for season in sorted(seasons):
        if season > min(seasons):
            # Simulate prior seasons to warm up elo/stats
            pass  # elo already updated incrementally below
        stats.reset_season()
        matchweek_counter.clear()

        for match in matches_by_season.get(season, []):
            h, a = match["home_team"], match["away_team"]
            fv = build_feature_vector(match, elo, stats, matchweek_counter, last_match_dates)

            # Update state AFTER feature extraction
            stats.update(match)
            elo.update(h, a, match["home_score"], match["away_score"], season)
            matchweek_counter[h] += 1
            matchweek_counter[a] += 1
            last_match_dates[h] = match["date"]
            last_match_dates[a] = match["date"]

            if fv is not None:
                X.append(fv)
                y.append(match["outcome"])

    return np.array(X, dtype=np.float32) if X else np.zeros((0, len(FEATURE_NAMES))), y

# ---------------------------------------------------------------------------
# Train final model on all seasons
# ---------------------------------------------------------------------------
def train_final_model(matches_by_season: dict, seasons: list):
    print("\nBuilding full training dataset...")
    X, y = build_dataset(matches_by_season, seasons)
    print(f"  Total samples: {len(X)}")
    if len(X) < 50:
        raise RuntimeError("Insufficient training data — check ESPN data fetch")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(multi_class="multinomial", C=0.8, solver="lbfgs",
                               max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    print(f"  Classes: {model.classes_}")
    print(f"  Train accuracy: {(model.predict(X_scaled) == np.array(y)).mean():.4f}")

    return model, scaler

# ---------------------------------------------------------------------------
# Export artifacts
# ---------------------------------------------------------------------------
def export_artifacts(model, scaler, cv_results: list, seasons: list, n_matches: int):
    os.makedirs("data/model", exist_ok=True)

    classes = list(model.classes_)
    coef_map = {cls: model.coef_[i].tolist() for i, cls in enumerate(classes)}
    intercept_map = {cls: float(model.intercept_[i]) for i, cls in enumerate(classes)}

    # coefficients.json — structure expected by metaModel.ts
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
        "version": "4.1.0",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "training_seasons": seasons,
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
    print(f"  coefficients.json — {len(FEATURE_NAMES)} features × {len(classes)} classes")
    print(f"  scaler.json       — StandardScaler params")
    print(f"  metadata.json     — CV results, provenance")
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
    print("MLS Oracle v4.1 — Training Pipeline")
    print("=" * 60)

    # Fetch historical matches
    matches = fetch_all_matches(TRAINING_SEASONS)

    if not matches:
        raise RuntimeError("No matches fetched — check ESPN API connectivity")

    # Group by season
    matches_by_season: dict = defaultdict(list)
    for m in matches:
        matches_by_season[m["season"]].append(m)

    for s, ms in sorted(matches_by_season.items()):
        print(f"  Season {s}: {len(ms)} matches")

    # Walk-forward CV
    print("\nRunning walk-forward cross-validation...")
    cv_results = walk_forward_cv(matches_by_season, TRAINING_SEASONS)

    # Train final model on all seasons
    print("\nTraining final model on all seasons...")
    model, scaler = train_final_model(matches_by_season, TRAINING_SEASONS)

    # Export
    export_artifacts(model, scaler, cv_results, TRAINING_SEASONS, len(matches))

    print("\nDone.")

if __name__ == "__main__":
    main()
