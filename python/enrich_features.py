#!/usr/bin/env python3
"""
MLS Oracle — Build enriched training dataset from ESPN game data.

Reads the matches cache (data/matches_cache.json) or fetches fresh from ESPN,
then computes rolling features per team and outputs data/training_data.csv
with one row per game (features computed BEFORE the game, label = outcome).

Features computed:
  - elo_diff: Elo rating differential (home - away)
  - ppg_diff: Points-per-game differential (rolling 10 games)
  - goals_for_diff: Goals scored per game differential (rolling 6)
  - goals_against_diff: Goals conceded per game differential (rolling 6)
  - form_5g_diff: Form over last 5 games (W=3, D=1, L=0, normalized)
  - rest_days_diff: Days since last match differential
  - home_advantage_diff: Team home-vs-away win rate gap differential
  - xg_for_diff: xG proxy differential (SOT-based)
  - xg_against_diff: xG against differential
  - possession_diff: Possession % differential
  - draw_tendency_diff: Draw rate over last 10 games
  - travel_distance: Normalized away team travel distance
  - midweek_flag: 1 if Tue/Wed/Thu
  - altitude_flag: 1 if home venue is high altitude
  - is_home: Always 1 (home perspective)
  - poisson_home/draw/away: Poisson 3-way probs from rolling xG
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

MATCHES_CACHE = DATA_DIR / "matches_cache.json"
OUTPUT_CSV = DATA_DIR / "training_data.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1"
HEADERS = {"User-Agent": "MLS-Oracle-Enricher/1.0"}

# Elo constants
K_FACTOR = 20
HOME_ADVANTAGE_ELO = 70
GOAL_CAP = 4
DEFAULT_ELO = 1500
OFFSEASON_REGRESS = 0.75

ALTITUDE_VENUES = {"COL": 5282, "RSL": 4327}

CITY_COORDS = {
    "ATL": (33.7554, -84.4009), "ATX": (30.3870, -97.7191),
    "CHI": (41.8623, -87.6167), "CIN": (39.1116, -84.5232),
    "CLB": (39.9675, -82.9913), "COL": (39.8061, -104.9776),
    "DAL": (32.9120, -97.0601), "DC":  (38.8682, -77.0122),
    "HOU": (29.7527, -95.4101), "KC":  (39.1219, -94.8232),
    "LA":  (34.0144, -118.2872), "LAFC": (34.0126, -118.2842),
    "MIA": (25.9580, -80.2389), "MIN": (44.9732, -93.1675),
    "MONT": (45.4678, -73.6748), "MTL": (45.4678, -73.6748),
    "NE":  (42.0914, -71.2643), "NSH": (36.1306, -86.7717),
    "NY":  (40.7315, -74.0685), "RBNY": (40.7315, -74.0685),
    "NYC": (40.7505, -73.9934), "ORL": (28.5392, -81.3890),
    "PHI": (39.9012, -75.1674), "POR": (45.5215, -122.6916),
    "RSL": (40.5830, -111.8927), "SJ":  (37.3519, -121.9250),
    "SEA": (47.5952, -122.3316), "SKC": (38.8895, -94.8234),
    "STL": (38.6322, -90.1987), "TOR": (43.6333, -79.4189),
    "VAN": (49.2767, -123.1108), "SD":  (32.7573, -117.1664),
}

TRAINING_SEASONS = list(range(2019, 2026))  # 2019-2025


# ---------------------------------------------------------------------------
# ESPN fetch (reused from train_model.py)
# ---------------------------------------------------------------------------
def espn_get(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                print(f"  Failed after 3 attempts: {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}


def parse_scoreboard_event(event):
    """Parse ESPN scoreboard event into a match dict."""
    try:
        comp = event["competitions"][0]
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
        home_team = home_comp.get("team", {}).get("abbreviation", "")
        away_team = away_comp.get("team", {}).get("abbreviation", "")
        if not home_team or not away_team:
            return None

        date = event.get("date", "")[:10]

        def get_stat(comp_data, name):
            for s in comp_data.get("statistics", []):
                if s.get("name", "") == name:
                    try:
                        return float(str(s.get("displayValue", "0")).replace("%", ""))
                    except (ValueError, TypeError):
                        pass
            return 0.0

        home_sot = get_stat(home_comp, "shotsOnTarget")
        away_sot = get_stat(away_comp, "shotsOnTarget")
        home_tot = get_stat(home_comp, "totalShots")
        away_tot = get_stat(away_comp, "totalShots")
        home_poss = get_stat(home_comp, "possessionPct")
        away_poss = get_stat(away_comp, "possessionPct")

        # xG proxy: SOT * 0.30 + off-target * 0.05
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
            "outcome": outcome,
            "event_id": event.get("id", ""),
        }
    except Exception:
        return None


def fetch_season_from_espn(season):
    """Fetch all MLS matches for a season from ESPN scoreboard API."""
    cache_file = CACHE_DIR / f"mls_season_{season}.json"
    if cache_file.exists():
        print(f"  Season {season}: loaded from cache")
        return json.loads(cache_file.read_text())

    all_matches = {}
    season_start = datetime(season, 2, 15)
    season_end = datetime(season, 12, 10)
    current = season_start
    match_days = 0

    while current <= season_end:
        weekday = current.weekday()
        if weekday in (1, 2, 4, 5, 6):  # Tue, Wed, Fri, Sat, Sun
            try:
                date_str = current.strftime("%Y%m%d")
                data = espn_get(f"{ESPN_BASE}/scoreboard", params={"dates": date_str, "limit": "100"})
                for ev in data.get("events", []):
                    parsed = parse_scoreboard_event(ev)
                    if parsed and parsed["event_id"] not in all_matches:
                        parsed["season"] = season
                        all_matches[parsed["event_id"]] = parsed
                        match_days += 1
                time.sleep(0.05)
            except Exception as e:
                print(f"    Warning: {current.date()} failed: {e}")
        current += timedelta(days=1)

    matches = sorted(all_matches.values(), key=lambda m: m["date"])
    cache_file.write_text(json.dumps(matches, indent=2))
    print(f"  Season {season}: {len(matches)} matches fetched")
    return matches


def load_or_fetch_matches():
    """Load matches from cache or fetch from ESPN."""
    # Try existing matches_cache.json first
    if MATCHES_CACHE.exists():
        with open(MATCHES_CACHE) as f:
            matches = json.load(f)
        if matches:
            print(f"Loaded {len(matches)} matches from {MATCHES_CACHE}")
            # Check which seasons we have
            existing_seasons = set(m.get("season", 0) for m in matches)
            missing = [s for s in TRAINING_SEASONS if s not in existing_seasons]
            if missing:
                print(f"Missing seasons: {missing} — fetching from ESPN...")
                for season in missing:
                    new_matches = fetch_season_from_espn(season)
                    matches.extend(new_matches)
                # Update cache
                with open(MATCHES_CACHE, "w") as f:
                    json.dump(matches, f)
                print(f"Updated cache: {len(matches)} total matches")
            return sorted(matches, key=lambda m: m["date"])

    # Fetch everything
    print("No cache found — fetching all seasons from ESPN...")
    all_matches = []
    for season in TRAINING_SEASONS:
        season_matches = fetch_season_from_espn(season)
        all_matches.extend(season_matches)

    with open(MATCHES_CACHE, "w") as f:
        json.dump(all_matches, f)
    print(f"Cached {len(all_matches)} matches to {MATCHES_CACHE}")
    return sorted(all_matches, key=lambda m: m["date"])


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Poisson 3-way
# ---------------------------------------------------------------------------
def poisson_3way(lambda_home, lambda_away, max_goals=8):
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, max(lambda_home, 0.01)) * poisson.pmf(a, max(lambda_away, 0.01))
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p
    total = home_win + draw + away_win
    if total == 0:
        return 0.33, 0.33, 0.34
    return home_win / total, draw / total, away_win / total


# ---------------------------------------------------------------------------
# Rolling team tracker
# ---------------------------------------------------------------------------
class TeamTracker:
    """Tracks rolling stats per team, computable from ESPN game results."""

    def __init__(self):
        self.goals_for = defaultdict(list)
        self.goals_against = defaultdict(list)
        self.xg_for = defaultdict(list)
        self.xg_against = defaultdict(list)
        self.poss = defaultdict(list)
        self.results = defaultdict(list)  # 3=W, 1=D, 0=L
        self.home_results = defaultdict(list)  # (is_home, points)
        self.last_match_date = {}
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.seasons_seen = defaultdict(int)

    def _rolling_avg(self, vals, n=6):
        if not vals:
            return 0.0
        recent = vals[-n:]
        return sum(recent) / len(recent)

    def get_features(self, team):
        """Get current rolling features for a team BEFORE a new game."""
        gf = self._rolling_avg(self.goals_for[team], 6)
        ga = self._rolling_avg(self.goals_against[team], 6)
        xgf = self._rolling_avg(self.xg_for[team], 6)
        xga = self._rolling_avg(self.xg_against[team], 6)
        possession = self._rolling_avg(self.poss[team], 6)

        # PPG over last 10
        pts = self.results[team][-10:]
        ppg = sum(pts) / len(pts) if pts else 1.5

        # Form over last 5
        form_5 = self.results[team][-5:]
        form_5g = sum(form_5) / (len(form_5) * 3) if form_5 else 0.5  # Normalized 0-1

        # Draw tendency last 10
        last_10 = self.results[team][-10:]
        draw_rate = sum(1 for p in last_10 if p == 1) / len(last_10) if last_10 else 0.25

        # Home advantage: home win% - away win%
        home_games = [(is_h, pts) for is_h, pts in self.home_results[team] if is_h]
        away_games = [(is_h, pts) for is_h, pts in self.home_results[team] if not is_h]
        home_wr = sum(1 for _, p in home_games[-15:] if p == 3) / max(len(home_games[-15:]), 1)
        away_wr = sum(1 for _, p in away_games[-15:] if p == 3) / max(len(away_games[-15:]), 1)
        home_adv = home_wr - away_wr

        n_matches = len(self.goals_for[team])

        return {
            "goals_for_avg": gf,
            "goals_against_avg": ga,
            "xg_for_avg": xgf,
            "xg_against_avg": xga,
            "possession_avg": possession,
            "ppg": ppg,
            "form_5g": form_5g,
            "draw_tendency": draw_rate,
            "home_advantage": home_adv,
            "n_matches": n_matches,
        }

    def update(self, match):
        """Update team stats AFTER feature extraction."""
        h, a = match["home_team"], match["away_team"]
        hg, ag = match["home_score"], match["away_score"]
        hxg, axg = match.get("home_xg", 0), match.get("away_xg", 0)
        hp, ap = match.get("home_poss", 0), match.get("away_poss", 0)

        # Goals
        self.goals_for[h].append(hg)
        self.goals_against[h].append(ag)
        self.goals_for[a].append(ag)
        self.goals_against[a].append(hg)

        # xG
        self.xg_for[h].append(hxg)
        self.xg_against[h].append(axg)
        self.xg_for[a].append(axg)
        self.xg_against[a].append(hxg)

        # Possession
        if hp > 0:
            self.poss[h].append(hp)
        if ap > 0:
            self.poss[a].append(ap)

        # Results (points)
        if hg > ag:
            self.results[h].append(3)
            self.results[a].append(0)
        elif hg == ag:
            self.results[h].append(1)
            self.results[a].append(1)
        else:
            self.results[h].append(0)
            self.results[a].append(3)

        # Home/away tracking
        h_pts = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts = 3 if ag > hg else (1 if hg == ag else 0)
        self.home_results[h].append((True, h_pts))
        self.home_results[a].append((False, a_pts))

        # Last match date
        self.last_match_date[h] = match["date"]
        self.last_match_date[a] = match["date"]

    def update_elo(self, match, season):
        """Update Elo ratings after a match."""
        h, a = match["home_team"], match["away_team"]

        # Offseason regression
        for team in [h, a]:
            if self.seasons_seen[team] < season and self.elo[team] != DEFAULT_ELO:
                current = self.elo[team]
                self.elo[team] = current * OFFSEASON_REGRESS + DEFAULT_ELO * (1 - OFFSEASON_REGRESS)
            self.seasons_seen[team] = season

        home_elo = self.elo[h]
        away_elo = self.elo[a]
        exp = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - HOME_ADVANTAGE_ELO) / 400))

        margin = abs(match["home_score"] - match["away_score"])
        mov = math.log(1 + min(margin, GOAL_CAP))

        if match["home_score"] > match["away_score"]:
            actual = 1.0
        elif match["home_score"] == match["away_score"]:
            actual = 0.5
        else:
            actual = 0.0

        delta = K_FACTOR * mov * (actual - exp)
        self.elo[h] += delta
        self.elo[a] -= delta

    def get_rest_days(self, team, date_str):
        if team not in self.last_match_date:
            return 7
        try:
            delta = datetime.strptime(date_str, "%Y-%m-%d") - datetime.strptime(
                self.last_match_date[team], "%Y-%m-%d"
            )
            return min(delta.days, 14)
        except Exception:
            return 7

    def reset_season(self):
        """Clear rolling stats between seasons (Elo persists)."""
        for d in [
            self.goals_for, self.goals_against, self.xg_for, self.xg_against,
            self.poss, self.results, self.home_results,
        ]:
            d.clear()
        self.last_match_date.clear()


# ---------------------------------------------------------------------------
# Feature columns (what we output)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
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
    "poisson_home",
    "poisson_draw",
    "poisson_away",
]


def build_training_data(matches):
    """Build enriched training dataset from match list."""
    tracker = TeamTracker()
    rows = []
    current_season = None
    min_matches = 3  # Need at least 3 games of history

    for match in matches:
        season = match.get("season", 0)
        h, a = match["home_team"], match["away_team"]
        date_str = match["date"]

        # Season reset
        if season != current_season:
            if current_season is not None:
                tracker.reset_season()
            current_season = season

        # Get features BEFORE updating
        h_stats = tracker.get_features(h)
        a_stats = tracker.get_features(a)

        # Only include if both teams have enough history
        if h_stats["n_matches"] >= min_matches and a_stats["n_matches"] >= min_matches:
            elo_diff = tracker.elo[h] - tracker.elo[a]
            rest_h = tracker.get_rest_days(h, date_str)
            rest_a = tracker.get_rest_days(a, date_str)

            # Travel distance
            h_coords = CITY_COORDS.get(h)
            a_coords = CITY_COORDS.get(a)
            travel_dist = 0.0
            if h_coords and a_coords:
                travel_dist = haversine_miles(a_coords[0], a_coords[1], h_coords[0], h_coords[1])

            # Midweek
            try:
                midweek = 1 if datetime.strptime(date_str, "%Y-%m-%d").weekday() in [1, 2, 3] else 0
            except Exception:
                midweek = 0

            # Altitude
            altitude = 1 if h in ALTITUDE_VENUES else 0

            # Poisson priors from rolling xG
            lambda_home = max(h_stats["xg_for_avg"] * 0.9 + a_stats["xg_against_avg"] * 0.1, 0.1)
            lambda_away = max(a_stats["xg_for_avg"] * 0.9 + h_stats["xg_against_avg"] * 0.1, 0.1)
            pois_h, pois_d, pois_a = poisson_3way(lambda_home, lambda_away)

            row = {
                "season": season,
                "game_date": date_str,
                "home_team": h,
                "away_team": a,
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "outcome": match["outcome"],
                # Features
                "elo_diff": round(elo_diff, 4),
                "ppg_diff": round(h_stats["ppg"] - a_stats["ppg"], 4),
                "goals_for_diff": round(h_stats["goals_for_avg"] - a_stats["goals_for_avg"], 4),
                "goals_against_diff": round(h_stats["goals_against_avg"] - a_stats["goals_against_avg"], 4),
                "form_5g_diff": round(h_stats["form_5g"] - a_stats["form_5g"], 4),
                "rest_days_diff": rest_h - rest_a,
                "home_advantage_diff": round(h_stats["home_advantage"] - a_stats["home_advantage"], 4),
                "xg_for_diff": round(h_stats["xg_for_avg"] - a_stats["xg_for_avg"], 4),
                "xg_against_diff": round(h_stats["xg_against_avg"] - a_stats["xg_against_avg"], 4),
                "possession_diff": round(h_stats["possession_avg"] - a_stats["possession_avg"], 4),
                "draw_tendency_diff": round(h_stats["draw_tendency"] - a_stats["draw_tendency"], 4),
                "travel_distance": round(travel_dist / 3000.0, 4),  # Normalized
                "midweek_flag": midweek,
                "altitude_flag": altitude,
                "is_home": 1.0,
                "poisson_home": round(pois_h, 4),
                "poisson_draw": round(pois_d, 4),
                "poisson_away": round(pois_a, 4),
            }
            rows.append(row)

        # Update AFTER feature extraction
        tracker.update(match)
        tracker.update_elo(match, season)

    return rows


def main():
    print("=" * 60)
    print("MLS Oracle — Enrich Features")
    print("=" * 60)

    matches = load_or_fetch_matches()
    print(f"\nTotal matches available: {len(matches)}")

    # Group by season for reporting
    by_season = defaultdict(int)
    for m in matches:
        by_season[m.get("season", "?")] += 1
    for s in sorted(by_season):
        print(f"  Season {s}: {by_season[s]} matches")

    print("\nBuilding enriched features...")
    rows = build_training_data(matches)

    df = pd.DataFrame(rows)
    print(f"\nTraining samples: {len(df)} (from {len(matches)} total matches)")
    print(f"  (Skipped early-season games where teams had <3 matches of history)")

    # Report feature coverage
    print(f"\nFeature columns ({len(FEATURE_COLUMNS)}):")
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            non_zero = (df[col] != 0).sum()
            print(f"  {col:25s}  non-zero: {non_zero:5d}/{len(df)}  "
                  f"mean={df[col].mean():+.4f}  std={df[col].std():.4f}")

    # Outcome distribution
    print(f"\nOutcome distribution:")
    for outcome in ["H", "D", "A"]:
        count = (df["outcome"] == outcome).sum()
        print(f"  {outcome}: {count} ({count/len(df)*100:.1f}%)")

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
