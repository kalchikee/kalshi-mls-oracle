#!/usr/bin/env python3
"""
MLS Playoff Data Fetcher — MLS Cup Playoffs, last 5 seasons, ESPN API.
seasontype=3, league usa.1 (MLS). Output: data/playoff_data.csv
"""
import sys, json, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

OUT_CSV   = DATA_DIR / "playoff_data.csv"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1"
HEADERS   = {"User-Agent": "MLS-Oracle/4.1"}
PLAYOFF_YEARS = [2020, 2021, 2022, 2023, 2024]  # MLS playoffs Oct-Dec
K_FACTOR   = 20.0
HOME_ADV   = 70.0
LEAGUE_ELO = 1500.0


def espn_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"  Failed: {e}"); return {}


def fetch_mls_playoff_games(year: int) -> list:
    cache = CACHE_DIR / f"mls_playoffs_{year}.json"
    if cache.exists(): return json.loads(cache.read_text())

    games = []
    start = datetime(year, 10, 1); end = datetime(year, 12, 20)
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        data = espn_get(f"{ESPN_BASE}/scoreboard?dates={date_str}&seasontype=3&limit=20")
        for ev in data.get("events", []):
            if not ev.get("status", {}).get("type", {}).get("completed", False): continue
            comps = ev.get("competitions", [{}])[0]
            home = next((c for c in comps.get("competitors",[]) if c.get("homeAway")=="home"), None)
            away = next((c for c in comps.get("competitors",[]) if c.get("homeAway")=="away"), None)
            if not home or not away: continue
            h_s = int(home.get("score",0) or 0); a_s = int(away.get("score",0) or 0)
            h_id = home.get("team",{}).get("abbreviation",""); a_id = away.get("team",{}).get("abbreviation","")
            if not h_id or not a_id: continue
            # Determine winner (may go to PK)
            h_winner = home.get("winner", False); a_winner = away.get("winner", False)
            label = 1 if h_winner else (0 if a_winner else (1 if h_s > a_s else 0))
            games.append({
                "game_id": ev.get("id",""), "game_date": current.strftime("%Y-%m-%d"),
                "home_team": h_id, "away_team": a_id,
                "home_score": h_s, "away_score": a_s,
                "home_winner": int(label), "season": year,
            })
        current += timedelta(days=1); time.sleep(0.2)

    seen = set(); unique = []
    for g in games:
        if g["game_id"] not in seen: seen.add(g["game_id"]); unique.append(g)
    cache.write_text(json.dumps(unique, indent=2))
    return unique


def main():
    print("MLS Playoff Data Fetcher"); print("=" * 40)
    all_rows = []
    elo = defaultdict(lambda: LEAGUE_ELO)

    for year in PLAYOFF_YEARS:
        print(f"\nYear {year}")
        games = fetch_mls_playoff_games(year)
        print(f"  Fetched {len(games)} playoff games")

        for g in games:
            h, a = g["home_team"], g["away_team"]
            h_elo = elo[h]; a_elo = elo[a]
            label = g["home_winner"]
            all_rows.append({
                "season": year, "game_id": g["game_id"], "game_date": g["game_date"],
                "home_team": h, "away_team": a,
                "home_score": g["home_score"], "away_score": g["away_score"],
                "label": label, "actual_outcome": "home" if label==1 else "away",
                "is_playoff": 1,
                "elo_diff": h_elo - a_elo,
            })
            exp = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADV)) / 400))
            elo[h] = h_elo + K_FACTOR * (label - exp)
            elo[a] = a_elo + K_FACTOR * ((1-label) - (1-exp))

    if not all_rows:
        print("\nNo data."); return
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} playoff games to {OUT_CSV}")
    print(f"Home win rate: {df['label'].mean():.3f}")

if __name__ == "__main__":
    main()
