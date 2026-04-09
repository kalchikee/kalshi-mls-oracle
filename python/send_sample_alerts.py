#!/usr/bin/env python3
"""
MLS Oracle v4.1 — Send sample Discord embeds
Sends one example prediction message and one example recap message
so you can preview the exact format in your Discord channel.
"""

import json
import os
import sys
from datetime import date

import requests

WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
if not WEBHOOK_URL:
    print("ERROR: DISCORD_WEBHOOK_URL not set")
    sys.exit(1)

MLS_BLUE  = 0x0053A0
MLS_GREEN = 0x27AE60
MLS_RED   = 0xE74C3C
MLS_GRAY  = 0x95A5A6

TODAY = str(date.today())


def send(payload: dict):
    r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
    if not r.ok:
        print(f"  ERROR {r.status_code}: {r.text}")
    else:
        print(f"  Sent OK ({r.status_code})")


# ─────────────────────────────────────────────────────────────────────────────
# EMBED 1 & 2: Matchday Predictions (morning briefing)
# ─────────────────────────────────────────────────────────────────────────────
print("Sending sample matchday predictions embed...")

picks_embed = {
    "title": f"⚽ MLS Oracle — Matchday Predictions | {TODAY}",
    "description": "6 matches today  ·  7-day accuracy: **54.2%**  ·  📈 Season: **23-18** (56.1%)  ·  ⚡ High Conv: **7-4** (63.6%)",
    "color": MLS_BLUE,
    "fields": [
        {
            "name": "⚡⚡ **LAFC @ ATL** | Pick: **ATL** (67.3%)  🟡 EARLY SEASON",
            "value": (
                "🏠 51.2% · 🤝 24.1% · ✈️ 24.7%\n"
                "⚽ Proj: **2-1** · xG: 1.82-1.31\n"
                "BTTS: 48.3% · O2.5: 61.7%\n"
                "✈️ 1,947mi · 💹 Edge: +8.1% (H)"
            ),
            "inline": False,
        },
        {
            "name": "⚡ **SEA @ COL** | Pick: **COL** (62.8%)",
            "value": (
                "🏠 54.1% · 🤝 22.3% · ✈️ 23.6%\n"
                "⚽ Proj: **2-1** · xG: 1.94-1.18\n"
                "BTTS: 44.1% · O2.5: 58.9%\n"
                "🏔️ Altitude · ✈️ 1,234mi"
            ),
            "inline": False,
        },
        {
            "name": "✅ **NYC @ MIA** | Pick: **MIA** (55.6%)",
            "value": (
                "🏠 46.8% · 🤝 26.1% · ✈️ 27.1%\n"
                "⚽ Proj: **1-1** · xG: 1.44-1.52\n"
                "BTTS: 52.7% · O2.5: 54.3%"
            ),
            "inline": False,
        },
        {
            "name": "✅ **POR @ SEA** | Pick: **SEA** (54.1%)",
            "value": (
                "🏠 48.3% · 🤝 25.8% · ✈️ 25.9%\n"
                "⚽ Proj: **2-1** · xG: 1.71-1.48\n"
                "BTTS: 50.2% · O2.5: 57.1%\n"
                "🟩 Turf"
            ),
            "inline": False,
        },
        {
            "name": "🪙 **CHI @ KC** | Pick: **Draw** (44.2%)",
            "value": (
                "🏠 34.7% · 🤝 37.1% · ✈️ 28.2%\n"
                "⚽ Proj: **1-1** · xG: 1.22-1.19\n"
                "BTTS: 45.8% · O2.5: 41.3%"
            ),
            "inline": False,
        },
        {
            "name": "🪙 **PHI @ NYC** | Pick: **PHI** (51.3%)",
            "value": (
                "🏠 41.2% · 🤝 28.4% · ✈️ 30.4%\n"
                "⚽ Proj: **2-1** · xG: 1.55-1.47\n"
                "BTTS: 49.6% · O2.5: 53.2%"
            ),
            "inline": False,
        },
    ],
    "footer": {
        "text": "⚡⚡ = Extreme conviction (65%+)  |  ⚡ = High conviction (60%+)  |  ✅ = Strong (52%+)  |  🪙 = Lean  |  MLS Oracle v4.1"
    },
    "timestamp": f"{TODAY}T14:00:00.000Z",
}

high_conv_embed = {
    "title": "💰 High-Conviction Plays — 2 today",
    "description": "Games where the model has ≥60% on a single outcome. Target accuracy: **65-70%**.",
    "color": MLS_GREEN,
    "fields": [
        {
            "name": "⚡⚡ EXTREME: LAFC @ ATL",
            "value": (
                "**Pick:** ATL · 67.3%\n"
                "**Why:** Model confidence: 67.3% · Away travel 1,947mi · Home rest advantage\n"
                "✈️ 1,947mi"
            ),
            "inline": False,
        },
        {
            "name": "⚡ HIGH CONVICTION: SEA @ COL",
            "value": (
                "**Pick:** COL · 62.8%\n"
                "**Why:** Model confidence: 62.8% · Altitude home advantage · Away travel 1,234mi\n"
                "🏔️ Altitude · ✈️ 1,234mi"
            ),
            "inline": False,
        },
    ],
    "footer": {
        "text": "Bet responsibly. Past performance ≠ future results. MLS Oracle v4.1"
    },
}

send({"embeds": [picks_embed, high_conv_embed]})

# Small delay to avoid Discord rate limit
import time
time.sleep(2)

# ─────────────────────────────────────────────────────────────────────────────
# EMBED 3: Evening Recap
# ─────────────────────────────────────────────────────────────────────────────
print("Sending sample evening recap embed...")

recap_embed = {
    "title": f"🌙 MLS Oracle — Results | {TODAY}",
    "color": MLS_GREEN,
    "fields": [
        {
            "name": "📊 Summary",
            "value": (
                "**🟢 Tonight: 4/6 correct (67%)**\n"
                "**⚡ High-conviction: 2/2 correct (100%)**\n"
                "Draw accuracy: 60% · DC accuracy: 83%\n"
                "3-way Brier: 0.4812\n"
                "📈 Season: **27-20** (57.4%)\n"
                "⚡ Season 60%+ picks: **9-4**"
            ),
            "inline": False,
        },
        {
            "name": "🎯 Match Results",
            "value": (
                "✅⚡ **LAFC @ ATL**: 1-2 *(ATL)* → ATL\n"
                "✅⚡ **SEA @ COL**: 0-2 *(COL)* → COL\n"
                "✅ **NYC @ MIA**: 1-2 *(MIA)* → MIA\n"
                "✅ **POR @ SEA**: 0-1 *(SEA)* → SEA\n"
                "❌ **CHI @ KC**: 2-1 *(Draw)* → KC\n"
                "❌ **PHI @ NYC**: 1-0 *(PHI)* → NYC"
            ),
            "inline": False,
        },
    ],
    "footer": {
        "text": "MLS Oracle v4.1 · 3-way Brier tracks calibration · Target: 52%+ match result"
    },
    "timestamp": f"{TODAY}T04:00:00.000Z",
}

send({"embeds": [recap_embed]})

print("\nDone — check your Discord channel.")
