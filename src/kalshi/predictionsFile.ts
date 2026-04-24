// Writes today's MLS predictions to predictions/YYYY-MM-DD.json.
// The kalshi-safety service fetches this file via GitHub raw URL to
// decide which picks to back on Kalshi.

import { mkdirSync, writeFileSync } from 'fs';
import { resolve } from 'path';
import type { Prediction } from '../types.js';

interface Pick {
  gameId: string;
  home: string;
  away: string;
  startTime?: string;
  pickedTeam: string;
  pickedSide: 'home' | 'away';
  modelProb: number;
  vegasProb?: number;
  edge?: number;
  confidenceTier?: string;
  extra?: Record<string, unknown>;
}

interface PredictionsFile {
  sport: 'MLS';
  date: string;
  generatedAt: string;
  picks: Pick[];
}

const MIN_PROB = parseFloat(process.env.KALSHI_MIN_PROB ?? '0.58');

function tierFor(prob: number): string {
  if (prob >= 0.68) return 'extreme';
  if (prob >= 0.60) return 'high_conviction';
  if (prob >= 0.52) return 'strong';
  if (prob >= 0.45) return 'lean';
  return 'uncertain';
}

export function writePredictionsFile(date: string, predictions: Prediction[]): string {
  const dir = resolve(process.cwd(), 'predictions');
  mkdirSync(dir, { recursive: true });
  const path = resolve(dir, `${date}.json`);

  const picks: Pick[] = [];
  for (const p of predictions) {
    // MLS is a 3-way market (H/D/A). Only emit a pick when home or away
    // dominates and clears MIN_PROB — skip draws (Kalshi markets are binary).
    const homeProb = p.home_prob;
    const awayProb = p.away_prob;
    const drawProb = p.draw_prob;

    const favorHome = homeProb >= awayProb;
    const sideProb = favorHome ? homeProb : awayProb;

    if (drawProb > sideProb) continue;
    if (sideProb < MIN_PROB) continue;

    const pickedTeam = favorHome ? p.home_team : p.away_team;
    const pickedSide: 'home' | 'away' = favorHome ? 'home' : 'away';
    const vegasProb = favorHome ? p.vegas_home_prob : p.vegas_away_prob;
    const edge = favorHome ? p.edge_home : p.edge_away;

    picks.push({
      gameId: `mls-${date}-${p.away_team}-${p.home_team}`,
      home: p.home_team,
      away: p.away_team,
      pickedTeam,
      pickedSide,
      modelProb: sideProb,
      vegasProb,
      edge,
      confidenceTier: tierFor(sideProb),
      extra: {
        gameId: p.game_id,
        matchweek: p.matchweek,
        venue: p.venue,
        drawProb,
        expectedScore: p.most_likely_score,
        over25_prob: p.over25_prob,
        btts_prob: p.btts_prob,
        early_season: p.early_season,
      },
    });
  }

  const file: PredictionsFile = {
    sport: 'MLS',
    date,
    generatedAt: new Date().toISOString(),
    picks,
  };
  writeFileSync(path, JSON.stringify(file, null, 2));
  return path;
}
