// MLS Oracle v4.1 — Results Processor
// Fetches completed match results, updates Elo ratings, computes 3-way accuracy metrics

import { logger } from '../logger.js';
import {
  getPredictionsByDate, updatePredictionResult,
  upsertGameResult, upsertAccuracyLog,
} from '../db/database.js';
import { updateEloAfterMatch } from '../features/eloEngine.js';
import { fetchCompletedResults } from '../api/mlsClient.js';
import type { Prediction, AccuracyLog, GameResult } from '../types.js';

export interface GameWithResult {
  prediction: Prediction;
  homeScore: number;
  awayScore: number;
  outcome: 'H' | 'D' | 'A';
}

export interface DayMetrics {
  accuracy: number;
  brier3way: number;
  drawAccuracy: number;
  doubleChanceAccuracy: number;
  highConvAccuracy: number | null;
}

// ─── Process results for a date ───────────────────────────────────────────────

export async function processResults(date: string): Promise<{
  games: GameWithResult[];
  metrics: DayMetrics;
}> {
  logger.info({ date }, 'Processing MLS results');

  const completedGames = await fetchCompletedResults(date);
  const predictions    = getPredictionsByDate(date);

  if (completedGames.length === 0) {
    logger.warn({ date }, 'No completed MLS matches found for this date');
    return { games: [], metrics: emptyMetrics() };
  }

  const games: GameWithResult[] = [];

  for (const result of completedGames) {
    const pred = predictions.find(
      p => p.home_team === result.home_team && p.away_team === result.away_team
    );

    if (!pred) {
      logger.debug({ home: result.home_team, away: result.away_team }, 'No prediction for result — showing result only');
      games.push({
        prediction: {
          game_id:      result.game_id,
          game_date:    date,
          home_team:    result.home_team,
          away_team:    result.away_team,
          venue:        result.venue,
          matchweek:    0,
          feature_vector: {} as never,
          poisson_home: 0.45, poisson_draw: 0.27, poisson_away: 0.28,
          home_prob:    0.45, draw_prob:    0.27, away_prob:    0.28,
          pick: 'H', pick_confidence: 0.45,
          home_xg: 0, away_xg: 0,
          most_likely_score: '1-1', btts_prob: 0, over25_prob: 0,
          early_season: false, matchweek_label: '',
          model_version: '4.1.0',
          actual_outcome: result.outcome,
          correct: undefined,
          created_at: new Date().toISOString(),
        } as Prediction,
        homeScore: result.home_score,
        awayScore: result.away_score,
        outcome: result.outcome,
      });
      continue;
    }

    // Determine correctness: prediction pick matches actual outcome
    const correct = pred.pick === result.outcome;

    // Update prediction in DB
    updatePredictionResult(pred.game_id, result.outcome, correct);
    pred.actual_outcome = result.outcome;
    pred.correct = correct;

    // Store game result
    upsertGameResult({
      game_id:    result.game_id,
      date:       result.date,
      home_team:  result.home_team,
      away_team:  result.away_team,
      home_score: result.home_score,
      away_score: result.away_score,
      outcome:    result.outcome,
      venue:      result.venue,
    });

    // Update Elo ratings
    updateEloAfterMatch(result.home_team, result.away_team, result.home_score, result.away_score);

    games.push({
      prediction: pred,
      homeScore: result.home_score,
      awayScore: result.away_score,
      outcome: result.outcome,
    });
  }

  // Compute accuracy metrics
  const metrics = computeMetrics(games);

  // Store accuracy log
  if (games.length > 0) {
    const accuracyLog: AccuracyLog = {
      date,
      brier_3way:            metrics.brier3way,
      accuracy:              metrics.accuracy,
      draw_accuracy:         metrics.drawAccuracy,
      double_chance_accuracy: metrics.doubleChanceAccuracy,
      high_conv_accuracy:    metrics.highConvAccuracy ?? 0,
      games_evaluated:       games.length,
    };
    upsertAccuracyLog(accuracyLog);
  }

  logger.info(
    { date, games: games.length, accuracy: metrics.accuracy.toFixed(3), brier: metrics.brier3way.toFixed(4) },
    'MLS results processed'
  );

  return { games, metrics };
}

// ─── Metric computation ───────────────────────────────────────────────────────

function computeMetrics(games: GameWithResult[]): DayMetrics {
  if (games.length === 0) return emptyMetrics();

  let correct = 0;
  let brierSum = 0;
  let drawsWithPrediction = 0;
  let drawsCorrect = 0;
  let dcCorrect = 0;    // double chance (X1 or X2)
  let dcTotal = 0;
  let hcCorrect = 0;
  let hcTotal = 0;

  for (const { prediction: pred, outcome } of games) {
    // Skip games with no actual prediction (no-pick placeholders)
    if (pred.correct === undefined) continue;

    if (pred.correct) correct++;

    // 3-way Brier score: sum of squared errors across all 3 classes
    const outcomeVec = { H: 0, D: 0, A: 0 };
    outcomeVec[outcome] = 1;
    const brierThisGame =
      (pred.home_prob - outcomeVec.H) ** 2 +
      (pred.draw_prob - outcomeVec.D) ** 2 +
      (pred.away_prob - outcomeVec.A) ** 2;
    brierSum += brierThisGame;

    // Draw accuracy: when pick is Draw, did it draw?
    if (pred.pick === 'D') {
      drawsWithPrediction++;
      if (outcome === 'D') drawsCorrect++;
    }

    // Double chance: X1 = home or draw, X2 = away or draw
    // We evaluate: did the model's DC pick (pick != away for X1, pick != home for X2) succeed?
    // Simplified: pick != A → X1 bet, correct if H or D
    if (pred.pick !== 'A') {
      dcTotal++;
      if (outcome === 'H' || outcome === 'D') dcCorrect++;
    } else {
      dcTotal++;
      if (outcome === 'A' || outcome === 'D') dcCorrect++;
    }

    // High conviction
    if (pred.pick_confidence >= 0.60) {
      hcTotal++;
      if (pred.correct) hcCorrect++;
    }
  }

  const n = games.filter(g => g.prediction.correct !== undefined).length;
  if (n === 0) return emptyMetrics();

  return {
    accuracy:            correct / n,
    brier3way:           brierSum / n,
    drawAccuracy:        drawsWithPrediction > 0 ? drawsCorrect / drawsWithPrediction : 0,
    doubleChanceAccuracy: dcTotal > 0 ? dcCorrect / dcTotal : 0,
    highConvAccuracy:    hcTotal > 0 ? hcCorrect / hcTotal : null,
  };
}

function emptyMetrics(): DayMetrics {
  return { accuracy: 0, brier3way: 0, drawAccuracy: 0, doubleChanceAccuracy: 0, highConvAccuracy: null };
}
