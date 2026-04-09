// MLS Oracle v4.1 — Daily Pipeline
// Orchestrates: Fetch → Features → Poisson → ML → Edge → Store → Print

import { logger } from './logger.js';
import { fetchSchedule } from './api/mlsClient.js';
import { computeFeatures, getBlendConfig } from './features/featureEngine.js';
import { runPoissonSimulation } from './models/poissonModel.js';
import { upsertPrediction, initDb } from './db/database.js';
import { loadModel, predict as mlPredict, isModelLoaded, getModelInfo } from './models/metaModel.js';
import { computeEdge, formatEdge } from './features/marketEdge.js';
import { initializeOdds, hasAnyOdds, getOddsForGame } from './api/oddsClient.js';
import { seedElos } from './features/eloEngine.js';
import type { MLSMatch, Prediction, PipelineOptions } from './types.js';

const MODEL_VERSION = '4.1.0';

// ─── Main pipeline ────────────────────────────────────────────────────────────

export async function runPipeline(options: PipelineOptions = {}): Promise<Prediction[]> {
  const today     = new Date().toISOString().split('T')[0];
  const gameDate  = options.date ?? today;

  logger.info({ gameDate, version: MODEL_VERSION }, '=== MLS Oracle v4.1 Pipeline Start ===');

  // 1. Initialize database
  await initDb();

  // 2. Seed Elo ratings
  seedElos();

  // 3. Attempt ML meta-model
  const modelLoaded = loadModel();
  if (modelLoaded) {
    const info = getModelInfo();
    logger.info(
      { version: info?.version, avgBrier: info?.avgBrier, trainDates: info?.trainDates },
      'Using ML meta-model (multinomial logistic regression)'
    );
  } else {
    logger.info('ML model not found — using Poisson simulation probabilities directly');
    logger.info('Run: npm run train  (python python/train_model.py)');
  }

  // 4. Initialize Vegas odds (3-way: H/D/A)
  await initializeOdds(gameDate);
  if (hasAnyOdds()) {
    logger.info('Vegas 3-way lines loaded — will compute H/D/A edge');
  }

  // 5. Fetch today's schedule
  const matches = await fetchSchedule(gameDate);
  if (matches.length === 0) {
    logger.warn({ gameDate }, 'No MLS matches found for date');
    return [];
  }

  logger.info({ gameDate, matches: matches.length }, 'Schedule fetched');

  // 6. Process each match
  const predictions: Prediction[] = [];
  let processed = 0;
  let failed = 0;

  for (const match of matches) {
    try {
      const pred = await processMatch(match, gameDate, modelLoaded);
      if (pred) {
        predictions.push(pred);
        processed++;
      }
    } catch (err) {
      failed++;
      logger.error(
        { err, matchId: match.matchId, home: match.homeTeam.teamAbbr, away: match.awayTeam.teamAbbr },
        'Failed to process match'
      );
    }
  }

  logger.info({ processed, failed, total: matches.length }, 'Pipeline complete');

  if (options.verbose !== false) {
    printPredictions(predictions, gameDate, modelLoaded);
  }

  return predictions;
}

// ─── Single match processing ──────────────────────────────────────────────────

async function processMatch(
  match: MLSMatch,
  gameDate: string,
  modelLoaded: boolean
): Promise<Prediction | null> {
  const homeAbbr = match.homeTeam.teamAbbr;
  const awayAbbr = match.awayTeam.teamAbbr;

  logger.info({ matchId: match.matchId, matchup: `${awayAbbr} @ ${homeAbbr}` }, 'Processing match');

  // Skip completed or live matches
  const statusLower = match.status.toLowerCase();
  if (statusLower.includes('final') || statusLower.includes('in progress') || statusLower.includes('live')) {
    logger.info({ status: match.status }, 'Skipping non-upcoming match');
    return null;
  }

  // ── Step A: Blending config (early-season / midseason) ─────────────────────
  const blend = getBlendConfig(match.matchweek);
  logger.debug({ matchweek: match.matchweek, blend: blend.label }, 'Blend config');

  // ── Step B: Compute feature vector ─────────────────────────────────────────
  const features = await computeFeatures(match, gameDate);

  // ── Step C: Poisson simulation → 3-way probabilities ───────────────────────
  const poissonResult = runPoissonSimulation(features);
  const { probs: poissonProbs } = poissonResult;

  // ── Step D: ML calibration (if model available) ─────────────────────────────
  let calibrated = poissonProbs;

  if (modelLoaded && isModelLoaded()) {
    calibrated = mlPredict(features, poissonProbs);
    logger.debug(
      {
        matchId: match.matchId,
        poisson: `H${(poissonProbs.home * 100).toFixed(1)}% D${(poissonProbs.draw * 100).toFixed(1)}% A${(poissonProbs.away * 100).toFixed(1)}%`,
        ml:      `H${(calibrated.home * 100).toFixed(1)}% D${(calibrated.draw * 100).toFixed(1)}% A${(calibrated.away * 100).toFixed(1)}%`,
      },
      'ML calibration applied'
    );
  }

  // ── Step E: Market edge computation ────────────────────────────────────────
  let vegasHomeProb: number | undefined;
  let vegasDrawProb: number | undefined;
  let vegasAwayProb: number | undefined;
  let edgeHome: number | undefined;
  let edgeDraw: number | undefined;
  let edgeAway: number | undefined;

  const matchupKey = `${awayAbbr}@${homeAbbr}`;
  const gameOdds = getOddsForGame(matchupKey);

  if (gameOdds) {
    vegasHomeProb = gameOdds.homeImpliedProb;
    vegasDrawProb = gameOdds.drawImpliedProb;
    vegasAwayProb = gameOdds.awayImpliedProb;
    edgeHome = calibrated.home - vegasHomeProb;
    edgeDraw = calibrated.draw - vegasDrawProb;
    edgeAway = calibrated.away - vegasAwayProb;

    const edgeResult = computeEdge(
      calibrated.home, calibrated.draw, calibrated.away,
      vegasHomeProb, vegasDrawProb, vegasAwayProb
    );
    logger.info({ matchId: match.matchId, matchup: matchupKey }, formatEdge(edgeResult));

    // Inject Vegas probs into features for ML use
    features.vegas_home_prob = vegasHomeProb;
    features.vegas_draw_prob = vegasDrawProb;
    features.vegas_away_prob = vegasAwayProb;
  }

  // ── Step F: Determine pick ────────────────────────────────────────────────
  type Pick = 'H' | 'D' | 'A';
  const pickArr: Array<{ outcome: Pick; prob: number }> = [
    { outcome: 'H', prob: calibrated.home },
    { outcome: 'D', prob: calibrated.draw },
    { outcome: 'A', prob: calibrated.away },
  ];
  const best = pickArr.reduce((a, b) => a.prob > b.prob ? a : b);

  // ── Step G: Build prediction record ────────────────────────────────────────
  const prediction: Prediction = {
    game_date: gameDate,
    game_id: match.matchId,
    home_team: homeAbbr,
    away_team: awayAbbr,
    venue: match.venue,
    matchweek: match.matchweek,
    feature_vector: features,

    poisson_home: poissonProbs.home,
    poisson_draw: poissonProbs.draw,
    poisson_away: poissonProbs.away,

    home_prob: calibrated.home,
    draw_prob: calibrated.draw,
    away_prob: calibrated.away,

    pick: best.outcome,
    pick_confidence: best.prob,

    home_xg: poissonResult.home_xg,
    away_xg: poissonResult.away_xg,
    most_likely_score: `${poissonResult.most_likely_score[0]}-${poissonResult.most_likely_score[1]}`,
    btts_prob:  poissonResult.btts_prob,
    over25_prob: poissonResult.over25_prob,

    vegas_home_prob: vegasHomeProb,
    vegas_draw_prob: vegasDrawProb,
    vegas_away_prob: vegasAwayProb,
    edge_home: edgeHome,
    edge_draw: edgeDraw,
    edge_away: edgeAway,

    early_season: blend.earlySeasonFlag,
    matchweek_label: blend.label,

    model_version: MODEL_VERSION,
    created_at: new Date().toISOString(),
  };

  // ── Step H: Store in DB ────────────────────────────────────────────────────
  upsertPrediction(prediction);

  return prediction;
}

// ─── Console output ───────────────────────────────────────────────────────────

function printPredictions(
  predictions: Prediction[],
  gameDate: string,
  mlModelActive = false,
): void {
  if (predictions.length === 0) {
    console.log(`\nNo predictions for ${gameDate}\n`);
    return;
  }

  const modelLabel = mlModelActive ? 'ML+Poisson' : 'Poisson Monte Carlo';
  const hasEdge    = predictions.some(p => p.edge_home !== undefined);
  const totalWidth = hasEdge ? 125 : 105;

  console.log('\n' + '═'.repeat(totalWidth));
  console.log(`  MLS ORACLE v4.1  ·  Matchday Predictions for ${gameDate}  ·  ${predictions.length} match${predictions.length !== 1 ? 'es' : ''}  ·  [${modelLabel}]`);
  console.log('═'.repeat(totalWidth));

  const header = [
    pad('MATCHUP', 24),
    pad('MW', 4),
    pad('HOME%', 7),
    pad('DRAW%', 7),
    pad('AWAY%', 7),
    pad('PICK', 6),
    pad('CONF', 6),
    pad('xG', 7),
    pad('PROJ', 5),
    pad('O2.5', 5),
    pad('FLAGS', 15),
  ];
  if (hasEdge) header.push(pad('EDGE', 8));

  console.log('\n' + header.join('  '));
  console.log('─'.repeat(totalWidth));

  const sorted = [...predictions].sort((a, b) => b.pick_confidence - a.pick_confidence);

  for (const p of sorted) {
    const matchup = `${p.away_team} @ ${p.home_team}`;
    const flags: string[] = [];
    if (p.feature_vector.altitude_flag === 1) flags.push('ALT');
    if (p.feature_vector.turf_flag === 1)     flags.push('TURF');
    if (p.feature_vector.cross_country_flag === 1) flags.push('CC');

    const marker = p.pick_confidence >= 0.65 ? ' ⚡⚡'
                 : p.pick_confidence >= 0.60 ? ' ⚡'
                 : p.pick_confidence >= 0.52 ? ' ✅'
                 : '';

    const row = [
      pad(matchup, 24),
      pad(String(p.matchweek || '?'), 4),
      pad((p.home_prob * 100).toFixed(1) + '%', 7),
      pad((p.draw_prob * 100).toFixed(1) + '%', 7),
      pad((p.away_prob * 100).toFixed(1) + '%', 7),
      pad(p.pick, 6),
      pad((p.pick_confidence * 100).toFixed(1) + '%', 6),
      pad(`${p.home_xg.toFixed(1)}-${p.away_xg.toFixed(1)}`, 7),
      pad(p.most_likely_score, 5),
      pad((p.over25_prob * 100).toFixed(0) + '%', 5),
      pad(flags.join(' ') || '—', 15),
    ];

    if (hasEdge) {
      if (p.edge_home !== undefined) {
        const bestEdge = [p.edge_home, p.edge_draw ?? 0, p.edge_away ?? 0]
          .reduce((a, b) => Math.abs(a) > Math.abs(b) ? a : b, 0);
        const sign = bestEdge >= 0 ? '+' : '';
        row.push(pad(`${sign}${(bestEdge * 100).toFixed(1)}%`, 8));
      } else {
        row.push(pad('—', 8));
      }
    }

    console.log(row.join('  ') + marker);
  }

  console.log('─'.repeat(totalWidth));
  console.log('\nLegend: ⚡⚡ = Extreme (65%+)  ⚡ = High Conviction (60%+)  ✅ = Strong (52%+)');
  console.log('FLAGS: ALT = Altitude  TURF = Artificial Turf  CC = Coast-to-Coast Travel');
  if (hasEdge) console.log('EDGE: model vs vig-removed Vegas probability');

  const avgConf = predictions.reduce((s, p) => s + p.pick_confidence, 0) / predictions.length;
  const highConv = predictions.filter(p => p.pick_confidence >= 0.60).length;
  const earlySeason = predictions.some(p => p.early_season);

  let summary = (
    `\nSummary: ${predictions.length} matches  |  avg confidence = ${(avgConf * 100).toFixed(1)}%` +
    `  |  ${highConv} high-conviction (60%+)`
  );
  if (earlySeason) summary += '  |  [EARLY SEASON — prior-season data blended]';
  console.log(summary);
  console.log('═'.repeat(totalWidth) + '\n');

  // High-conviction picks section
  const hcPicks = sorted.filter(p => p.pick_confidence >= 0.60);
  if (hcPicks.length > 0) {
    console.log('─'.repeat(72));
    console.log('  HIGH-CONVICTION PICKS (60%+ on single outcome)');
    console.log('─'.repeat(72));
    for (const p of hcPicks) {
      const matchup = `${p.away_team} @ ${p.home_team}`;
      const pickStr = p.pick === 'H' ? p.home_team
                    : p.pick === 'A' ? p.away_team
                    : 'Draw';
      console.log(
        `  ${pad(matchup, 24)}  Pick: ${pad(pickStr, 12)}  ${(p.pick_confidence * 100).toFixed(1)}%` +
        (p.edge_home !== undefined ? `  xG: ${p.home_xg.toFixed(2)}-${p.away_xg.toFixed(2)}` : '')
      );
    }
    console.log('─'.repeat(72) + '\n');
  }
}

function pad(str: string, width: number): string {
  if (str.length >= width) return str.slice(0, width);
  return str + ' '.repeat(width - str.length);
}

export { processMatch };
