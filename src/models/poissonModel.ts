// MLS Oracle v4.1 — Poisson Goal Simulation Engine
// Estimates lambda (xG) for each team, simulates 50,000 matches via Poisson distribution
// Produces P(home win), P(draw), P(away win), BTTS, Over 2.5 goals

import type { FeatureVector, PoissonResult, ThreeWayProbs } from '../types.js';

const N_SIMULATIONS = 50_000;

// MLS league averages per match
const LEAGUE_AVG_HOME_GOALS = 1.50;   // ~1.5 per spec (+0.35 home adv over away)
const LEAGUE_AVG_AWAY_GOALS = 1.25;
const LEAGUE_AVG_GOALS      = LEAGUE_AVG_HOME_GOALS + LEAGUE_AVG_AWAY_GOALS; // 2.75, rounds to ~2.9

// Attack/defense strength reference
const LEAGUE_AVG_XG_FOR     = 1.35;
const LEAGUE_AVG_XG_AGAINST = 1.35;

// ─── Poisson random number (Knuth algorithm) ──────────────────────────────────
// For lambda <= 30, Knuth is accurate and fast.

function poissonRandom(lambda: number): number {
  if (lambda <= 0) return 0;
  if (lambda > 30) {
    // For large lambda, approximate with Normal
    const normal = Math.sqrt(lambda) * (Math.random() - 0.5) * 2 * Math.sqrt(3);
    return Math.max(0, Math.round(lambda + normal));
  }
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1.0;
  do {
    k++;
    p *= Math.random();
  } while (p > L);
  return k - 1;
}

// ─── Lambda estimation ────────────────────────────────────────────────────────
// Per spec formula:
// λ_home = attack_str × opp_def_weakness × home_adv × altitude_adj × fatigue_adj × league_avg

export interface LambdaEstimate {
  lambdaHome: number;   // expected goals for home team
  lambdaAway: number;   // expected goals for away team
}

export function estimateLambdas(features: FeatureVector): LambdaEstimate {
  // Attack strength = (xG for per game) / league average
  // Encoded in features as diffs: xg_for_diff = home_xgFor - away_xgFor
  // Recover individual team values relative to league avg
  const homeXgFor  = Math.max(0.3, LEAGUE_AVG_XG_FOR  + features.xg_for_diff  / 2);
  const awayXgFor  = Math.max(0.3, LEAGUE_AVG_XG_FOR  - features.xg_for_diff  / 2);
  const homeXgAg   = Math.max(0.3, LEAGUE_AVG_XG_AGAINST + features.xg_against_diff / 2);
  const awayXgAg   = Math.max(0.3, LEAGUE_AVG_XG_AGAINST - features.xg_against_diff / 2);

  // Attack and defense strength ratios relative to league average
  const homeAttack = homeXgFor  / LEAGUE_AVG_XG_FOR;
  const awayAttack = awayXgFor  / LEAGUE_AVG_XG_FOR;
  const homeDefWeak = homeXgAg  / LEAGUE_AVG_XG_AGAINST; // high = weak defense
  const awayDefWeak = awayXgAg  / LEAGUE_AVG_XG_AGAINST;

  // Base lambdas (attack × opponent defense weakness × league average)
  let lambdaHome = homeAttack * awayDefWeak * LEAGUE_AVG_HOME_GOALS;
  let lambdaAway = awayAttack * homeDefWeak * LEAGUE_AVG_AWAY_GOALS;

  // ── Altitude adjustment (per spec: 3-5% xG reduction for away visiting altitude) ──
  if (features.altitude_flag === 1) {
    // altitude_penalty ≈ 0.04-0.05 (encoded in feature vector)
    const altPenalty = Math.max(0, Math.min(0.08, features.altitude_penalty));
    lambdaAway *= (1 - altPenalty);
    // First-half emphasis: partial benefit for home team too
    lambdaHome *= (1 + altPenalty * 0.3);
  }

  // ── Artificial turf adjustment ────────────────────────────────────────────────
  // Turf increases pace, slightly favors home team accustomed to surface
  if (features.turf_flag === 1) {
    lambdaHome *= 1.03;
    lambdaAway *= 1.01;
  }

  // ── Travel fatigue ─────────────────────────────────────────────────────────────
  // Per spec: travel > 1,500 miles reduces away team xG by 3-5%
  if (features.cross_country_flag === 1) {
    lambdaAway *= 0.96; // ~4% reduction for coast-to-coast travel
  } else if (features.travel_distance_diff > 800) {
    lambdaAway *= 0.98; // ~2% reduction for moderate travel
  }

  // ── Midweek + travel compound fatigue ────────────────────────────────────────
  // Per spec: coast-to-coast + midweek = combined 5-7% penalty
  if (features.midweek_flag === 1 && features.cross_country_flag === 1) {
    lambdaAway *= 0.97; // additional 3% on top of cross_country (total ~7%)
  }

  // ── Rest days ──────────────────────────────────────────────────────────────────
  const restDiff = features.rest_days_diff;
  if (restDiff >= 3) {
    lambdaHome *= 1.02; // home team well-rested
    lambdaAway *= 0.98;
  } else if (restDiff <= -3) {
    lambdaHome *= 0.98;
    lambdaAway *= 1.02;
  }

  // ── Elo-based quality adjustment ───────────────────────────────────────────────
  // Strong Elo lead → expect more goals from the stronger team
  const eloDiffNorm = Math.max(-400, Math.min(400, features.elo_diff)) / 400;
  lambdaHome *= (1 + eloDiffNorm * 0.10);
  lambdaAway *= (1 - eloDiffNorm * 0.10);

  // ── DP availability ────────────────────────────────────────────────────────────
  // Missing a DP is a larger hit than in uncapped leagues (50-60% of output)
  const dpDiff = features.dp_available_diff;
  if (dpDiff > 0) {
    lambdaHome *= (1 + dpDiff * 0.04); // each available DP = ~4% xG boost
  } else if (dpDiff < 0) {
    lambdaAway *= (1 + (-dpDiff) * 0.04);
  }

  // ── Cup congestion ─────────────────────────────────────────────────────────────
  if (features.cup_congestion === 1) {
    lambdaHome *= 0.97;
    lambdaAway *= 0.97;
  }

  // Clamp to valid range [0.2, 4.0]
  lambdaHome = Math.max(0.2, Math.min(4.0, lambdaHome));
  lambdaAway = Math.max(0.2, Math.min(4.0, lambdaAway));

  return { lambdaHome, lambdaAway };
}

// ─── Monte Carlo simulation ───────────────────────────────────────────────────

export function runPoissonSimulation(features: FeatureVector): PoissonResult {
  const { lambdaHome, lambdaAway } = estimateLambdas(features);

  let homeWins = 0;
  let draws = 0;
  let awayWins = 0;
  let btts = 0;       // both teams to score
  let over25 = 0;     // over 2.5 total goals

  // Score frequency map for most-likely score
  const scoreFreq = new Map<string, number>();

  for (let i = 0; i < N_SIMULATIONS; i++) {
    const homeGoals = poissonRandom(lambdaHome);
    const awayGoals = poissonRandom(lambdaAway);

    if (homeGoals > awayGoals) homeWins++;
    else if (homeGoals === awayGoals) draws++;
    else awayWins++;

    if (homeGoals > 0 && awayGoals > 0) btts++;
    if (homeGoals + awayGoals > 2.5) over25++;

    // Track score frequency (cap at 7 goals per team for map size)
    const key = `${Math.min(homeGoals, 7)}-${Math.min(awayGoals, 7)}`;
    scoreFreq.set(key, (scoreFreq.get(key) ?? 0) + 1);
  }

  const probs: ThreeWayProbs = {
    home: homeWins / N_SIMULATIONS,
    draw: draws    / N_SIMULATIONS,
    away: awayWins / N_SIMULATIONS,
  };

  // Most likely scoreline
  let topScore = '1-1';
  let topFreq = 0;
  for (const [score, freq] of scoreFreq.entries()) {
    if (freq > topFreq) {
      topFreq = freq;
      topScore = score;
    }
  }

  return {
    probs,
    home_xg: lambdaHome,
    away_xg: lambdaAway,
    most_likely_score: [
      Number(topScore.split('-')[0]),
      Number(topScore.split('-')[1]),
    ],
    btts_prob:  btts  / N_SIMULATIONS,
    over25_prob: over25 / N_SIMULATIONS,
    simulations: N_SIMULATIONS,
  };
}
