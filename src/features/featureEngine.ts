// MLS Oracle v4.1 — Feature Engineering
// 40+ features including MLS-unique: altitude, turf, travel, DP impact, cup congestion
// All diff features = home − away (positive favors home team)

import { logger } from '../logger.js';
import type { MLSMatch, MLSTeam, FeatureVector, BlendConfig } from '../types.js';
import {
  fetchAllTeamStats, fetchTeamLastGameDate,
  isHighAltitude, isArtificialTurf, isExpansionTeam,
  travelDistance, ALTITUDE_VENUES, CONFERENCE,
} from '../api/mlsClient.js';
import { getEloDiff } from './eloEngine.js';

// ─── MLS league averages ──────────────────────────────────────────────────────

const LEAGUE_AVG_XG_FOR     = 1.35;  // xG per game league average
const LEAGUE_AVG_XG_AGAINST = 1.35;
const LEAGUE_AVG_PPG        = 1.35;  // MLS points per game average (~W:D:L ≈ 45:27:28)
const LEAGUE_DRAW_RATE      = 0.27;  // ~27% draws in MLS

// ─── Matchweek blending schedule ─────────────────────────────────────────────
// Per spec: MW 1-2 = 80/20, MW 3-4 = 60/40, MW 5-7 = 40/60, MW 8-12 = 25/75, MW 13+ = 10/90

export function getBlendConfig(matchweek: number): BlendConfig {
  if (matchweek <= 2)  return { priorWeight: 0.80, currentWeight: 0.20, label: 'EARLY SEASON', earlySeasonFlag: true };
  if (matchweek <= 4)  return { priorWeight: 0.60, currentWeight: 0.40, label: 'EARLY SEASON', earlySeasonFlag: true };
  if (matchweek <= 7)  return { priorWeight: 0.40, currentWeight: 0.60, label: 'Blending',     earlySeasonFlag: false };
  if (matchweek <= 12) return { priorWeight: 0.25, currentWeight: 0.75, label: '',              earlySeasonFlag: false };
  return                      { priorWeight: 0.10, currentWeight: 0.90, label: '',              earlySeasonFlag: false };
}

// ─── Rest days calculation ────────────────────────────────────────────────────

function computeRestDays(lastGameDate: string | null, matchDate: string): number {
  if (!lastGameDate) return 5; // assume 5 days rest if unknown
  const last = new Date(lastGameDate);
  const match = new Date(matchDate);
  return Math.round((match.getTime() - last.getTime()) / (1000 * 60 * 60 * 24));
}

// ─── Midweek flag ─────────────────────────────────────────────────────────────

function isMidweek(dateStr: string): boolean {
  const dayOfWeek = new Date(dateStr + 'T12:00:00Z').getUTCDay(); // 0=Sun, 3=Wed, 5=Fri
  return dayOfWeek === 2 || dayOfWeek === 3; // Tuesday or Wednesday
}

// ─── Cup congestion ───────────────────────────────────────────────────────────
// Simplified: flag August as Leagues Cup period; US Open Cup = mid-May to August
function isCupCongestion(dateStr: string): boolean {
  const month = new Date(dateStr + 'T12:00:00Z').getUTCMonth() + 1; // 1-indexed
  return month === 8; // August = Leagues Cup
}

// ─── Altitude adjustment ──────────────────────────────────────────────────────
// Per spec: visiting team xGA increased by 5-8% in first half, normalizing in second half
// Effective full-match impact: ~3-4% xG reduction for away team

export function altitudePenalty(homeAbbr: string): number {
  const altitude = ALTITUDE_VENUES[homeAbbr];
  if (!altitude) return 0;
  // Scale: 4,327 ft (RSL) → ~0.04 penalty, 5,282 ft (COL) → ~0.05 penalty
  return altitude / 100000; // normalized penalty (0.04-0.05)
}

// ─── Main feature computation ─────────────────────────────────────────────────

export async function computeFeatures(
  match: MLSMatch,
  gameDate: string
): Promise<FeatureVector> {
  const homeAbbr = match.homeTeam.teamAbbr;
  const awayAbbr = match.awayTeam.teamAbbr;

  logger.debug({ home: homeAbbr, away: awayAbbr }, 'Computing MLS features');

  // Fetch all data in parallel
  const [teamStats, homeLastGame, awayLastGame] = await Promise.all([
    fetchAllTeamStats(),
    fetchTeamLastGameDate(homeAbbr, gameDate),
    fetchTeamLastGameDate(awayAbbr, gameDate),
  ]);

  const home = teamStats.get(homeAbbr) ?? defaultTeam(homeAbbr);
  const away = teamStats.get(awayAbbr) ?? defaultTeam(awayAbbr);

  // ── Elo ──────────────────────────────────────────────────────────────────────
  const eloDiff = getEloDiff(homeAbbr, awayAbbr);

  // ── xG features ──────────────────────────────────────────────────────────────
  const xgForDiff     = home.xgFor      - away.xgFor;
  const xgAgainstDiff = home.xgAgainst  - away.xgAgainst;
  const xgDiff        = home.xgDiff     - away.xgDiff;
  const xptsDiff      = home.xPts       - away.xPts;
  const ppgDiff       = home.winPct * 3 - away.winPct * 3; // convert to PPG equiv

  // ── Style ─────────────────────────────────────────────────────────────────────
  const possessionDiff = home.possession - away.possession;
  const passAccDiff    = home.passAccuracy - away.passAccuracy;

  // ── Form ──────────────────────────────────────────────────────────────────────
  const form5Diff         = home.form5xPts - away.form5xPts;
  const overperformDiff   = home.overperformance - away.overperformance;

  // ── Draw tendency ─────────────────────────────────────────────────────────────
  const drawTendencyDiff = home.drawRate - away.drawRate;

  // ── Home advantage (season home vs away split) ────────────────────────────────
  // Proxy: MLS home advantage is real, especially with travel
  const homeAdvDiff = 0.35; // consistent ~+0.35 goals home advantage in MLS

  // ── Venue environment ─────────────────────────────────────────────────────────
  const altFlag    = isHighAltitude(homeAbbr) ? 1 : 0;
  const altPenalty = altitudePenalty(homeAbbr);
  const turfFlag   = isArtificialTurf(homeAbbr) ? 1 : 0;

  // ── Travel ────────────────────────────────────────────────────────────────────
  const awayTravelMiles = travelDistance(awayAbbr, homeAbbr);
  const crossCountryFlag = awayTravelMiles > 1500 ? 1 : 0;

  // ── Rest days ─────────────────────────────────────────────────────────────────
  const homeRestDays = computeRestDays(homeLastGame, gameDate);
  const awayRestDays = computeRestDays(awayLastGame, gameDate);
  const restDaysDiff = homeRestDays - awayRestDays;

  // ── Schedule context ──────────────────────────────────────────────────────────
  const midweekFlag    = isMidweek(gameDate) ? 1 : 0;
  const cupCongestion  = isCupCongestion(gameDate) ? 1 : 0;

  // ── Designated Player impact ──────────────────────────────────────────────────
  const dpImpactDiff    = home.dpImpact    - away.dpImpact;
  const dpAvailableDiff = home.dpAvailable - away.dpAvailable;

  // ── Salary proxy ──────────────────────────────────────────────────────────────
  // No public salary data available; use xG performance as proxy
  const rosterSalaryDiff = home.xgFor - away.xgFor; // normalized placeholder

  // ── Context ───────────────────────────────────────────────────────────────────
  const sameConference  = (CONFERENCE[homeAbbr] === CONFERENCE[awayAbbr]) ? 1 : 0;
  const expansionFlag   = (isExpansionTeam(homeAbbr) || isExpansionTeam(awayAbbr)) ? 1 : 0;

  // Playoff position diff: use points gap from playoff line (6th in each conference)
  // Approximation: higher xPts = closer to playoff spot → positive = home better
  const playoffPosDiff = home.xPts - away.xPts;

  // Manager tenure diff
  const managerTenureDiff = home.managerTenure - away.managerTenure;

  const features: FeatureVector = {
    elo_diff: eloDiff,
    xg_for_diff: xgForDiff,
    xg_against_diff: xgAgainstDiff,
    xg_diff: xgDiff,
    xpts_diff: xptsDiff,
    ppg_diff: ppgDiff,
    possession_diff: possessionDiff,
    pass_pct_diff: passAccDiff,
    form_5g_xpts_diff: form5Diff,
    overperformance_diff: overperformDiff,
    draw_tendency_diff: drawTendencyDiff,
    is_home: 1,
    home_advantage_diff: homeAdvDiff,
    altitude_flag: altFlag,
    altitude_penalty: altPenalty,
    turf_flag: turfFlag,
    travel_distance_diff: awayTravelMiles,
    cross_country_flag: crossCountryFlag,
    rest_days_diff: restDaysDiff,
    midweek_flag: midweekFlag,
    cup_congestion: cupCongestion,
    dp_impact_diff: dpImpactDiff,
    dp_available_diff: dpAvailableDiff,
    roster_salary_diff: rosterSalaryDiff,
    conference_diff: sameConference,
    expansion_flag: expansionFlag,
    playoff_position_diff: playoffPosDiff,
    manager_tenure_diff: managerTenureDiff,
    vegas_home_prob: 0,
    vegas_draw_prob: 0,
    vegas_away_prob: 0,
  };

  logger.debug(
    {
      home: homeAbbr, away: awayAbbr,
      eloDiff: eloDiff.toFixed(0),
      xgDiff: xgDiff.toFixed(2),
      altFlag, turfFlag,
      travelMiles: awayTravelMiles.toFixed(0),
      crossCountry: crossCountryFlag,
      restDaysDiff,
    },
    'MLS features computed'
  );

  return features;
}

// ─── Default team fallback ────────────────────────────────────────────────────

function defaultTeam(abbr: string): MLSTeam {
  return {
    teamId: 0, teamAbbr: abbr, teamName: abbr,
    conference: CONFERENCE[abbr] ?? 'East',
    w: 0, d: 0, l: 0, gp: 0, pts: 0, winPct: 0.5,
    goalsFor: 1.45, goalsAgainst: 1.45, goalDiff: 0,
    xgFor: LEAGUE_AVG_XG_FOR, xgAgainst: LEAGUE_AVG_XG_AGAINST, xgDiff: 0,
    shotsFor: 12, shotsOnTargetFor: 4.5,
    shotsAgainst: 12, shotsOnTargetAgainst: 4.5,
    possession: 50, passAccuracy: 75,
    xPts: LEAGUE_AVG_PPG,
    dpImpact: 0, dpAvailable: 3,
    form5xPts: LEAGUE_AVG_PPG,
    overperformance: 0, drawRate: LEAGUE_DRAW_RATE,
    managerTenure: 24,
  };
}
