// MLS Oracle v4.1 — Soccer Elo Rating Engine
// K-factor: 20 | Margin capped at 4 goals | Home advantage: +70 Elo
// Offseason regression: 75% carry + 25% league mean
// Expansion teams seeded at league_mean − 150

import { getElo, upsertElo, getAllElos } from '../db/database.js';
import { logger } from '../logger.js';

export const LEAGUE_MEAN_ELO = 1500;
const K_FACTOR = 20;
const HOME_ADVANTAGE_ELO = 70;   // Soccer home advantage is smaller than NBA
const GOAL_CAP = 4;              // Cap margin of victory at 4 goals (per MLS spec)
const EXPANSION_PENALTY = 150;   // Expansion teams start below mean

export const ALL_MLS_ABBRS = [
  'ATL', 'CHI', 'CLB', 'CLT', 'CIN', 'DC', 'MIA', 'MTL', 'NE', 'NSH',
  'NYC', 'NYRB', 'ORL', 'PHI', 'TOR',
  'ATX', 'COL', 'DAL', 'HOU', 'KC', 'LA', 'LAFC', 'MIN', 'POR', 'RSL',
  'SD', 'SEA', 'SJ', 'STL', 'VAN',
];

// Teams that are in their first or second year (limited historical data)
const EXPANSION_TEAMS = new Set(['SD']);

// ─── Seed all teams with default Elo if not present ──────────────────────────

export function seedElos(): void {
  for (const abbr of ALL_MLS_ABBRS) {
    const existing = getElo(abbr);
    if (existing === LEAGUE_MEAN_ELO) {
      const rating = EXPANSION_TEAMS.has(abbr)
        ? LEAGUE_MEAN_ELO - EXPANSION_PENALTY
        : LEAGUE_MEAN_ELO;
      upsertElo({ teamAbbr: abbr, rating, updatedAt: new Date().toISOString() });
    }
  }
}

// ─── Expected win probability (home perspective) ──────────────────────────────

export function eloWinProb(homeElo: number, awayElo: number): number {
  // Home advantage baked into expected score
  const eloDiff = (homeElo + HOME_ADVANTAGE_ELO) - awayElo;
  return 1 / (1 + Math.pow(10, -eloDiff / 400));
}

export function getEloDiff(homeAbbr: string, awayAbbr: string): number {
  return getElo(homeAbbr) - getElo(awayAbbr);
}

// ─── Update Elo after a match ─────────────────────────────────────────────────

export function updateEloAfterMatch(
  homeAbbr: string,
  awayAbbr: string,
  homeScore: number,
  awayScore: number
): void {
  const homeElo = getElo(homeAbbr);
  const awayElo = getElo(awayAbbr);

  const homeExpected = eloWinProb(homeElo, awayElo);

  // Soccer outcome: 1 = home win, 0.5 = draw, 0 = away win
  const homeActual = homeScore > awayScore ? 1.0 : homeScore === awayScore ? 0.5 : 0.0;

  // Margin of victory multiplier: log(1 + min(margin, 4))
  const margin = Math.abs(homeScore - awayScore);
  const cappedMargin = Math.min(margin, GOAL_CAP);
  const movMultiplier = cappedMargin > 0 ? Math.log(1 + cappedMargin) : 1.0;

  const adjustedK = K_FACTOR * movMultiplier;

  const homeNewElo = homeElo + adjustedK * (homeActual - homeExpected);
  const awayNewElo = awayElo + adjustedK * ((1 - homeActual) - (1 - homeExpected));

  const now = new Date().toISOString();
  upsertElo({ teamAbbr: homeAbbr, rating: Math.round(homeNewElo), updatedAt: now });
  upsertElo({ teamAbbr: awayAbbr, rating: Math.round(awayNewElo), updatedAt: now });

  logger.debug(
    {
      home: homeAbbr, away: awayAbbr,
      score: `${homeScore}-${awayScore}`,
      homeElo: homeNewElo.toFixed(0),
      awayElo: awayNewElo.toFixed(0),
    },
    'Elo updated after MLS match'
  );
}

// ─── Offseason regression ─────────────────────────────────────────────────────
// Call once at the start of each new MLS season.
// Formula per spec: new_season_elo = 0.75 × prior_final + 0.25 × league_mean

export function applyOffseasonRegression(): void {
  const ratings = getAllElos();
  const now = new Date().toISOString();

  for (const r of ratings) {
    const newRating = 0.75 * r.rating + 0.25 * LEAGUE_MEAN_ELO;
    upsertElo({ teamAbbr: r.teamAbbr, rating: Math.round(newRating), updatedAt: now });
  }

  logger.info({ teams: ratings.length }, 'Offseason Elo regression applied (75% carry + 25% mean)');
}

// ─── Elo-based 3-way probabilities (rough approximation) ─────────────────────
// Convert binary Elo win probability into H/D/A using draw-rate adjustment.
// MLS draws ~25-28% of matches.

export function eloThreeWayProbs(
  homeAbbr: string,
  awayAbbr: string,
  leagueDrawRate = 0.27
): { home: number; draw: number; away: number } {
  const homeWinProb = eloWinProb(getElo(homeAbbr), getElo(awayAbbr));

  // Allocate draws proportionally from both win probabilities
  const drawShare = leagueDrawRate;
  const remainingHome = 1 - drawShare;

  // Map Elo win prob to H/A split within non-draw outcomes
  const home = homeWinProb * remainingHome;
  const away = (1 - homeWinProb) * remainingHome;
  const draw = drawShare;

  return { home, draw, away };
}
