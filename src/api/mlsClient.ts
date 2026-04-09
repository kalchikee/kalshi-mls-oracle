// MLS Oracle v4.1 — ESPN-based MLS API Client
// Uses ESPN's public soccer/usa.1 APIs (no key required).
// Fetches schedule, team stats, and injuries for MLS.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type { MLSMatch, MLSTeam, GameResult } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CACHE_DIR = process.env.CACHE_DIR ?? resolve(__dirname, '../../cache');
const CACHE_TTL_MS = (Number(process.env.CACHE_TTL_HOURS ?? 6)) * 60 * 60 * 1000;

mkdirSync(CACHE_DIR, { recursive: true });

const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1';

// ─── MLS team abbreviations ───────────────────────────────────────────────────
// All 30 MLS teams (2025 season — includes San Diego FC expansion)

export const ALL_MLS_ABBRS = [
  // Eastern Conference
  'ATL', 'CHI', 'CLB', 'CLT', 'CIN', 'DC', 'MIA', 'MTL', 'NE', 'NSH',
  'NYC', 'NYRB', 'ORL', 'PHI', 'TOR',
  // Western Conference
  'ATX', 'COL', 'DAL', 'HOU', 'KC', 'LA', 'LAFC', 'MIN', 'POR', 'RSL',
  'SD', 'SEA', 'SJ', 'STL', 'VAN',
];

export const CONFERENCE: Record<string, 'East' | 'West'> = {
  ATL: 'East', CHI: 'East', CLB: 'East', CLT: 'East', CIN: 'East',
  DC: 'East', MIA: 'East', MTL: 'East', NE: 'East', NSH: 'East',
  NYC: 'East', NYRB: 'East', ORL: 'East', PHI: 'East', TOR: 'East',
  ATX: 'West', COL: 'West', DAL: 'West', HOU: 'West', KC: 'West',
  LA: 'West', LAFC: 'West', MIN: 'West', POR: 'West', RSL: 'West',
  SD: 'West', SEA: 'West', SJ: 'West', STL: 'West', VAN: 'West',
};

// ESPN display name → our abbreviation
const ESPN_NAME_TO_ABBR: Record<string, string> = {
  'Atlanta United FC': 'ATL',
  'Chicago Fire FC': 'CHI',
  'Columbus Crew': 'CLB',
  'Charlotte FC': 'CLT',
  'FC Cincinnati': 'CIN',
  'D.C. United': 'DC',
  'Inter Miami CF': 'MIA',
  'CF Montréal': 'MTL',
  'CF Montreal': 'MTL',
  'New England Revolution': 'NE',
  'Nashville SC': 'NSH',
  'New York City FC': 'NYC',
  'New York Red Bulls': 'NYRB',
  'Orlando City SC': 'ORL',
  'Philadelphia Union': 'PHI',
  'Toronto FC': 'TOR',
  'Austin FC': 'ATX',
  'Colorado Rapids': 'COL',
  'FC Dallas': 'DAL',
  'Houston Dynamo FC': 'HOU',
  'Sporting Kansas City': 'KC',
  'LA Galaxy': 'LA',
  'Los Angeles FC': 'LAFC',
  'Minnesota United FC': 'MIN',
  'Portland Timbers': 'POR',
  'Real Salt Lake': 'RSL',
  'San Diego FC': 'SD',
  'Seattle Sounders FC': 'SEA',
  'San Jose Earthquakes': 'SJ',
  'St. Louis City SC': 'STL',
  'Vancouver Whitecaps FC': 'VAN',
};

// ESPN abbreviation normalization (ESPN sometimes uses abbreviated forms)
const ESPN_ABBR_FIX: Record<string, string> = {
  'ATL': 'ATL', 'CHI': 'CHI', 'CLB': 'CLB', 'CLT': 'CLT', 'CIN': 'CIN',
  'DC': 'DC', 'MIA': 'MIA', 'MTL': 'MTL', 'NE': 'NE', 'NSH': 'NSH',
  'NYC': 'NYC', 'NY': 'NYRB', 'NYRB': 'NYRB', 'ORL': 'ORL', 'PHI': 'PHI',
  'TOR': 'TOR', 'ATX': 'ATX', 'COL': 'COL', 'DAL': 'DAL', 'HOU': 'HOU',
  'SKC': 'KC', 'KC': 'KC', 'LA': 'LA', 'LAFC': 'LAFC', 'MIN': 'MIN',
  'POR': 'POR', 'RSL': 'RSL', 'SD': 'SD', 'SEA': 'SEA', 'SJ': 'SJ',
  'STL': 'STL', 'VAN': 'VAN',
};

function normalizeAbbr(raw: string): string {
  return ESPN_ABBR_FIX[raw] ?? raw;
}

function resolveTeamAbbr(displayName: string, abbreviation: string): string {
  if (ESPN_NAME_TO_ABBR[displayName]) return ESPN_NAME_TO_ABBR[displayName];
  return normalizeAbbr(abbreviation);
}

// ─── Altitude and turf flags ──────────────────────────────────────────────────

// Altitude venues: home team abbreviation → altitude in feet
export const ALTITUDE_VENUES: Record<string, number> = {
  COL: 5282,  // Dick's Sporting Goods Park (Commerce City, CO)
  RSL: 4327,  // America First Field (Sandy, UT)
};

// Artificial turf venues
export const TURF_VENUES = new Set(['SEA', 'NE', 'VAN']);

// Expansion teams (Year 1 or Year 2 — limited historical data)
export const EXPANSION_TEAMS = new Set(['SD']); // San Diego FC (2025)

// ─── City coordinates for travel distance ────────────────────────────────────
// [latitude, longitude]

export const CITY_COORDS: Record<string, [number, number]> = {
  ATL:  [33.7553, -84.4006],   // Atlanta
  ATX:  [30.2268, -97.7489],   // Austin
  CHI:  [41.8623, -87.6167],   // Chicago
  CLB:  [39.9670, -82.9929],   // Columbus
  CLT:  [35.2277, -80.8529],   // Charlotte
  CIN:  [39.1034, -84.5120],   // Cincinnati
  COL:  [39.8051, -104.8820],  // Commerce City (Denver area)
  DAL:  [32.8869, -97.0541],   // Frisco (Dallas area)
  DC:   [38.8722, -77.0127],   // Washington DC
  HOU:  [29.7563, -95.4100],   // Houston
  KC:   [39.1237, -94.8314],   // Kansas City
  LA:   [33.8644, -118.2611],  // Los Angeles (Galaxy)
  LAFC: [34.0139, -118.2857],  // Los Angeles (LAFC)
  MIA:  [25.8000, -80.1870],   // Fort Lauderdale
  MIN:  [44.9537, -93.1638],   // Saint Paul
  MTL:  [45.5630, -73.5507],   // Montreal
  NE:   [42.0909, -71.2643],   // Foxborough
  NSH:  [36.1308, -86.7667],   // Nashville
  NYC:  [40.7505, -73.9934],   // New York City
  NYRB: [40.7367, -74.1502],   // Harrison NJ
  ORL:  [28.5416, -81.3888],   // Orlando
  PHI:  [40.0068, -75.1638],   // Chester PA
  POR:  [45.5212, -122.6914],  // Portland
  RSL:  [40.5831, -111.8948],  // Sandy UT
  SD:   [32.7157, -117.1611],  // San Diego
  SEA:  [47.5952, -122.3316],  // Seattle
  SJ:   [37.3519, -121.9258],  // San Jose
  STL:  [38.6314, -90.2078],   // St. Louis
  TOR:  [43.6333, -79.4167],   // Toronto
  VAN:  [49.2767, -123.1125],  // Vancouver
};

// Great circle distance in miles
export function travelDistance(fromAbbr: string, toAbbr: string): number {
  const from = CITY_COORDS[fromAbbr];
  const to = CITY_COORDS[toAbbr];
  if (!from || !to) return 0;

  const R = 3958.8; // Earth radius in miles
  const dLat = ((to[0] - from[0]) * Math.PI) / 180;
  const dLon = ((to[1] - from[1]) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((from[0] * Math.PI) / 180) *
    Math.cos((to[0] * Math.PI) / 180) *
    Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function isHighAltitude(homeAbbr: string): boolean {
  return homeAbbr in ALTITUDE_VENUES;
}

export function isArtificialTurf(homeAbbr: string): boolean {
  return TURF_VENUES.has(homeAbbr);
}

export function isExpansionTeam(abbr: string): boolean {
  return EXPANSION_TEAMS.has(abbr);
}

// ─── Cache helpers ────────────────────────────────────────────────────────────

function cacheKey(url: string): string {
  return url.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 200) + '.json';
}

function readCache<T>(key: string): T | null {
  const path = resolve(CACHE_DIR, key);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try { return JSON.parse(readFileSync(path, 'utf-8')) as T; }
  catch { return null; }
}

function writeCache(key: string, data: unknown): void {
  try { writeFileSync(resolve(CACHE_DIR, key), JSON.stringify(data), 'utf-8'); }
  catch (err) { logger.warn({ err }, 'Failed to write cache'); }
}

// ─── Fetch with retry ─────────────────────────────────────────────────────────

async function fetchWithRetry<T>(url: string, attempts = 3): Promise<T> {
  const key = cacheKey(url);
  const cached = readCache<T>(key);
  if (cached !== null) { logger.debug({ url }, 'Cache HIT'); return cached; }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      const resp = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
        signal: AbortSignal.timeout(15000),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = (await resp.json()) as T;
      writeCache(key, data);
      return data;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < attempts - 1) await new Promise(r => setTimeout(r, (attempt + 1) * 2000));
    }
  }
  throw lastError ?? new Error(`Failed: ${url}`);
}

// ─── ESPN scoreboard types ────────────────────────────────────────────────────

interface ESPNCompetitor {
  homeAway: 'home' | 'away';
  score?: string;
  team: { id: string; abbreviation: string; displayName: string };
}

interface ESPNEvent {
  id: string;
  date: string;
  status: { type: { name: string; description: string } };
  season?: { slug?: string };
  week?: { number?: number };
  competitions: Array<{
    competitors: ESPNCompetitor[];
    venue?: { fullName?: string; address?: { city?: string } };
  }>;
}

interface ESPNScoreboardResp { events?: ESPNEvent[] }

// ─── Schedule ─────────────────────────────────────────────────────────────────

export async function fetchSchedule(date: string): Promise<MLSMatch[]> {
  const dateStr = date.replace(/-/g, '');
  const url = `${ESPN_BASE}/scoreboard?dates=${dateStr}&limit=30`;

  let data: ESPNScoreboardResp;
  try {
    data = await fetchWithRetry<ESPNScoreboardResp>(url);
  } catch (err) {
    logger.warn({ err, date }, 'Failed to fetch MLS schedule — returning empty');
    return [];
  }

  const events = data.events ?? [];
  if (events.length === 0) {
    logger.info({ date }, 'No MLS matches on ESPN scoreboard');
    return [];
  }

  const matches: MLSMatch[] = [];

  for (const event of events) {
    const comp = event.competitions[0];
    if (!comp) continue;

    const home = comp.competitors.find(c => c.homeAway === 'home');
    const away = comp.competitors.find(c => c.homeAway === 'away');
    if (!home || !away) continue;

    const homeAbbr = resolveTeamAbbr(home.team.displayName, home.team.abbreviation);
    const awayAbbr = resolveTeamAbbr(away.team.displayName, away.team.abbreviation);

    // Skip non-MLS teams (e.g. Leagues Cup Liga MX opponents)
    if (!ALL_MLS_ABBRS.includes(homeAbbr) && !ALL_MLS_ABBRS.includes(awayAbbr)) {
      logger.debug({ homeAbbr, awayAbbr }, 'Skipping non-MLS match');
      continue;
    }

    const statusDesc = event.status.type.description;
    const matchweek = event.week?.number ?? 0;

    matches.push({
      matchId:   event.id,
      matchDate: date,
      matchTime: event.date,
      status:    statusDesc,
      matchweek,
      homeTeam: {
        teamId:   Number(home.team.id),
        teamAbbr: homeAbbr,
        teamName: home.team.displayName,
        score:    home.score !== undefined ? Number(home.score) : undefined,
      },
      awayTeam: {
        teamId:   Number(away.team.id),
        teamAbbr: awayAbbr,
        teamName: away.team.displayName,
        score:    away.score !== undefined ? Number(away.score) : undefined,
      },
      venue:     comp.venue?.fullName ?? '',
      venueCity: comp.venue?.address?.city ?? '',
    });
  }

  logger.info({ date, matches: matches.length }, 'MLS schedule fetched (ESPN)');
  return matches;
}

// ─── Team stats ───────────────────────────────────────────────────────────────

interface ESPNSoccerStat {
  name: string;
  value: number | string;
}

interface ESPNSoccerStatGroup {
  name: string;
  stats: ESPNSoccerStat[];
}

interface ESPNTeamStatsResp {
  results?: { stats?: { categories?: ESPNSoccerStatGroup[] } };
  team?: {
    record?: { items?: Array<{ summary?: string; stats?: ESPNSoccerStat[] }> };
  };
}

function getStatValue(groups: ESPNSoccerStatGroup[], statName: string): number {
  for (const grp of groups) {
    const s = grp.stats.find(s => s.name === statName);
    if (s !== undefined) return Number(s.value) || 0;
  }
  return 0;
}

let _teamStatsCache: Map<string, MLSTeam> | null = null;
let _teamStatsCacheTime = 0;

async function fetchTeamById(espnId: string, abbr: string): Promise<{ abbr: string; stats: ESPNSoccerStatGroup[]; record: string } | null> {
  const url = `${ESPN_BASE}/teams/${espnId}/statistics`;
  try {
    const data = await fetchWithRetry<ESPNTeamStatsResp>(url);
    const cats = data.results?.stats?.categories ?? [];
    const recordItems = data.team?.record?.items ?? [];
    const overall = recordItems.find(r => r.summary !== undefined)?.summary ?? '0-0-0';
    return { abbr, stats: cats, record: overall };
  } catch {
    return null;
  }
}

// Discover ESPN team IDs from the teams endpoint
async function fetchTeamIds(): Promise<Map<string, string>> {
  const url = `${ESPN_BASE}/teams?limit=50`;
  const map = new Map<string, string>();

  try {
    const data = await fetchWithRetry<{ sports?: Array<{ leagues?: Array<{ teams?: Array<{ team: { id: string; abbreviation: string; displayName: string } }> }> }> }>(url);
    const teams = data.sports?.[0]?.leagues?.[0]?.teams ?? [];
    for (const { team } of teams) {
      const abbr = resolveTeamAbbr(team.displayName, team.abbreviation);
      map.set(abbr, team.id);
    }
    logger.info({ teams: map.size }, 'MLS team IDs fetched from ESPN');
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch MLS team IDs — using fallback');
  }

  return map;
}

// Compute xG proxy from shots (ESPN MLS often lacks direct xG)
function computeXG(shotsOnTarget: number, shotsOffTarget: number): number {
  // On-target: ~0.30 xG average; Off-target: ~0.05 xG
  return shotsOnTarget * 0.30 + shotsOffTarget * 0.05;
}

// Parse W-D-L record string
function parseRecord(record: string): { w: number; d: number; l: number } {
  const parts = record.split('-').map(Number);
  if (parts.length === 3) return { w: parts[0], d: parts[1], l: parts[2] };
  if (parts.length === 2) return { w: parts[0], d: 0, l: parts[1] };
  return { w: 0, d: 0, l: 0 };
}

export async function fetchAllTeamStats(): Promise<Map<string, MLSTeam>> {
  const now = Date.now();
  if (_teamStatsCache && now - _teamStatsCacheTime < CACHE_TTL_MS) return _teamStatsCache;

  // 1. Discover ESPN team IDs
  const teamIds = await fetchTeamIds();

  // 2. Fetch stats for all teams in parallel
  const entries = Array.from(teamIds.entries());
  const results = await Promise.allSettled(
    entries.map(([abbr, espnId]) => fetchTeamById(espnId, abbr))
  );

  const teamMap = new Map<string, MLSTeam>();
  const LEAGUE_AVG_GOALS = 1.45; // per game (for attack/defense strength normalization)

  for (const result of results) {
    if (result.status !== 'fulfilled' || !result.value) continue;
    const { abbr, stats, record } = result.value;
    if (!ALL_MLS_ABBRS.includes(abbr)) continue;

    const { w, d, l } = parseRecord(record);
    const gp = w + d + l;

    // Extract stats from ESPN categories
    const goalsFor     = getStatValue(stats, 'goalsFor')       || getStatValue(stats, 'goals') || 0;
    const goalsAgainst = getStatValue(stats, 'goalsAgainst')   || 0;
    const shotsFor     = getStatValue(stats, 'shots')          || getStatValue(stats, 'shotsTotal') || 0;
    const shotsOT      = getStatValue(stats, 'shotsOnTarget')  || 0;
    const shotsOTAgainst = getStatValue(stats, 'shotsOnTargetAgainst') || 0;
    const shotsAgainst = getStatValue(stats, 'shotsAgainst')   || 0;
    const possession   = getStatValue(stats, 'possessionPct')  || getStatValue(stats, 'possession') || 50;
    const passAcc      = getStatValue(stats, 'passingAccuracy')|| getStatValue(stats, 'passAccuracy') || 75;

    // Per-game rates
    const gpSafe = Math.max(1, gp);
    const gfPg   = goalsFor     / gpSafe;
    const gaPg   = goalsAgainst / gpSafe;
    const shotsFPg  = shotsFor  / gpSafe;
    const shotsOTPg = shotsOT   / gpSafe;
    const shotsAPg  = shotsAgainst / gpSafe;
    const shotsOTAPg = shotsOTAgainst / gpSafe;

    // xG proxies
    const xgFor     = computeXG(shotsOTPg, shotsFPg - shotsOTPg);
    const xgAgainst = computeXG(shotsOTAPg, shotsAPg - shotsOTAPg);

    // Clamp to reasonable range
    const xgForClamped     = Math.max(0.3, Math.min(3.5, xgFor));
    const xgAgainstClamped = Math.max(0.3, Math.min(3.5, xgAgainst));

    // Expected points from xG (rough: win when xGF > xGA, probability via Poisson)
    const xgDiff = xgForClamped - xgAgainstClamped;
    const xPts   = 1.0 + xgDiff * 0.8; // approx expected pts per game

    // Points per game (W=3, D=1)
    const pts  = 3 * w + d;
    const ppg  = pts / gpSafe;

    // Draw tendency
    const drawRate = d / gpSafe;

    // Overperformance (goals vs xG)
    const overperformance = gfPg - xgForClamped;

    // DP impact proxy: teams with high goals_for relative to shots = efficient DPs
    const dpImpact = Math.max(0, (gfPg - LEAGUE_AVG_GOALS) * 1.5);

    teamMap.set(abbr, {
      teamId:        Number(teamIds.get(abbr) ?? 0),
      teamAbbr:      abbr,
      teamName:      abbr,
      conference:    CONFERENCE[abbr] ?? 'East',
      w, d, l, gp, pts, winPct: gp > 0 ? (w + 0.5 * d) / gp : 0.5,
      goalsFor:      gfPg,
      goalsAgainst:  gaPg,
      goalDiff:      gfPg - gaPg,
      xgFor:         xgForClamped,
      xgAgainst:     xgAgainstClamped,
      xgDiff:        xgForClamped - xgAgainstClamped,
      shotsFor:      shotsFPg,
      shotsOnTargetFor: shotsOTPg,
      shotsAgainst:  shotsAPg,
      shotsOnTargetAgainst: shotsOTAPg,
      possession:    possession > 1 ? possession : possession * 100, // normalize to %
      passAccuracy:  passAcc > 1 ? passAcc : passAcc * 100,
      xPts:          Math.max(0.2, Math.min(3.0, xPts)),
      dpImpact,
      dpAvailable:   3,           // default: all DPs available; updated by injury feed
      form5xPts:     xPts,        // ESPN doesn't have rolling stats — use season avg
      overperformance,
      drawRate,
      managerTenure: 24,          // default 2 years; not available from ESPN
    });
  }

  // Fallback: seed missing teams with league-average defaults
  for (const abbr of ALL_MLS_ABBRS) {
    if (!teamMap.has(abbr)) {
      teamMap.set(abbr, defaultTeam(abbr));
    }
  }

  if (teamMap.size > 0) {
    _teamStatsCache = teamMap;
    _teamStatsCacheTime = now;
    logger.info({ teams: teamMap.size }, 'MLS team stats loaded (ESPN)');
  }

  return teamMap;
}

function defaultTeam(abbr: string): MLSTeam {
  const isExpansion = isExpansionTeam(abbr);
  return {
    teamId: 0, teamAbbr: abbr, teamName: abbr,
    conference: CONFERENCE[abbr] ?? 'East',
    w: 0, d: 0, l: 0, gp: 0, pts: 0, winPct: isExpansion ? 0.35 : 0.5,
    goalsFor: 1.45, goalsAgainst: 1.45, goalDiff: 0,
    xgFor: 1.35, xgAgainst: 1.35, xgDiff: 0,
    shotsFor: 12, shotsOnTargetFor: 4.5,
    shotsAgainst: 12, shotsOnTargetAgainst: 4.5,
    possession: 50, passAccuracy: 75,
    xPts: isExpansion ? 0.8 : 1.0,
    dpImpact: 0, dpAvailable: 3,
    form5xPts: isExpansion ? 0.7 : 1.0,
    overperformance: 0, drawRate: 0.27,
    managerTenure: isExpansion ? 6 : 24,
  };
}

// ─── Last game date (for rest days) ──────────────────────────────────────────

export async function fetchTeamLastGameDate(teamAbbr: string, beforeDate: string): Promise<string | null> {
  // Fetch last 30 days of matches for this team via ESPN
  const today = new Date(beforeDate);
  const startDate = new Date(today);
  startDate.setDate(startDate.getDate() - 30);
  const startStr = startDate.toISOString().split('T')[0].replace(/-/g, '');
  const endStr   = beforeDate.replace(/-/g, '');

  const url = `${ESPN_BASE}/scoreboard?dates=${startStr}-${endStr}&limit=100`;

  try {
    const data = await fetchWithRetry<ESPNScoreboardResp>(url);
    const events = (data.events ?? []).filter(e =>
      e.status.type.name === 'STATUS_FINAL' &&
      e.competitions[0]?.competitors?.some(c => {
        const abbr = resolveTeamAbbr(c.team.displayName, c.team.abbreviation);
        return abbr === teamAbbr;
      })
    );

    let latestDate: string | null = null;
    for (const event of events) {
      const d = event.date.split('T')[0];
      if (d < beforeDate && (!latestDate || d > latestDate)) latestDate = d;
    }
    return latestDate;
  } catch {
    return null;
  }
}

// ─── Completed results ────────────────────────────────────────────────────────

export async function fetchCompletedResults(date: string): Promise<GameResult[]> {
  const dateStr = date.replace(/-/g, '');
  const url = `${ESPN_BASE}/scoreboard?dates=${dateStr}&limit=30`;

  try {
    const data = await fetchWithRetry<ESPNScoreboardResp>(url);
    const events = data.events ?? [];
    const results: GameResult[] = [];

    for (const event of events) {
      if (event.status.type.name !== 'STATUS_FINAL') continue;

      const comp = event.competitions[0];
      if (!comp) continue;

      const home = comp.competitors.find(c => c.homeAway === 'home');
      const away = comp.competitors.find(c => c.homeAway === 'away');
      if (!home || !away) continue;

      const homeAbbr = resolveTeamAbbr(home.team.displayName, home.team.abbreviation);
      const awayAbbr = resolveTeamAbbr(away.team.displayName, away.team.abbreviation);
      const homeScore = Number(home.score ?? 0);
      const awayScore = Number(away.score ?? 0);

      const outcome: 'H' | 'D' | 'A' =
        homeScore > awayScore ? 'H' :
        homeScore < awayScore ? 'A' : 'D';

      results.push({
        game_id:    event.id,
        date,
        home_team:  homeAbbr,
        away_team:  awayAbbr,
        home_score: homeScore,
        away_score: awayScore,
        outcome,
        venue:      comp.venue?.fullName ?? '',
      });
    }

    logger.info({ date, results: results.length }, 'Completed MLS results fetched (ESPN)');
    return results;
  } catch (err) {
    logger.warn({ err, date }, 'Failed to fetch completed MLS results');
    return [];
  }
}
