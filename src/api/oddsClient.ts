// MLS Oracle v4.1 — Vegas Odds Client (3-way: Home / Draw / Away)
// Uses The Odds API (free tier: 500 requests/month)
// Falls back to manual vegas_lines.json if API not configured.

import { existsSync, readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MANUAL_LINES_PATH = resolve(__dirname, '../../data/vegas_lines.json');

// ─── Manual lines JSON format ─────────────────────────────────────────────────
// data/vegas_lines.json (optional, create manually):
// {
//   "2026-04-10": {
//     "LAFC@SEA": { "homeML": -130, "awayML": 310, "drawML": 240 },
//     ...
//   }
// }

interface ManualLines {
  [date: string]: {
    [matchupKey: string]: { homeML: number; awayML: number; drawML?: number };
  };
}

interface OddsApiOutcome {
  name: string;
  price: number;
}

interface OddsApiBookmaker {
  key: string;
  markets: Array<{ key: string; outcomes: OddsApiOutcome[] }>;
}

interface OddsApiGame {
  id: string;
  home_team: string;
  away_team: string;
  bookmakers: OddsApiBookmaker[];
}

// ─── Moneyline conversions ────────────────────────────────────────────────────

export function mlToImplied(ml: number): number {
  if (ml > 0) return 100 / (ml + 100);
  return Math.abs(ml) / (Math.abs(ml) + 100);
}

export interface ThreeWayImplied {
  homeProb: number;
  drawProb: number;
  awayProb: number;
  vig: number;
}

export function removeVig3Way(homeML: number, drawML: number, awayML: number): ThreeWayImplied {
  const rawHome = mlToImplied(homeML);
  const rawDraw = mlToImplied(drawML);
  const rawAway = mlToImplied(awayML);
  const total = rawHome + rawDraw + rawAway;
  const vig = total - 1.0;
  return {
    homeProb: rawHome / total,
    drawProb: rawDraw / total,
    awayProb: rawAway / total,
    vig,
  };
}

// ─── Persistent odds storage ──────────────────────────────────────────────────

export interface GameOdds {
  homeImpliedProb: number;
  drawImpliedProb: number;
  awayImpliedProb: number;
  homeML: number;
  drawML: number;
  awayML: number;
}

let _gameOddsMap: Map<string, GameOdds> | null = null;

export function getOddsForGame(matchupKey: string): GameOdds | null {
  return _gameOddsMap?.get(matchupKey) ?? null;
}

export function hasAnyOdds(): boolean {
  return (_gameOddsMap?.size ?? 0) > 0;
}

// ─── Load manual lines ────────────────────────────────────────────────────────

export function loadManualLines(date: string): Map<string, { homeML: number; drawML: number; awayML: number }> {
  if (!existsSync(MANUAL_LINES_PATH)) return new Map();

  try {
    const raw = readFileSync(MANUAL_LINES_PATH, 'utf-8');
    const lines = JSON.parse(raw) as ManualLines;
    const dayLines = lines[date];
    if (!dayLines) return new Map();

    const map = new Map<string, { homeML: number; drawML: number; awayML: number }>();
    for (const [key, val] of Object.entries(dayLines)) {
      map.set(key, { homeML: val.homeML, drawML: val.drawML ?? 280, awayML: val.awayML });
    }
    return map;
  } catch (err) {
    logger.warn({ err }, 'Failed to parse manual vegas_lines.json');
    return new Map();
  }
}

// ─── The Odds API ─────────────────────────────────────────────────────────────

// MLS team name → abbreviation (for Odds API matching)
const ODDS_NAME_TO_ABBR: Record<string, string> = {
  'Atlanta United': 'ATL',
  'Chicago Fire': 'CHI',
  'Columbus Crew': 'CLB',
  'Charlotte FC': 'CLT',
  'FC Cincinnati': 'CIN',
  'DC United': 'DC',
  'Inter Miami CF': 'MIA',
  'CF Montréal': 'MTL',
  'Montreal Impact': 'MTL',
  'New England Revolution': 'NE',
  'Nashville SC': 'NSH',
  'New York City FC': 'NYC',
  'New York Red Bulls': 'NYRB',
  'Orlando City': 'ORL',
  'Philadelphia Union': 'PHI',
  'Toronto FC': 'TOR',
  'Austin FC': 'ATX',
  'Colorado Rapids': 'COL',
  'FC Dallas': 'DAL',
  'Houston Dynamo': 'HOU',
  'Sporting Kansas City': 'KC',
  'LA Galaxy': 'LA',
  'LAFC': 'LAFC',
  'Minnesota United': 'MIN',
  'Portland Timbers': 'POR',
  'Real Salt Lake': 'RSL',
  'San Diego FC': 'SD',
  'Seattle Sounders FC': 'SEA',
  'San Jose Earthquakes': 'SJ',
  'St. Louis City SC': 'STL',
  'Vancouver Whitecaps': 'VAN',
};

function resolveTeamAbbrFromOdds(name: string): string | null {
  return ODDS_NAME_TO_ABBR[name] ?? null;
}

export async function loadOddsApiLines(
  _date: string
): Promise<Map<string, { homeML: number; drawML: number; awayML: number }>> {
  const apiKey = process.env.ODDS_API_KEY;
  if (!apiKey) {
    logger.debug('ODDS_API_KEY not set — skipping live odds fetch');
    return new Map();
  }

  const url = `https://api.the-odds-api.com/v4/sports/soccer_usa_mls/odds/?apiKey=${apiKey}&regions=us&markets=h2h&dateFormat=iso&oddsFormat=american`;

  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (!resp.ok) {
      logger.warn({ status: resp.status }, 'Odds API returned error');
      return new Map();
    }

    const games = (await resp.json()) as OddsApiGame[];
    const map = new Map<string, { homeML: number; drawML: number; awayML: number }>();

    for (const game of games) {
      const allHomeOdds: number[] = [];
      const allDrawOdds: number[] = [];
      const allAwayOdds: number[] = [];

      for (const book of game.bookmakers) {
        const h2h = book.markets.find(m => m.key === 'h2h');
        if (!h2h) continue;
        const homeOut = h2h.outcomes.find(o => o.name === game.home_team);
        const awayOut = h2h.outcomes.find(o => o.name === game.away_team);
        const drawOut = h2h.outcomes.find(o => o.name === 'Draw');
        if (homeOut) allHomeOdds.push(homeOut.price);
        if (awayOut) allAwayOdds.push(awayOut.price);
        if (drawOut) allDrawOdds.push(drawOut.price);
      }

      if (allHomeOdds.length === 0) continue;

      const avgHome = allHomeOdds.reduce((a, b) => a + b, 0) / allHomeOdds.length;
      const avgAway = allAwayOdds.reduce((a, b) => a + b, 0) / allAwayOdds.length;
      const avgDraw = allDrawOdds.length > 0
        ? allDrawOdds.reduce((a, b) => a + b, 0) / allDrawOdds.length
        : 280; // fallback typical draw odds

      const homeAbbr = resolveTeamAbbrFromOdds(game.home_team);
      const awayAbbr = resolveTeamAbbrFromOdds(game.away_team);
      if (homeAbbr && awayAbbr) {
        map.set(`${awayAbbr}@${homeAbbr}`, {
          homeML: Math.round(avgHome),
          drawML: Math.round(avgDraw),
          awayML: Math.round(avgAway),
        });
      }
    }

    logger.info({ games: map.size }, 'MLS Odds API lines loaded');
    return map;
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch Odds API lines');
    return new Map();
  }
}

// ─── Initialize odds for a date ───────────────────────────────────────────────

export async function initializeOdds(date: string): Promise<void> {
  _gameOddsMap = new Map();

  const manualLines = loadManualLines(date);
  if (manualLines.size > 0) {
    logger.info({ lines: manualLines.size }, 'Using manual Vegas lines');
    for (const [key, line] of manualLines.entries()) {
      const { homeProb, drawProb, awayProb } = removeVig3Way(line.homeML, line.drawML, line.awayML);
      _gameOddsMap.set(key, {
        homeImpliedProb: homeProb,
        drawImpliedProb: drawProb,
        awayImpliedProb: awayProb,
        homeML: line.homeML,
        drawML: line.drawML,
        awayML: line.awayML,
      });
    }
    return;
  }

  const apiLines = await loadOddsApiLines(date);
  for (const [key, line] of apiLines.entries()) {
    const { homeProb, drawProb, awayProb } = removeVig3Way(line.homeML, line.drawML, line.awayML);
    _gameOddsMap.set(key, {
      homeImpliedProb: homeProb,
      drawImpliedProb: drawProb,
      awayImpliedProb: awayProb,
      homeML: line.homeML,
      drawML: line.drawML,
      awayML: line.awayML,
    });
  }
}
