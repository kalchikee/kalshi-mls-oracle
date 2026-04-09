// MLS Oracle v4.1 — SQLite Database Layer (sql.js — pure JS, no native build)

import initSqlJs, { type Database as SqlJsDatabase } from 'sql.js';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { Prediction, GameResult, AccuracyLog, EloRating } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DB_PATH = resolve(
  process.env.DB_PATH
    ? process.env.DB_PATH.startsWith('.')
      ? resolve(__dirname, '../../', process.env.DB_PATH)
      : process.env.DB_PATH
    : resolve(__dirname, '../../data/mls_oracle.db')
);

mkdirSync(dirname(DB_PATH), { recursive: true });

let _db: SqlJsDatabase | null = null;
let _SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

// ─── Initialization ───────────────────────────────────────────────────────────

export async function initDb(): Promise<SqlJsDatabase> {
  if (_db) return _db;

  _SQL = await initSqlJs();

  if (existsSync(DB_PATH)) {
    const fileBuffer = readFileSync(DB_PATH);
    _db = new _SQL.Database(fileBuffer);
  } else {
    _db = new _SQL.Database();
  }

  initializeSchema(_db);
  persistDb();
  return _db;
}

export function getDb(): SqlJsDatabase {
  if (!_db) throw new Error('Database not initialized. Call initDb() first.');
  return _db;
}

export function persistDb(): void {
  if (!_db) return;
  const data = _db.export();
  writeFileSync(DB_PATH, Buffer.from(data));
}

function run(sql: string, params: (string | number | null | undefined)[] = []): void {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.run(params.map(p => p === undefined ? null : p));
  stmt.free();
  persistDb();
}

function queryAll<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T[] {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results: T[] = [];
  while (stmt.step()) {
    results.push(stmt.getAsObject() as T);
  }
  stmt.free();
  return results;
}

function queryOne<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T | undefined {
  const results = queryAll<T>(sql, params);
  return results[0];
}

// ─── Schema ───────────────────────────────────────────────────────────────────

function initializeSchema(db: SqlJsDatabase): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS elo_ratings (
      team_abbr TEXT PRIMARY KEY,
      rating REAL NOT NULL DEFAULT 1500,
      games_played INTEGER NOT NULL DEFAULT 0,
      season TEXT NOT NULL DEFAULT '2025',
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      game_id TEXT NOT NULL,
      game_date TEXT NOT NULL,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      venue TEXT NOT NULL DEFAULT '',
      matchweek INTEGER NOT NULL DEFAULT 0,
      feature_vector TEXT NOT NULL,

      -- Poisson raw probabilities
      poisson_home REAL NOT NULL DEFAULT 0,
      poisson_draw REAL NOT NULL DEFAULT 0,
      poisson_away REAL NOT NULL DEFAULT 0,

      -- Calibrated probabilities
      home_prob REAL NOT NULL,
      draw_prob REAL NOT NULL,
      away_prob REAL NOT NULL,

      -- Pick
      pick TEXT NOT NULL DEFAULT 'H',
      pick_confidence REAL NOT NULL DEFAULT 0,

      -- Expected goals
      home_xg REAL NOT NULL DEFAULT 0,
      away_xg REAL NOT NULL DEFAULT 0,
      most_likely_score TEXT NOT NULL DEFAULT '',
      btts_prob REAL NOT NULL DEFAULT 0,
      over25_prob REAL NOT NULL DEFAULT 0,

      -- Vegas
      vegas_home_prob REAL,
      vegas_draw_prob REAL,
      vegas_away_prob REAL,
      edge_home REAL,
      edge_draw REAL,
      edge_away REAL,

      -- Labels
      early_season INTEGER NOT NULL DEFAULT 0,
      matchweek_label TEXT NOT NULL DEFAULT '',
      model_version TEXT NOT NULL DEFAULT '4.1.0',

      -- Results
      actual_outcome TEXT,
      correct INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS accuracy_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      date TEXT NOT NULL UNIQUE,
      brier_3way REAL NOT NULL DEFAULT 0,
      accuracy REAL NOT NULL DEFAULT 0,
      draw_accuracy REAL NOT NULL DEFAULT 0,
      double_chance_accuracy REAL NOT NULL DEFAULT 0,
      high_conv_accuracy REAL NOT NULL DEFAULT 0,
      games_evaluated INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS game_results (
      game_id TEXT PRIMARY KEY,
      date TEXT NOT NULL,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      home_score INTEGER NOT NULL,
      away_score INTEGER NOT NULL,
      outcome TEXT NOT NULL DEFAULT 'H',
      venue TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS model_registry (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      version TEXT NOT NULL UNIQUE,
      model_type TEXT NOT NULL DEFAULT 'poisson',
      train_dates TEXT NOT NULL DEFAULT '',
      test_brier REAL NOT NULL DEFAULT 0,
      test_accuracy REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);

  const cnt = queryOne<{ cnt: number }>('SELECT COUNT(*) as cnt FROM model_registry');
  if (!cnt || cnt.cnt === 0) {
    db.run(
      `INSERT OR IGNORE INTO model_registry (version, model_type, train_dates, test_brier, test_accuracy)
       VALUES (?, ?, ?, ?, ?)`,
      ['4.1.0', 'poisson-only', '2019-01-01/2024-12-31', 0, 0]
    );
  }
}

// ─── Elo helpers ──────────────────────────────────────────────────────────────

export function upsertElo(rating: EloRating): void {
  run(
    `INSERT INTO elo_ratings (team_abbr, rating, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT(team_abbr) DO UPDATE SET
       rating = excluded.rating,
       updated_at = excluded.updated_at`,
    [rating.teamAbbr, rating.rating, rating.updatedAt]
  );
}

export function getElo(teamAbbr: string): number {
  const row = queryOne<{ rating: number }>(
    'SELECT rating FROM elo_ratings WHERE team_abbr = ?',
    [teamAbbr]
  );
  return row?.rating ?? 1500;
}

export function getAllElos(): EloRating[] {
  return queryAll<{ team_abbr: string; rating: number; updated_at: string }>(
    'SELECT team_abbr, rating, updated_at FROM elo_ratings ORDER BY rating DESC'
  ).map(r => ({ teamAbbr: r.team_abbr, rating: r.rating, updatedAt: r.updated_at }));
}

// ─── Prediction helpers ───────────────────────────────────────────────────────

export function upsertPrediction(pred: Prediction): void {
  run(`DELETE FROM predictions WHERE game_id = ? AND model_version = ?`, [pred.game_id, pred.model_version]);
  run(
    `INSERT INTO predictions (
       game_id, game_date, home_team, away_team, venue, matchweek,
       feature_vector,
       poisson_home, poisson_draw, poisson_away,
       home_prob, draw_prob, away_prob,
       pick, pick_confidence,
       home_xg, away_xg, most_likely_score, btts_prob, over25_prob,
       vegas_home_prob, vegas_draw_prob, vegas_away_prob,
       edge_home, edge_draw, edge_away,
       early_season, matchweek_label, model_version, created_at
     ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
    [
      pred.game_id, pred.game_date, pred.home_team, pred.away_team, pred.venue, pred.matchweek,
      JSON.stringify(pred.feature_vector),
      pred.poisson_home, pred.poisson_draw, pred.poisson_away,
      pred.home_prob, pred.draw_prob, pred.away_prob,
      pred.pick, pred.pick_confidence,
      pred.home_xg, pred.away_xg, pred.most_likely_score, pred.btts_prob, pred.over25_prob,
      pred.vegas_home_prob ?? null, pred.vegas_draw_prob ?? null, pred.vegas_away_prob ?? null,
      pred.edge_home ?? null, pred.edge_draw ?? null, pred.edge_away ?? null,
      pred.early_season ? 1 : 0, pred.matchweek_label, pred.model_version, pred.created_at,
    ]
  );
}

export function getPredictionsByDate(date: string): Prediction[] {
  const rows = queryAll<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE game_date = ? ORDER BY pick_confidence DESC',
    [date]
  );
  return rows.map(row => ({
    ...row,
    feature_vector: JSON.parse(row.feature_vector as string),
    early_season: (row.early_season as number) === 1,
  })) as Prediction[];
}

export function updatePredictionResult(gameId: string, outcome: 'H' | 'D' | 'A', correct: boolean): void {
  run(
    `UPDATE predictions SET actual_outcome = ?, correct = ? WHERE game_id = ?`,
    [outcome, correct ? 1 : 0, gameId]
  );
}

// ─── Game result helpers ──────────────────────────────────────────────────────

export function upsertGameResult(result: GameResult): void {
  run(
    `INSERT INTO game_results (game_id, date, home_team, away_team, home_score, away_score, outcome, venue)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(game_id) DO UPDATE SET
       home_score = excluded.home_score,
       away_score = excluded.away_score,
       outcome = excluded.outcome`,
    [result.game_id, result.date, result.home_team, result.away_team,
     result.home_score, result.away_score, result.outcome, result.venue]
  );
}

// ─── Accuracy helpers ─────────────────────────────────────────────────────────

export function upsertAccuracyLog(log: AccuracyLog): void {
  run(
    `INSERT INTO accuracy_log (date, brier_3way, accuracy, draw_accuracy, double_chance_accuracy, high_conv_accuracy, games_evaluated)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(date) DO UPDATE SET
       brier_3way = excluded.brier_3way,
       accuracy = excluded.accuracy,
       draw_accuracy = excluded.draw_accuracy,
       double_chance_accuracy = excluded.double_chance_accuracy,
       high_conv_accuracy = excluded.high_conv_accuracy,
       games_evaluated = excluded.games_evaluated`,
    [log.date, log.brier_3way, log.accuracy, log.draw_accuracy,
     log.double_chance_accuracy, log.high_conv_accuracy, log.games_evaluated]
  );
}

export function getRecentAccuracy(days = 30): AccuracyLog[] {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - days);
  const dateStr = cutoff.toISOString().split('T')[0];
  return queryAll<AccuracyLog>(
    'SELECT * FROM accuracy_log WHERE date >= ? ORDER BY date DESC',
    [dateStr]
  );
}

export interface SeasonRecord {
  correct: number;
  total: number;
  draws_correct: number;
  high_conv_correct: number;
  high_conv_total: number;
}

function getCurrentSeasonStartDate(): string {
  // MLS season runs late February to early December
  const now = new Date();
  const year = now.getFullYear();
  // Season starts in Feb; if before Feb 15, use prior year
  return (now.getMonth() + 1) >= 2 ? `${year}-02-15` : `${year - 1}-02-15`;
}

export function getSeasonRecord(): SeasonRecord {
  const seasonStart = getCurrentSeasonStartDate();
  const all = queryOne<{ correct: number; total: number; draws_correct: number }>(
    `SELECT
       SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
       COUNT(*) as total,
       SUM(CASE WHEN correct = 1 AND actual_outcome = 'D' THEN 1 ELSE 0 END) as draws_correct
     FROM predictions WHERE correct IS NOT NULL AND game_date >= ?`,
    [seasonStart]
  );
  const hc = queryOne<{ correct: number; total: number }>(
    `SELECT
       SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
       COUNT(*) as total
     FROM predictions
     WHERE correct IS NOT NULL AND game_date >= ? AND pick_confidence >= 0.60`,
    [seasonStart]
  );
  return {
    correct:          all?.correct       ?? 0,
    total:            all?.total         ?? 0,
    draws_correct:    all?.draws_correct ?? 0,
    high_conv_correct: hc?.correct       ?? 0,
    high_conv_total:   hc?.total         ?? 0,
  };
}

export function closeDb(): void {
  if (_db) {
    persistDb();
    _db.close();
    _db = null;
  }
}
