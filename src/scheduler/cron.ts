// MLS Oracle v4.1 — Cron Scheduler
// 10 AM ET daily: matchday predictions (if match within 48hrs)
// Midnight ET daily: evening results recap
//
// MLS Season: late February → early December (MLS Cup)
// Leagues Cup disruption: ~August (Liga MX matches)
// Active window: Feb 20 – Dec 15
//
// Usage: npm run scheduler  (long-running process)

import 'dotenv/config';
import cron from 'node-cron';
import { logger } from '../logger.js';
import { runPipeline } from '../pipeline.js';
import { initDb, getPredictionsByDate, closeDb } from '../db/database.js';
import { sendMorningBriefing, sendSeasonAlert } from '../alerts/discord.js';

// ─── MLS season window ────────────────────────────────────────────────────────
// Active: Feb 20 – Dec 15 (regular season + playoffs + MLS Cup)
// Dormant: Dec 16 – Feb 19

const SEASON_START_MONTH = 2;   // February
const SEASON_START_DAY   = 20;
const SEASON_END_MONTH   = 12;  // December
const SEASON_END_DAY     = 15;

// Leagues Cup: August 1-31 (optional pause)
const LEAGUES_CUP_MONTH  = 8;

function isMLSSeason(date: Date = new Date()): boolean {
  const month = date.getMonth() + 1;
  const day   = date.getDate();

  if (month > SEASON_END_MONTH) return false;
  if (month < SEASON_START_MONTH) return false;
  if (month === SEASON_START_MONTH && day < SEASON_START_DAY) return false;
  if (month === SEASON_END_MONTH   && day > SEASON_END_DAY)   return false;
  return true;
}

function isLeaguesCupPeriod(date: Date = new Date()): boolean {
  return (date.getMonth() + 1) === LEAGUES_CUP_MONTH;
}

// ─── Transition state ─────────────────────────────────────────────────────────

let lastSeasonState: boolean | null = null;
let leaguesCupAlertSent = false;

function todayStr(): string {
  return new Date().toISOString().split('T')[0];
}

// ─── Morning routine (10 AM ET daily) ────────────────────────────────────────

async function runMorningRoutine(): Promise<void> {
  const date = todayStr();
  const now = new Date();

  // Handle season transitions
  const active = isMLSSeason(now);
  if (lastSeasonState !== null && lastSeasonState !== active) {
    await sendSeasonAlert(active ? 'start' : 'end');
  }
  lastSeasonState = active;

  if (!active) {
    logger.info({ date }, '[Scheduler] MLS off-season — skipping morning routine');
    return;
  }

  // Leagues Cup notification (once per August)
  if (isLeaguesCupPeriod(now) && !leaguesCupAlertSent) {
    await sendSeasonAlert('leagues_cup');
    leaguesCupAlertSent = true;
  }
  if (!isLeaguesCupPeriod(now)) leaguesCupAlertSent = false;

  logger.info({ date }, '[Scheduler] MLS morning routine starting');

  try {
    const predictions = await runPipeline({ date, verbose: false });

    if (predictions.length === 0) {
      logger.info({ date }, '[Scheduler] No MLS matches today — skipping Discord');
      return;
    }

    await sendMorningBriefing(date);
    logger.info({ date, matches: predictions.length }, '[Scheduler] Morning routine complete');
  } catch (err) {
    logger.error({ err, date }, '[Scheduler] Morning routine failed');
  }
}

// ─── Evening recap (midnight ET daily) ───────────────────────────────────────

async function runEveningRoutine(): Promise<void> {
  const date = todayStr();

  if (!isMLSSeason()) {
    logger.info({ date }, '[Scheduler] MLS off-season — skipping recap');
    return;
  }

  logger.info({ date }, '[Scheduler] MLS evening routine starting');

  try {
    const { sendEveningRecap } = await import('../alerts/discord.js');
    const { processResults }   = await import('../alerts/results.js');

    // Recap covers yesterday's matches (games played last evening)
    const yesterday = new Date(date + 'T12:00:00Z');
    yesterday.setUTCDate(yesterday.getUTCDate() - 1);
    const recapDate = yesterday.toISOString().split('T')[0];

    const { games, metrics } = await processResults(recapDate);

    if (games.length === 0) {
      logger.info({ recapDate }, '[Scheduler] No completed matches — skipping recap');
      return;
    }

    await sendEveningRecap(recapDate, games, metrics);
    logger.info({ recapDate, games: games.length }, '[Scheduler] Evening routine complete');
  } catch (err) {
    logger.error({ err, date }, '[Scheduler] Evening routine failed');
  }
}

// ─── Start scheduler ──────────────────────────────────────────────────────────

async function startScheduler(): Promise<void> {
  logger.info('[Scheduler] MLS Oracle v4.1 Scheduler starting...');

  await initDb();

  lastSeasonState = isMLSSeason();
  logger.info(
    { active: lastSeasonState },
    lastSeasonState
      ? '[Scheduler] MLS season is ACTIVE'
      : '[Scheduler] MLS OFF-SEASON — messages suppressed until Feb 20'
  );

  // 10 AM ET daily (= 14:00 UTC, adjusting for EDT = UTC-4)
  // Cron runs in UTC: 14:00 UTC = 10:00 AM EDT / 9:00 AM EST
  cron.schedule('0 14 * * *', () => {
    logger.info('[Scheduler] 10 AM ET fired (matchday check)');
    void runMorningRoutine();
  }, { timezone: 'America/New_York' });

  // Midnight ET daily (= 04:00 UTC next day)
  cron.schedule('0 4 * * *', () => {
    logger.info('[Scheduler] Midnight ET fired (evening recap)');
    void runEveningRoutine();
  }, { timezone: 'America/New_York' });

  logger.info('[Scheduler] Cron running: 10 AM morning + midnight recap (ET). Active Feb 20 – Dec 15.');

  // If already past 10 AM with no predictions and season is active, run immediately
  const now  = new Date();
  const hour = now.getHours();
  const date = todayStr();

  if (isMLSSeason(now) && hour >= 10 && getPredictionsByDate(date).length === 0) {
    logger.info('[Scheduler] Running morning routine now (missed window)');
    void runMorningRoutine();
  }
}

// ─── Graceful shutdown ────────────────────────────────────────────────────────

process.on('SIGINT', () => {
  logger.info('[Scheduler] SIGINT — shutting down');
  closeDb();
  process.exit(0);
});

process.on('SIGTERM', () => {
  logger.info('[Scheduler] SIGTERM — shutting down');
  closeDb();
  process.exit(0);
});

process.on('unhandledRejection', (reason) => {
  logger.error({ reason }, '[Scheduler] Unhandled rejection');
});

process.on('uncaughtException', (err) => {
  logger.error({ err }, '[Scheduler] Uncaught exception');
  closeDb();
  process.exit(1);
});

startScheduler();
