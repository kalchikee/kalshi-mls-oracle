// MLS Oracle v4.1 — CLI Entry Point
// Usage:
//   npm start                              → predictions for today
//   npm start -- --date 2026-04-10        → predictions for specific date
//   npm start -- --alert morning          → send matchday picks to Discord
//   npm start -- --alert recap            → send evening recap to Discord
//   npm start -- --help                   → show help

import 'dotenv/config';
import { logger } from './logger.js';
import { runPipeline } from './pipeline.js';
import { closeDb, initDb } from './db/database.js';
import type { PipelineOptions } from './types.js';

// ─── CLI argument parsing ─────────────────────────────────────────────────────

type AlertMode = 'morning' | 'recap' | null;

function parseArgs(): PipelineOptions & { help: boolean; alertMode: AlertMode } {
  const args = process.argv.slice(2);
  const opts: PipelineOptions & { help: boolean; alertMode: AlertMode } = {
    help: false,
    verbose: true,
    forceRefresh: false,
    alertMode: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--help':
      case '-h':
        opts.help = true;
        break;
      case '--date':
      case '-d':
        opts.date = args[++i];
        break;
      case '--force-refresh':
      case '-f':
        opts.forceRefresh = true;
        break;
      case '--quiet':
      case '-q':
        opts.verbose = false;
        break;
      case '--alert':
      case '-a': {
        const mode = args[++i];
        if (mode === 'morning' || mode === 'recap') {
          opts.alertMode = mode as AlertMode;
        } else {
          console.error(`Unknown alert mode: "${mode}". Use "morning" or "recap".`);
          process.exit(1);
        }
        break;
      }
      default:
        if (/^\d{4}-\d{2}-\d{2}$/.test(arg)) {
          opts.date = arg;
        }
    }
  }

  return opts;
}

function printHelp(): void {
  console.log(`
MLS Oracle v4.1 — Poisson ML Prediction Engine
===============================================

USAGE:
  npm start [options]
  node --loader ts-node/esm src/index.ts [options]

OPTIONS:
  --date, -d YYYY-MM-DD        Run predictions for a specific date (default: today)
  --force-refresh, -f          Bypass cache and re-fetch all data
  --quiet, -q                  Suppress prediction table output
  --alert, -a <morning|recap>  Send a Discord alert
  --help, -h                   Show this help message

EXAMPLES:
  npm start                              # Today's predictions
  npm start -- --date 2026-04-10        # Specific date
  npm run alerts:morning                 # Send matchday picks to Discord
  npm run alerts:recap                   # Send evening recap to Discord
  npm run scheduler                      # Start the long-running cron scheduler
  npm run train                          # Train the ML meta-model (Python)
  npm run backtest                       # Run walk-forward backtest (Python)

OUTPUT:
  Predictions stored in ./data/mls_oracle.db (SQLite)
  Cache files in ./cache/
  Logs in ./logs/

ENVIRONMENT (.env):
  DISCORD_WEBHOOK_URL    Discord webhook (required for alerts)
  ODDS_API_KEY           The Odds API key (optional — live 3-way Vegas lines)
  LOG_LEVEL              Logging level (default: info)

ARCHITECTURE:
  ESPN API → Feature Vector (40+ features) → Poisson Simulation (50k matches)
  → ML Multinomial LR (H/D/A) → Edge Detection → SQLite

MLS-SPECIFIC FEATURES:
  Altitude (COL/RSL), artificial turf (SEA/NE/VAN), travel distance (max ~2,600mi),
  cross-country flag, Designated Player impact, Leagues Cup congestion, expansion flags

TARGET ACCURACY:
  Match result: 50-54% | Double chance: 58-62% | High-conviction (60%+): 65-70%
`);
}

// ─── Alert handlers ───────────────────────────────────────────────────────────

async function runMorningAlert(date: string): Promise<void> {
  const { sendMorningBriefing } = await import('./alerts/discord.js');
  await initDb();
  const predictions = await runPipeline({ date, verbose: false });
  await sendMorningBriefing(date);

  if (predictions.length === 0) {
    console.log(`\nNo MLS matches on ${date} — no alert sent.\n`);
  }
}

async function runRecapAlert(date: string): Promise<void> {
  const { sendEveningRecap } = await import('./alerts/discord.js');
  const { processResults }   = await import('./alerts/results.js');

  await initDb();

  // If explicit --date was passed, use it directly.
  // Otherwise (scheduled midnight run), recap yesterday's matches.
  const explicitDate = process.argv.includes('--date') || process.argv.includes('-d');
  let recapDate = date;
  if (!explicitDate) {
    const d = new Date(date + 'T12:00:00Z');
    d.setUTCDate(d.getUTCDate() - 1);
    recapDate = d.toISOString().split('T')[0];
  }

  logger.info({ recapDate }, 'Running MLS evening recap');

  const { games, metrics } = await processResults(recapDate);
  await sendEveningRecap(recapDate, games, metrics);
}

// ─── Entry point ──────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  if (opts.date && !/^\d{4}-\d{2}-\d{2}$/.test(opts.date)) {
    logger.error({ date: opts.date }, 'Invalid date format. Use YYYY-MM-DD');
    process.exit(1);
  }

  const date = opts.date ?? new Date().toISOString().split('T')[0];

  logger.info(
    { date, version: '4.1.0', pid: process.pid, alertMode: opts.alertMode ?? 'pipeline' },
    'MLS Oracle starting'
  );

  try {
    if (opts.alertMode === 'morning') {
      await runMorningAlert(date);
      return;
    }

    if (opts.alertMode === 'recap') {
      await runRecapAlert(date);
      return;
    }

    // Force refresh: clear cache
    if (opts.forceRefresh) {
      logger.info('Force refresh: clearing cache');
      const { readdirSync, unlinkSync } = await import('fs');
      const cacheDir = process.env.CACHE_DIR ?? './cache';
      try {
        const files = readdirSync(cacheDir);
        for (const file of files) {
          if (file.endsWith('.json')) unlinkSync(`${cacheDir}/${file}`);
        }
        logger.info({ cleared: files.length }, 'Cache cleared');
      } catch {
        // Cache dir may not exist yet
      }
    }

    const predictions = await runPipeline(opts);

    if (predictions.length === 0) {
      console.log(`\nNo MLS matches scheduled for ${date}.\n`);
      process.exit(0);
    }

    logger.info({ predictions: predictions.length }, 'Pipeline completed successfully');

  } catch (err) {
    logger.error({ err }, 'Fatal error');
    process.exit(1);
  } finally {
    closeDb();
  }
}

process.on('unhandledRejection', (reason) => {
  logger.error({ reason }, 'Unhandled promise rejection');
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  logger.error({ err }, 'Uncaught exception');
  closeDb();
  process.exit(1);
});

main();
