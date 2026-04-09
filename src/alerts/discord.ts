// MLS Oracle v4.1 — Discord Webhook Alert Module
// Color: Blue sidebar #0053A0 (MLS brand color)
// Matchday Picks embed: 3-way probs, altitude/turf flags, travel, DP status
// Evening Recap embed: results, accuracy tracking

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getPredictionsByDate, initDb, getRecentAccuracy, getSeasonRecord } from '../db/database.js';
import { getConfidenceTier, confidenceEmoji } from '../features/marketEdge.js';
import type { Prediction } from '../types.js';

// ─── MLS brand colors ─────────────────────────────────────────────────────────

const MLS_BLUE    = 0x0053A0; // MLS primary blue
const MLS_GREEN   = 0x27AE60; // good night
const MLS_RED     = 0xE74C3C; // bad night
const MLS_GRAY    = 0x95A5A6; // neutral
const MLS_AMBER   = 0xF39C12; // early season

// ─── Discord types ────────────────────────────────────────────────────────────

interface DiscordField {
  name: string;
  value: string;
  inline?: boolean;
}

interface DiscordEmbed {
  title?: string;
  description?: string;
  color?: number;
  fields?: DiscordField[];
  footer?: { text: string };
  timestamp?: string;
}

interface DiscordPayload {
  content?: string;
  embeds: DiscordEmbed[];
}

// ─── Webhook sender ───────────────────────────────────────────────────────────

async function sendWebhook(payload: DiscordPayload): Promise<boolean> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    logger.warn('DISCORD_WEBHOOK_URL not set — skipping Discord alert');
    return false;
  }

  try {
    const resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });

    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Discord webhook error');
      return false;
    }

    logger.info('Discord alert sent');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send Discord webhook');
    return false;
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pct(prob: number): string {
  return (prob * 100).toFixed(1) + '%';
}

function pickLabel(pred: Prediction): string {
  switch (pred.pick) {
    case 'H': return pred.home_team;
    case 'A': return pred.away_team;
    case 'D': return 'Draw';
  }
}

function venueFlags(pred: Prediction): string {
  const fv = pred.feature_vector;
  const flags: string[] = [];
  if (fv.altitude_flag === 1) flags.push('🏔️ Altitude');
  if (fv.turf_flag === 1)     flags.push('🟩 Turf');
  if (fv.cross_country_flag === 1) flags.push(`✈️ ${Math.round(fv.travel_distance_diff)}mi`);
  else if (fv.travel_distance_diff > 500) flags.push(`✈️ ${Math.round(fv.travel_distance_diff)}mi`);
  return flags.join(' · ') || '';
}

function earlySeasonBadge(pred: Prediction): string {
  if (!pred.matchweek_label) return '';
  if (pred.matchweek_label === 'EARLY SEASON') return ' 🟡 EARLY SEASON';
  if (pred.matchweek_label === 'Blending')     return ' 🟠 Blending';
  return '';
}

// ─── MORNING: Matchday Predictions ───────────────────────────────────────────

export async function sendMorningBriefing(date: string): Promise<boolean> {
  await initDb();
  const predictions = getPredictionsByDate(date);

  if (predictions.length === 0) {
    logger.warn({ date }, 'No predictions for morning briefing');
    return false;
  }

  // Sort by confidence descending
  const sorted = [...predictions].sort((a, b) => b.pick_confidence - a.pick_confidence);

  const highConv  = sorted.filter(p => p.pick_confidence >= 0.60);
  const recentAcc = getRecentAccuracy(7);
  const avgAcc    = recentAcc.length > 0
    ? recentAcc.reduce((s, a) => s + a.accuracy, 0) / recentAcc.length
    : null;
  const season = getSeasonRecord();

  // ── Embed 1: All Picks ────────────────────────────────────────────────────
  const picksFields: DiscordField[] = [];

  for (const pred of sorted) {
    const matchup = `${pred.away_team} @ ${pred.home_team}`;
    const conf    = confidenceEmoji(pred.pick_confidence);
    const flags   = venueFlags(pred);
    const esBadge = earlySeasonBadge(pred);

    // Build 3-way probability line
    const probLine = `🏠 ${pct(pred.home_prob)} · 🤝 ${pct(pred.draw_prob)} · ✈️ ${pct(pred.away_prob)}`;
    const scoreLine = `⚽ Proj: **${pred.most_likely_score}** · xG: ${pred.home_xg.toFixed(2)}-${pred.away_xg.toFixed(2)}`;
    const bttsLine  = `BTTS: ${pct(pred.btts_prob)} · O2.5: ${pct(pred.over25_prob)}`;

    let valueStr = [probLine, scoreLine, bttsLine].join('\n');
    if (flags) valueStr += `\n${flags}`;

    // Edge info if available
    if (pred.edge_home !== undefined && (
      Math.abs(pred.edge_home) >= 0.05 ||
      Math.abs(pred.edge_draw ?? 0) >= 0.05 ||
      Math.abs(pred.edge_away ?? 0) >= 0.05
    )) {
      const bestEdgeSide = [
        { side: 'H', edge: pred.edge_home ?? 0 },
        { side: 'D', edge: pred.edge_draw ?? 0 },
        { side: 'A', edge: pred.edge_away ?? 0 },
      ].reduce((a, b) => Math.abs(a.edge) > Math.abs(b.edge) ? a : b);
      const sign = bestEdgeSide.edge >= 0 ? '+' : '';
      valueStr += `\n💹 Edge: ${sign}${(bestEdgeSide.edge * 100).toFixed(1)}% (${bestEdgeSide.side})`;
    }

    picksFields.push({
      name: `${conf} **${matchup}** | Pick: **${pickLabel(pred)}** (${pct(pred.pick_confidence)})${esBadge}`,
      value: valueStr,
      inline: false,
    });
  }

  const seasonLine = season.total > 0
    ? `📈 Season: **${season.correct}-${season.total - season.correct}** (${((season.correct / season.total) * 100).toFixed(1)}%)` +
      (season.high_conv_total > 0
        ? `  ·  ⚡ High Conv: **${season.high_conv_correct}-${season.high_conv_total - season.high_conv_correct}** (${((season.high_conv_correct / season.high_conv_total) * 100).toFixed(1)}%)`
        : '')
    : '📈 Season: **0-0** (tracking starts tonight)';

  const descParts = [
    `${predictions.length} match${predictions.length !== 1 ? 'es' : ''} today`,
    avgAcc !== null ? `7-day accuracy: **${(avgAcc * 100).toFixed(1)}%**` : '',
    seasonLine,
  ].filter(Boolean);

  const embedColor = predictions.some(p => p.early_season) ? MLS_AMBER : MLS_BLUE;

  const picksEmbed: DiscordEmbed = {
    title: `⚽ MLS Oracle — Matchday Predictions | ${date}`,
    description: descParts.join('  ·  '),
    color: embedColor,
    fields: picksFields.slice(0, 20), // Discord limit
    footer: {
      text: [
        '⚡⚡ = Extreme conviction (65%+)',
        '⚡ = High conviction (60%+)',
        '✅ = Strong (52%+)',
        '🪙 = Lean',
        'MLS Oracle v4.1',
      ].join('  |  '),
    },
    timestamp: new Date().toISOString(),
  };

  // ── Embed 2: High-Conviction Picks ────────────────────────────────────────
  let hcEmbed: DiscordEmbed;

  if (highConv.length === 0) {
    hcEmbed = {
      title: '💰 High-Conviction Plays',
      description: '**No picks today clear the 60% threshold.** Wait for a better matchup.',
      color: MLS_GRAY,
    };
  } else {
    const hcFields: DiscordField[] = highConv.map(pred => {
      const matchup = `${pred.away_team} @ ${pred.home_team}`;
      const conf    = confidenceEmoji(pred.pick_confidence);
      const tier    = getConfidenceTier(pred.pick_confidence);
      const tierLabel = tier === 'high_conviction' ? '⚡ HIGH CONVICTION' : '⚡⚡ EXTREME';
      const flagStr = venueFlags(pred);
      const why: string[] = [`Model confidence: ${pct(pred.pick_confidence)}`];
      if (pred.feature_vector.altitude_flag === 1) why.push('Altitude home advantage');
      if (pred.feature_vector.cross_country_flag === 1) why.push(`Away travel ${Math.round(pred.feature_vector.travel_distance_diff)}mi`);
      if (pred.feature_vector.dp_available_diff > 0) why.push('Home DP advantage');
      if (pred.feature_vector.rest_days_diff >= 3) why.push('Home rest advantage');
      return {
        name: `${conf} ${tierLabel}: ${matchup}`,
        value: [
          `**Pick:** ${pickLabel(pred)} · ${pct(pred.pick_confidence)}`,
          `**Why:** ${why.join(' · ')}`,
          flagStr ? flagStr : '',
        ].filter(Boolean).join('\n'),
        inline: false,
      };
    });

    hcEmbed = {
      title: `💰 High-Conviction Plays — ${highConv.length} today`,
      description: 'Games where the Poisson model has ≥60% on a single outcome. Target accuracy: **65-70%**.',
      color: MLS_GREEN,
      fields: hcFields,
      footer: { text: 'Bet responsibly. Past performance ≠ future results. MLS Oracle v4.1' },
    };
  }

  return sendWebhook({ embeds: [picksEmbed, hcEmbed] });
}

// ─── EVENING: Results Recap ───────────────────────────────────────────────────

export async function sendEveningRecap(
  date: string,
  games: Array<{
    prediction: Prediction;
    homeScore: number;
    awayScore: number;
    outcome: 'H' | 'D' | 'A';
  }>,
  metrics: {
    accuracy: number;
    brier3way: number;
    drawAccuracy: number;
    doubleChanceAccuracy: number;
    highConvAccuracy: number | null;
  }
): Promise<boolean> {
  const season = getSeasonRecord();

  if (games.length === 0) {
    return sendWebhook({
      embeds: [{
        title: `🌙 MLS Oracle — Recap | ${date}`,
        description: 'No completed matches found. Results may still be in progress.',
        color: MLS_GRAY,
        timestamp: new Date().toISOString(),
      }],
    });
  }

  const gradedGames = games.filter(g => g.prediction.correct !== undefined);
  const correct = gradedGames.filter(g => g.prediction.correct).length;
  const total   = gradedGames.length;
  const accPct  = total > 0 ? (correct / total) * 100 : 0;

  const recapColor = total === 0 ? MLS_GRAY
    : accPct >= 60 ? MLS_GREEN
    : accPct >= 45 ? MLS_GRAY
    : MLS_RED;

  const accEmoji = total === 0 ? '⚪' : accPct >= 60 ? '🟢' : accPct >= 45 ? '🟡' : '🔴';

  // High conviction picks performance
  const hcGames   = gradedGames.filter(g => g.prediction.pick_confidence >= 0.60);
  const hcCorrect = hcGames.filter(g => g.prediction.correct).length;

  // Game-by-game lines
  const gameLines = games.map(({ prediction: pred, homeScore, awayScore, outcome }) => {
    const noPred = pred.correct === undefined && pred.home_prob === pred.draw_prob;
    const outcomeStr = outcome === 'H' ? pred.home_team : outcome === 'A' ? pred.away_team : 'Draw';
    if (noPred) {
      return `⚪ **${pred.away_team} @ ${pred.home_team}**: ${awayScore}-${homeScore} *(no pick — ${outcomeStr})*`;
    }
    const isCorrect = pred.correct ? '✅' : '❌';
    const wasHC = pred.pick_confidence >= 0.60 ? ' ⚡' : '';
    const picked = pickLabel(pred);
    return `${isCorrect}${wasHC} **${pred.away_team} @ ${pred.home_team}**: ${awayScore}-${homeScore} *(${picked})* → ${outcomeStr}`;
  }).join('\n');

  const seasonSummary = season.total > 0
    ? `📈 Season: **${season.correct}-${season.total - season.correct}** (${((season.correct / season.total) * 100).toFixed(1)}%)` +
      (season.high_conv_total > 0
        ? `\n⚡ Season 60%+ picks: **${season.high_conv_correct}-${season.high_conv_total - season.high_conv_correct}**`
        : '')
    : '📈 Season record: **0-0**';

  const summaryLines = [
    total > 0
      ? `**${accEmoji} Tonight: ${correct}/${total} correct (${accPct.toFixed(0)}%)**`
      : '**⚪ No picks graded tonight**',
    hcGames.length > 0
      ? `**⚡ High-conviction: ${hcCorrect}/${hcGames.length} correct (${hcGames.length > 0 ? ((hcCorrect / hcGames.length) * 100).toFixed(0) : 0}%)**`
      : '**⚡ No high-conviction picks tonight**',
    metrics.highConvAccuracy !== null
      ? `Calibrated HC accuracy: ${(metrics.highConvAccuracy * 100).toFixed(0)}%`
      : '',
    `Draw accuracy: ${(metrics.drawAccuracy * 100).toFixed(0)}% · DC accuracy: ${(metrics.doubleChanceAccuracy * 100).toFixed(0)}%`,
    `3-way Brier: ${metrics.brier3way.toFixed(4)}`,
    seasonSummary,
  ].filter(Boolean).join('\n');

  const embed: DiscordEmbed = {
    title: `🌙 MLS Oracle — Results | ${date}`,
    color: recapColor,
    fields: [
      { name: '📊 Summary',     value: summaryLines, inline: false },
      { name: '🎯 Match Results', value: gameLines || 'No results.', inline: false },
    ],
    footer: { text: 'MLS Oracle v4.1 · 3-way Brier tracks calibration · Target: 52%+ match result' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── Season lifecycle alerts ──────────────────────────────────────────────────

export async function sendSeasonAlert(type: 'start' | 'end' | 'leagues_cup'): Promise<void> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) return;

  const embeds: Record<string, DiscordEmbed> = {
    start: {
      title: '⚽ MLS Season Is Underway!',
      description: 'MLS Oracle v4.1 is **online** — daily matchday predictions and recap embeds. Target: **52%+ match result** · **65-70%** on high-conviction picks.',
      color: MLS_BLUE,
      timestamp: new Date().toISOString(),
    },
    end: {
      title: '🏆 MLS Cup Season Complete',
      description: 'MLS Oracle is entering **off-season mode**. No daily messages until late February. Thanks for following along!',
      color: MLS_GRAY,
      timestamp: new Date().toISOString(),
    },
    leagues_cup: {
      title: '🌎 Leagues Cup Begins',
      description: 'MLS regular season pauses for **Leagues Cup** (MLS vs Liga MX). Oracle will resume MLS predictions post-tournament.',
      color: MLS_AMBER,
      timestamp: new Date().toISOString(),
    },
  };

  try {
    await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ embeds: [embeds[type]] }),
      signal: AbortSignal.timeout(10000),
    });
    logger.info({ type }, 'Season lifecycle alert sent');
  } catch (err) {
    logger.warn({ err }, 'Failed to send season lifecycle alert');
  }
}
