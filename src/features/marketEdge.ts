// MLS Oracle v4.1 — Market Edge Detection (3-way: H/D/A)
// Edge tiers: <3% = none, 3-6% = small, 6-10% = meaningful, 10-15% = large, ≥15% = extreme

import type { ThreeWayEdge, EdgeCategory } from '../types.js';

// ─── Compute 3-way edge ───────────────────────────────────────────────────────

export function computeEdge(
  modelHome: number,
  modelDraw: number,
  modelAway: number,
  vegasHome: number,
  vegasDraw: number,
  vegasAway: number
): ThreeWayEdge {
  const edgeHome = modelHome - vegasHome;
  const edgeDraw = modelDraw - vegasDraw;
  const edgeAway = modelAway - vegasAway;

  // Best edge is the largest absolute edge across all three markets
  const edges = [
    { side: 'H' as const, val: edgeHome },
    { side: 'D' as const, val: edgeDraw },
    { side: 'A' as const, val: edgeAway },
  ];
  const best = edges.reduce((a, b) => Math.abs(a.val) > Math.abs(b.val) ? a : b);

  const absEdge = Math.abs(best.val);
  let edgeCategory: EdgeCategory;
  if (absEdge < 0.03)      edgeCategory = 'none';
  else if (absEdge < 0.06) edgeCategory = 'small';
  else if (absEdge < 0.10) edgeCategory = 'meaningful';
  else if (absEdge < 0.15) edgeCategory = 'large';
  else                     edgeCategory = 'extreme';

  return {
    modelHomeProb: modelHome, modelDrawProb: modelDraw, modelAwayProb: modelAway,
    vegasHomeProb: vegasHome, vegasDrawProb: vegasDraw, vegasAwayProb: vegasAway,
    edgeHome, edgeDraw, edgeAway,
    bestEdge: best.val,
    bestEdgeSide: absEdge >= 0.03 ? best.side : null,
    edgeCategory,
  };
}

export function formatEdge(edge: ThreeWayEdge): string {
  const sign = edge.bestEdge >= 0 ? '+' : '';
  const pct = (edge.bestEdge * 100).toFixed(1);
  const side = edge.bestEdgeSide ?? '-';
  const tier = edge.edgeCategory.toUpperCase();
  return (
    `Best edge: ${sign}${pct}% on ${side} [${tier}] | ` +
    `Model: H${(edge.modelHomeProb * 100).toFixed(1)}% D${(edge.modelDrawProb * 100).toFixed(1)}% A${(edge.modelAwayProb * 100).toFixed(1)}% | ` +
    `Vegas: H${(edge.vegasHomeProb * 100).toFixed(1)}% D${(edge.vegasDrawProb * 100).toFixed(1)}% A${(edge.vegasAwayProb * 100).toFixed(1)}%`
  );
}

// ─── Confidence tiers ─────────────────────────────────────────────────────────
// Soccer has 3-way outcomes so probabilities are naturally lower.
// 60%+ on any single outcome is very high conviction in soccer.

export type ConfidenceTier =
  | 'coin_flip'       // max prob < 40%
  | 'lean'            // 40-50%
  | 'strong'          // 50-60%
  | 'high_conviction' // 60%+
  ;

export function getConfidenceTier(pickProb: number): ConfidenceTier {
  if (pickProb >= 0.60) return 'high_conviction';
  if (pickProb >= 0.50) return 'strong';
  if (pickProb >= 0.40) return 'lean';
  return 'coin_flip';
}

export function confidenceEmoji(pickProb: number): string {
  if (pickProb >= 0.65) return '⚡⚡';
  if (pickProb >= 0.60) return '⚡';
  if (pickProb >= 0.52) return '✅';
  if (pickProb >= 0.45) return '🪙';
  return '❓';
}

export function shouldAlert(pickProb: number): boolean {
  return pickProb >= 0.60;
}
