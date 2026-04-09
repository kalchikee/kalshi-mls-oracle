// MLS Oracle v4.1 — ML Meta-Model (Phase 3)
// Multinomial Logistic Regression for 3-way outcome (H/D/A).
// Falls back to Poisson simulation probabilities if model files are absent.
// Train with: python python/train_model.py → exports to data/model/

import { existsSync, readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type { FeatureVector, ThreeWayProbs } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODEL_DIR = resolve(__dirname, '../../data/model');

// ─── JSON artifact shapes ─────────────────────────────────────────────────────

// Multinomial LR: one coefficient vector per class (H, D, A)
interface MultiCoefficientsJson {
  home: Record<string, number>;   // { feature_name: coeff, ..., _intercept: n }
  draw: Record<string, number>;
  away: Record<string, number>;
}

interface ScalerJson {
  feature_names: string[];
  mean: number[];
  scale: number[];
}

interface ModelMetadataJson {
  version: string;
  model_type: string;
  feature_names: string[];
  train_dates: string;
  avg_brier: number;
  avg_accuracy: number;
  trained_at: string;
}

// ─── Internal model state ─────────────────────────────────────────────────────

interface LoadedModel {
  featureNames: string[];
  coeffsHome: Float64Array;
  coeffsDrawArr: Float64Array;
  coeffsAway: Float64Array;
  interceptHome: number;
  interceptDraw: number;
  interceptAway: number;
  scalerMean: Float64Array;
  scalerScale: Float64Array;
  metadata: ModelMetadataJson;
}

let _model: LoadedModel | null = null;

export function isModelLoaded(): boolean {
  return _model !== null;
}

export function getModelInfo(): { version: string; avgBrier: number; trainDates: string } | null {
  if (!_model) return null;
  return {
    version: _model.metadata.version,
    avgBrier: _model.metadata.avg_brier,
    trainDates: _model.metadata.train_dates,
  };
}

// ─── Load model from disk ─────────────────────────────────────────────────────

export function loadModel(): boolean {
  const coeffPath  = resolve(MODEL_DIR, 'coefficients.json');
  const scalerPath = resolve(MODEL_DIR, 'scaler.json');
  const metaPath   = resolve(MODEL_DIR, 'metadata.json');

  if (!existsSync(coeffPath) || !existsSync(scalerPath) || !existsSync(metaPath)) {
    logger.info('ML model files not found — using Poisson simulation fallback');
    logger.info(`Expected: ${MODEL_DIR}/coefficients.json, scaler.json, metadata.json`);
    logger.info('Run: npm run train  (python python/train_model.py) to train the model');
    return false;
  }

  try {
    const coeffs: MultiCoefficientsJson = JSON.parse(readFileSync(coeffPath, 'utf-8'));
    const scaler: ScalerJson            = JSON.parse(readFileSync(scalerPath, 'utf-8'));
    const meta: ModelMetadataJson       = JSON.parse(readFileSync(metaPath, 'utf-8'));

    const featureNames = scaler.feature_names;
    const n = featureNames.length;

    const buildCoeffArr = (classCoeffs: Record<string, number>) => {
      const arr = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        arr[i] = classCoeffs[featureNames[i]] ?? 0;
      }
      return arr;
    };

    _model = {
      featureNames,
      coeffsHome:    buildCoeffArr(coeffs.home),
      coeffsDrawArr: buildCoeffArr(coeffs.draw),
      coeffsAway:    buildCoeffArr(coeffs.away),
      interceptHome: coeffs.home['_intercept'] ?? 0,
      interceptDraw: coeffs.draw['_intercept'] ?? 0,
      interceptAway: coeffs.away['_intercept'] ?? 0,
      scalerMean:    new Float64Array(scaler.mean),
      scalerScale:   new Float64Array(scaler.scale),
      metadata:      meta,
    };

    logger.info({ version: meta.version, features: n, avgBrier: meta.avg_brier }, 'ML meta-model loaded');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to load ML model — falling back to Poisson');
    _model = null;
    return false;
  }
}

// ─── Softmax function ─────────────────────────────────────────────────────────

function softmax(scores: [number, number, number]): [number, number, number] {
  const maxScore = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - maxScore)); // subtract max for numerical stability
  const sum = exps.reduce((a, b) => a + b, 0);
  return [exps[0] / sum, exps[1] / sum, exps[2] / sum] as [number, number, number];
}

// ─── Build feature array ──────────────────────────────────────────────────────

function buildFeatureArray(features: FeatureVector, featureNames: string[]): Float64Array {
  const arr = new Float64Array(featureNames.length);
  const fv = features as unknown as Record<string, number>;
  for (let i = 0; i < featureNames.length; i++) {
    arr[i] = fv[featureNames[i]] ?? 0;
  }
  return arr;
}

// ─── Predict 3-way probabilities ─────────────────────────────────────────────

export function predict(
  features: FeatureVector,
  poissonProbs: ThreeWayProbs
): ThreeWayProbs {
  if (!_model) return poissonProbs;

  const { featureNames, coeffsHome, coeffsDrawArr, coeffsAway,
          interceptHome, interceptDraw, interceptAway,
          scalerMean, scalerScale } = _model;

  const rawFeatures = buildFeatureArray(features, featureNames);
  const n = featureNames.length;

  // Inject Poisson probs as training features (if they're in the feature set)
  const poissonHomeIdx = featureNames.indexOf('poisson_home');
  const poissonDrawIdx = featureNames.indexOf('poisson_draw');
  const poissonAwayIdx = featureNames.indexOf('poisson_away');
  if (poissonHomeIdx >= 0) rawFeatures[poissonHomeIdx] = poissonProbs.home;
  if (poissonDrawIdx >= 0) rawFeatures[poissonDrawIdx] = poissonProbs.draw;
  if (poissonAwayIdx >= 0) rawFeatures[poissonAwayIdx] = poissonProbs.away;

  let logitHome = interceptHome;
  let logitDraw = interceptDraw;
  let logitAway = interceptAway;

  for (let i = 0; i < n; i++) {
    if (scalerScale[i] <= 0) continue;
    const z = Math.max(-3, Math.min(3, (rawFeatures[i] - scalerMean[i]) / scalerScale[i]));
    logitHome += coeffsHome[i]    * z;
    logitDraw += coeffsDrawArr[i] * z;
    logitAway += coeffsAway[i]    * z;
  }

  const [home, draw, away] = softmax([logitHome, logitDraw, logitAway]);

  // Clamp each probability
  const clamp = (p: number) => Math.max(0.02, Math.min(0.96, p));
  const rawHome = clamp(home);
  const rawDraw = clamp(draw);
  const rawAway = clamp(away);

  // Re-normalize after clamping
  const total = rawHome + rawDraw + rawAway;
  return {
    home: rawHome / total,
    draw: rawDraw / total,
    away: rawAway / total,
  };
}
