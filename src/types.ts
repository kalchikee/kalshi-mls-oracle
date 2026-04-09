// MLS Oracle v4.1 — Core Type Definitions
// Three-way outcome (H/D/A), xG model, Designated Player impact, altitude & turf

// ─── MLS Team ─────────────────────────────────────────────────────────────────

export interface MLSTeam {
  teamId: number;
  teamAbbr: string;
  teamName: string;
  conference: 'East' | 'West';

  // Season record
  w: number;
  d: number;
  l: number;
  gp: number;
  pts: number;
  winPct: number;  // (W + 0.5*D) / GP — points rate

  // Goals
  goalsFor: number;     // per game
  goalsAgainst: number; // per game
  goalDiff: number;     // per game

  // xG (computed from shots data)
  xgFor: number;        // xG for per game
  xgAgainst: number;    // xG against per game
  xgDiff: number;       // xGF - xGA per game

  // Shooting
  shotsFor: number;     // shots per game
  shotsOnTargetFor: number;
  shotsAgainst: number;
  shotsOnTargetAgainst: number;

  // Style
  possession: number;   // avg possession %
  passAccuracy: number; // pass completion %

  // Expected points (3W + 1D) / 3
  xPts: number;         // expected pts per game from xG

  // Home/Away splits
  homeGoalsFor?: number;
  homeGoalsAgainst?: number;
  awayGoalsFor?: number;
  awayGoalsAgainst?: number;

  // Designated Player impact (estimated)
  dpImpact: number;     // weighted xG contribution of DPs
  dpAvailable: number;  // 0-3 — how many DPs are available

  // Form (last 5 matches xPts)
  form5xPts: number;

  // Overperformance: goals - xG (regression signal)
  overperformance: number;

  // Draw tendency
  drawRate: number;     // % of matches drawn

  // Manager tenure (months)
  managerTenure: number;
}

// ─── MLS Match ────────────────────────────────────────────────────────────────

export interface MLSMatch {
  matchId: string;
  matchDate: string;    // YYYY-MM-DD
  matchTime: string;    // ISO datetime
  status: string;       // 'Scheduled' | 'In Progress' | 'Final'
  matchweek: number;
  homeTeam: MLSMatchTeam;
  awayTeam: MLSMatchTeam;
  venue: string;
  venueCity: string;
}

export interface MLSMatchTeam {
  teamId: number;
  teamAbbr: string;
  teamName: string;
  score?: number;
}

// ─── Feature vector ───────────────────────────────────────────────────────────
// All diff features = home - away (positive = home advantage)

export interface FeatureVector {
  // Team strength
  elo_diff: number;               // home Elo - away Elo

  // xG-based (primary features per spec)
  xg_for_diff: number;            // xGF per 90 (home - away)
  xg_against_diff: number;        // xGA per 90 (home - away; higher = home worse defense)
  xg_diff: number;                // (xGF - xGA) per 90 (home - away)
  xpts_diff: number;              // xPts per match (home - away)
  ppg_diff: number;               // points per game (home - away)

  // Style
  possession_diff: number;        // avg possession % (home - away)
  pass_pct_diff: number;          // pass completion % (home - away)

  // Form
  form_5g_xpts_diff: number;      // last 5 matches xPts (home - away)
  overperformance_diff: number;   // goals - xG regression signal (home - away)

  // Draw prediction
  draw_tendency_diff: number;     // draw rate % (home - away)

  // Venue
  is_home: number;                // always 1 (vector from home perspective)
  home_advantage_diff: number;    // home vs away point gap (home - away)

  // MLS-unique environment features
  altitude_flag: number;          // 1 if playing at altitude (COL or RSL home)
  altitude_penalty: number;       // numeric penalty for away team (0-10% xG reduction)
  turf_flag: number;              // 1 if artificial turf venue (SEA, NE, VAN)

  // MLS-unique travel features
  travel_distance_diff: number;   // away team travel distance in miles
  cross_country_flag: number;     // 1 if away travel > 1,500 miles
  rest_days_diff: number;         // days since last match (home - away)
  midweek_flag: number;           // 1 if Wednesday match

  // MLS-unique congestion
  cup_congestion: number;         // 1 if US Open Cup / Leagues Cup in last 7 days

  // Designated Player impact
  dp_impact_diff: number;         // DP xG+xA contribution (home - away)
  dp_available_diff: number;      // number of DPs available (home - away, max 3)

  // Salary / investment
  roster_salary_diff: number;     // relative salary proxy (home - away)

  // Context
  conference_diff: number;        // 1 if same conference, 0 if cross-conference
  expansion_flag: number;         // 1 if either team is an expansion team (Year 1-2)

  // Playoff / season context
  playoff_position_diff: number;  // distance from playoff line (home - away)
  manager_tenure_diff: number;    // manager months at club (home - away)

  // Vegas (filled at prediction time if available)
  vegas_home_prob: number;        // vig-removed implied probability (0 if unavailable)
  vegas_draw_prob: number;        // vig-removed draw probability
  vegas_away_prob: number;        // vig-removed away probability
}

// ─── Model outputs ────────────────────────────────────────────────────────────

export interface ThreeWayProbs {
  home: number;   // P(home win)
  draw: number;   // P(draw)
  away: number;   // P(away win)
}

export interface PoissonResult {
  probs: ThreeWayProbs;
  home_xg: number;    // expected goals for home
  away_xg: number;    // expected goals for away
  most_likely_score: [number, number]; // [home, away]
  btts_prob: number;  // both teams to score
  over25_prob: number; // over 2.5 goals
  simulations: number;
}

export interface Prediction {
  game_date: string;
  game_id: string;
  home_team: string;
  away_team: string;
  venue: string;
  matchweek: number;
  feature_vector: FeatureVector;

  // Raw Poisson probabilities
  poisson_home: number;
  poisson_draw: number;
  poisson_away: number;

  // Calibrated (ML model if available, else Poisson)
  home_prob: number;
  draw_prob: number;
  away_prob: number;

  // Pick (max probability outcome)
  pick: 'H' | 'D' | 'A';
  pick_confidence: number;

  // Expected goals
  home_xg: number;
  away_xg: number;
  most_likely_score: string;
  btts_prob: number;
  over25_prob: number;

  // Vegas edge (if available)
  vegas_home_prob?: number;
  vegas_draw_prob?: number;
  vegas_away_prob?: number;
  edge_home?: number;
  edge_draw?: number;
  edge_away?: number;

  // Early season label
  early_season: boolean;
  matchweek_label: string; // 'EARLY SEASON' | 'Blending' | ''

  model_version: string;

  // Results
  actual_outcome?: 'H' | 'D' | 'A';
  correct?: boolean;
  created_at: string;
}

export interface EloRating {
  teamAbbr: string;
  rating: number;
  updatedAt: string;
}

export interface AccuracyLog {
  date: string;
  brier_3way: number;         // multiclass Brier score (sum of squared errors across 3 classes)
  accuracy: number;           // correct match result %
  draw_accuracy: number;      // draw prediction accuracy
  double_chance_accuracy: number; // X1 or X2 accuracy
  high_conv_accuracy: number; // accuracy on 60%+ picks
  games_evaluated: number;
}

export interface GameResult {
  game_id: string;
  date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  outcome: 'H' | 'D' | 'A';
  venue: string;
}

export interface PipelineOptions {
  date?: string;
  forceRefresh?: boolean;
  verbose?: boolean;
}

// ─── Edge detection ───────────────────────────────────────────────────────────

export type EdgeCategory = 'none' | 'small' | 'meaningful' | 'large' | 'extreme';

export interface ThreeWayEdge {
  modelHomeProb: number;
  modelDrawProb: number;
  modelAwayProb: number;
  vegasHomeProb: number;
  vegasDrawProb: number;
  vegasAwayProb: number;
  edgeHome: number;
  edgeDraw: number;
  edgeAway: number;
  bestEdge: number;
  bestEdgeSide: 'H' | 'D' | 'A' | null;
  edgeCategory: EdgeCategory;
}

// ─── Matchweek blending ───────────────────────────────────────────────────────

export interface BlendConfig {
  priorWeight: number;   // 0.0 - 1.0
  currentWeight: number; // 1 - priorWeight
  label: string;         // 'EARLY SEASON' | 'Blending' | ''
  earlySeasonFlag: boolean;
}
