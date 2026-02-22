// ============================================
// QUANTCORE GENETIC ENGINE - TYPE DEFINITIONS
// ============================================

export type IndicatorType = 
  | 'RSI' | 'MACD' | 'BB' | 'SMA' | 'EMA' | 'STOCH' | 'ADX' 
  | 'ATR' | 'CCI' | 'MOM' | 'ROC' | 'WILLR' | 'CMO' | 'MFI'
  | 'OBV' | 'VWAP' | 'SUPERTREND' | 'ICHIMOKU' | 'KELTNER' | 'DONCHIAN'
  | 'TRIX' | 'HURST' | 'COMPOSITE' | 'ELDERRAY' | 'KST' | 'PIVOT'
  | 'ELDERRAY' | 'MOMENTUM';

export type MarketRegime = 'bull' | 'bear' | 'sideways' | 'volatile' | 'any';
export type TradeSide = 'long' | 'short' | 'both';
export type SignalSource = 'entry' | 'exit' | 'filter';

export interface StrategyIndicator {
  type: IndicatorType;
  enabled: boolean;
  params: Record<string, number | boolean | string>;
}

export interface RiskParams {
  stop_loss_pct: number;
  take_profit_pct: number;
  position_size_pct: number;
  max_positions: number;
  use_trailing_stop: boolean;
  trailing_distance_pct: number;
  trailing_type?: 'fixed' | 'atr' | 'sar';
  trailing_atr_periods?: number;
  use_time_exit: boolean;
  max_bars_held: number;
  stop_type?: 'fixed' | 'atr' | 'sar' | 'chandelier';
  atr_multiplier?: number;
  sar_acceleration?: number;
  chandelier_periods?: number;
  chandelier_multiplier?: number;
  use_martingale?: boolean;
  martingale_factor?: number;
  kelly_fraction?: number;
  max_pyramid?: number;
}

export interface FilterParams {
  volume_confirmation: boolean;
  volume_multiplier: number;
  volume_spike_required?: boolean;
  volume_spike_multiplier?: number;
  volatility_filter: boolean;
  volatility_threshold: number;
  regime_filter: MarketRegime;
  adx_required?: boolean;
  adx_threshold?: number;
  session_filter?: string;
  time_filter: { enabled: boolean; sessions: string[] };
  min_liquidity: number;
}

export interface AdvancedParams {
  use_compounding: boolean;
  compounding_method: 'fixed' | 'kelly' | 'martingale' | 'pyramid';
  kelly_fraction: number;
  martingale_multiplier: number;
  max_consecutive_losses: number;
  partial_exit_enabled: boolean;
  partial_exit_pct: number;
}

export interface StrategyParams {
  id: string;
  name: string;
  version: string;
  created_at: number;
  updated_at: number;
  indicator: IndicatorType;
  trade_side: TradeSide;
  signal_inverted?: boolean;
  require_confirmation?: boolean;
  use_divergence?: boolean | 'bullish' | 'bearish';
  entry_indicators: StrategyIndicator[];
  exit_indicators: StrategyIndicator[];
  filter_indicators: StrategyIndicator[];
  risk: RiskParams;
  filters: FilterParams;
  advanced: AdvancedParams;
  description: string;
  tags: string[];
  author: string;
}

export interface Mutant {
  id: string;
  name: string;
  generation: number;
  parent_id: string | null;
  ancestor_ids: string[];
  params: StrategyParams;
  fitness: number;
  fitness_components: FitnessComponents;
  total_return_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  win_rate: number;
  max_drawdown: number;
  profit_factor: number;
  calmar_ratio: number;
  total_trades: number;
  expectancy: number;
  mutation_description: string;
  mutation_count: number;
  crossover_count: number;
  created_at: number;
  backtest_result?: any;
}

export interface FitnessComponents {
  return_score: number;
  sharpe_score: number;
  winrate_score: number;
  drawdown_score: number;
  profit_factor_score: number;
  robustness_score: number;
}

export interface EvolutionConfig {
  population_size: number;
  elitism_count: number;
  generations: number;
  mutation_rate: number;
  adaptive_mutation: boolean;
  min_mutation_rate: number;
  max_mutation_rate: number;
  selection_method: SelectionMethod;
  tournament_size: number;
  crossover_rate: number;
  crossover_method: CrossoverMethod;
  diversity_threshold: number;
  niche_preservation: boolean;
  early_stopping: boolean;
  patience: number;
  min_improvement: number;
  log_interval: number;
  save_all: boolean;
}

export type SelectionMethod = 'tournament' | 'roulette' | 'rank' | 'nsga2' | 'spea2';
export type CrossoverMethod = 'single_point' | 'two_point' | 'uniform' | 'simulated_binary' | 'linear';

export interface EvolutionState {
  generation: number;
  population: Mutant[];
  best_mutant: Mutant | null;
  hall_of_fame: Mutant[];
  history: EvolutionHistoryEntry[];
  is_running: boolean;
  is_paused: boolean;
  progress: number;
  elapsed_time: number;
  fitness_over_time: number[];
}

export interface EvolutionHistoryEntry {
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  worst_fitness: number;
  diversity: number;
  best_return: number;
  best_sharpe: number;
}

export interface BacktestConfig {
  symbol: string;
  timeframe: string;
  start_date?: number;
  end_date?: number;
  initial_balance: number;
  commission_pct: number;
  slippage_pct: number;
}

export type MutationCategory = 'risk' | 'indicator' | 'entry' | 'exit' | 'filter' | 'regime' | 'position' | 'signal' | 'stop';

export const MUTATION_CATEGORIES = {
  RISK: 'risk',
  INDICATOR: 'indicator',
  ENTRY: 'entry',
  EXIT: 'exit',
  FILTER: 'filter',
  REGIME: 'regime',
  POSITION: 'position',
  SIGNAL: 'signal',
} as const;

export const DEFAULT_EVOLUTION_CONFIG: EvolutionConfig = {
  population_size: 50,
  elitism_count: 5,
  generations: 50,
  mutation_rate: 0.3,
  adaptive_mutation: true,
  min_mutation_rate: 0.1,
  max_mutation_rate: 0.6,
  selection_method: 'tournament',
  tournament_size: 4,
  crossover_rate: 0.7,
  crossover_method: 'two_point',
  diversity_threshold: 0.3,
  niche_preservation: true,
  early_stopping: true,
  patience: 10,
  min_improvement: 0.01,
  log_interval: 1,
  save_all: false,
};

export function createDefaultStrategyParams(indicator: IndicatorType = 'RSI'): StrategyParams {
  const defaults: Record<string, Record<string, number | boolean>> = {
    RSI: { period: 14, oversold: 30, overbought: 70 },
    MACD: { fast: 12, slow: 26, signal: 9 },
    BB: { period: 20, stdDev: 2 },
    SMA: { period: 20 },
    EMA: { period: 20 },
    STOCH: { k: 14, d: 3, overbought: 80, oversold: 20 },
    ADX: { period: 14, threshold: 25 },
    ATR: { period: 14, multiplier: 2 },
    CCI: { period: 20, overbought: 100, oversold: -100 },
    MOM: { period: 10, threshold: 0 },
    ROC: { period: 12, threshold: 0 },
    WILLR: { period: 14, overbought: -20, oversold: -80 },
    SUPERTREND: { period: 10, multiplier: 3 },
    ICHIMOKU: { conversion: 9, base: 26, spanB: 52, displacement: 26 },
    KELTNER: { period: 20, multiplier: 2 },
    DONCHIAN: { period: 20, upper: 20, lower: 20 },
    TRIX: { period: 15, signal: 9 },
  };
  
  return {
    id: `strat_${Date.now()}`,
    name: `${indicator} Strategy`,
    version: '1.0.0',
    created_at: Date.now(),
    updated_at: Date.now(),
    indicator,
    trade_side: 'both',
    entry_indicators: [{ type: indicator, enabled: true, params: defaults[indicator] || { period: 14 } }],
    exit_indicators: [],
    filter_indicators: [],
    risk: {
      stop_loss_pct: 3,
      take_profit_pct: 6,
      position_size_pct: 5,
      max_positions: 1,
      use_trailing_stop: false,
      trailing_distance_pct: 2,
      use_time_exit: false,
      max_bars_held: 12,
    },
    filters: {
      volume_confirmation: false,
      volume_multiplier: 1.5,
      volatility_filter: false,
      volatility_threshold: 2,
      regime_filter: 'any',
      time_filter: { enabled: false, sessions: [] },
      min_liquidity: 10000,
    },
    advanced: {
      use_compounding: false,
      compounding_method: 'fixed',
      kelly_fraction: 0.25,
      martingale_multiplier: 2,
      max_consecutive_losses: 5,
      partial_exit_enabled: false,
      partial_exit_pct: 50,
    },
    description: '',
    tags: [],
    author: 'QuantCore',
  };
}
