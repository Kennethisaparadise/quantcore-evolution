// ============================================
// QUANTCORE SEED STRATEGIES
// Recovered from original build
// ============================================

export interface SeedStrategy {
  name: string;
  indicator: string;
  description?: string;
  type?: string;
  
  // Core parameters
  period?: number;
  fast_period?: number;
  slow_period?: number;
  signal_period?: number;
  
  // Multi-indicator strategies
  primary_indicator?: string;
  secondary_indicator?: string;
  
  // Multi-timeframe
  use_mtf?: boolean;
  htf_timeframe?: string;
  ltf_timeframe?: string;
  htf_ma_period?: number;
  htf_use_ema?: boolean;
  
  // Regime detection
  regime_detection?: boolean;
  regime_modes?: Record<string, unknown>;
  
  // Entry confirmations
  entry_confirmations?: Record<string, boolean>;
  
  // RSI params (alternative naming for composite/other strategies)
  rsi_oversold?: number;
  rsi_overbought?: number;
  
  // Trend indicators
  use_ema?: boolean;
  ema_fast?: number;
  ema_slow?: number;
  ema_medium?: number;
  
  // MACD params
  macd_fast?: number;
  macd_slow?: number;
  macd_signal?: number;
  macd_confirm?: string;
  
  // Bollinger params
  bb_period?: number;
  bb_std_dev?: number;
  
  // ATR params
  atr_stop_multiplier?: number;
  atr_tp_multiplier?: number;
  
  // BB squeeze
  bb_squeeze_threshold?: number;
  
  // Volume
  volume_ma_period?: number;
  volume_spike_mult?: number;
  
  // Position sizing
  position_sizing?: string;
  confidence_levels?: Record<string, unknown>;
  
  // Confidence-based sizing
  high_confidence_size?: number;
  medium_confidence_size?: number;
  low_confidence_size?: number;
  confidence_adx_threshold?: number;
  
  // Adaptive stops
  use_adaptive_sl?: boolean;
  trend_stop_loss?: number;
  reversion_stop_loss?: number;
  trend_take_profit?: number;
  reversion_take_profit?: number;
  
  // Threshold params
  overbought?: number;
  oversold?: number;
  std_dev?: number;
  multiplier?: number;
  
  // Entry/Exit rules (as strings for display)
  entry_long?: string;
  exit_long?: string;
  entry_short?: string;
  exit_short?: string;
  
  // Risk management
  stop_loss_pct: number;
  take_profit_pct: number;
  position_size_pct: number;
  
  // Additional params
  lookback?: number;
  volume_period?: number;
  volume_multiplier?: number;
  k_period?: number;
  d_period?: number;
  adx_threshold?: number;
  atr_multiplier?: number;
  breakout_threshold?: number;
  retracement_levels?: number[];
  
  // Tenkan/Kijun for Ichimoku
  tenkan_period?: number;
  kijun_period?: number;
  senkou_span_b?: number;
  
  // EMA periods
  ema_period?: number;
  atr_period?: number;
  
  // Advanced
  rsi_period?: number;
  stochastic_period?: number;
  williams_period?: number;
  
  // Regime-based
  bull_strategy?: string;
  bear_strategy?: string;
  sideways_strategy?: string;
  
  // Filters
  volume_confirmation?: boolean;
  max_daily_trades?: number;
  use_session_filter?: boolean;
  active_sessions?: string[];
  use_trailing_stop?: boolean;
  use_time_exit?: boolean;
  max_bars_held?: number;
}

export const SEED_STRATEGIES: Record<string, SeedStrategy> = {
  // === CORE STRATEGIES ===
  rsi: {
    name: 'RSI Mean Reversion',
    indicator: 'RSI',
    entry_long: 'RSI < 30',
    exit_long: 'RSI > 70',
    entry_short: 'RSI > 70',
    exit_short: 'RSI < 30',
    period: 14,
    overbought: 70,
    oversold: 30,
    stop_loss_pct: 3,
    take_profit_pct: 6,
    position_size_pct: 5,
  },
  
  ma_cross: {
    name: 'MA Crossover',
    indicator: 'SMA',
    fast_period: 10,
    slow_period: 50,
    entry_long: 'Fast MA crosses above Slow MA',
    exit_long: 'Fast MA crosses below Slow MA',
    stop_loss_pct: 4,
    take_profit_pct: 8,
    position_size_pct: 5,
  },
  
  bollinger: {
    name: 'Bollinger Band Squeeze',
    indicator: 'Bollinger Bands',
    period: 20,
    std_dev: 2,
    entry_long: 'Price touches lower band + squeeze release',
    exit_long: 'Price touches upper band',
    stop_loss_pct: 3,
    take_profit_pct: 6,
    position_size_pct: 5,
  },
  
  momentum: {
    name: 'Momentum Breakout',
    indicator: 'Momentum',
    lookback: 20,
    entry_long: 'Momentum > 0 and increasing',
    exit_long: 'Momentum < 0',
    stop_loss_pct: 5,
    take_profit_pct: 10,
    position_size_pct: 4,
  },
  
  vwap: {
    name: 'VWAP Reversion',
    indicator: 'VWAP',
    entry_long: 'Price < VWAP - 1 std dev',
    exit_long: 'Price > VWAP',
    entry_short: 'Price > VWAP + 1 std dev',
    exit_short: 'Price < VWAP',
    std_dev: 1,
    stop_loss_pct: 2,
    take_profit_pct: 4,
    position_size_pct: 6,
  },
  
  high_52w: {
    name: '52-Week High Breakout',
    indicator: '52W High',
    lookback: 252,
    entry_long: 'Price breaks above 52W high with volume',
    exit_long: 'Price drops 5% from peak',
    stop_loss_pct: 5,
    take_profit_pct: 15,
    position_size_pct: 3,
  },
  
  macd: {
    name: 'MACD Crossover',
    indicator: 'MACD',
    fast_period: 12,
    slow_period: 26,
    signal_period: 9,
    entry_long: 'MACD crosses above signal line',
    exit_long: 'MACD crosses below signal line',
    stop_loss_pct: 4,
    take_profit_pct: 8,
    position_size_pct: 5,
  },
  
  stochastic: {
    name: 'Stochastic Oscillator',
    indicator: 'Stochastic',
    k_period: 14,
    d_period: 3,
    oversold: 20,
    overbought: 80,
    entry_long: 'Stochastic K crosses above D below 20',
    exit_long: 'Stochastic K crosses below D above 80',
    stop_loss_pct: 3,
    take_profit_pct: 6,
    position_size_pct: 5,
  },
  
  williams_r: {
    name: 'Williams %R',
    indicator: 'Williams %R',
    period: 14,
    oversold: -80,
    overbought: -20,
    entry_long: '%R < -80 (oversold)',
    exit_long: '%R > -20 (overbought)',
    stop_loss_pct: 3,
    take_profit_pct: 6,
    position_size_pct: 5,
  },
  
  adx: {
    name: 'ADX Trend Strength',
    indicator: 'ADX',
    period: 14,
    adx_threshold: 25,
    entry_long: 'ADX > 25 and +DI > -DI',
    exit_long: 'ADX < 20 or +DI < -DI',
    stop_loss_pct: 4,
    take_profit_pct: 8,
    position_size_pct: 4,
  },
  
  atr_trailing: {
    name: 'ATR Trailing Stop',
    indicator: 'ATR',
    period: 14,
    atr_multiplier: 2.5,
    entry_long: 'Price closes above ATR trail',
    exit_long: 'Price closes below ATR trail',
    stop_loss_pct: 2,
    take_profit_pct: 10,
    position_size_pct: 5,
  },
  
  ema_cross: {
    name: 'EMA Crossover',
    indicator: 'EMA',
    fast_period: 9,
    slow_period: 21,
    entry_long: 'Fast EMA crosses above Slow EMA',
    exit_long: 'Fast EMA crosses below Slow EMA',
    stop_loss_pct: 4,
    take_profit_pct: 8,
    position_size_pct: 5,
  },
  
  volume_spike: {
    name: 'Volume Spike Breakout',
    indicator: 'Volume',
    volume_period: 20,
    volume_multiplier: 2,
    entry_long: 'Volume > 2x average + price breakout',
    exit_long: 'Volume returns to normal',
    stop_loss_pct: 3,
    take_profit_pct: 6,
    position_size_pct: 4,
  },
  
  fib_retrace: {
    name: 'Fibonacci Retracement',
    indicator: 'Fibonacci',
    retracement_levels: [0.236, 0.382, 0.5, 0.618, 0.786],
    entry_long: 'Price bounces at fib support level',
    exit_long: 'Price hits next resistance level',
    stop_loss_pct: 4,
    take_profit_pct: 12,
    position_size_pct: 4,
  },
  
  ichimoku: {
    name: 'Ichimoku Cloud',
    indicator: 'Ichimoku',
    tenkan_period: 9,
    kijun_period: 26,
    senkou_span_b: 52,
    entry_long: 'Price above cloud + Tenkan crosses above Kijun',
    exit_long: 'Price below cloud or Tenkan crosses below Kijun',
    stop_loss_pct: 5,
    take_profit_pct: 10,
    position_size_pct: 4,
  },
  
  supertrend: {
    name: 'SuperTrend',
    indicator: 'SuperTrend',
    period: 10,
    multiplier: 3,
    entry_long: 'Price closes above SuperTrend',
    exit_long: 'Price closes below SuperTrend',
    stop_loss_pct: 3,
    take_profit_pct: 6,
    position_size_pct: 5,
  },
  
  // === ADVANCED INDICATORS ===
  keltner: {
    name: 'Keltner Channel',
    indicator: 'Keltner',
    ema_period: 20,
    atr_period: 10,
    multiplier: 2,
    entry_long: 'Price breaks above upper Keltner band',
    exit_long: 'Price returns to middle band',
    stop_loss_pct: 3,
    take_profit_pct: 8,
    position_size_pct: 5,
  },
  
  donchian: {
    name: 'Donchian Channel',
    indicator: 'Donchian',
    period: 20,
    breakout_threshold: 0.5,
    entry_long: 'Price breaks above upper Donchian band',
    exit_long: 'Price drops below middle band',
    stop_loss_pct: 4,
    take_profit_pct: 10,
    position_size_pct: 4,
  },
  
  elder_ray: {
    name: 'Elder-Ray',
    indicator: 'ElderRay',
    ema_period: 13,
    lookback: 25,
    entry_long: 'Bull Power > 0 + Bull divergence',
    exit_long: 'Bear Power > 0 or price below EMA',
    stop_loss_pct: 3,
    take_profit_pct: 8,
    position_size_pct: 5,
  },
  
  trix: {
    name: 'TRIX Oscillator',
    indicator: 'TRIX',
    period: 15,
    signal_period: 9,
    entry_long: 'TRIX crosses above signal line',
    exit_long: 'TRIX crosses below signal line',
    stop_loss_pct: 3,
    take_profit_pct: 7,
    position_size_pct: 5,
  },
  
  chaikin: {
    name: 'Chaikin Oscillator',
    indicator: 'Chaikin',
    fast_period: 3,
    slow_period: 10,
    entry_long: 'Chaikin crosses above 0',
    exit_long: 'Chaikin crosses below 0',
    stop_loss_pct: 3,
    take_profit_pct: 6,
    position_size_pct: 5,
  },
  
  mass_index: {
    name: 'Mass Index',
    indicator: 'MassIndex',
    fast_period: 9,
    slow_period: 25,
    entry_long: 'Mass Index > 27 (reversal signal)',
    exit_long: 'Mass Index drops below 26.5',
    stop_loss_pct: 4,
    take_profit_pct: 8,
    position_size_pct: 4,
  },
  
  dpo: {
    name: 'Detrended Price Osc',
    indicator: 'DPO',
    period: 20,
    k_period: 10,
    entry_long: 'DPO crosses above 0',
    exit_long: 'DPO crosses below 0',
    stop_loss_pct: 3,
    take_profit_pct: 6,
    position_size_pct: 5,
  },
  
  kst: {
    name: 'Know Sure Thing (KST)',
    indicator: 'KST',
    fast_period: 10,
    slow_period: 15,
    signal_period: 9,
    entry_long: 'KST crosses above signal',
    exit_long: 'KST crosses below signal',
    stop_loss_pct: 4,
    take_profit_pct: 8,
    position_size_pct: 5,
  },
  
  hurst: {
    name: 'Hurst Exponent',
    indicator: 'Hurst',
    lookback: 100,
    period: 20,
    entry_long: 'Hurst > 0.5 (trending market)',
    exit_long: 'Hurst < 0.5 (mean reverting)',
    stop_loss_pct: 4,
    take_profit_pct: 10,
    position_size_pct: 4,
  },
  
  composite_index: {
    name: 'Composite Index',
    indicator: 'Composite',
    rsi_period: 14,
    stochastic_period: 14,
    williams_period: 14,
    entry_long: 'Composite < -100 (oversold)',
    exit_long: 'Composite > 100 (overbought)',
    stop_loss_pct: 3,
    take_profit_pct: 7,
    position_size_pct: 5,
  },
  
  // === SPECIAL STRATEGIES ===
  cashflow: {
    name: 'Cash Flow: Consistent Wins',
    indicator: 'RSI_BB_COMBO',
    description: 'High win rate, low drawdown - steady daily income',
    
    // Core settings - Conservative for consistency
    primary_indicator: 'RSI',
    secondary_indicator: 'Bollinger Bands',
    rsi_period: 7,
    rsi_oversold: 35,
    rsi_overbought: 65,
    bb_period: 20,
    bb_std_dev: 2,
    
    entry_long: 'RSI < 35 AND price near lower BB',
    entry_short: 'RSI > 65 AND price near upper BB',
    exit_long: 'RSI > 50',
    exit_short: 'RSI < 50',
    
    stop_loss_pct: 1.5,
    take_profit_pct: 2.5,
    position_size_pct: 6,
    
    volume_confirmation: true,
    volume_multiplier: 1.3,
    max_daily_trades: 10,
    use_time_exit: true,
    max_bars_held: 5,
    use_session_filter: true,
    active_sessions: ['london', 'new_york'],
  },
  
  regime_adaptive: {
    name: 'Regime Adaptive',
    indicator: 'REGIME',
    description: 'Auto-detect market regime and adapt strategy',
    
    // Strategy Selection by Regime
    bull_strategy: 'trend_following',
    bear_strategy: 'trend_following_short',
    sideways_strategy: 'mean_reversion',
    
    // Trend-Following Parameters
    fast_period: 10,
    slow_period: 50,
    breakout_threshold: 0.02,
    atr_period: 14,
    atr_multiplier: 2.5,
    
    // Mean Reversion Parameters
    rsi_period: 14,
    rsi_oversold: 30,
    rsi_overbought: 70,
    
    // Position Sizing by Confidence
    position_size_pct: 5,
    
    stop_loss_pct: 3,
    take_profit_pct: 6,
  },
  
  ultima: {
    name: 'ULTIMA: The Ultimate Strategy',
    indicator: 'ULTIMA',
    description: 'Combines the best elements from ALL strategies - Multi-timeframe, regime-aware, adaptive',
    type: 'ultimate_meta',
    
    // Multi-Timeframe
    use_mtf: true,
    fast_period: 15,
    slow_period: 200,
    
    // Regime Detection
    regime_detection: true,
    
    // Entry Confirmations
    entry_long: 'Multiple confirmations (trend + momentum + volume)',
    
    // Indicators
    ema_fast: 9,
    ema_slow: 21,
    rsi_period: 14,
    rsi_oversold: 30,
    rsi_overbought: 70,
    macd_fast: 12,
    macd_slow: 26,
    macd_signal: 9,
    atr_period: 14,
    atr_multiplier: 2.5,
    bb_period: 20,
    bb_std_dev: 2,
    
    // Position sizing
    position_size_pct: 5,
    
    stop_loss_pct: 3,
    take_profit_pct: 10,
  },
};

// Export for use in the app
export default SEED_STRATEGIES;
