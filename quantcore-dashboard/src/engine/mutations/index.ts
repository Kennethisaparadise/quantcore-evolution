// ============================================
// QUANTCORE GENETIC ENGINE - MUTATION OPERATORS
// ============================================

import type { StrategyParams, MutationCategory } from '../types';

export interface MutationOperator {
  id: string;
  category: MutationCategory;
  label: string;
  description: string;
  icon: string;
  probability: number;
  apply: (params: StrategyParams) => StrategyParams;
}

const clamp = (val: number, min: number, max: number) => Math.max(min, Math.min(max, val));
const randomRange = (min: number, max: number) => Math.random() * (max - min) + min;
const randomChoice = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const randomInt = (min: number, max: number) => Math.floor(randomRange(min, max));

const riskMutations: MutationOperator[] = [
  { id: 'tighten_stop_loss', category: 'risk', label: 'Tighten Stop Loss', description: 'Reduce stop loss percentage', icon: '🛡️', probability: 0.15, apply: p => ({ ...p, risk: { ...p.risk, stop_loss_pct: clamp(p.risk.stop_loss_pct * randomRange(0.5, 0.8), 0.5, 10) }, updated_at: Date.now() }) },
  { id: 'widen_stop_loss', category: 'risk', label: 'Widen Stop Loss', description: 'Increase stop loss room', icon: '🎯', probability: 0.15, apply: p => ({ ...p, risk: { ...p.risk, stop_loss_pct: clamp(p.risk.stop_loss_pct * randomRange(1.2, 1.5), 0.5, 15) }, updated_at: Date.now() }) },
  { id: 'increase_take_profit', category: 'risk', label: 'Increase Take Profit', description: 'Aim for bigger wins', icon: '💰', probability: 0.15, apply: p => ({ ...p, risk: { ...p.risk, take_profit_pct: clamp(p.risk.take_profit_pct * randomRange(1.2, 1.6), 2, 30) }, updated_at: Date.now() }) },
  { id: 'decrease_take_profit', category: 'risk', label: 'Decrease Take Profit', description: 'Secure profits faster', icon: '⚡', probability: 0.15, apply: p => ({ ...p, risk: { ...p.risk, take_profit_pct: clamp(p.risk.take_profit_pct * randomRange(0.5, 0.8), 1, 25) }, updated_at: Date.now() }) },
  { id: 'increase_position_size', category: 'risk', label: 'Larger Positions', description: 'Take bigger positions', icon: '📈', probability: 0.1, apply: p => ({ ...p, risk: { ...p.risk, position_size_pct: clamp(p.risk.position_size_pct + randomRange(1, 3), 1, 20) }, updated_at: Date.now() }) },
  { id: 'decrease_position_size', category: 'risk', label: 'Smaller Positions', description: 'Reduce risk', icon: '📉', probability: 0.1, apply: p => ({ ...p, risk: { ...p.risk, position_size_pct: clamp(p.risk.position_size_pct - randomRange(1, 2), 1, 15) }, updated_at: Date.now() }) },
  { id: 'enable_trailing_stop', category: 'risk', label: 'Enable Trailing Stop', description: 'Lock in profits', icon: '🦾', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, use_trailing_stop: true, trailing_distance_pct: randomRange(1, 4) }, updated_at: Date.now() }) },
  { id: 'disable_trailing_stop', category: 'risk', label: 'Disable Trailing', description: 'Use fixed stops', icon: '⏹️', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, use_trailing_stop: false }, updated_at: Date.now() }) },
  { id: 'enable_time_exit', category: 'risk', label: 'Enable Time Exit', description: 'Exit after N bars', icon: '⏰', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, use_time_exit: true, max_bars_held: Math.floor(randomRange(5, 30)) }, updated_at: Date.now() }) },
  { id: 'adjust_trailing', category: 'risk', label: 'Adjust Trailing', description: 'Modify trailing distance', icon: '📏', probability: 0.06, apply: p => ({ ...p, risk: { ...p.risk, trailing_distance_pct: clamp(p.risk.trailing_distance_pct + randomRange(-1, 1), 0.5, 5) }, updated_at: Date.now() }) },
];

const indicatorMutations: MutationOperator[] = [
  { id: 'rsi_faster', category: 'indicator', label: 'RSI Faster', description: 'Decrease RSI period', icon: '🚀', probability: 0.12, apply: p => p.indicator === 'RSI' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, period: clamp((i.params.period as number || 14) - randomInt(1, 4), 5, 30) } })), updated_at: Date.now() } : p },
  { id: 'rsi_slower', category: 'indicator', label: 'RSI Slower', description: 'Increase RSI period', icon: '🐢', probability: 0.12, apply: p => p.indicator === 'RSI' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, period: clamp((i.params.period as number || 14) + randomInt(1, 5), 5, 30) } })), updated_at: Date.now() } : p },
  { id: 'rsi_oversold', category: 'indicator', label: 'Adjust RSI Oversold', description: 'Modify oversold level', icon: '📊', probability: 0.1, apply: p => p.indicator === 'RSI' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, oversold: clamp((i.params.oversold as number || 30) + randomInt(-10, 10), 10, 40) } })), updated_at: Date.now() } : p },
  { id: 'rsi_overbought', category: 'indicator', label: 'Adjust RSI Overbought', description: 'Modify overbought level', icon: '📈', probability: 0.1, apply: p => p.indicator === 'RSI' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, overbought: clamp((i.params.overbought as number || 70) + randomInt(-10, 10), 60, 90) } })), updated_at: Date.now() } : p },
  { id: 'macd_fast', category: 'indicator', label: 'MACD Fast Adjust', description: 'Modify fast period', icon: '⚡', probability: 0.1, apply: p => p.indicator === 'MACD' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, fast: clamp((i.params.fast as number || 12) + randomInt(-3, 3), 5, 20) } })), updated_at: Date.now() } : p },
  { id: 'macd_slow', category: 'indicator', label: 'MACD Slow Adjust', description: 'Modify slow period', icon: '🐘', probability: 0.1, apply: p => p.indicator === 'MACD' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, slow: clamp((i.params.slow as number || 26) + randomInt(-5, 5), 15, 50) } })), updated_at: Date.now() } : p },
  { id: 'macd_signal', category: 'indicator', label: 'MACD Signal Adjust', description: 'Modify signal line', icon: '📉', probability: 0.08, apply: p => p.indicator === 'MACD' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, signal: clamp((i.params.signal as number || 9) + randomInt(-2, 2), 5, 15) } })), updated_at: Date.now() } : p },
  { id: 'bb_period', category: 'indicator', label: 'BB Period Adjust', description: 'Modify BB period', icon: '🎳', probability: 0.1, apply: p => p.indicator === 'BB' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, period: clamp((i.params.period as number || 20) + randomInt(-5, 5), 10, 50) } })), updated_at: Date.now() } : p },
  { id: 'bb_stddev', category: 'indicator', label: 'BB StdDev Adjust', description: 'Modify standard deviation', icon: '📐', probability: 0.08, apply: p => p.indicator === 'BB' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, stdDev: clamp((i.params.stdDev as number || 2) + randomRange(-0.5, 0.5), 1, 4) } })), updated_at: Date.now() } : p },
  { id: 'st_period', category: 'indicator', label: 'Supertrend Period', description: 'Modify ST period', icon: '🌊', probability: 0.1, apply: p => p.indicator === 'SUPERTREND' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, period: clamp((i.params.period as number || 10) + randomInt(-3, 3), 5, 20) } })), updated_at: Date.now() } : p },
  { id: 'st_multiplier', category: 'indicator', label: 'Supertrend Multiplier', description: 'Modify ST multiplier', icon: '✖️', probability: 0.1, apply: p => p.indicator === 'SUPERTREND' ? { ...p, entry_indicators: p.entry_indicators.map(i => ({ ...i, params: { ...i.params, multiplier: clamp((i.params.multiplier as number || 3) + randomRange(-1, 1), 1, 5) } })), updated_at: Date.now() } : p },
];

const filterMutations: MutationOperator[] = [
  { id: 'enable_volume', category: 'filter', label: 'Volume Filter', description: 'Require volume confirmation', icon: '📊', probability: 0.2, apply: p => ({ ...p, filters: { ...p.filters, volume_confirmation: true, volume_multiplier: randomRange(1.2, 2.0) }, updated_at: Date.now() }) },
  { id: 'disable_volume', category: 'filter', label: 'No Volume Filter', description: 'Remove volume requirement', icon: '🔇', probability: 0.15, apply: p => ({ ...p, filters: { ...p.filters, volume_confirmation: false }, updated_at: Date.now() }) },
  { id: 'volume_mult', category: 'filter', label: 'Volume Multiplier', description: 'Adjust volume threshold', icon: '🔊', probability: 0.15, apply: p => ({ ...p, filters: { ...p.filters, volume_multiplier: clamp(p.filters.volume_multiplier + randomRange(-0.3, 0.3), 1.0, 3.0) }, updated_at: Date.now() }) },
  { id: 'bull_only', category: 'regime', label: 'Bull Market Only', description: 'Trade bull markets', icon: '🐂', probability: 0.15, apply: p => ({ ...p, filters: { ...p.filters, regime_filter: 'bull' }, updated_at: Date.now() }) },
  { id: 'bear_only', category: 'regime', label: 'Bear Market Only', description: 'Trade bear markets', icon: '🐻', probability: 0.15, apply: p => ({ ...p, filters: { ...p.filters, regime_filter: 'bear' }, updated_at: Date.now() }) },
  { id: 'any_regime', category: 'regime', label: 'Any Market', description: 'Trade all conditions', icon: '🌐', probability: 0.1, apply: p => ({ ...p, filters: { ...p.filters, regime_filter: 'any' }, updated_at: Date.now() }) },
];

const tradeSideMutations: MutationOperator[] = [
  { id: 'long_only', category: 'position', label: 'Long Only', description: 'Only long positions', icon: '⬆️', probability: 0.25, apply: p => ({ ...p, trade_side: 'long', updated_at: Date.now() }) },
  { id: 'short_only', category: 'position', label: 'Short Only', description: 'Only short positions', icon: '⬇️', probability: 0.25, apply: p => ({ ...p, trade_side: 'short', updated_at: Date.now() }) },
  { id: 'both_sides', category: 'position', label: 'Both Directions', description: 'Long and short', icon: '⬆️⬇️', probability: 0.5, apply: p => ({ ...p, trade_side: 'both', updated_at: Date.now() }) },
];

// -------- Position Management (Kelly, Martingale, Pyramiding) --------
const positionMutations: MutationOperator[] = [
  { id: 'kelly_full', category: 'position', label: 'Full Kelly', description: 'Use full Kelly sizing', icon: '🎰', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, kelly_fraction: 1.0 }, updated_at: Date.now() }) },
  { id: 'kelly_half', category: 'position', label: 'Half Kelly', description: 'Conservative Kelly', icon: '🎲', probability: 0.1, apply: p => ({ ...p, risk: { ...p.risk, kelly_fraction: 0.5 }, updated_at: Date.now() }) },
  { id: 'kelly_quarter', category: 'position', label: 'Quarter Kelly', description: 'Very conservative sizing', icon: '🔒', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, kelly_fraction: 0.25 }, updated_at: Date.now() }) },
  { id: 'martingale_double', category: 'position', label: 'Martingale x2', description: 'Double size after loss', icon: '↔️', probability: 0.06, apply: p => ({ ...p, risk: { ...p.risk, use_martingale: true, martingale_factor: 2.0 }, updated_at: Date.now() }) },
  { id: 'martingale_1.5', category: 'position', label: 'Martingale x1.5', description: 'Moderate martingale', icon: '↔️', probability: 0.06, apply: p => ({ ...p, risk: { ...p.risk, use_martingale: true, martingale_factor: 1.5 }, updated_at: Date.now() }) },
  { id: 'martingale_disable', category: 'position', label: 'Disable Martingale', description: 'Fixed position sizing', icon: '⛔', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, use_martingale: false }, updated_at: Date.now() }) },
  { id: 'pyramid_add', category: 'position', label: 'Enable Pyramiding', description: 'Add to winning positions', icon: '🔺', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, max_pyramid: randomInt(2, 4) }, updated_at: Date.now() }) },
  { id: 'pyramid_remove', category: 'position', label: 'Disable Pyramiding', description: 'Single position only', icon: '⬛', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, max_pyramid: 0 }, updated_at: Date.now() }) },
];

// -------- Signal Modifications --------
const signalMutations: MutationOperator[] = [
  { id: 'invert_signal', category: 'signal', label: 'Invert Signal', description: 'Flip long/short signals', icon: '🔄', probability: 0.1, apply: p => ({ ...p, signal_inverted: !p.signal_inverted, updated_at: Date.now() }) },
  { id: 'require_confirm', category: 'signal', label: 'Require Confirmation', description: 'Wait for next bar', icon: '✅', probability: 0.12, apply: p => ({ ...p, require_confirmation: true, updated_at: Date.now() }) },
  { id: 'immediate_entry', category: 'signal', label: 'Immediate Entry', description: 'Enter on signal bar', icon: '⚡', probability: 0.12, apply: p => ({ ...p, require_confirmation: false, updated_at: Date.now() }) },
  { id: 'divergence_bull', category: 'signal', label: 'Bullish Divergence', description: 'Entry on bullish divergence', icon: '🐂', probability: 0.08, apply: p => ({ ...p, use_divergence: 'bullish', updated_at: Date.now() }) },
  { id: 'divergence_bear', category: 'signal', label: 'Bearish Divergence', description: 'Entry on bearish divergence', icon: '🐻', probability: 0.08, apply: p => ({ ...p, use_divergence: 'bearish', updated_at: Date.now() }) },
  { id: 'divergence_disable', category: 'signal', label: 'No Divergence', description: 'Disable divergence filter', icon: '🚫', probability: 0.1, apply: p => ({ ...p, use_divergence: false, updated_at: Date.now() }) },
];

// -------- Enhanced Stop Loss --------
const stopLossMutations: MutationOperator[] = [
  { id: 'atr_stop', category: 'stop', label: 'ATR Stop', description: 'Use ATR-based stops', icon: '📊', probability: 0.1, apply: p => ({ ...p, risk: { ...p.risk, stop_type: 'atr', atr_multiplier: randomRange(1.5, 3.0) }, updated_at: Date.now() }) },
  { id: 'sar_stop', category: 'stop', label: 'SAR Stop', description: 'Use Parabolic SAR', icon: '⭕', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, stop_type: 'sar', sar_acceleration: randomRange(0.02, 0.1) }, updated_at: Date.now() }) },
  { id: 'chandelier_stop', category: 'stop', label: 'Chandelier Stop', description: 'Chandelier exit', icon: '💡', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, stop_type: 'chandelier', chandelier_periods: randomInt(20, 30), chandelier_multiplier: randomRange(2.5, 4.0) }, updated_at: Date.now() }) },
  { id: 'fixed_stop', category: 'stop', label: 'Fixed Stop', description: 'Use percentage stop', icon: '📏', probability: 0.15, apply: p => ({ ...p, risk: { ...p.risk, stop_type: 'fixed' }, updated_at: Date.now() }) },
  { id: 'trailing_atr', category: 'stop', label: 'Trailing ATR', description: 'ATR-trailing stop', icon: '🎢', probability: 0.1, apply: p => ({ ...p, risk: { ...p.risk, use_trailing_stop: true, trailing_type: 'atr', trailing_atr_periods: randomInt(14, 28) }, updated_at: Date.now() }) },
  { id: 'trailing_sar', category: 'stop', label: 'Trailing SAR', description: 'SAR-trailing stop', icon: '⭕', probability: 0.08, apply: p => ({ ...p, risk: { ...p.risk, use_trailing_stop: true, trailing_type: 'sar' }, updated_at: Date.now() }) },
];

// -------- Advanced Filters --------
const advancedFilterMutations: MutationOperator[] = [
  { id: 'adx_filter', category: 'filter', label: 'ADX Trend Filter', description: 'Require strong trend', icon: '💨', probability: 0.1, apply: p => ({ ...p, filters: { ...p.filters, adx_required: true, adx_threshold: randomInt(20, 35) }, updated_at: Date.now() }) },
  { id: 'adx_disable', category: 'filter', label: 'Disable ADX', description: 'Remove ADX filter', icon: '🚫', probability: 0.1, apply: p => ({ ...p, filters: { ...p.filters, adx_required: false }, updated_at: Date.now() }) },
  { id: 'volume_spike', category: 'filter', label: 'Volume Spike', description: 'Require volume spike', icon: '📈', probability: 0.1, apply: p => ({ ...p, filters: { ...p.filters, volume_spike_required: true, volume_spike_multiplier: randomRange(1.5, 2.5) }, updated_at: Date.now() }) },
  { id: 'time_filter', category: 'filter', label: 'Session Filter', description: 'Trade specific sessions', icon: '🕐', probability: 0.08, apply: p => ({ ...p, filters: { ...p.filters, session_filter: randomChoice(['asian', 'london', 'ny', 'asian_london']) }, updated_at: Date.now() }) },
  { id: 'time_disable', category: 'filter', label: 'No Time Filter', description: 'All sessions', icon: '🌍', probability: 0.1, apply: p => ({ ...p, filters: { ...p.filters, session_filter: 'any' }, updated_at: Date.now() }) },
  { id: 'sideways_only', category: 'regime', label: 'Sideways Market', description: 'Range-bound markets', icon: '↔️', probability: 0.12, apply: p => ({ ...p, filters: { ...p.filters, regime_filter: 'sideways' }, updated_at: Date.now() }) },
];

// -------- Advanced Indicators --------
const advancedIndicatorMutations: MutationOperator[] = [
  { id: 'keltner_channel', category: 'indicator', label: 'Keltner Channel', description: 'Add Keltner entry', icon: '📐', probability: 0.08, apply: p => ({ ...p, indicator: 'KELTNER', entry_indicators: [{ type: 'KELTNER', enabled: true, params: { period: randomInt(20, 30), multiplier: randomRange(2, 3) } }], updated_at: Date.now() }) },
  { id: 'donchian_channel', category: 'indicator', label: 'Donchian Channel', description: 'Donchian breakout', icon: '🟦', probability: 0.08, apply: p => ({ ...p, indicator: 'DONCHIAN', entry_indicators: [{ type: 'DONCHIAN', enabled: true, params: { period: randomInt(20, 40) } }], updated_at: Date.now() }) },
  { id: 'elder_ray', category: 'indicator', label: 'Elder-Ray', description: 'Bull/Bear power', icon: '🔦', probability: 0.06, apply: p => ({ ...p, indicator: 'ELDERRAY', entry_indicators: [{ type: 'ELDERRAY', enabled: true, params: { period: randomInt(13, 26) } }], updated_at: Date.now() }) },
  { id: 'trix', category: 'indicator', label: 'TRIX', description: 'Triple smoothed ROC', icon: '📉', probability: 0.06, apply: p => ({ ...p, indicator: 'TRIX', entry_indicators: [{ type: 'TRIX', enabled: true, params: { period: randomInt(15, 30), signal: randomInt(5, 15) } }], updated_at: Date.now() }) },
  { id: 'hurst_exponent', category: 'indicator', label: 'Hurst Mode', description: 'Trend/reversal mode', icon: '🧠', probability: 0.05, apply: p => ({ ...p, indicator: 'HURST', entry_indicators: [{ type: 'HURST', enabled: true, params: { lookback: randomInt(50, 200) } }], updated_at: Date.now() }) },
  { id: 'ichimoku_cloud', category: 'indicator', label: 'Ichimoku', description: 'Cloud-based entry', icon: '☁️', probability: 0.06, apply: p => ({ ...p, indicator: 'ICHIMOKU', entry_indicators: [{ type: 'ICHIMOKU', enabled: true, params: { conversion: 9, base: 26, spanB: 52 } }], updated_at: Date.now() }) },
  { id: 'stochastic', category: 'indicator', label: 'Stochastic', description: 'Stochastic oscillator', icon: '🎯', probability: 0.1, apply: p => ({ ...p, indicator: 'STOCH', entry_indicators: [{ type: 'STOCH', enabled: true, params: { k: randomInt(14, 21), d: randomInt(3, 7), oversold: randomInt(20, 30), overbought: randomInt(70, 80) } }], updated_at: Date.now() }) },
  { id: 'cci', category: 'indicator', label: 'CCI', description: 'Commodity Channel Index', icon: '📦', probability: 0.08, apply: p => ({ ...p, indicator: 'CCI', entry_indicators: [{ type: 'CCI', enabled: true, params: { period: randomInt(14, 28) } }], updated_at: Date.now() }) },
  { id: 'williams_r', category: 'indicator', label: 'Williams %R', description: 'Williams percent range', icon: '🎪', probability: 0.08, apply: p => ({ ...p, indicator: 'WILLR', entry_indicators: [{ type: 'WILLR', enabled: true, params: { period: randomInt(14, 28) } }], updated_at: Date.now() }) },
  { id: 'momentum', category: 'indicator', label: 'Momentum', description: 'Rate of change', icon: '💨', probability: 0.1, apply: p => ({ ...p, indicator: 'MOM', entry_indicators: [{ type: 'MOM', enabled: true, params: { period: randomInt(10, 20), threshold: randomRange(0.02, 0.05) } }], updated_at: Date.now() }) },
];

export const ALL_MUTATIONS: MutationOperator[] = [
  ...riskMutations,
  ...indicatorMutations,
  ...filterMutations,
  ...tradeSideMutations,
  ...positionMutations,
  ...signalMutations,
  ...stopLossMutations,
  ...advancedFilterMutations,
  ...advancedIndicatorMutations,
];

export class MutationManager {
  private mutations: MutationOperator[];
  
  constructor(customMutations?: MutationOperator[]) {
    this.mutations = customMutations || ALL_MUTATIONS;
  }
  
  selectMutation(): MutationOperator {
    return randomChoice(this.mutations);
  }
  
  applyMutation(params: StrategyParams, mutation?: MutationOperator): StrategyParams {
    const op = mutation || this.selectMutation();
    return { ...op.apply(params), updated_at: Date.now() };
  }
  
  getAllMutations(): MutationOperator[] {
    return [...this.mutations];
  }
}

export default MutationManager;
