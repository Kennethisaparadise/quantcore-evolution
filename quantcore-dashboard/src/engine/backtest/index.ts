// ============================================
// QUANTCORE GENETIC ENGINE - BACKTEST ENGINE
// ============================================

import type { StrategyParams, BacktestConfig } from '../types';

export interface OHLCV {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  id: string;
  entry: number;
  exit: number;
  side: 'long' | 'short';
  size: number;
  pnl: number;
  pnlPercent: number;
  commission: number;
  slippage: number;
  entryTime: number;
  exitTime: number;
  barsHeld: number;
}

export interface BacktestResult {
  total_return_pct: number;
  total_return_abs: number;
  annualized_return_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  calmar_ratio: number;
  win_rate: number;
  loss_rate: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  avg_win: number;
  avg_loss: number;
  largest_win: number;
  largest_loss: number;
  avg_win_loss_ratio: number;
  profit_factor: number;
  expectancy: number;
  expectancy_pct: number;
  avg_bars_held: number;
  avg_trade_duration_ms: number;
  consecutive_wins: number;
  consecutive_losses: number;
  max_consecutive_wins: number;
  max_consecutive_losses: number;
  trades: Trade[];
  equity_curve: number[];
  drawdown_curve: number[];
}

export interface Position {
  entry: number;
  side: 'long' | 'short';
  size: number;
  entryTime: number;
  entryBar: number;
  stopLoss?: number;
  takeProfit?: number;
}

// -------- Technical Indicators --------

export function calculateRSI(data: OHLCV[], period: number = 14): (number | null)[] {
  const rsi: (number | null)[] = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period) { rsi.push(null); continue; }
    let gains = 0, losses = 0;
    for (let j = i - period + 1; j <= i; j++) {
      const change = data[j].close - data[j - 1].close;
      if (change > 0) gains += change;
      else losses -= change;
    }
    const avgGain = gains / period;
    const avgLoss = losses / period;
    if (avgLoss === 0) rsi.push(100);
    else rsi.push(100 - (100 / (1 + avgGain / avgLoss)));
  }
  return rsi;
}

export function calculateEMA(data: OHLCV[], period: number): (number | null)[] {
  const ema: (number | null)[] = [];
  const k = 2 / (period + 1);
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) { ema.push(null); continue; }
    if (i === period - 1) {
      const sma = data.slice(0, period).reduce((s, d) => s + d.close, 0) / period;
      ema.push(sma);
      continue;
    }
    ema.push(data[i].close * k + (ema[i - 1] as number) * (1 - k));
  }
  return ema;
}

export function calculateSMA(data: OHLCV[], period: number): (number | null)[] {
  const sma: (number | null)[] = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) { sma.push(null); continue; }
    sma.push(data.slice(i - period + 1, i + 1).reduce((s, d) => s + d.close, 0) / period);
  }
  return sma;
}

export function calculateBB(data: OHLCV[], period: number = 20, stdDev: number = 2): { upper: (number | null)[], middle: (number | null)[], lower: (number | null)[] } {
  const sma = calculateSMA(data, period);
  const upper: (number | null)[] = [], lower: (number | null)[] = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) { upper.push(null); lower.push(null); continue; }
    const mean = sma[i] as number;
    const slice = data.slice(i - period + 1, i + 1);
    const variance = slice.reduce((s, d) => s + Math.pow(d.close - mean, 2), 0) / period;
    upper.push(mean + stdDev * Math.sqrt(variance));
    lower.push(mean - stdDev * Math.sqrt(variance));
  }
  return { upper, middle: sma, lower };
}

export function calculateMACD(data: OHLCV[], fast: number = 12, slow: number = 26, signal: number = 9): { macd: (number | null)[], signal: (number | null)[] } {
  const emaFast = calculateEMA(data, fast);
  const emaSlow = calculateEMA(data, slow);
  const macd: (number | null)[] = [];
  for (let i = 0; i < data.length; i++) {
    if (emaFast[i] === null || emaSlow[i] === null) { macd.push(null); continue; }
    macd.push((emaFast[i] as number) - (emaSlow[i] as number));
  }
  const signalLine = calculateEMAWithNull(macd, signal);
  return { macd, signal: signalLine };
}

function calculateEMAWithNull(data: (number | null)[], period: number): (number | null)[] {
  const ema: (number | null)[] = [];
  const k = 2 / (period + 1);
  let sum = 0, count = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] === null) { ema.push(null); continue; }
    count++;
    if (count < period) { sum += data[i] as number; ema.push(null); continue; }
    if (count === period) { sum += data[i] as number; ema.push(sum / period); sum = 0; continue; }
    ema.push((data[i] as number) * k + (ema[i - 1] as number) * (1 - k));
  }
  return ema;
}

export function calculateATR(data: OHLCV[], period: number = 14): (number | null)[] {
  const atr: (number | null)[] = [];
  const trueRanges: number[] = [];
  for (let i = 0; i < data.length; i++) {
    if (i === 0) { trueRanges.push(data[i].high - data[i].low); atr.push(null); continue; }
    const tr = Math.max(data[i].high - data[i].low, Math.abs(data[i].high - data[i - 1].close), Math.abs(data[i].low - data[i - 1].close));
    trueRanges.push(tr);
    if (i < period) { atr.push(null); continue; }
    if (i === period) { atr.push(trueRanges.slice(1).reduce((s, v) => s + v, 0) / period); continue; }
    atr.push((atr[i - 1] as number) * (period - 1) / period + tr);
  }
  return atr;
}

export function calculateSupertrend(data: OHLCV[], period: number = 10, multiplier: number = 3): { direction: (1 | -1 | null)[], values: (number | null)[] } {
  const atr = calculateATR(data, period);
  const direction: (1 | -1 | null)[] = [];
  const values: (number | null)[] = [];
  let trend: 1 | -1 = 1;
  for (let i = 0; i < data.length; i++) {
    if (atr[i] === null) { direction.push(null); values.push(null); continue; }
    const hl2 = (data[i].high + data[i].low) / 2;
    const upper = hl2 + multiplier * (atr[i] as number);
    const lower = hl2 - multiplier * (atr[i] as number);
    if (i > 0) {
      if (data[i].close > (values[i - 1] as number || 0)) trend = 1;
      else if (data[i].close < (values[i - 1] as number || Infinity)) trend = -1;
    }
    direction.push(trend);
    values.push(trend === 1 ? lower : upper);
  }
  return { direction, values };
}

// -------- Signal Generation --------

export function generateEntrySignal(params: StrategyParams, data: OHLCV[], barIndex: number): 'long' | 'short' | null {
  const { indicator, trade_side, filters } = params;
  const lookback = 50;
  if (barIndex < lookback) return null;
  
  const curr = data[barIndex];
  
  if (filters.volume_confirmation) {
    const slice = data.slice(Math.max(0, barIndex - 20), barIndex);
    const avgVolume = slice.reduce((s, d) => s + d.volume, 0) / slice.length;
    if (curr.volume < avgVolume * filters.volume_multiplier) return null;
  }
  
  if (indicator === 'RSI') {
    const rsiPeriod = (params.entry_indicators[0]?.params.period as number) || 14;
    const rsi = calculateRSI(data.slice(Math.max(0, barIndex - rsiPeriod), barIndex + 1), rsiPeriod);
    const currRsi = rsi[rsi.length - 1], prevRsi = rsi[rsi.length - 2];
    if (currRsi === null || prevRsi === null) return null;
    const oversold = (params.entry_indicators[0]?.params.oversold as number) || 30;
    const overbought = (params.entry_indicators[0]?.params.overbought as number) || 70;
    if (trade_side !== 'short' && currRsi < oversold && prevRsi >= oversold) return filters.regime_filter === 'bear' ? null : 'long';
    if (trade_side !== 'long' && currRsi > overbought && prevRsi <= overbought) return filters.regime_filter === 'bull' ? null : 'short';
  }
  
  if (indicator === 'MACD') {
    const fast = (params.entry_indicators[0]?.params.fast as number) || 12;
    const slow = (params.entry_indicators[0]?.params.slow as number) || 26;
    const { macd, signal } = calculateMACD(data.slice(0, barIndex + 1), fast, slow);
    const currM = macd[macd.length - 1], prevM = macd[macd.length - 2];
    const currS = signal[signal.length - 1], prevS = signal[signal.length - 2];
    if (currM === null || prevM === null || currS === null || prevS === null) return null;
    if (trade_side !== 'short' && prevM <= prevS && currM > currS) return filters.regime_filter === 'bear' ? null : 'long';
    if (trade_side !== 'long' && prevM >= prevS && currM < currS) return filters.regime_filter === 'bull' ? null : 'short';
  }
  
  if (indicator === 'BB') {
    const period = (params.entry_indicators[0]?.params.period as number) || 20;
    const bb = calculateBB(data.slice(0, barIndex + 1), period);
    const lower = bb.lower[bb.lower.length - 1], upper = bb.upper[bb.upper.length - 1];
    if (lower === null || upper === null) return null;
    if (trade_side !== 'short' && curr.close < lower) return 'long';
    if (trade_side !== 'long' && curr.close > upper) return 'short';
  }
  
  if (indicator === 'SUPERTREND') {
    const period = (params.entry_indicators[0]?.params.period as number) || 10;
    const mult = (params.entry_indicators[0]?.params.multiplier as number) || 3;
    const { direction } = calculateSupertrend(data.slice(0, barIndex + 1), period, mult);
    const currD = direction[direction.length - 1], prevD = direction[direction.length - 2];
    if (currD === null || prevD === null) return null;
    if (trade_side !== 'short' && prevD === -1 && currD === 1) return 'long';
    if (trade_side !== 'long' && prevD === 1 && currD === -1) return 'short';
  }
  
  return null;
}

// -------- Main Backtest Engine --------

export function runBacktest(params: StrategyParams, data: OHLCV[], config: Partial<BacktestConfig> = {}): BacktestResult {
  const { initial_balance = 10000, commission_pct = 0.1, slippage_pct = 0.05 } = config;
  
  const trades: Trade[] = [];
  let position: Position | null = null;
  let balance = initial_balance;
  let equity = initial_balance;
  let peakEquity = initial_balance;
  let maxDrawdownPct = 0;
  let consecutiveWins = 0, consecutiveLosses = 0;
  let maxConsecutiveWins = 0, maxConsecutiveLosses = 0;
  let totalBars = 0, totalDuration = 0;
  const equityCurve: number[] = [initial_balance];
  
  for (let i = 1; i < data.length; i++) {
    const curr = data[i];
    const signal = generateEntrySignal(params, data, i);
    
    if (!position && signal) {
      const price = curr.close * (1 + slippage_pct / 100);
      const commission = price * (commission_pct / 100);
      const size = (balance * params.risk.position_size_pct / 100) / price;
      position = {
        entry: price + commission, side: signal, size, entryTime: curr.timestamp, entryBar: i,
        stopLoss: signal === 'long' ? price * (1 - params.risk.stop_loss_pct / 100) : price * (1 + params.risk.stop_loss_pct / 100),
        takeProfit: signal === 'long' ? price * (1 + params.risk.take_profit_pct / 100) : price * (1 - params.risk.take_profit_pct / 100),
      };
    }
    
    if (position) {
      const price = curr.close;
      let exitPrice = price, shouldExit = false;
      
      const pnlPct = position.side === 'long' ? (price - position.entry) / position.entry * 100 : (position.entry - price) / position.entry * 100;
      
      if (position.side === 'long' && price <= (position.stopLoss || 0)) { exitPrice = position.stopLoss! * (1 - slippage_pct / 100); shouldExit = true; }
      else if (position.side === 'short' && price >= (position.stopLoss || Infinity)) { exitPrice = position.stopLoss! * (1 + slippage_pct / 100); shouldExit = true; }
      else if (position.side === 'long' && price >= (position.takeProfit || Infinity)) { exitPrice = position.takeProfit!; shouldExit = true; }
      else if (position.side === 'short' && price <= (position.takeProfit || 0)) { exitPrice = position.takeProfit!; shouldExit = true; }
      else if (params.risk.use_time_exit && i - position.entryBar >= params.risk.max_bars_held) shouldExit = true;
      
      if (shouldExit) {
        const exitCommission = exitPrice * (commission_pct / 100);
        const pnl = position.side === 'long' ? (exitPrice - position.entry - exitCommission) * position.size : (position.entry - exitPrice - exitCommission) * position.size;
        
        trades.push({ id: `t_${i}`, entry: position.entry, exit: exitPrice, side: position.side, size: position.size, pnl, pnlPercent: (pnl / balance) * 100, commission: commission_pct, slippage: slippage_pct, entryTime: position.entryTime, exitTime: curr.timestamp, barsHeld: i - position.entryBar });
        
        balance += pnl;
        totalBars += i - position.entryBar;
        totalDuration += curr.timestamp - position.entryTime;
        
        if (pnl > 0) { consecutiveWins++; consecutiveLosses = 0; if (consecutiveWins > maxConsecutiveWins) maxConsecutiveWins = consecutiveWins; }
        else { consecutiveLosses++; consecutiveWins = 0; if (consecutiveLosses > maxConsecutiveLosses) maxConsecutiveLosses = consecutiveLosses; }
        position = null;
      }
    }
    
    equity = balance;
    if (peakEquity < equity) peakEquity = equity;
    const dd = (peakEquity - equity) / peakEquity * 100;
    if (dd > maxDrawdownPct) maxDrawdownPct = dd;
    equityCurve.push(equity);
  }
  
  const wins = trades.filter(t => t.pnl > 0), losses = trades.filter(t => t.pnl <= 0);
  const avgWin = wins.length ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0;
  const avgLoss = losses.length ? Math.abs(losses.reduce((s, t) => s + t.pnl, 0) / losses.length) : 1;
  const returns = trades.map(t => t.pnlPercent);
  const avgReturn = returns.length ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
  const stdDev = returns.length > 1 ? Math.sqrt(returns.map(r => Math.pow(r - avgReturn, 2)).reduce((a, b) => a + b, 0) / (returns.length - 1)) : 1;
  const negReturns = returns.filter(r => r < 0);
  const downstdDev = negReturns.length ? Math.sqrt(negReturns.map(r => r * r).reduce((a, b) => a + b, 0) / negReturns.length) : 1;
  
  const totalReturn = ((balance - initial_balance) / initial_balance) * 100;
  const years = data.length / (365 * 24);
  const annualizedReturn = years > 0 ? (Math.pow(balance / initial_balance, 1 / years) - 1) * 100 : 0;
  
  return {
    total_return_pct: totalReturn,
    total_return_abs: balance - initial_balance,
    annualized_return_pct: annualizedReturn,
    sharpe_ratio: stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0,
    sortino_ratio: downstdDev > 0 ? (avgReturn / downstdDev) * Math.sqrt(252) : 0,
    max_drawdown: peakEquity - balance,
    max_drawdown_pct: maxDrawdownPct,
    calmar_ratio: maxDrawdownPct > 0 ? totalReturn / maxDrawdownPct : 0,
    win_rate: trades.length ? (wins.length / trades.length) * 100 : 0,
    loss_rate: trades.length ? (losses.length / trades.length) * 100 : 0,
    total_trades: trades.length,
    winning_trades: wins.length,
    losing_trades: losses.length,
    avg_win: avgWin,
    avg_loss: avgLoss,
    largest_win: wins.length ? Math.max(...wins.map(t => t.pnl)) : 0,
    largest_loss: losses.length ? Math.min(...losses.map(t => t.pnl)) : 0,
    avg_win_loss_ratio: avgLoss > 0 ? avgWin / avgLoss : 0,
    profit_factor: avgLoss > 0 ? (avgWin * wins.length) / (avgLoss * losses.length) : 0,
    expectancy: trades.length ? (wins.length / trades.length) * avgWin - (losses.length / trades.length) * avgLoss : 0,
    expectancy_pct: avgReturn,
    avg_bars_held: trades.length ? totalBars / trades.length : 0,
    avg_trade_duration_ms: trades.length ? totalDuration / trades.length : 0,
    consecutive_wins: consecutiveWins,
    consecutive_losses: consecutiveLosses,
    max_consecutive_wins: maxConsecutiveWins,
    max_consecutive_losses: maxConsecutiveLosses,
    trades,
    equity_curve: equityCurve,
    drawdown_curve: equityCurve.map((e, i) => (peakEquity - e) / peakEquity * 100),
  };
}

export function generateMockData(symbol: string, bars: number = 500): OHLCV[] {
  const basePrice: Record<string, number> = { BTC: 50000, ETH: 3000, SOL: 100, AVAX: 35, ARB: 1.2, OP: 2.5 };
  const volatility: Record<string, number> = { BTC: 0.02, ETH: 0.025, SOL: 0.035, AVAX: 0.04, ARB: 0.05, OP: 0.05 };
  const price = basePrice[symbol] || 100;
  const vol = volatility[symbol] || 0.03;
  
  const data: OHLCV[] = [];
  let currentPrice = price, trend = 0;
  for (let i = 0; i < bars; i++) {
    trend = trend * 0.95 + (Math.random() - 0.5) * 0.1;
    const change = trend + (Math.random() - 0.5) * vol;
    const open = currentPrice;
    const close = currentPrice * (1 + change);
    data.push({ timestamp: Date.now() - (bars - i) * 3600000, open, high: Math.max(open, close) * (1 + Math.random() * vol * 0.5), low: Math.min(open, close) * (1 - Math.random() * vol * 0.5), close, volume: 1000000 + Math.random() * 2000000 });
    currentPrice = close;
  }
  return data;
}
