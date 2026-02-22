// ============================================
// QUANTCORE GENETIC ENGINE - EXPORT MODULE
// ============================================

import type { Mutant, StrategyParams } from '../types';
import type { BacktestResult, Trade } from '../backtest';

export interface ExportOptions {
  includeTrades?: boolean;
  includeEquityCurve?: boolean;
  includeMetadata?: boolean;
  prettyPrint?: boolean;
}

const round = (num: number, decimals: number): number => Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);

// -------- JSON Export --------

export function exportToJSON(mutant: Mutant, result: BacktestResult, options: ExportOptions = {}): string {
  const { includeTrades = true, includeEquityCurve = false, includeMetadata = true, prettyPrint = true } = options;
  
  const exportData = {
    id: mutant.params.id,
    name: mutant.params.name,
    version: mutant.params.version,
    author: mutant.params.author,
    created_at: new Date(mutant.params.created_at).toISOString(),
    updated_at: new Date(mutant.params.updated_at).toISOString(),
    indicator: mutant.params.indicator,
    trade_side: mutant.params.trade_side,
    market_regime: mutant.params.filters.regime_filter,
    tags: mutant.params.tags,
    parameters: {
      entry: mutant.params.entry_indicators.reduce((acc, ind) => ({ ...acc, [ind.type]: ind.params }), {}),
      risk: {
        stop_loss_pct: mutant.params.risk.stop_loss_pct,
        take_profit_pct: mutant.params.risk.take_profit_pct,
        position_size_pct: mutant.params.risk.position_size_pct,
        use_trailing_stop: mutant.params.risk.use_trailing_stop,
        trailing_distance_pct: mutant.params.risk.trailing_distance_pct,
        use_time_exit: mutant.params.risk.use_time_exit,
        max_bars_held: mutant.params.risk.max_bars_held,
      },
      filters: {
        volume_confirmation: mutant.params.filters.volume_confirmation,
        volume_multiplier: mutant.params.filters.volume_multiplier,
        regime_filter: mutant.params.filters.regime_filter,
      },
    },
    statistics: {
      total_return_pct: round(result.total_return_pct, 2),
      annualized_return_pct: round(result.annualized_return_pct, 2),
      sharpe_ratio: round(result.sharpe_ratio, 2),
      sortino_ratio: round(result.sortino_ratio, 2),
      max_drawdown_pct: round(result.max_drawdown_pct, 2),
      calmar_ratio: round(result.calmar_ratio, 2),
      win_rate: round(result.win_rate, 2),
      profit_factor: round(result.profit_factor, 2),
      total_trades: result.total_trades,
      winning_trades: result.winning_trades,
      losing_trades: result.losing_trades,
      avg_win: round(result.avg_win, 2),
      avg_loss: round(result.avg_loss, 2),
      largest_win: round(result.largest_win, 2),
      largest_loss: round(result.largest_loss, 2),
      expectancy: round(result.expectancy, 2),
      max_consecutive_wins: result.max_consecutive_wins,
      max_consecutive_losses: result.max_consecutive_losses,
    },
    trades: includeTrades ? result.trades.map((t: Trade) => ({
      id: t.id, entry: round(t.entry, 2), exit: round(t.exit, 2), side: t.side,
      pnl: round(t.pnl, 2), pnl_percent: round(t.pnlPercent, 2),
      entry_time: new Date(t.entryTime).toISOString(), exit_time: new Date(t.exitTime).toISOString(), bars_held: t.barsHeld,
    })) : undefined,
    metadata: includeMetadata ? {
      generation: mutant.generation, fitness: round(mutant.fitness, 2),
      fitness_components: mutant.fitness_components,
      mutation_count: mutant.mutation_count, crossover_count: mutant.crossover_count,
      parent_id: mutant.parent_id, ancestor_ids: mutant.ancestor_ids,
    } : undefined,
  };
  
  return JSON.stringify(exportData, null, prettyPrint ? 2 : 0);
}

// -------- PineScript Export --------

export function exportToPineScript(mutant: Mutant, version: number = 5): string {
  const { params } = mutant;
  const ind = params.indicator;
  const indParams = params.entry_indicators[0]?.params || {};
  
  return `//@version=${version}
indicator("${params.name}", overlay=true)

// Strategy: ${params.name}
// Return: ${round(mutant.total_return_pct, 2)}% | Win: ${round(mutant.win_rate, 1)}% | Trades: ${mutant.total_trades}
// Sharpe: ${round(mutant.sharpe_ratio, 2)} | DD: ${round(mutant.max_drawdown, 2)}%

// Inputs
stopLoss = input.float(${params.risk.stop_loss_pct}, "Stop Loss %") / 100
takeProfit = input.float(${params.risk.take_profit_pct}, "Take Profit %") / 100
posSize = input.float(${params.risk.position_size_pct}, "Position Size %")

${ind === 'RSI' ? `rsiPeriod = input.int(${indParams.period || 14}, "RSI Period")
rsiOversold = input.float(${indParams.oversold || 30}, "RSI Oversold")
rsiOverbought = input.float(${indParams.overbought || 70}, "RSI Overbought")
rsiValue = ta.rsi(close, rsiPeriod)` : ''}
${ind === 'MACD' ? `macdFast = input.int(${indParams.fast || 12}, "MACD Fast")
macdSlow = input.int(${indParams.slow || 26}, "MACD Slow")
macdSignal = input.int(${indParams.signal || 9}, "MACD Signal")
[macdLine, signalLine, _] = ta.macd(close, macdFast, macdSlow, macdSignal)` : ''}
${ind === 'BB' ? `bbPeriod = input.int(${indParams.period || 20}, "BB Period")
bbStdDev = input.float(${indParams.stdDev || 2}, "BB StdDev")
[bbUpper, bbMiddle, bbLower] = ta.bb(close, bbPeriod, bbStdDev)` : ''}
${ind === 'SUPERTREND' ? `stPeriod = input.int(${indParams.period || 10}, "ST Period")
stMult = input.float(${indParams.multiplier || 3}, "ST Multiplier")
[stDir, stVal] = ta.supertrend(stMult, stPeriod)` : ''}

// Conditions
${ind === 'RSI' ? `longCond = rsiValue < rsiOversold
shortCond = rsiValue > rsiOverbought` : ''}
${ind === 'MACD' ? `longCond = ta.crossover(macdLine, signalLine)
shortCond = ta.crossunder(macdLine, signalLine)` : ''}
${ind === 'BB' ? `longCond = close < bbLower
shortCond = close > bbUpper` : ''}
${ind === 'SUPERTREND' ? `longCond = ta.change(stDir) < 0
shortCond = ta.change(stDir) > 0` : ''}

// Strategy
if (longCond${params.trade_side === 'short' ? ' && false' : ''})
    strategy.entry("Long", strategy.long, qty_percent=posSize)
if (shortCond${params.trade_side === 'long' ? ' && false' : ''})
    strategy.entry("Short", strategy.short, qty_percent=posSize)

strategy.exit("Long Exit", "Long", stop=strategy.position_avg_price*(1-stopLoss), limit=strategy.position_avg_price*(1+takeProfit))
strategy.exit("Short Exit", "Short", stop=strategy.position_avg_price*(1+stopLoss), limit=strategy.position_avg_price*(1-takeProfit))

// Plots
${ind === 'RSI' ? `plot(rsiValue, "RSI", color=color.purple)
hline(rsiOversold, "OS", color=color.green)
hline(rsiOverbought, "OB", color=color.red)` : ''}
${ind === 'SUPERTREND' ? `plot(stVal, "ST", color=stDir>0?color.red:color.green, linewidth=2)` : ''}
`;
}

// -------- Python Export --------

export function exportToPython(mutant: Mutant, result: BacktestResult): string {
  const { params } = mutant;
  const indParams = params.entry_indicators[0]?.params || {};
  
  return `#!/usr/bin/env python3
"""
Strategy: ${params.name}
Generated by QuantCore Genetic Engine
Return: ${round(mutant.total_return_pct, 2)}% | Win: ${round(mutant.win_rate, 1)}% | Sharpe: ${round(mutant.sharpe_ratio, 2)}
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    stop_loss_pct: float = ${params.risk.stop_loss_pct}
    take_profit_pct: float = ${params.risk.take_profit_pct}
    position_size_pct: float = ${params.risk.position_size_pct}
    use_trailing_stop: bool = ${String(params.risk.use_trailing_stop).toLowerCase()}
    volume_filter: bool = ${String(params.filters.volume_confirmation).toLowerCase()}
    ${params.indicator === 'RSI' ? `rsi_period: int = ${indParams.period || 14}
    rsi_oversold: float = ${indParams.oversold || 30}
    rsi_overbought: float = ${indParams.overbought || 70}` : ''}
    ${params.indicator === 'MACD' ? `macd_fast: int = ${indParams.fast || 12}
    macd_slow: int = ${indParams.slow || 26}
    macd_signal: int = ${indParams.signal || 9}` : ''}

STATS = {
    'total_return_pct': ${round(result.total_return_pct, 2)},
    'sharpe_ratio': ${round(result.sharpe_ratio, 2)},
    'win_rate': ${round(result.win_rate, 2)},
    'profit_factor': ${round(result.profit_factor, 2)},
    'max_drawdown_pct': ${round(result.max_drawdown_pct, 2)},
    'total_trades': ${result.total_trades},
    'winning_trades': ${result.winning_trades},
    'losing_trades': ${result.losing_trades},
    'avg_win': ${round(result.avg_win, 2)},
    'avg_loss': ${round(result.avg_loss, 2)},
    'expectancy': ${round(result.expectancy, 2)},
}

class Strategy:
    def __init__(self, config: Config):
        self.config = config
        self.position = None
        self.trades: List[dict] = []
        self.balance = 10000.0
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))
    
    def calculate_macd(self, prices: pd.Series):
        ema_fast = prices.ewm(span=self.config.macd_fast).mean()
        ema_slow = prices.ewm(span=self.config.macd_slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.config.macd_signal).mean()
        return macd, signal
    
    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        if idx < 50: return None
        ${params.indicator === 'RSI' ? `rsi = self.calculate_rsi(df['close'], self.config.rsi_period)
        if rsi.iloc[idx] < self.config.rsi_oversold: return 'long'
        if rsi.iloc[idx] > self.config.rsi_overbought: return 'short'` : ''}
        ${params.indicator === 'MACD' ? `macd, signal = self.calculate_macd(df['close'])
        if macd.iloc[idx-1] <= signal.iloc[idx-1] and macd.iloc[idx] > signal.iloc[idx]: return 'long'
        if macd.iloc[idx-1] >= signal.iloc[idx-1] and macd.iloc[idx] < signal.iloc[idx]: return 'short'` : ''}
        return None
    
    def run_backtest(self, df: pd.DataFrame) -> dict:
        for i in range(1, len(df)):
            signal = self.generate_signal(df, i)
            price = df['close'].iloc[i]
            if self.position is None and signal:
                size = (self.balance * self.config.position_size_pct / 100) / price
                self.position = {'entry': price, 'side': signal, 'size': size, 'bar': i}
            if self.position:
                pnl_pct = (price - self.position['entry'])/self.position['entry']*100 if self.position['side']=='long' else (self.position['entry']-price)/self.position['entry']*100
                if pnl_pct <= -self.config.stop_loss_pct or pnl_pct >= self.config.take_profit_pct:
                    pnl = (price - self.position['entry'])*self.position['size'] if self.position['side']=='long' else (self.position['entry']-price)*self.position['size']
                    self.trades.append({**self.position, 'exit': price, 'pnl': pnl, 'pnl_pct': pnl_pct})
                    self.balance += pnl
                    self.position = None
        return {'return_pct': (self.balance-10000)/100, 'trades': len(self.trades), 'balance': self.balance}

if __name__ == '__main__':
    config = Config()
    strat = Strategy(config)
    # df = pd.read_csv('data.csv')
    # results = strat.run_backtest(df)
`;
}

// -------- CSV Export --------

export function exportTradesToCSV(trades: Trade[]): string {
  const headers = 'ID,Entry,Exit,Side,Size,PnL,PnL%,Entry Time,Exit Time,Bars Held';
  const rows = trades.map(t => `${t.id},${round(t.entry,2)},${round(t.exit,2)},${t.side},${round(t.size,6)},${round(t.pnl,2)},${round(t.pnlPercent,2)},${new Date(t.entryTime).toISOString()},${new Date(t.exitTime).toISOString()},${t.barsHeld}`);
  return [headers, ...rows].join('\n');
}

// -------- Statistics Summary --------

export function generateStatisticsSummary(mutant: Mutant, result: BacktestResult): string {
  const s = result;
  return `
╔═══════════════════════════════════════════════════════════╗
║  STRATEGY: ${mutant.params.name.substring(0, 48).padEnd(48)}║
╠═══════════════════════════════════════════════════════════╣
║  Return: ${String(round(s.total_return_pct, 2)) + '%'.padEnd(8)} | Sharpe: ${String(round(s.sharpe_ratio, 2)).padEnd(8)} | Win: ${String(round(s.win_rate, 1)) + '%'.padEnd(7)}║
║  Trades: ${String(s.total_trades).padEnd(8)} | Wins: ${String(s.winning_trades).padEnd(6)} | Losses: ${String(s.losing_trades).padEnd(6)}        ║
║  Avg Win: $${String(round(s.avg_win, 2)).padEnd(7)} | Avg Loss: $${String(round(s.avg_loss, 2)).padEnd(7)}            ║
║  Max DD: ${String(round(s.max_drawdown_pct, 2)) + '%'.padEnd(8)} | Profit Factor: ${String(round(s.profit_factor, 2)).padEnd(6)}             ║
║  Gen: ${String(mutant.generation).padEnd(4)} | Fitness: ${String(round(mutant.fitness, 2)).padEnd(8)} | Mutations: ${String(mutant.mutation_count).padEnd(4)}           ║
╚═══════════════════════════════════════════════════════════╝
`.trim();
}

// -------- Population Export --------

export function exportPopulationToJSON(population: Mutant[], results: Map<string, BacktestResult>): string {
  const exports = population.map(m => {
    const r = results.get(m.id);
    return r ? JSON.parse(exportToJSON(m, r)) : null;
  }).filter(Boolean);
  return JSON.stringify({ exported_at: new Date().toISOString(), count: exports.length, strategies: exports }, null, 2);
}
