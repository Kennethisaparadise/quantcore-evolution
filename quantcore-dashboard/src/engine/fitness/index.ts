// ============================================
// QUANTCORE GENETIC ENGINE - FITNESS FUNCTIONS
// ============================================

import type { BacktestResult } from '../backtest';
import type { FitnessComponents, Mutant } from '../types';

// -------- Fitness Calculator --------

export interface FitnessWeights {
  return_weight: number;
  sharpe_weight: number;
  winrate_weight: number;
  drawdown_weight: number;
  profit_factor_weight: number;
  robustness_weight: number;
  trades_weight: number;
}

export const DEFAULT_FITNESS_WEIGHTS: FitnessWeights = {
  return_weight: 0.25,
  sharpe_weight: 0.25,
  winrate_weight: 0.15,
  drawdown_weight: 0.15,
  profit_factor_weight: 0.10,
  robustness_weight: 0.05,
  trades_weight: 0.05,
};

export function calculateFitness(result: BacktestResult, weights: Partial<FitnessWeights> = {}): { fitness: number; components: FitnessComponents } {
  const w = { ...DEFAULT_FITNESS_WEIGHTS, ...weights };
  
  // Normalize return score (cap at 200%)
  const returnScore = Math.min(Math.abs(result.total_return_pct) / 100, 2) * 100;
  
  // Sharpe ratio score (cap at 3.0)
  const sharpeScore = Math.min(Math.max(result.sharpe_ratio, 0), 3) / 3 * 100;
  
  // Win rate score
  const winrateScore = result.win_rate;
  
  // Drawdown penalty (inverse - lower is better)
  const drawdownScore = Math.max(0, 100 - result.max_drawdown_pct * 2);
  
  // Profit factor score (cap at 3.0)
  const profitFactorScore = Math.min(result.profit_factor, 3) / 3 * 100;
  
  // Robustness: combination of consistency
  const consistencyScore = (result.win_rate * 0.5) + (result.profit_factor > 1 ? 50 : 0);
  const robustnessScore = Math.min(consistencyScore, 100);
  
  // Minimum trades bonus (prefer strategies with meaningful sample sizes)
  const tradesScore = result.total_trades >= 10 ? Math.min(result.total_trades / 50 * 100, 100) : result.total_trades / 10 * 100;
  
  const components: FitnessComponents = {
    return_score: returnScore,
    sharpe_score: sharpeScore,
    winrate_score: winrateScore,
    drawdown_score: drawdownScore,
    profit_factor_score: profitFactorScore,
    robustness_score: robustnessScore,
  };
  
  const fitness = 
    (returnScore * w.return_weight) +
    (sharpeScore * w.sharpe_weight) +
    (winrateScore * w.winrate_weight) +
    (drawdownScore * w.drawdown_weight) +
    (profitFactorScore * w.profit_factor_weight) +
    (robustnessScore * w.robustness_weight) +
    (tradesScore * w.trades_weight);
  
  return { fitness, components };
}

// -------- Multi-Objective Fitness (for NSGA2) --------

export interface MultiObjectiveFitness {
  return_fitness: number;
  sharpe_fitness: number;
  drawdown_fitness: number;
  winrate_fitness: number;
}

export function calculateMultiObjective(result: BacktestResult): MultiObjectiveFitness {
  return {
    return_fitness: result.total_return_pct,
    sharpe_fitness: result.sharpe_ratio,
    drawdown_fitness: -result.max_drawdown_pct, // Negative because lower is better
    winrate_fitness: result.win_rate,
  };
}

// -------- Diversity Score --------

export function calculateDiversity(population: Mutant[]): number {
  if (population.length < 2) return 1;
  
  let totalDiff = 0;
  let comparisons = 0;
  
  for (let i = 0; i < population.length; i++) {
    for (let j = i + 1; j < population.length; j++) {
      const diff = compareMutants(population[i], population[j]);
      totalDiff += diff;
      comparisons++;
    }
  }
  
  return comparisons > 0 ? totalDiff / comparisons : 1;
}

function compareMutants(a: Mutant, b: Mutant): number {
  let diff = 0;
  
  // Risk parameters
  diff += Math.abs(a.params.risk.stop_loss_pct - b.params.risk.stop_loss_pct) / 10;
  diff += Math.abs(a.params.risk.take_profit_pct - b.params.risk.take_profit_pct) / 20;
  diff += Math.abs(a.params.risk.position_size_pct - b.params.risk.position_size_pct) / 15;
  
  // Indicator parameters
  const aPeriod = (a.params.entry_indicators[0]?.params.period as number) || 14;
  const bPeriod = (b.params.entry_indicators[0]?.params.period as number) || 14;
  diff += Math.abs(aPeriod - bPeriod) / 30;
  
  // Trade side
  if (a.params.trade_side !== b.params.trade_side) diff += 1;
  
  // Filters
  if (a.params.filters.volume_confirmation !== b.params.filters.volume_confirmation) diff += 0.5;
  if (a.params.filters.regime_filter !== b.params.filters.regime_filter) diff += 0.5;
  
  return Math.min(diff, 1);
}

// -------- Fitness Ranker --------

export function rankByFitness(population: Mutant[]): Mutant[] {
  return [...population].sort((a, b) => b.fitness - a.fitness);
}

export function getBestMutant(population: Mutant[]): Mutant | null {
  if (population.length === 0) return null;
  return population.reduce((best, m) => m.fitness > best.fitness ? m : best);
}

export function getWorstMutant(population: Mutant[]): Mutant | null {
  if (population.length === 0) return null;
  return population.reduce((worst, m) => m.fitness < worst.fitness ? m : worst);
}

export function getAverageFitness(population: Mutant[]): number {
  if (population.length === 0) return 0;
  return population.reduce((sum, m) => sum + m.fitness, 0) / population.length;
}
