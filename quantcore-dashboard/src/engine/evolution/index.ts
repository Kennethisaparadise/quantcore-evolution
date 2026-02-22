// ============================================
// QUANTCORE GENETIC ENGINE - EVOLUTION ENGINE
// ============================================

import type { Mutant, StrategyParams, EvolutionConfig, EvolutionState, EvolutionHistoryEntry } from '../types';
import type { BacktestResult } from '../backtest';
import { runBacktest, generateMockData, type OHLCV } from '../backtest';
import { calculateFitness, calculateDiversity } from '../fitness';
import { tournamentSelect, selectElites } from '../selection';
import { crossover } from '../crossover';
import { MutationManager } from '../mutations';

export interface EvolutionCallbacks {
  onGeneration?: (state: EvolutionState) => void;
  onNewBest?: (mutant: Mutant) => void;
  onComplete?: (state: EvolutionState) => void;
  onError?: (error: Error) => void;
}

export class EvolutionEngine {
  private config: EvolutionConfig;
  private state: EvolutionState;
  private mutationManager: MutationManager;
  private callbacks: EvolutionCallbacks;
  private isRunning: boolean = false;
  private isPaused: boolean = false;
  private abortController: AbortController | null = null;
  private hallOfFame: Mutant[] = [];
  private bestFitnessHistory: number[] = [];

  constructor(config: Partial<EvolutionConfig> = {}, callbacks: EvolutionCallbacks = {}) {
    this.config = { population_size: 50, elitism_count: 5, generations: 50, mutation_rate: 0.3, adaptive_mutation: true, min_mutation_rate: 0.1, max_mutation_rate: 0.6, selection_method: 'tournament', tournament_size: 4, crossover_rate: 0.7, crossover_method: 'two_point', diversity_threshold: 0.3, niche_preservation: true, early_stopping: true, patience: 10, min_improvement: 0.01, log_interval: 1, save_all: false, ...config };
    this.callbacks = callbacks;
    this.mutationManager = new MutationManager();
    this.state = this.createInitialState();
  }

  private createInitialState(): EvolutionState {
    return { generation: 0, population: [], best_mutant: null, hall_of_fame: [], history: [], is_running: false, is_paused: false, progress: 0, elapsed_time: 0, fitness_over_time: [] };
  }

  initializePopulation(seedStrategies: StrategyParams[] = []): Mutant[] {
    const population: Mutant[] = [];
    seedStrategies.forEach((params, idx) => population.push(this.createMutant(params, 0, 'seed', `Seed ${idx + 1}`)));
    while (population.length < this.config.population_size) {
      const baseParams = seedStrategies.length > 0 ? JSON.parse(JSON.stringify(seedStrategies[Math.floor(Math.random() * seedStrategies.length)])) : this.createDefaultParams();
      let mutatedParams = { ...baseParams };
      let mutationDesc = '';
      for (let i = 0; i < 2; i++) { const op = this.mutationManager.selectMutation(); mutatedParams = op.apply(mutatedParams); mutationDesc = op.label; }
      population.push(this.createMutant(mutatedParams, 0, 'seed', `Random ${mutationDesc}`));
    }
    this.state.population = population;
    return population;
  }

  private createMutant(params: StrategyParams, generation: number, parentId: string, mutationDesc: string): Mutant {
    return { id: `gen${generation}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`, name: params.name, generation, parent_id: parentId === 'seed' ? null : parentId, ancestor_ids: [], params: { ...params, id: `strat_${Date.now()}`, updated_at: Date.now() }, fitness: 0, fitness_components: { return_score: 0, sharpe_score: 0, winrate_score: 0, drawdown_score: 0, profit_factor_score: 0, robustness_score: 0 }, total_return_pct: 0, sharpe_ratio: 0, sortino_ratio: 0, win_rate: 0, max_drawdown: 0, profit_factor: 0, calmar_ratio: 0, total_trades: 0, expectancy: 0, mutation_description: mutationDesc, mutation_count: 0, crossover_count: 0, created_at: Date.now() };
  }

  private createDefaultParams(): StrategyParams {
    const indicators = [
      { type: 'RSI' as const, params: { period: 14, oversold: 30, overbought: 70 } },
      { type: 'MACD' as const, params: { fast: 12, slow: 26, signal: 9 } },
      { type: 'BB' as const, params: { period: 20, stdDev: 2 } },
      { type: 'SUPERTREND' as const, params: { period: 10, multiplier: 3 } }
    ];
    const ind = indicators[Math.floor(Math.random() * indicators.length)];
    const params = ind.params as unknown as Record<string, number>;
    return { 
      id: `strat_${Date.now()}`, 
      name: `${ind.type} Strategy`, 
      version: '1.0.0', 
      created_at: Date.now(), 
      updated_at: Date.now(), 
      indicator: ind.type, 
      trade_side: Math.random() < 0.3 ? 'long' : Math.random() < 0.5 ? 'short' : 'both', 
      entry_indicators: [{ type: ind.type, enabled: true, params }], 
      exit_indicators: [], 
      filter_indicators: [], 
      risk: { stop_loss_pct: 2 + Math.random() * 4, take_profit_pct: 4 + Math.random() * 8, position_size_pct: 3 + Math.random() * 7, max_positions: 1, use_trailing_stop: Math.random() < 0.3, trailing_distance_pct: 1 + Math.random() * 3, use_time_exit: Math.random() < 0.2, max_bars_held: 10 + Math.floor(Math.random() * 20) }, 
      filters: { volume_confirmation: Math.random() < 0.4, volume_multiplier: 1.2 + Math.random() * 1.5, volatility_filter: false, volatility_threshold: 2, regime_filter: Math.random() < 0.3 ? 'bull' : Math.random() < 0.5 ? 'bear' : 'any', time_filter: { enabled: false, sessions: [] }, min_liquidity: 10000 }, 
      advanced: { use_compounding: false, compounding_method: 'fixed', kelly_fraction: 0.25, martingale_multiplier: 2, max_consecutive_losses: 5, partial_exit_enabled: false, partial_exit_pct: 50 }, 
      description: '', 
      tags: [], 
      author: 'QuantCore' 
    };
  }

  async evolve(data: OHLCV[] | string = 'BTC'): Promise<EvolutionState> {
    if (this.isRunning) throw new Error('Evolution already running');
    this.isRunning = true; this.state.is_running = true; this.abortController = new AbortController();
    const startTime = Date.now();
    const ohlcvData = Array.isArray(data) ? data : generateMockData(data);
    try {
      if (this.state.population.length === 0) this.initializePopulation();
      for (let gen = 0; gen < this.config.generations; gen++) {
        if (this.abortController.signal.aborted) break;
        while (this.isPaused) { if (this.abortController.signal.aborted) break; await new Promise(r => setTimeout(r, 100)); }
        await this.evaluatePopulation(ohlcvData);
        const currentBest = this.state.population.reduce((best, m) => m.fitness > best.fitness ? m : best);
        if (!this.state.best_mutant || currentBest.fitness > this.state.best_mutant.fitness) {
          this.state.best_mutant = { ...currentBest };
          this.hallOfFame.push({ ...currentBest });
          if (this.hallOfFame.length > 20) this.hallOfFame.sort((a, b) => b.fitness - a.fitness).pop();
          this.callbacks.onNewBest?.(this.state.best_mutant);
        }
        this.state.history.push({ generation: gen + 1, best_fitness: currentBest.fitness, avg_fitness: this.state.population.reduce((s, m) => s + m.fitness, 0) / this.state.population.length, worst_fitness: this.state.population.reduce((min, m) => Math.min(min, m.fitness), Infinity), diversity: calculateDiversity(this.state.population), best_return: currentBest.total_return_pct, best_sharpe: currentBest.sharpe_ratio });
        this.state.fitness_over_time.push(currentBest.fitness);
        this.bestFitnessHistory.push(currentBest.fitness);
        if (this.config.early_stopping && this.shouldStopEarly()) break;
        if (gen < this.config.generations - 1) this.state.population = this.createNextGeneration();
        this.state.generation = gen + 1; this.state.progress = ((gen + 1) / this.config.generations) * 100; this.state.elapsed_time = Date.now() - startTime;
        this.callbacks.onGeneration?.({ ...this.state });
        await new Promise(r => setTimeout(r, 10));
      }
      this.state.is_running = false; this.state.hall_of_fame = this.hallOfFame;
      this.callbacks.onComplete?.(this.state);
    } catch (error) { this.state.is_running = false; this.callbacks.onError?.(error as Error); }
    this.isRunning = false;
    return this.state;
  }

  private async evaluatePopulation(data: OHLCV[]): Promise<void> {
    for (const mutant of this.state.population) {
      try {
        const result = runBacktest(mutant.params, data);
        const { fitness, components } = calculateFitness(result);
        Object.assign(mutant, { fitness, fitness_components: components, total_return_pct: result.total_return_pct, sharpe_ratio: result.sharpe_ratio, sortino_ratio: result.sortino_ratio, win_rate: result.win_rate, max_drawdown: result.max_drawdown, profit_factor: result.profit_factor, calmar_ratio: result.calmar_ratio, total_trades: result.total_trades, expectancy: result.expectancy, backtest_result: result });
      } catch (e) { mutant.fitness = 0; }
    }
    this.state.population.sort((a, b) => b.fitness - a.fitness);
  }

  private createNextGeneration(): Mutant[] {
    const nextGen: Mutant[] = [];
    const elites = selectElites(this.state.population, this.config.elitism_count);
    nextGen.push(...elites.map(e => ({ ...e, id: `${e.id}_elite` })));
    while (nextGen.length < this.config.population_size) {
      const parent1 = tournamentSelect(this.state.population, this.config.tournament_size);
      const parent2 = tournamentSelect(this.state.population, this.config.tournament_size);
      let child = Math.random() < this.config.crossover_rate ? crossover(parent1, parent2, this.config.crossover_method) : { ...parent1, id: `${parent1.id}_copy`, generation: parent1.generation + 1 };
      if (Math.random() < this.config.mutation_rate) { const op = this.mutationManager.selectMutation(); child.params = op.apply(child.params); child.mutation_description = op.label; child.mutation_count = (child.mutation_count || 0) + 1; }
      child.generation = this.state.generation + 1; child.fitness = 0;
      nextGen.push(child);
    }
    return nextGen;
  }

  private shouldStopEarly(): boolean {
    if (this.bestFitnessHistory.length < this.config.patience + 1) return false;
    const recent = this.bestFitnessHistory.slice(-this.config.patience);
    return (recent[recent.length - 1] - recent[0]) < this.config.min_improvement * recent[0];
  }

  pause(): void { this.isPaused = true; this.state.is_paused = true; }
  resume(): void { this.isPaused = false; this.state.is_paused = false; }
  stop(): void { this.abortController?.abort(); this.isRunning = false; this.isPaused = false; this.state.is_running = false; }
  reset(): void { this.stop(); this.state = this.createInitialState(); this.hallOfFame = []; this.bestFitnessHistory = []; }
  getState(): EvolutionState { return { ...this.state }; }
  getBestMutant(): Mutant | null { return this.state.best_mutant; }
  getHallOfFame(): Mutant[] { return [...this.hallOfFame]; }
  getHistory(): EvolutionHistoryEntry[] { return [...this.state.history]; }
}

export default EvolutionEngine;
