// ============================================
// QUANTCORE GENETIC ENGINE - MAIN EXPORTS
// ============================================

// Types
export * from './types';

// Backtest
export { runBacktest, generateMockData, calculateRSI, calculateEMA, calculateSMA, calculateBB, calculateMACD, calculateATR, calculateSupertrend, generateEntrySignal } from './backtest';
export type { OHLCV, Trade, BacktestResult, Position } from './backtest';

// Mutations
export { MutationManager, ALL_MUTATIONS } from './mutations';
export type { MutationOperator } from './mutations';

// Fitness
export { calculateFitness, calculateDiversity, rankByFitness, getBestMutant, getAverageFitness, DEFAULT_FITNESS_WEIGHTS } from './fitness';
export type { FitnessWeights, MultiObjectiveFitness } from './fitness';

// Selection
export { tournamentSelect, rouletteSelect, rankSelect, select, selectElites, nsga2Select, assignNSGA2Ranks } from './selection';

// Crossover
export { crossover, singlePointCrossover, twoPointCrossover, uniformCrossover, linearCrossover, simulatedBinaryCrossover } from './crossover';

// Evolution
export { EvolutionEngine } from './evolution';
export type { EvolutionCallbacks } from './evolution';

// Export
export { exportToJSON, exportToPineScript, exportToPython, exportTradesToCSV, generateStatisticsSummary, exportPopulationToJSON } from './export';
export type { ExportOptions } from './export';

// Default config
export { DEFAULT_EVOLUTION_CONFIG, createDefaultStrategyParams, MUTATION_CATEGORIES } from './types';
