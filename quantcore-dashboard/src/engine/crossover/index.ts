// ============================================
// QUANTCORE GENETIC ENGINE - CROSSOVER OPERATORS
// ============================================

import type { StrategyParams, Mutant, CrossoverMethod } from '../types';

// -------- Utility --------

const randomChoice = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const randomKey = <T extends object>(obj: T): keyof T => randomChoice(Object.keys(obj) as (keyof T)[]);

// -------- Single Point Crossover --------

export function singlePointCrossover(parent1: Mutant, parent2: Mutant): Mutant {
  const child1Params = JSON.parse(JSON.stringify(parent1.params));
  const child2Params = JSON.parse(JSON.stringify(parent2.params));
  
  // Swap at parameter level
  const keys1 = Object.keys(child1Params.risk);
  const keys2 = Object.keys(child2Params.risk);
  const swapKey = randomChoice([...keys1, ...keys2]) as keyof typeof child1Params.risk;
  
  if (swapKey && swapKey in child1Params.risk && swapKey in child2Params.risk) {
    const temp = (child1Params.risk as any)[swapKey];
    (child1Params.risk as any)[swapKey] = (child2Params.risk as any)[swapKey];
    (child2Params.risk as any)[swapKey] = temp;
  }
  
  return createChild(parent1, parent2, child1Params, 'single_point');
}

// -------- Two Point Crossover --------

export function twoPointCrossover(parent1: Mutant, parent2: Mutant): Mutant {
  const childParams = JSON.parse(JSON.stringify(parent1.params));
  const parent2Params = JSON.parse(JSON.stringify(parent2.params));
  
  // Two point crossover on risk parameters
  const riskKeys = Object.keys(childParams.risk) as (keyof typeof childParams.risk)[];
  const idx1 = Math.floor(Math.random() * riskKeys.length);
  const idx2 = idx1 + Math.floor(Math.random() * (riskKeys.length - idx1));
  
  for (let i = idx1; i < idx2 && i < riskKeys.length; i++) {
    const key = riskKeys[i];
    const temp = (childParams.risk as any)[key];
    (childParams.risk as any)[key] = (parent2Params.risk as any)[key];
    (parent2Params.risk as any)[key] = temp;
  }
  
  return createChild(parent1, parent2, childParams, 'two_point');
}

// -------- Uniform Crossover --------

export function uniformCrossover(parent1: Mutant, parent2: Mutant): Mutant {
  const childParams = JSON.parse(JSON.stringify(parent1.params));
  const parent2Params = JSON.parse(JSON.stringify(parent2.params));
  
  // Randomly swap each risk parameter
  for (const key of Object.keys(childParams.risk) as (keyof typeof childParams.risk)[]) {
    if (Math.random() < 0.5) {
      const temp = (childParams.risk as any)[key];
      (childParams.risk as any)[key] = (parent2Params.risk as any)[key];
      (parent2Params.risk as any)[key] = temp;
    }
  }
  
  // Also try indicator parameters
  if (childParams.entry_indicators.length > 0 && parent2Params.entry_indicators.length > 0) {
    for (const key of Object.keys(childParams.entry_indicators[0].params) as (keyof any)[]) {
      if (Math.random() < 0.4 && key in parent2Params.entry_indicators[0].params) {
        const temp = childParams.entry_indicators[0].params[key];
        childParams.entry_indicators[0].params[key] = parent2Params.entry_indicators[0].params[key];
        parent2Params.entry_indicators[0].params[key] = temp;
      }
    }
  }
  
  return createChild(parent1, parent2, childParams, 'uniform');
}

// -------- Linear Crossover --------

export function linearCrossover(parent1: Mutant, parent2: Mutant): Mutant {
  const childParams = JSON.parse(JSON.stringify(parent1.params));
  const parent2Params = JSON.parse(JSON.stringify(parent2.params));
  
  const alpha = Math.random();
  
  // Blend risk parameters
  for (const key of Object.keys(childParams.risk) as (keyof typeof childParams.risk)[]) {
    const v1 = (childParams.risk as any)[key];
    const v2 = (parent2Params.risk as any)[key];
    if (typeof v1 === 'number' && typeof v2 === 'number') {
      (childParams.risk as any)[key] = v1 * alpha + v2 * (1 - alpha);
    }
  }
  
  // Blend indicator parameters
  if (childParams.entry_indicators.length > 0 && parent2Params.entry_indicators.length > 0) {
    for (const key of Object.keys(childParams.entry_indicators[0].params) as (keyof any)[]) {
      const v1 = childParams.entry_indicators[0].params[key];
      const v2 = parent2Params.entry_indicators[0].params[key];
      if (typeof v1 === 'number' && typeof v2 === 'number') {
        childParams.entry_indicators[0].params[key] = v1 * alpha + v2 * (1 - alpha);
      }
    }
  }
  
  return createChild(parent1, parent2, childParams, 'linear');
}

// -------- Simulated Binary Crossover (SBX) --------

export function simulatedBinaryCrossover(parent1: Mutant, parent2: Mutant): Mutant {
  const childParams = JSON.parse(JSON.stringify(parent1.params));
  const parent2Params = JSON.parse(JSON.stringify(parent2.params));
  
  const eta = 20; // Distribution index
  
  // SBX for risk parameters
  for (const key of Object.keys(childParams.risk) as (keyof typeof childParams.risk)[]) {
    const v1 = (childParams.risk as any)[key];
    const v2 = (parent2Params.risk as any)[key];
    if (typeof v1 === 'number' && typeof v2 === 'number') {
      if (Math.random() < 0.5) {
        const diff = Math.abs(v1 - v2);
        if (diff > 0.0001) {
          let u = Math.random();
          const gamma = u <= 0.5 
            ? Math.pow(2 * u, 1 / (eta + 1))
            : Math.pow(1 / (2 - 2 * u), 1 / (eta + 1));
          const newV1 = 0.5 * ((1 + gamma) * v1 + (1 - gamma) * v2);
          const newV2 = 0.5 * ((1 - gamma) * v1 + (1 + gamma) * v2);
          (childParams.risk as any)[key] = newV1;
          (parent2Params.risk as any)[key] = newV2;
        }
      }
    }
  }
  
  return createChild(parent1, parent2, childParams, 'simulated_binary');
}

// -------- Helper Functions --------

function createChild(parent1: Mutant, parent2: Mutant, params: StrategyParams, method: CrossoverMethod): Mutant {
  return {
    id: `gen${parent1.generation + 1}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    name: `${parent1.name} × ${parent2.name}`,
    generation: Math.max(parent1.generation, parent2.generation) + 1,
    parent_id: parent1.id,
    ancestor_ids: [...new Set([...parent1.ancestor_ids, parent1.id, parent2.id])].slice(-20),
    params: {
      ...params,
      id: `strat_${Date.now()}`,
      created_at: Date.now(),
      updated_at: Date.now(),
    },
    fitness: 0,
    fitness_components: { return_score: 0, sharpe_score: 0, winrate_score: 0, drawdown_score: 0, profit_factor_score: 0, robustness_score: 0 },
    total_return_pct: 0,
    sharpe_ratio: 0,
    sortino_ratio: 0,
    win_rate: 0,
    max_drawdown: 0,
    profit_factor: 0,
    calmar_ratio: 0,
    total_trades: 0,
    expectancy: 0,
    mutation_description: `Crossover (${method})`,
    mutation_count: 0,
    crossover_count: (parent1.crossover_count || 0) + (parent2.crossover_count || 0) + 1,
    created_at: Date.now(),
  };
}

// -------- Crossover Factory --------

export function crossover(parent1: Mutant, parent2: Mutant, method: CrossoverMethod): Mutant {
  switch (method) {
    case 'single_point':
      return singlePointCrossover(parent1, parent2);
    case 'two_point':
      return twoPointCrossover(parent1, parent2);
    case 'uniform':
      return uniformCrossover(parent1, parent2);
    case 'linear':
      return linearCrossover(parent1, parent2);
    case 'simulated_binary':
      return simulatedBinaryCrossover(parent1, parent2);
    default:
      return twoPointCrossover(parent1, parent2);
  }
}
