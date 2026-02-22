// ============================================
// QUANTCORE GENETIC ENGINE - SELECTION STRATEGIES
// ============================================

import type { Mutant, SelectionMethod } from '../types';

// -------- Tournament Selection --------

export function tournamentSelect(population: Mutant[], tournamentSize: number): Mutant {
  const tournament: Mutant[] = [];
  const used = new Set<number>();
  
  while (tournament.length < tournamentSize) {
    const idx = Math.floor(Math.random() * population.length);
    if (!used.has(idx)) {
      used.add(idx);
      tournament.push(population[idx]);
    }
  }
  
  return tournament.reduce((best, m) => m.fitness > best.fitness ? m : best);
}

// -------- Roulette Wheel Selection --------

export function rouletteSelect(population: Mutant[]): Mutant {
  const totalFitness = population.reduce((sum, m) => sum + Math.max(0, m.fitness), 0);
  if (totalFitness === 0) return population[Math.floor(Math.random() * population.length)];
  
  let rand = Math.random() * totalFitness;
  
  for (const mutant of population) {
    rand -= Math.max(0, mutant.fitness);
    if (rand <= 0) return mutant;
  }
  
  return population[population.length - 1];
}

// -------- Rank Selection --------

export function rankSelect(population: Mutant[]): Mutant {
  const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
  const n = sorted.length;
  
  // Linear rank weights
  const weights = sorted.map((_, i) => n - i);
  const totalWeight = weights.reduce((s, w) => s + w, 0);
  
  let rand = Math.random() * totalWeight;
  
  for (let i = 0; i < n; i++) {
    rand -= weights[i];
    if (rand <= 0) return sorted[i];
  }
  
  return sorted[n - 1];
}

// -------- Selection Factory --------

export function select(population: Mutant[], method: SelectionMethod, tournamentSize: number = 4): Mutant {
  switch (method) {
    case 'tournament':
      return tournamentSelect(population, tournamentSize);
    case 'roulette':
      return rouletteSelect(population);
    case 'rank':
      return rankSelect(population);
    default:
      return tournamentSelect(population, tournamentSize);
  }
}

// -------- Elitism --------

export function selectElites(population: Mutant[], count: number): Mutant[] {
  const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
  return sorted.slice(0, count);
}

// -------- NSGA2 Selection (simplified) --------

export interface NSGA2Rank {
  mutant: Mutant;
  rank: number;
  crowdingDistance: number;
}

export function assignNSGA2Ranks(population: Mutant[]): NSGA2Rank[] {
  const objectives = ['total_return_pct', 'sharpe_ratio', 'win_rate', 'max_drawdown'];
  
  // Assign Pareto ranks
  const ranks: NSGA2Rank[] = population.map(m => ({ mutant: m, rank: 0, crowdingDistance: 0 }));
  let currentRank = 0;
  
  while (true) {
    const dominated: number[] = [];
    
    for (let i = 0; i < ranks.length; i++) {
      if (ranks[i].rank > 0) continue;
      
      let isDominated = false;
      for (let j = 0; j < ranks.length; j++) {
        if (i === j || ranks[j].rank > 0) continue;
        
        let dominates = true;
        for (const obj of objectives) {
          const vi = (ranks[i].mutant as any)[obj];
          const vj = (ranks[j].mutant as any)[obj];
          if (obj === 'max_drawdown') {
            if (vi < vj) { dominates = false; break; }
          } else {
            if (vi < vj) { dominates = false; break; }
          }
        }
        if (dominates) { isDominated = true; break; }
      }
      
      if (!isDominated) {
        dominated.push(i);
      }
    }
    
    if (dominated.length === 0) break;
    
    for (const idx of dominated) {
      ranks[idx].rank = currentRank;
    }
    currentRank++;
  }
  
  // Calculate crowding distance for each rank
  const rankGroups = new Map<number, Mutant[]>();
  for (const r of ranks) {
    if (!rankGroups.has(r.rank)) rankGroups.set(r.rank, []);
    rankGroups.get(r.rank)!.push(r.mutant);
  }
  
  for (const [rank, mutants] of rankGroups) {
    const rankIndices = ranks.filter(r => r.rank === rank);
    
    for (const obj of objectives) {
      const sorted = [...mutants].sort((a, b) => (b as any)[obj] - (a as any)[obj]);
      
      // Boundary points get infinite distance
      rankIndices.find(r => r.mutant.id === sorted[0].id)!.crowdingDistance = Infinity;
      rankIndices.find(r => r.mutant.id === sorted[sorted.length - 1].id)!.crowdingDistance = Infinity;
      
      // Calculate for others
      const objRange = (sorted[0] as any)[obj] - (sorted[sorted.length - 1] as any)[obj];
      if (objRange === 0) continue;
      
      for (let i = 1; i < sorted.length - 1; i++) {
        const idx = rankIndices.findIndex(r => r.mutant.id === sorted[i].id)!;
        const distance = ((sorted[i + 1] as any)[obj] - (sorted[i - 1] as any)[obj]) / objRange;
        rankIndices[idx].crowdingDistance += distance;
      }
    }
  }
  
  return ranks;
}

export function nsga2Select(population: Mutant[], count: number): Mutant[] {
  const ranked = assignNSGA2Ranks(population);
  
  return ranked
    .sort((a, b) => {
      if (a.rank !== b.rank) return a.rank - b.rank;
      return b.crowdingDistance - a.crowdingDistance;
    })
    .slice(0, count)
    .map(r => r.mutant);
}
