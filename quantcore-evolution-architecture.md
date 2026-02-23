# QuantCore Evolution Engine - Architecture Document

**Version:** 1.0  
**Date:** 2026-02-22  
**Status:** Draft

---

## 1. Overview

The QuantCore Evolution Engine is a genetic algorithm-based system for discovering, optimizing, and evolving quantitative trading strategies. It combines multiple genetic operators, multi-objective fitness evaluation, and regime-aware selection to generate robust trading strategies across market conditions.

### 1.1 Core Philosophy
- **First-principles evolution**: Strategies evolve through natural selection, not brute-force parameter sweeps
- **Diversity preservation**: Pareto-optimal fronts maintain variety in the population
- **Regime awareness**: Fitness evaluation accounts for different market conditions (bull, bear, sideways)
- **Continuous adaptation**: Self-tuning mutation rates and meta-learning components

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      QuantCore Evolution Engine                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Frontend   │  │   Backend    │  │   Compute Cluster   │ │
│  │   (React)    │◄─┤   (Node.js)  │◄─┤   (Worker Pool)     │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│         │                  │                     │             │
│         ▼                  ▼                     ▼             │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Persistence Layer (localStorage/API)        │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Components

| Component | Responsibility |
|-----------|----------------|
| **Frontend** | UI for configuration, visualization, strategy management |
| **Backend** | Evolution loop, genetic operators, fitness evaluation |
| **Worker Pool** | Parallel strategy evaluation across multiple workers |
| **Persistence** | Strategy storage, run history, lineage tracking |

---

## 3. Core Data Structures

### 3.1 Genotype (Strategy)

```typescript
interface Strategy {
  id: string;
  name: string;
  version: string;
  genes: Gene[];
  createdAt: number;
  lineage: {
    parentId?: string;
    generation: number;
    mutations: MutationLog[];
  };
  fitness?: Fitness;
}

interface Gene {
  category: 'entry' | 'exit' | 'indicator' | 'filter' | 'stop' | 'position';
  type: string;
  params: Record<string, number | string | boolean>;
  enabled: boolean;
}
```

### 3.2 Fitness

```typescript
interface Fitness {
  return: number;        // Total return %
  sharpe: number;        // Sharpe ratio
  maxDrawdown: number;   // Maximum drawdown %
  winRate: number;        // Win rate %
  trades: number;        // Total trades
  robustness: number;    // Regime consistency score
  paretoRank: number;    // Dominance ranking
  crowdingDistance: number; // Diversity metric
}
```

---

## 4. Evolution Loop

### 4.1 Epoch Cycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Initialize │ ──► │  Evaluate   │ ──► │  Select     │ ──► │ Reproduce   │
│  Population │     │  Fitness    │     │  Parents    │     │  & Mutate   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                                                         │
       │     ┌──────────────────────────────────────────────────┘
       ▼     ▼
┌─────────────┐
│  Converged?  │ ──► (Yes) ──► Terminate / Export Best
└─────────────┘
       │
       │ (No)
       ▼
┌─────────────┐
│  Next Gen   │ ──► [Evaluate]
└─────────────┘
```

### 4.2 Generation Steps

1. **Initialization**: Create N random strategies from gene pool
2. **Evaluation**: Run backtest for each strategy, compute fitness
3. **Ranking**: Apply Pareto dominance, compute crowding distance
4. **Selection**: Tournament selection with fitness + diversity weighting
5. **Crossover**: Blend parent strategies
6. **Mutation**: Apply operators with configured probability
7. **Replacement**: Elitism (keep top K) + new offspring
8. **Convergence Check**: Monitor diversity, fitness plateau

---

## 5. Genetic Operators

### 5.1 Mutation Categories

| Category | Operators | Description |
|----------|-----------|-------------|
| **Entry** | `entry_rsi_cross`, `entry_macd_cross`, `entry_breakout` | Entry signal generation |
| **Exit** | `exit_rsi_cross`, `exit_trailing_stop`, `exit_time_based` | Exit signal generation |
| **Indicator** | `add_sma`, `add_ema`, `add_rsi`, `add_macd` | Technical indicators |
| **Filter** | `filter_adx`, `filter_volume`, `filter_regime` | Market condition filters |
| **Stop** | `stop_atr`, `stop_percentage`, `stop_sar` | Stop-loss mechanisms |
| **Position** | `position_kelly`, `position_fixed`, `position_martingale` | Position sizing |

### 5.2 Mutation Operators (50+)

#### Core Mutations
- `mutate_threshold`: Adjust indicator thresholds
- `mutate_indicator_period`: Change lookback periods
- `mutate_indicator_type`: Swap indicator family (SMA ↔ EMA)

#### Position Management
- `mutate_kelly_fraction`: Adjust Kelly criterion multiplier
- `mutate_martingale_factor`: Change doubling factor
- `mutate_pyramiding`: Enable/disable pyramid scaling

#### Signal Modifications
- `invert_signal`: Flip entry/exit logic
- `combine_signals`: AND/OR multiple signals
- `add_divergence`: Detect indicator divergence

#### Stop Enhancements
- `add_trailing_stop`: Implement trailing stop
- `adjust_atr_multiplier`: Modify ATR-based stops
- `add_chandelier_stop`: Chandelier exit implementation

#### Filters
- `add_adx_filter`: Only trade when ADX > threshold
- `add_regime_filter`: Filter by detected regime
- `add_volume_spike`: Volume confirmation requirement

### 5.3 Crossover Operators

```typescript
// Uniform crossover - random gene selection from either parent
function uniformCrossover(parentA: Strategy, parentB: Strategy): Strategy

// Segment crossover - swap gene segments
function segmentCrossover(parentA: Strategy, parentB: Strategy, start: number, end: number): Strategy

// Parameter crossover - blend numeric parameters
function parameterCrossover(parentA: Strategy, parentB: Strategy): Strategy
```

### 5.4 Selection

```typescript
// Tournament selection with diversity weighting
function tournamentSelect(population: Strategy[], k: number, diversityWeight: number): Strategy

// Rank-based selection
function rankSelect(population: Strategy[]): Strategy

// Fitness proportional (roulette wheel)
function fitnessProportionalSelect(population: Strategy[]): Strategy
```

---

## 6. Fitness Engine

### 6.1 Multi-Objective Evaluation

The fitness engine evaluates strategies across multiple objectives:

```typescript
interface FitnessWeights {
  return: number;      // Weight for total return
  sharpe: number;      // Weight for risk-adjusted return
  drawdown: number;   // Penalty for large drawdowns
  robustness: number; // Weight for regime consistency
  trades: number;     // Preference for more/less trading
}
```

### 6.2 Backtest Runner

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Market    │ ──► │   Signal   │ ──► │   Order     │ ──► │   P&L       │
│    Data     │     │  Generator  │     │  Execution  │     │  Calculator │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **Data Fetch**: Load OHLCV data (Binance API)
2. **Signal Generation**: Apply strategy genes to generate entry/exit
3. **Execution**: Simulate order fills (with slippage/fees)
4. **P&L Calculation**: Compute equity curve, returns, metrics

### 6.3 Regime Detection

```typescript
type Regime = 'bull' | 'bear' | 'sideways';

function detectRegime(prices: number[]): Regime {
  // SMA slope + volatility analysis
  // Returns detected market regime
}
```

Fitness includes regime-specific scoring to reward strategies that perform across conditions.

---

## 7. Diversity & Convergence

### 7.1 Pareto Dominance

```
Individual A dominates B if:
  ∀ objective: A.score ≥ B.score
  ∧ ∃ objective: A.score > B.score
```

### 7.2 Crowding Distance

Measures how close an individual is to its neighbors in objective space. Higher distance = more diverse.

### 7.3 Convergence Detection

```typescript
interface ConvergenceMetrics {
  avgFitnessChange: number;    // Fitness delta over last N generations
  paretoFrontSpread: number;   // Diversity of Pareto front
  populationDiversity: number; // Gene pool variety
}

function isConverged(metrics: ConvergenceMetrics, thresholds: Thresholds): boolean
```

---

## 8. Self-Tuning (Meta-Learning)

### 8.1 Adaptive Mutation Rate

Uses Thompson Sampling to balance exploration vs exploitation:

```typescript
// Track mutation operator success rates
interface MutationStats {
  operator: string;
  successes: number;  // Improved fitness
  attempts: number;
}

// Thompson Sampling for operator selection
function selectMutationOperator(stats: MutationStats[]): string
```

### 8.2 Historical Learning

- Store mutation outcomes in history
- Learn which operators work for which strategy types
- Bias mutation selection toward historically successful operators

---

## 9. Parallelization

### 9.1 Worker Pool Architecture

```
          ┌─────────────────┐
          │   Main Thread   │
          │  (Orchestrator) │
          └────────┬────────┘
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Worker 1 │ │ Worker 2 │ │ Worker N │
│ (Eval)   │ │ (Eval)   │ │ (Eval)   │
└──────────┘ └──────────┘ └──────────┘
```

### 9.2 Distribution Strategy

- **Island Model**: Multiple populations evolve in parallel, occasional migration
- **Global Model**: Single population, work distributed across workers

---

## 10. Persistence

### 10.1 Strategy Storage

```typescript
interface StoredStrategy {
  id: string;
  strategy: Strategy;
  savedAt: number;
  tags: string[];
  notes: string;
}
```

### 10.2 Run History

```typescript
interface EvolutionRun {
  id: string;
  config: EvolutionConfig;
  generations: GenerationSummary[];
  bestStrategy: string;  // Strategy ID
  startedAt: number;
  endedAt?: number;
  status: 'running' | 'completed' | 'converged' | 'terminated';
}
```

---

## 11. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/evolution/start` | POST | Start evolution run |
| `/api/evolution/stop` | POST | Stop evolution run |
| `/api/evolution/status` | GET | Get run status |
| `/api/strategies` | GET | List saved strategies |
| `/api/strategies/:id` | GET | Get strategy details |
| `/api/strategies` | POST | Save new strategy |
| `/api/backtest` | POST | Run backtest on strategy |

---

## 12. Configuration

```typescript
interface EvolutionConfig {
  population: {
    size: number;           // Number of individuals
    eliteCount: number;     // Best to keep unchanged
  };
  evolution: {
    maxGenerations: number;
    convergenceGenerations: number;
    mutationRate: number;
    crossoverRate: number;
  };
  fitness: {
    weights: FitnessWeights;
    minTrades: number;
  };
  operators: {
    enabled: string[];      // Mutation operators to use
    customWeights?: Record<string, number>;
  };
}
```

---

## 13. Future Enhancements

- [ ] Distributed island model across nodes
- [ ] Real-time paper trading integration
- [ ] Multi-timeframe strategy evolution
- [ ] Natural language strategy generation (LLM)
- [ ] Reinforcement learning hybrid operators
- [ ] Strategy ensemble evolution

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **Individual** | A single strategy in the population |
| **Gene** | A component of a strategy (indicator, rule, etc.) |
| **Fitness** | Score measuring strategy quality |
| **Pareto Front** | Set of non-dominated solutions |
| **Crowding Distance** | Measure of solution diversity |
| **Epoch** | One complete evolution cycle |
| **Generation** | One iteration of selection + reproduction |
| **Elitism** | Preserving best individuals unchanged |

---

*Document Version: 1.0*  
*Last Updated: 2026-02-22*
