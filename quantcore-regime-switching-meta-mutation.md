# QuantCore Regime-Switching Meta-Mutation

**Version:** 1.0  
**Date:** 2026-02-22  
**Status:** Draft  
**Priority:** Tier 1 - Must Build

---

## 1. Overview

The Regime-Switching Meta-Mutation enables the Genetic Algorithm to evolve not just individual trading strategies, but the *rules for switching between strategies* based on detected market regimes. This transforms the engine from "evolve a strategy" to "evolve a portfolio of strategies + when to use each."

### Core Philosophy
- **Regimes are latent states** — the market is always in some regime (bull, bear, sideways, high-vol, low-vol, etc.)
- **No single strategy works everywhere** — trend followers win in bull, mean-reverters win in range
- **Let the GA discover the mapping** — don't hardcode "RSI for bear, MA for bull" — let evolution find what works

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REGIME-SWITCHING META-MUTATION                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐     ┌──────────────────┐     ┌───────────────┐ │
│  │  Regime Detector │ ──► │ Strategy Pool    │ ──► │  Switcher     │ │
│  │  (HMM/Cluster)   │     │ (Multiple Strats)│     │  (Evolved)    │ │
│  └──────────────────┘     └──────────────────┘     └───────────────┘ │
│           │                        │                        │          │
│           ▼                        ▼                        ▼          │
│  ┌──────────────────┐     ┌──────────────────┐     ┌───────────────┐ │
│  │ Vol, Trend, Corr │     │ Strategy A, B, C│     │ Confidence   │ │
│  │ Features         │     │ Per Regime      │     │ Threshold    │ │
│  └──────────────────┘     └──────────────────┘     └───────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Structures

### 3.1 Regime State

```typescript
interface RegimeState {
  type: 'bull' | 'bear' | 'sideways' | 'high_vol' | 'low_vol' | 'trending' | 'ranging';
  confidence: number;           // 0-1, how certain the detector is
  features: RegimeFeatures;
  startedAt: number;           // timestamp when regime started
}

interface RegimeFeatures {
  volatility: number;         // ATR percentile or std dev
  trendStrength: number;      // ADX or linear regression R²
  correlationBreadth: number;  // Avg correlation to market
  volumeProfile: number;      // Volume regime
  returns: number[];           // Recent returns for distribution
}
```

### 3.2 Meta-Strategy (The Evolvable Unit)

```typescript
interface MetaStrategy {
  id: string;
  name: string;
  
  // The strategy pool — multiple strategies this meta-strategy can use
  strategies: Strategy[];
  
  // Evolution rules (these are the MUTABLE GENES)
  regimeMapping: RegimeMapping[];
  switchConfig: SwitchConfig;
  
  // Fitness (evaluated per-regime)
  regimeFitness: Record<RegimeState['type'], number>;
  overallFitness: number;
}

interface RegimeMapping {
  regime: RegimeState['type'];
  strategyId: string;           // Which strategy to use
  confidenceThreshold: number; // Min confidence to switch
}

interface SwitchConfig {
  minHoldTime: number;          // Min bars to hold before switching
  reentryCooldown: number;      // Bars to wait before re-entering
  transitionMode: 'hard' | 'smooth' | 'blend';
  hysteresis: number;           // Buffer to prevent flip-flopping
}
```

---

## 4. Regime Detection Methods

### 4.1 Hidden Markov Model (HMM)

```python
# States: Bull, Bear, Sideways, HighVol, LowVol
# Observations: returns, volatility, volume

class RegimeHMM:
    def __init__(self, n_states=4):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full")
    
    def fit(self, price_data: pd.DataFrame):
        features = self._extract_features(price_data)
        self.model.fit(features)
    
    def predict(self, price_data) -> list[RegimeState]:
        features = self._extract_features(price_data)
        states = self.model.predict(features)
        return [self._state_to_regime(s) for s in states]
    
    def _extract_features(self, data):
        return np.column_stack([
            data['returns'],
            data['volatility'],      # ATR or rolling std
            data['volume'] / data['volume'].mean(),
            data['trend_strength'],  # Linear regression R²
        ])
```

### 4.2 K-Means Clustering

```python
class RegimeCluster:
    def __init__(self, n_clusters=4):
        self.kmeans = KMeans(n_clusters=n_clusters)
    
    def fit_predict(self, data) -> list[RegimeState]:
        features = self._extract_features(data)
        clusters = self.kmeans.fit_predict(features)
        
        # Map clusters to regime types based on feature centroids
        return [self._cluster_to_regime(c, features[i]) for i, c in enumerate(clusters)]
```

### 4.3 Rule-Based (Lightweight)

```python
def detect_regime(prices, period=50) -> RegimeState:
    returns = prices.pct_change()
    volatility = returns.rolling(period).std()
    trend = linear_regression(prices).r_squared
    
    if volatility > volatility.quantile(0.8):
        return RegimeState(type='high_vol', confidence=0.8)
    elif trend > 0.7:
        return RegimeState(type='trending', confidence=0.75)
    elif abs(returns.mean()) < 0.001:
        return RegimeState(type='sideways', confidence=0.7)
    # ... etc
```

---

## 5. Mutation Operators

### 5.1 Regime Classification Mutations

| Operator | Description | Parameters |
|----------|-------------|------------|
| `mutate_regime_threshold` | Adjust volatility/trend thresholds for regime classification | `threshold_delta: float` |
| `mutate_regime_features` | Add/remove features from regime detection | `feature: string`, `weight: float` |
| `mutate_detection_method` | Switch between HMM, K-Means, Rule-Based | `method: string` |
| `mutate_smoothing` | Adjust regime detection smoothing window | `window: int` |

### 5.2 Strategy Selection Mutations

| Operator | Description | Parameters |
|----------|-------------|------------|
| `assign_strategy_to_regime` | Change which strategy is assigned to a regime | `regime: string`, `strategy_id: string` |
| `swap_strategies` | Swap strategies between two regimes | `regime_a: string`, `regime_b: string` |
| `add_strategy_to_pool` | Add a new strategy to the meta-strategy pool | `strategy: Strategy` |
| `remove_strategy` | Remove a strategy from the pool | `strategy_id: string` |

### 5.3 Transition Logic Mutations

| Operator | Description | Parameters |
|----------|-------------|------------|
| `mutate_confidence_threshold` | Change min confidence to switch strategies | `threshold: float` |
| `mutate_min_hold_time` | Adjust minimum bars before switch | `bars: int` |
| `mutate_hysteresis` | Add buffer to prevent flip-flopping | `buffer: float` |
| `mutate_transition_mode` | Switch between hard/smooth/blend | `mode: string` |
| `mutate_cooldown` | Adjust reentry cooldown | `bars: int` |

---

## 6. Fitness Evaluation

### 6.1 Per-Regime Fitness

```python
def evaluate_meta_strategy(meta: MetaStrategy, data: DataFrame) -> dict:
    regime_sequence = detect_regimes(data)
    
    regime_returns = {r: [] for r in REGIME_TYPES}
    
    for i in range(len(data)):
        regime = regime_sequence[i]
        
        # Get active strategy for this regime
        strategy = get_strategy_for_regime(meta, regime)
        
        # Generate signal and execute
        signal = strategy.generate_signal(data[:i])
        pnl = execute(signal, data[i])
        
        regime_returns[regime.type].append(pnl)
    
    # Calculate fitness per regime
    fitness = {}
    for regime, returns in regime_returns.items():
        if len(returns) >= MIN_TRADES:
            fitness[regime] = calculate_sharpe(returns)
    
    # Overall fitness = weighted average across regimes
    # Penalty for strategy switching (avoid flip-flopping)
    switch_penalty = calculate_switch_penalty(meta, regime_sequence)
    
    meta.regimeFitness = fitness
    meta.overallFitness = weighted_average(fitness.values()) - switch_penalty
    
    return fitness
```

### 6.2 Switch Penalty

```python
def calculate_switch_penalty(meta: MetaStrategy, regime_sequence: list) -> float:
    switches = count_regime_changes(regime_sequence)
    
    # Penalize excessive switching
    if switches > MAX_ACCEPTABLE_SWITCHES:
        return (switches - MAX_ACCEPTABLE_SWITCHES) * PENALTY_FACTOR
    
    return 0
```

---

## 7. Example Meta-Strategy (Evolved)

```json
{
  "name": "Surfer + Sniper Combo",
  "strategies": [
    { "id": "trend_1", "type": "EMA_Cross", "regimes": ["bull", "trending"] },
    { "id": "mean_rev_1", "type": "RSI_Reversion", "regimes": ["sideways", "low_vol"] },
    { "id": "vol_adapt_1", "type": "ATR_Breakout", "regimes": ["high_vol", "bear"] }
  ],
  "regimeMapping": [
    { "regime": "bull", "strategyId": "trend_1", "confidenceThreshold": 0.6 },
    { "regime": "bear", "strategyId": "vol_adapt_1", "confidenceThreshold": 0.7 },
    { "regime": "sideways", "strategyId": "mean_rev_1", "confidenceThreshold": 0.5 },
    { "regime": "high_vol", "strategyId": "vol_adapt_1", "confidenceThreshold": 0.8 }
  ],
  "switchConfig": {
    "minHoldTime": 5,
    "reentryCooldown": 3,
    "transitionMode": "smooth",
    "hysteresis": 0.1
  },
  "regimeFitness": {
    "bull": 2.1,
    "bear": 1.4,
    "sideways": 1.8,
    "high_vol": 0.9
  }
}
```

---

## 8. Integration with Existing Engine

### 8.1 Modified Evolution Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                      EVOLVED LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│  1. Initialize: Create N meta-strategies (each with 2-4      │
│     child strategies)                                           │
│                                                                  │
│  2. For each generation:                                        │
│     a. Detect regime sequence for training period              │
│     b. For each meta-strategy:                                 │
│        - Evaluate each child strategy per regime               │
│        - Apply switch logic                                     │
│        - Calculate per-regime fitness                           │
│        - Aggregate overall fitness                             │
│                                                                  │
│  3. Selection: Tournament select meta-strategies              │
│                                                                  │
│  4. Mutation:                                                  │
│     - Standard strategy mutations (existing)                   │
│     - NEW: Regime mapping mutations                             │
│     - NEW: Switch config mutations                             │
│                                                                  │
│  5. Crossover:                                                 │
│     - Swap strategy pools between meta-strategies             │
│     - Swap regime mappings                                      │
│                                                                  │
│  6. Elitism: Keep top meta-strategies unchanged                │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Backward Compatibility

- Existing single-strategy evolution continues to work
- Meta-strategies are a NEW evolution mode
- Users can choose: "Evolve Single Strategy" or "Evolve Meta-Strategy"

---

## 9. Configuration

```typescript
interface RegimeEvolutionConfig {
  // Regime Detection
  detectionMethod: 'hmm' | 'kmeans' | 'rule-based';
  nRegimes: number;                    // Default: 4
  lookbackPeriod: number;               // Bars for feature calculation
  
  // Strategy Pool
  minStrategiesPerMeta: number;         // Default: 2
  maxStrategiesPerMeta: number;         // Default: 4
  
  // Switching
  defaultConfidenceThreshold: number;   // Default: 0.6
  defaultMinHoldTime: number;           // Default: 5 bars
  allowTransitionModes: string[];      // ['hard', 'smooth', 'blend']
  
  // Fitness
  regimeFitnessWeight: number;          // Weight per-regime vs overall
  switchPenaltyFactor: number;         // Penalty for flip-flopping
  
  // Mutation Rates
  regimeMutationRate: number;           // Default: 0.15
  switchMutationRate: number;          // Default: 0.1
}
```

---

## 10. UI Updates

### 10.1 New Tab: "Meta-Strategy Evolution"

- Toggle between "Single Strategy" and "Meta-Strategy" mode
- Visualizer: Show regime detection + active strategy per regime
- Stats: Per-regime performance breakdown
- Switch timeline: When did the strategy switch? Why?

### 10.2 Visualization

```
┌──────────────────────────────────────────────────────────────┐
│  REGIME TIMELINE (Last 100 bars)                             │
├──────────────────────────────────────────────────────────────┤
│  ████████████████░░░░░░░░░███████████████░░░░░░░░░░░░░░░░ │
│  ↑              ↑                                            │
│  BULL           SIDEWAYS                                     │
│  ─────────────────────────────────────────────────────────  │
│  Active Strategy: EMA_Cross (bull) → RSI_Reversion (sideways)│
│  Switches: 3 | Confidence: 0.72 | Hold Time: 8 bars         │
└──────────────────────────────────────────────────────────────┘
```

---

## 11. Implementation Roadmap

### Phase 1: Core Regime Detection
- [ ] Implement regime detector (HMM + rule-based)
- [ ] Add regime features (volatility, trend, correlation)
- [ ] Create regime timeline visualization

### Phase 2: Meta-Strategy Structure
- [ ] Define MetaStrategy data structure
- [ ] Implement regime mapping storage
- [ ] Create switch config storage

### Phase 3: Evolution Integration
- [ ] Add regime fitness evaluation
- [ ] Implement regime mapping mutations
- [ ] Implement switch config mutations
- [ ] Add crossover for meta-strategies

### Phase 4: UI/UX
- [ ] Add Meta-Strategy mode toggle
- [ ] Create regime timeline visualizer
- [ ] Add per-regime performance stats

---

## 12. Expected Outcomes

- **More robust strategies** — Evolved to work across regimes, not just fitted to one
- **Reduced overfitting** — Fitness evaluation includes regime diversity
- **Adaptive to regime changes** — No manual "when to switch" rules needed
- **"Ride trends, fade ranges"** — The algorithm learns when to do each

---

*This is the killer feature. It transforms QuantCore from a single-strategy optimizer into an adaptive, regime-aware trading system.*
