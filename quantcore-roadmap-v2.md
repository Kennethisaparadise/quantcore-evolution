# QuantCore Implementation Roadmap v2.0

*Based on expert feedback - Feb 2026*

---

## Current Status

✅ **COMPLETED:**
- Regime-Switching Meta-Mutation Engine (Tier 1)
- React UI with mode toggle
- Basic switch config & regime mappings

---

## Priority Updates (Based on Expert Feedback)

### 🔥 IMMEDIATE IMPROVEMENTS TO CURRENT CODE

#### 1. Evolved Regime Detection (Enhance Tier 1)

**Problem:** Currently using fixed HMM/k-means. Expert says: let the GA EVOLVE the detection pipeline.

**Add to MetaEvolutionEngine:**
```python
# Evolvable regime detection config
@dataclass
class RegimeDetectionConfig:
    n_regimes: int = 4                          # Evolve: 2-6
    feature_weights: Dict[str, float] = None     # Evolve: which features matter
    method: str = "hmm"                         # Evolve: hmm/kmeans/rule
    volatility_weight: float = 1.0
    trend_weight: float = 1.0
    volume_weight: float = 1.0
    correlation_weight: float = 1.0
```

**New Mutation Operators:**
- `mutate_n_regimes` — change number of regimes
- `mutate_feature_weight` — adjust feature importance
- `mutate_detection_method` — switch algorithms

#### 2. Transition Cost Penalty (Enhance Tier 1)

**Already added:** Basic switch penalty
**Enhance:** Make it evolvable per regime

```python
@dataclass
class TransitionCost:
    bull_to_bear: float = 0.1
    bull_to_sideways: float = 0.05
    # ... all pairs
```

---

### 📋 PHASE 2: FRACTAL TIME SERIES (Tier 2 → Move Up)

Expert says this is "filthy rich" when combined with cycles.

**New Mutation Category: `timeframe`**

Operators:
- `tf_15m_1h_sync` — require both timeframes aligned
- `tf_1h_4h_alignment` — larger timeframe confirmation
- `tf_daily_confirmation` — daily filter on intraday
- `fractal_harmonic_entry` — combine cycle + timeframe

**Integration:**
- Use existing `generateSineWave` + `getCycleSignal`
- New: check signal on multiple timeframes before entry

---

### 🛡️ PHASE 3: EVOLUTIONARY ADVERSARIAL VALIDATOR (Tier 2 → Move Up)

Expert: "This is your shield against curve-fitting."

**Implementation:**
```python
class AdversarialValidator:
    def generate_synthetic_paths(self, data, n_paths=100, method='garch'):
        # GARCH-based path generation
        # Bootstrap residuals
        # Regime-specific shocks
        
    def evaluate_worst_case(self, strategy, paths):
        # Run strategy on ALL synthetic paths
        # Fitness = WORST-CASE Sharpe (not average!)
        # This breeds antifragility
```

**New Mutation Operators:**
- `add_crash_hedge` — protective puts in crash regimes
- `add_liquidity_exit` — exit before gap events
- `add_gap_protection` — stop adjustment for overnight gaps

---

### 📊 PHASE 4: ORDER FLOW SHADOW (Tier 3 → Keep)

**Implementation using tick rule:**
```python
def simulate_order_flow(bars):
    """Disaggregate volume into buy/sell using tick rule"""
    for bar in bars:
        if bar.close > bar.open:
            bar.buy_volume = bar.volume
            bar.sell_volume = 0
        else:
            bar.buy_volume = 0
            bar.sell_volume = bar.volume
    
    # Cumulative delta
    bar.cumulative_delta = cumsum(buy_volume - sell_volume)
```

**Operators:**
- `delta_divergence` — price low vs delta low
- `vwap_cross` — VWAP level signals
- `imbalance_threshold` — buy/sell pressure

---

### 💰 PHASE 5: ADAPTIVE POSITION SIZING v2

**Conviction Score System:**
```python
def calculate_conviction(signal_strength, volatility, regime):
    # RSI extremity: 0-1
    # MACD histogram slope: 0-1
    # Regime confidence: 0-1
    
    score = (signal_strength * 0.4 + 
             (1/volatility) * 0.3 + 
             regime.confidence * 0.3)
    
    # Map to position size
    return map_score_to_size(score)
```

**Evolve:** mapping function parameters

---

### 🔄 PHASE 6: META-EVOLUTION SCHEDULER

**Self-tuning system:**
```python
class MetaEvolutionScheduler:
    def __init__(self):
        self.stagnation_threshold = 5  # generations
        self.current_mutation_rate = 0.3
        
    def on_generation_end(self, fitness_history):
        if fitness_stagnated():
            # Increase mutation, inject diversity
            self.current_mutation_rate *= 1.5
            self.inject_random_seeds()
        else:
            # Decrease mutation, fine-tune
            self.current_mutation_rate *= 0.9
```

---

## Updated Priority Order

| Phase | Feature | Reason | Priority |
|-------|---------|--------|----------|
| 1a | Enhance Regime Detection (evolvable) | Core differentiator | 🔴 NOW |
| 1b | Transition cost (evolvable) | Prevent whipsaw | 🔴 NOW |
| 2 | Fractal Timeframe Mutator | High impact, low data req | 🟠 Week 2 |
| 3 | Adversarial Validator | Anti-overfitting shield | 🟠 Week 2 |
| 4 | Order Flow Shadow | Intraday edge | 🟡 Week 3 |
| 5 | Adaptive Position v2 | Fine-tune returns | 🟡 Week 3 |
| 6 | Meta-Evolution Scheduler | Self-tuning | 🟢 Week 4 |

---

## New Prompt Assessment

| Prompt | Assessment |
|--------|------------|
| **11: Fractal Regime Combiner** | ⭐ MERGE WITH PHASE 2 - combine fractal with regime |
| **12: Adversarial Noise Injector** | ⭐ PART OF PHASE 3 - enhance validator |
| **13: Sentiment-Flow Fusion** | 🟡 Later - requires sentiment data |
| **14: Regime-Adaptive Position** | ⭐ MERGE WITH PHASE 5 |
| **15: Meta-Evolution Scheduler** | ⭐ PHASE 6 - essential for production |

---

## Implementation Notes

### Key Insight from Expert:
> "Let the algorithm discover whether it needs 2 regimes or 5"

This means the regime detection itself should be EVOLVED, not fixed. The GA should choose:
- How many regimes (2-6)
- Which features to use (vol, trend, volume, correlation)
- How to weight them
- Which algorithm (HMM, k-means, rule-based)

This transforms the system from "I think there are 4 regimes" to "let the data tell us."

---

*This is the difference between a toy and a weapon.*
