# QuantCore Evolution Engine - Best Practices Guide

**Version:** 1.0  
**Date:** 2026-02-22  
**Purpose:** Master the Evolution Engine for maximum strategy discovery

---

## 1. Getting Started Right

### 1.1 Initial Population

**Do:** Start with diverse, random seeds
- The initial population quality heavily influences convergence
- Use 100-200 individuals minimum for meaningful evolution

**Don't:** Start with cloned strategies
- Low initial diversity → premature convergence
- Avoid loading 10 copies of the same strategy as seed

**Pro Tip:** Use the "diverse seed" option to inject random gene variations from the start.

---

## 2. Fitness Function Mastery

### 2.1 Weight Tuning

The default weights are a starting point. Tune based on your goals:

| Goal | Return | Sharpe | Drawdown | Robustness |
|------|--------|--------|----------|------------|
| **Aggressive growth** | 0.5 | 0.2 | 0.1 | 0.2 |
| **Capital preservation** | 0.2 | 0.3 | 0.4 | 0.1 |
| **Balanced** | 0.3 | 0.3 | 0.2 | 0.2 |
| **Regime-proof** | 0.2 | 0.2 | 0.2 | 0.4 |

### 2.2 Minimum Trades Rule

**Critical:** Set `minTrades ≥ 20`

Why?
- A strategy with 3 trades and +50% return is noise
- You need statistical significance
- Fitness without enough trades = overfitting

**Pro Tip:** Increase to 30-50 for longer timeframes (daily/weekly).

### 2.3 Robustness Scoring

Enable regime-aware fitness:
- Strategies get scored separately in bull/bear/sideways
- Final score = weighted average across regimes
- Rewards consistency, not just aggregate return

---

## 3. Mutation Operator Strategy

### 3.1 Operator Selection

Not all mutations are equal. Start with these core operators:

**Essential (Always On):**
- `mutate_threshold` — fine-tune parameters
- `mutate_indicator_period` — adjust lookbacks
- `add_trailing_stop` — risk management

**Advanced (Enable After Gen 20+):**
- `invert_signal` — flip logic for counter-strategies
- `add_divergence` — detect divergences
- `combine_signals` — AND/OR logic

### 3.2 Mutation Rate Tuning

| Phase | Rate | Rationale |
|-------|------|-----------|
| **Early (Gen 1-10)** | 0.3-0.4 | High exploration, find promising regions |
| **Mid (Gen 10-30)** | 0.15-0.25 | Balance exploration/exploitation |
| **Late (Gen 30+)** | 0.05-0.15 | Fine-tune best performers |

**Pro Tip:** Enable **Adaptive Mutation** — the engine auto-adjusts based on convergence.

### 3.3 Crossover Strategy

- **Start with low crossover (0.2)** — crossover can break good gene combinations
- **Increase to 0.4** in later generations when you have strong parents
- **Use segment crossover** for swapping indicator blocks
- **Avoid uniform crossover** early on — too disruptive

---

## 4. Convergence Detection

### 4.1 When to Stop

The engine converges when:

1. **Diversity drops below 10%** — population becomes too similar
2. **Avg fitness change < 0.1%** over 5 generations — diminishing returns
3. **Pareto front stabilizes** — no new non-dominated solutions

**Don't:** Run past convergence — you'll get diminishing returns and potential overfitting.

### 4.2 Convergence Signals

Watch for these in the UI:
- 📉 Pareto front line flattening
- 📊 Crowding distance decreasing
- 🎯 Average fitness plateauing

---

## 5. Population Management

### 5.1 Size Guidelines

| Population Size | Best For |
|-----------------|----------|
| 50 | Quick experiments, idea testing |
| 100-150 | Standard production runs |
| 200+ | Complex strategies, many objectives |

**Rule of Thumb:** Population size ≥ 10 × number of objectives

### 5.2 Elitism

Keep **5-10%** of top performers unchanged each generation.

- Too little (1-2): Risk losing best solutions
- Too much (20%+): Reduces diversity, slows evolution

**Pro Tip:** Start with 10% elitism, reduce to 5% if convergence is slow.

---

## 6. Backtest Configuration

### 6.1 Data Settings

| Setting | Recommended | Why |
|---------|-------------|-----|
| **Lookback bars** | 1000-2000 | Enough history for regime detection |
| **Timeframe** | Match your target | Don't optimize on 1h and trade on 1d |
| **Commission** | 0.1% (conservative) | Realistic execution costs |
| **Slippage** | 0.05% | Spread + slippage |

### 6.2 Walk-Forward Validation

**Critical Best Practice:**

1. Train on first 70% of data
2. Validate on next 15%
3. Test on final 15%

If validation fails → the strategy is likely overfitting.

---

## 7. Advanced Techniques

### 7.1 Island Model (Coming Soon)

- Run 3-4 parallel populations
- Each "island" uses different operator settings
- Occasional migration (every 10 generations)
- Prevents premature convergence

### 7.2 Multi-Stage Evolution

**Stage 1: Discovery (Gen 1-20)**
- High mutation (0.4)
- Focus: Return + basic fitness
- Goal: Find promising strategy archetypes

**Stage 2: Refinement (Gen 21-50)**
- Medium mutation (0.2)
- Focus: Add robustness, reduce drawdown
- Goal: Make strategies tradeable

**Stage 3: Polish (Gen 51+)**
- Low mutation (0.1)
- Fine-tune parameters
- Goal: Maximum sharpe ratio

### 7.3 Ensemble Strategies

- Evolve strategies in parallel with different objectives
- Combine top performers from different Pareto fronts
- Use correlation analysis to ensure diversity

### 7.4 Injection

When evolution stalls:
1. Identify "dead zones" in gene space
2. Manually create strategies targeting those areas
3. Inject into population (10-20%)

---

## 8. Common Mistakes to Avoid

### ❌ Overfitting
**Symptom:** 95%+ win rate, perfect Sharpe in backtest
**Solution:** Increase minTrades, enable walk-forward validation

### ❌ Premature Convergence
**Symptom:** Population becomes identical by Gen 10
**Solution:** Increase mutation rate, inject diversity

### ❌ Ignoring Regime Changes
**Symptom:** Great backtest, terrible live performance
**Solution:** Enable regime-aware fitness, test on multiple regimes

### ❌ Too Many Parameters
**Symptom:** Complex strategy, fragile performance
**Solution:** Limit genes per strategy, prefer simple signals

### ❌ Cherry-Picking
**Symptom:** Running evolution 50 times, keeping only best
**Solution:** Use statistical comparison, document all runs

---

## 9. Workflow Recommendations

### Daily Workflow
1. **Morning:** Check overnight evolution runs
2. **Review:** Examine Pareto front, identify new candidates
3. **Save:** Archive top 5 strategies with tags
4. **Inject:** Add fresh diversity if converging

### Weekly Deep Dive
1. Run walk-forward validation on saved strategies
2. Compare against benchmarks
3. Adjust fitness weights based on goals
4. Document lessons learned

### Monthly Reset
1. Clear poor performers from archive
2. Analyze which operators performed best
3. Tune operator weights for next month's runs
4. Review and update minTrades/data settings

---

## 10. Debugging Strategies

### Strategy Won't Evolve?
- ✅ Check if mutation rate is too low
- ✅ Verify fitness weights aren't zero
- ✅ Ensure data is loading correctly

### All Strategies Converge to Same Type?
- ✅ Enable diverse seed initialization
- ✅ Increase crossover diversity
- ✅ Reduce elitism percentage

### Amazing Backtest, Terrible Live?
- ✅ Run walk-forward validation
- ✅ Check for look-ahead bias
- ✅ Increase minTrades requirement
- ✅ Test on unseen data

### Fitness Stuck at Zero?
- ✅ Check signal generation logic
- ✅ Verify indicator calculations
- ✅ Review entry/exit conditions

---

## 11. Quick Reference Cheatsheet

```
┌─────────────────────────────────────────────────────────────┐
│                    EVOLUTION QUICK START                    │
├─────────────────────────────────────────────────────────────┤
│  Population:    100-150                                     │
│  Max Gen:       50-100                                      │
│  Mutation:      0.2-0.3 (adaptive)                          │
│  Crossover:     0.2-0.4                                      │
│  Elitism:       5-10%                                        │
│  Min Trades:    20-30                                        │
│  Lookback:      1000-2000 bars                              │
└─────────────────────────────────────────────────────────────┘

STAGES:
  Gen 1-10   → High mutation (0.4) - Explore
  Gen 10-30  → Medium mutation (0.2) - Refine
  Gen 30+    → Low mutation (0.1) - Polish

RED FLAGS:
  ⚠️ Win rate > 95% → Likely overfitting
  ⚠️ Convergence < Gen 10 → Too much diversity loss
  ⚠️ Same top strategy every run → Operator bias
```

---

## 12. Keyboard Shortcuts (UI)

| Shortcut | Action |
|----------|--------|
| `Space` | Start/Stop evolution |
| `R` | Reset population |
| `S` | Save current best |
| `1-5` | Switch tabs |
| `Ctrl+S` | Quick save |

---

## 13. Further Resources

- **Architecture Doc:** `quantcore-evolution-architecture.md`
- **Genetic Algorithms:** Goldberg, "Genetic Algorithms in Search, Optimization, and Machine Learning"
- **Quantitative Trading:** Chan, "Quantitative Trading"
- **Swarm Intelligence:** Kennedy & Eberhart, "Particle Swarm Optimization"

---

*Happy Evolving! 🚀*
