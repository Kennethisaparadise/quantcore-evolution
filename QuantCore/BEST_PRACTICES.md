# QuantCore Evolution - Best Practices Guide

## Overview

QuantCore is an advanced algorithmic trading platform with 9 integrated modules that work together to create a comprehensive trading intelligence system.

## The 9 Modules

| # | Module | Dimension | File |
|----|--------|-----------|------|
| 1 | Regime-Switching | WHEN to trade | `regime_detector.py` |
| 2 | Fractal Time Series | WHAT TF | `fractal_mutator.py` |
| 3 | Adversarial Validator | SURVIVE chaos | `adversarial_validator.py` |
| 4 | Timeline Visualization | SEE thinking | UI built-in |
| 5 | Dynamic Regime Count | SELF-DISCOVER | `meta_regime_evolution.py` |
| 6 | Order Flow Shadow | MICROSTRUCTURE | `order_flow_shadow.py` |
| 7 | Sentiment Divergence | EXTERNAL | `sentiment_divergence.py` |
| 8 | Correlation Pairs | RELATIONSHIPS | `correlation_pairs.py` |
| 9 | Live Trading | EXECUTION | `live_trading.py` |
| 10 | Real-Time Adaptation | SELF-IMPROVE | `realtime_adaptation.py` |
| 11 | Compounding Engine | COMPOUND | `compounding_engine.py` |
| 12 | Order Book Imbalance | IMBALANCE | `microstructure.py` |
| 13 | Iceberg Hunter | HIDDEN | `microstructure.py` |
| 14 | Spoofing Detector | MANIPULATION | `microstructure.py` |
| 15 | Latency Arbitrage | SPEED | `microstructure.py` |
| 16 | Adversarial Algo Hunter | PREDATOR | `microstructure.py` |
| 17 | Microstructure Patterns | TICK-LEVEL | `microstructure.py` |

---

## Best Practices

### 1. Starting with Seed Strategies

**Recommended Starting Seeds:**
- RSI Mean Reversion (rsi) - Classic mean reversion
- MA Crossover (ma_cross) - Trend following baseline
- Bollinger Band Squeeze (bollinger) - Volatility strategy

**For Higher Returns:**
- Momentum Stack - Stack RSI + MACD + ATR
- Asymmetric 3:1+ - 3:1 risk:reward ratio
- Volatility Expansion - Capture big moves

**Cycle Strategies (Buy Bottom, Sell Top):**
- Sine Wave Cycle - Classic cycle trading
- Ehlers Sine Wave - Hilbert transform enhanced
- MESA Cycle - MESA Phasor analysis

### 2. Mutation Selection

**Conservative Evolution:**
```
- tighten_stop_loss
- decrease_position_size  
- enable_trailing_stop
- adx_filter
```

**Aggressive Evolution:**
```
- high_return_15m
- momentum_burst
- explosive_breakout
- ultra_aggressive
```

**Regime-Aware:**
```
- bull_only / bear_only
- sideways_only
- adx_filter
```

### 3. Evolution Parameters

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| Population | 50 | 100 | 200 |
| Generations | 10 | 20 | 50 |
| Mutation Rate | 0.05 | 0.1 | 0.2 |
| Elitism | 5 | 10 | 20 |

### 4. Fitness Function Weights

Default multi-objective:
- Return: 40%
- Sharpe Ratio: 30%
- Drawdown: 20%
- Win Rate: 10%

For consistency:
- Return: 20%
- Sharpe: 40%
- Drawdown: 30%
- Win Rate: 10%

### 5. Regime Detection

**Recommended Configuration:**
- Regimes: 3-5 (Bull, Bear, Sideways, High Vol, Low Vol)
- Lookback: 100 bars minimum
- Use HMM for probabilistic regime detection

### 6. Adversarial Validation

**Always enable before live trading:**
- GARCH volatility stress testing
- Flash crash injection (gap down 10%+)
- Sequence of losers simulation

Target: Strategies that survive worst-case scenarios.

### 7. Sentiment Integration

**Quick Start:**
1. Enable Fear & Greed index (simplest)
2. Set confirmation mode to "AND" with price signals
3. Use regime-specific sentiment thresholds

**Advanced:**
- Add news sentiment API
- Social media buzz tracking
- On-chain metrics (whale transactions)

### 8. Pair Trading

**Popular Pairs:**
- BTC/ETH - Crypto majors
- BTC/SOL - BTC vs altcoin
- Gold/Silver - Commodities
- SPY/QQQ - US equities

**Regime-Specific:**
- Bull: Momentum pairs (BTC/ETH, SOL/ETH)
- Bear: Defensive pairs (BTC/USDT, Gold)
- High Vol: Only high correlation (>0.8)

### 9. Live Trading

**Paper Trading First:**
1. Test for 2+ weeks in paper mode
2. Verify performance matches backtest
3. Check circuit breakers trigger correctly

**Risk Limits (recommended):**
- Max Daily Loss: 5%
- Max Drawdown: 20%
- Max Positions: 5
- Position Size: 10-20% max

**Going Live:**
- Start with small capital (10% of planned)
- Monitor for 2 weeks
- Scale up gradually

---

## Common Pitfalls

### Overfitting
- Don't optimize too many parameters
- Use out-of-sample validation
- Enable adversarial testing

### Regime Blindness
- Always use regime-aware filters
- Test across different market conditions
- Don't trust backtests in only one regime

### Position Sizing
- Never risk more than 2% per trade
- Use Kelly fraction ≤ 0.25
- Scale positions by confidence

### Correlation Traps
- Don't over-correlate your pairs
- Test correlation decay scenarios
- Use regime-specific pair selection

---

## File Structure

```
QuantCore/
├── engine/
│   ├── backtest/           # Backtesting engine
│   ├── mutation.py         # Core mutations
│   ├── regime_detector.py  # Regime detection (HMM)
│   ├── fractal_mutator.py # Multi-TF analysis
│   ├── adversarial_validator.py # Stress testing
│   ├── order_flow_shadow.py    # Order flow
│   ├── sentiment_divergence.py  # Sentiment
│   ├── correlation_pairs.py    # Pairs trading
│   └── live_trading.py        # Execution
├── strategies/            # Saved strategies
└── config.json           # Configuration
```

---

## Quick Start

```python
from engine.evolution import EvolutionEngine
from engine.regime_detector import RegimeDetector
from engine.live_trading import TradingOrchestrator, create_trading_config

# 1. Create evolution engine
engine = EvolutionEngine(
    population=100,
    generations=20,
    mutation_rate=0.1
)

# 2. Run evolution
best = engine.evolve(
    seeds=['rsi', 'ma_cross', 'bollinger'],
    mutations=['tighten_stop_loss', 'momentum_burst'],
    data=market_data
)

# 3. Setup regime detection
regime = RegimeDetector(num_regimes=3)
current_regime = regime.detect(market_data)

# 4. Setup live trading
config = create_trading_config(mode='paper', initial_capital=10000)
orchestrator = TradingOrchestrator(config)

# 5. Execute signals
orchestrator.execute_signal('BTCUSDT', direction=1, quantity=0.1)
```

---

## Version History

- **v1.2.0** - Live Trading + Paper Trading (Feb 2026)
- **v1.1.0** - Correlation Pairs (Feb 2026)
- **v1.0.0** - Sentiment Divergence (Feb 2026)
- **v0.9.0** - Order Flow Shadow (Feb 2026)
- **v0.8.0** - Meta-Regime Evolution (Feb 2026)
- **v0.7.0** - Adversarial Validator (Feb 2026)
- **v0.6.0** - Fractal Time Series (Feb 2026)
