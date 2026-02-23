# QuantCore Version History

## Current Version: v2.5.0 (2026-02-23)

### What's New in v2.5.0 - Withdrawal Optimizer

#### v2.5.0 - Withdrawal Optimizer (`engine/withdrawal_optimizer.py`)
- **Fixed percent** - 4% rule
- **Dynamic withdrawals** - adjust based on performance
- **Tax-aware** - prefer long-term gains
- **Drawdown protection** - pause if >15% DD
- **Goal-based planning** - retirement, house, etc.
- **Withdrawal mutations** - evolve rate, frequency, strategy

---

## v2.4.0 (2026-02-23) - Multi-Account Allocator

#### v2.4.0 - Multi-Account Allocator (`engine/multi_account.py`)
- **Multiple account types** - taxable, IRA, Roth, offshore
- **Strategy-to-account allocation** - based on tax efficiency
- **Asset location optimization** - high turnover → Roth, long-term → taxable
- **Automatic rebalancing** - when drift exceeds threshold
- **Fund transfers** between accounts with tax checking
- **Tax efficiency scoring** - measures portfolio tax efficiency

---

## v2.3.0 (2026-02-23) - Tax-Aware Optimization

#### v2.3.0 - Tax-Aware Optimization (`engine/tax_aware.py`)
- **Holding period tracking** for every position
- **Tax-loss harvesting** with automatic detection
- **Wash sale avoidance** - 30 day window tracking
- **After-tax fitness function** - optimize for after-tax returns
- **Short-term vs long-term** tax rate calculation
- **Tax mutations** - evolve harvest threshold, rates

---

## v2.2.0 (2026-02-23) - Fee-Aware Execution

#### v2.2.0 - Fee-Aware Execution (`engine/fee_aware.py`)
- **Configurable fee schedules** per exchange
- **30-day volume tracking** for tier determination
- **Maker/Taker detection** based on order type and price
- **Fee comparison** across 6 exchanges
- **Fee-aware position sizing** with capital buffers
- **BNB/loyalty discounts** support

**Exchanges:** Binance, Coinbase, Alpaca, dYdX, Hyperliquid, Robinhood

---

## v2.1.0 (2026-02-23) - Level 2: Order Book Warfare

#### v2.1.0 - Microstructure (`engine/microstructure.py`)
- **Order Book Imbalance Detector** - bid vs ask volume at top N levels
- **Iceberg Order Hunter** - detect hidden liquidity
- **Spoofing/Layering Detector** - identify manipulation
- **Latency Arbitrage Engine** - cross-exchange price differences
- **Adversarial Algo Hunter** - exploit stop hunts, momentum ignition
- **Microstructure Pattern Recognition** - tick-level patterns

---

## v2.0.0 (2026-02-23) - "The Complete System"

### New Features:

#### Core Evolution (from v1.x)
- ✅ Regime-Switching (HMM-based, WHEN to trade)
- ✅ Fractal Time Series (multi-TF consensus, WHAT TF)
- ✅ Adversarial Validator (stress testing, SURVIVE)
- ✅ Timeline Visualization (SEE thinking)
- ✅ Dynamic Regime Count (SELF-DISCOVER regimes)

#### v1.7.0 - Order Flow Shadow (`engine/order_flow_shadow.py`)
- Microstructure analysis
- Cumulative delta detection
- Large trade identification
- Stop hunt anticipation

#### v1.8.0 - Sentiment Divergence (`engine/sentiment_divergence.py`)
- Multiple sentiment sources (news, social, on-chain, Fear/Greed)
- Sentiment-based signals (extremes, momentum, divergence)
- Sentiment-Flow Fusion (killer combo: delta + sentiment)
- Regime-aware sentiment

#### v1.9.0 - Correlation Pairs (`engine/correlation_pairs.py`)
- Cointegration detection (Engle-Granger)
- Rolling correlation tracking
- Hedge ratio evolution
- Pair rotation strategies

#### v2.0.0 - Live Trading + Paper Trading (`engine/live_trading.py`)
- Paper mode (simulated with slippage)
- Live exchange connectors (Binance, Hyperliquid)
- Order management system
- Position tracking & PnL
- Risk guards & circuit breakers
- Kelly/Fixed/Volatility position sizing

#### v2.0.0 - Real-Time Adaptation (`engine/realtime_adaptation.py`)
- Streaming data ingestion
- Incremental regime learning
- Scheduled mini-evolution
- Underperformer replacement
- Risk-aware updates

#### v2.0.0 - Compounding Engine (`engine/compounding_engine.py`)
- Equity curve tracking
- Dynamic position sizing (Kelly-based)
- Auto-reinvestment logic
- Drawdown protection
- Portfolio allocator (Equal, Risk Parity, Kelly, Inverse Vol)
- Compounding-aware fitness (CAGR, Ulcer, MAR, Calmar)

### Module Summary (11 Modules)

| # | Module | Dimension | File |
|----|--------|-----------|------|
| 1 | Regime-Switching | WHEN | `regime_detector.py` |
| 2 | Fractal Time Series | WHAT TF | `fractal_mutator.py` |
| 3 | Adversarial Validator | SURVIVE | `adversarial_validator.py` |
| 4 | Timeline Visualization | SEE | UI built-in |
| 5 | Dynamic Regime Count | SELF-DISCOVER | `meta_regime_evolution.py` |
| 6 | Order Flow Shadow | MICRO | `order_flow_shadow.py` |
| 7 | Sentiment Divergence | FEEL | `sentiment_divergence.py` |
| 8 | Correlation Pairs | RELATE | `correlation_pairs.py` |
| 9 | Live Trading | EXECUTE | `live_trading.py` |
| 10 | Real-Time Adaptation | SELF-IMPROVE | `realtime_adaptation.py` |
| 11 | Compounding Engine | COMPOUND | `compounding_engine.py` |

### Documentation
- `BEST_PRACTICES.md` - Complete usage guide
- `QUICKSTART.md` - Getting started
- `RESOURCES.md` - External resources
- `STRUCTURE.md` - File organization

---

## v1.2.0 (2026-02-22)
- Genetic Programming (tree-based strategy discovery)
- Parallel GA (distributed evaluation)

---

## v1.1.0 (2026-02-22)
- Parallel evaluation with Dask/multiprocessing
