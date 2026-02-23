# 🐙 QuantCore Evolution Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Version-2.1.0-green.svg" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Modules-17-red.svg" alt="Modules">
</p>

> **17-module self-compounding trading intelligence that evolves in real time and reads order books.**

QuantCore is a next-generation algorithmic trading system that combines genetic evolution, regime detection, microstructure analysis, and real-time adaptation into a single self-improving organism.

## 🏃‍♂️ Quick Start

```bash
# Clone the repo
git clone https://github.com/Kennethisaparadise/quantcore-evolution.git
cd quantcore-evolution

# Install dependencies
pip install -r requirements.txt

# Run paper trading (test it out!)
python run_paper_trading.py --symbol BTCUSDT --capital 10000 --cycles 100 --interval 60
```

**Go Live (⚠️ Real Money):**
```bash
python run_paper_trading.py --mode live --capital 1000 --symbol BTCUSDT
```

## 🧠 The 17 Heads

| # | Module | Dimension | Description |
|----|--------|-----------|-------------|
| 1 | Regime-Switching | WHEN | HMM-based market state detection |
| 2 | Fractal Time Series | WHAT TF | Multi-timeframe consensus |
| 3 | Adversarial Validator | SURVIVE | Stress-test against chaos |
| 4 | Timeline Visualization | SEE | Watch the system think |
| 5 | Dynamic Regime Count | SELF-DISCOVER | Auto-detect optimal regimes |
| 6 | Order Flow Shadow | MICRO | Read whale footprints |
| 7 | Sentiment Divergence | FEEL | External market mood |
| 8 | Correlation Pairs | RELATE | Multi-asset relationships |
| 9 | Live Trading | EXECUTE | Paper & live execution |
| 10 | Real-Time Adaptation | SELF-IMPROVE | Evolves while trading |
| 11 | Compounding Engine | COMPOUND | Auto-reinvest profits |
| 12 | Order Book Imbalance | IMBALANCE | Bid/ask depth analysis |
| 13 | Iceberg Hunter | HIDDEN | Detect hidden liquidity |
| 14 | Spoofing Detector | MANIPULATION | Catch fake orders |
| 15 | Latency Arbitrage | SPEED | Cross-exchange arbitrage |
| 16 | Adversarial Algo Hunter | PREDATOR | Hunt other bots |
| 17 | Microstructure Patterns | TICK-LEVEL | Recognize patterns |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTCORE EVOLUTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   REGIME     │  │   FRACTAL    │  │  ADVERSARIAL │       │
│  │  SWITCHING   │  │ TIME SERIES  │  │  VALIDATOR   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                  │                  │                │
│  ┌──────▼──────────────────▼──────────────────▼───────┐      │
│  │              EVOLUTION ENGINE                        │      │
│  │         (Genetic Algorithm + Mutations)             │      │
│  └──────────────────────┬───────────────────────────────┘      │
│                         │                                      │
│  ┌──────────────────────▼───────────────────────────────┐      │
│  │              MODULE ORCHESTRATOR                      │      │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │      │
│  │  │ORDER   │ │SENTIMENT│ │CORREL │ │COMPUND │       │      │
│  │  │FLOW    │ │        │ │PAIRS  │ │ENGINE  │       │      │
│  │  └────────┘ └────────┘ └────────┘ └────────┘       │      │
│  └──────────────────────┬───────────────────────────────┘      │
│                         │                                      │
│  ┌──────────────────────▼───────────────────────────────┐      │
│  │            EXECUTION LAYER                            │      │
│  │     ┌─────────┐        ┌─────────┐                  │      │
│  │     │ PAPER   │        │  LIVE   │                  │      │
│  │     │ TRADING │        │ TRADING │                  │      │
│  │     └─────────┘        └─────────┘                  │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## ⚙️ Configuration

Edit `config.json` for your setup:

```json
{
  "trading": {
    "mode": "paper",
    "exchange": "binance",
    "initial_capital": 10000
  },
  "risk": {
    "max_drawdown": 0.20,
    "max_daily_loss": 0.05,
    "max_positions": 5
  },
  "compounding": {
    "base_risk_per_trade": 0.02,
    "kelly_fraction": 0.25,
    "reinvestment_rate": 0.80
  }
}
```

## 📊 Module Details

### Core Evolution
- **60+ Seed Strategies** - RSI, MACD, Bollinger, SuperTrend, Ichimoku, Cycle/Sine Wave
- **70+ Mutation Operators** - Risk, Indicators, Filters, Position, Signals, Stops
- **Multi-objective Fitness** - Return, Sharpe, Drawdown, Win Rate

### Market Intelligence
- **Regime Detection** - HMM-based Bull/Bear/Sideways/HighVol/LowVol
- **Order Flow** - Cumulative delta, large trades, stop hunt detection
- **Sentiment** - Fear & Greed, news, social, on-chain
- **Correlations** - Cointegration, pairs trading, rotation

### Execution
- **Paper Trading** - Simulated with realistic slippage
- **Live Trading** - Binance, Hyperliquid connectors
- **Risk Guards** - Circuit breakers, max drawdown, position limits

## 🔧 Development

```bash
# Run evolution UI
cd quantcore-evolution
npm install
npm run dev

# Run Python engine directly
python main.py --evolution --population 100 --generations 20
```

## 📈 Results (Paper Trading)

*Running live paper trading - updates coming soon!*

| Metric | Value |
|--------|-------|
| Sharpe Ratio | TBD |
| CAGR | TBD |
| Max Drawdown | TBD |
| Win Rate | TBD |

## 🤝 Contributing

Contributions welcome! Open an issue or PR.

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/quantcore-evolution.git

# Create feature branch
git checkout -b feature/amazing-new-module

# Commit and push
git add .
git commit -m "Add amazing new module"
git push origin feature/amazing-new-module
```

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This is **not financial advice**. Trading is risky. Past performance ≠ future results. Use at your own risk.

---

**Star if you think this is cool** ⭐ | [Report Issues](https://github.com/Kennethisaparadise/quantcore-evolution/issues)
