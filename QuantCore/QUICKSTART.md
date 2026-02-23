# QuantCore - Quick Start Guide

## Installation

```bash
git clone <repo>
cd QuantCore
pip install -r requirements.txt
```

## Configuration

1. Copy the example config:
```bash
cp config/config.example.json config/config.json
```

2. Add your API keys:
- OpenAlgo for Indian markets
- Polymarket API key (optional for public data)

## Usage

### Download Market Data
```bash
python main.py --data --symbol RELIANCE --exchange NSE --start 2024-01-01
```

### Run Backtest
```bash
python main.py --backtest --strategy strategies/examples.py --data data/csv/NSE_RELIANCE_1d_20240101_20250101.csv
```

### Launch Dashboard
```bash
python main.py --dashboard
```

## Strategy Development

Create a strategy by extending `StrategyBase`:

```python
from engine.base import StrategyBase

class MyStrategy(StrategyBase):
    def init(self):
        self.sma = self.indicator('sma', period=20)
    
    def next(self):
        if self.crossover(self.sma, self.data['close']):
            self.buy()
        elif self.crossunder(self.sma, self.data['close']):
            self.sell()
```

## Supported Markets

| Market | Adapter | Data Types |
|--------|---------|------------|
| India | OpenAlgo | Equity, F&O |
| Prediction | Polymarket | Binary, Scalar |
| Crypto | CCXT | Spot, Futures |
| Forex | CCXT | Spot |

## Risk Management

QuantCore implements:
- Position limits (max 25% per position)
- Daily loss limits (max 5%)
- Drawdown protection
- Order frequency limits
