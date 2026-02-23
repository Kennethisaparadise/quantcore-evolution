# QuantCore - Multi-Market Quantitative Trading Platform

A production-ready, modular algorithmic trading platform supporting multiple markets (Indian, Crypto, Forex, Prediction Markets) with strategy engine, backtesting, and risk management.

## Features

### Core Components
- **Market Data Pipeline** - Multi-source data ingestion (OpenAlgo, Polymarket, Generic APIs)
- **Strategy Engine** - Python-based strategy development with hot-reload support
- **Central Risk Management** - Position limits, stop-loss, drawdown controls
- **Execution Engine** - Unified order management across brokers
- **Trade Ledger** - Complete audit trail of all trades
- **Backtesting Harness** - VectorBT-powered backtesting
- **Dashboard** - Real-time monitoring and control

### Supported Markets
| Market | Adapter | Data Types |
|--------|---------|------------|
| India (NSE/BSE) | OpenAlgo | Equity, F&O, Commodities, Currency |
| Prediction Markets | Polymarket | Binary, Categorical, Scalar |
| Crypto | Generic (CCXT) | Spot, Futures, Perps |
| Forex | Generic | Spot, Forwards |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/quantcore/QuantCore.git
cd QuantCore

# Install dependencies
pip install -r requirements.txt

# Configure your broker/API keys
cp config/config.example.json config/config.json
# Edit config.json with your credentials

# Run the dashboard
python -m dashboard.app

# Or run a backtest
python -m backtest.run --strategy examples/ma_crossover.py --start 2024-01-01 --end 2024-12-31
```

## Project Structure

```
QuantCore/
├── adapters/           # Market data adapters
│   ├── openalgo/      # Indian market data (NSE/BSE)
│   ├── polymarket/    # Prediction market data
│   └── generic/       # Generic CCXT-based adapters
├── backtest/          # VectorBT backtesting engine
├── config/            # Configuration files
├── data/              # Data storage
│   ├── csv/           # Raw CSV downloads
│   └── parquet/       # Processed parquet files
├── dashboard/         # React dashboard
├── engine/            # Strategy execution engine
├── logs/              # Application logs
├── risk/              # Risk management module
├── strategies/        # Strategy scripts
├── utils/             # Utility functions
└── ledger/            # Trade ledger/Journal
```

## Strategy Example

```python
from engine.base import Strategy

class MACrossover(Strategy):
    params = {
        'fast_period': 10,
        'slow_period': 20,
    }
    
    def init(self):
        self.sma_fast = self.indicator('sma', period=self.params.fast_period)
        self.sma_slow = self.indicator('sma', period=self.params.slow_period)
    
    def next(self):
        if self.crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif self.crossover(self.sma_slow, self.sma_fast):
            self.sell()
```

## API Usage

```python
from adapters.openalgo import OpenAlgoAdapter
from engine.backtest import BacktestEngine

# Download data
adapter = OpenAlgoAdapter(api_key="your_key")
data = adapter.download("RELIANCE", start="2024-01-01", end="2024-12-31")

# Run backtest
engine = BacktestEngine()
results = engine.run(
    strategy="strategies/ma_crossover.py",
    data=data,
    cash=100000,
    fees=0.001
)
print(results.total_return)
```

## Configuration

See `config/config.example.json` for configuration options:

- Broker API keys
- Risk management parameters
- Data storage settings
- Dashboard configuration
- Logging preferences

## Risk Management

QuantCore implements centralized risk controls:

- **Position Limits** - Max % of portfolio per position
- **Drawdown Limits** - Max daily/monthly loss thresholds
- **Order Limits** - Max orders per day
- **Slippage Controls** - Max acceptable slippage
- **Stop Losses** - Automatic stop-loss execution

## Dashboard

The QuantCore dashboard provides:

- Real-time position monitoring
- P&L visualization
- Strategy performance metrics
- Trade history
- Risk exposure alerts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: https://docs.quantcore.io
- Issues: GitHub Issues
- Discord: https://discord.gg/quantcore
