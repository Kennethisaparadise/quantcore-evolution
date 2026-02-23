# QuantCore Evolution - Best Practices

## Overview

QuantCore is a 34-headed self-evolving trading system that trades BTC/USDT. This document captures best practices for running, maintaining, and expanding the system.

## System Architecture

### The 34 Heads

| Head | Module | Description |
|------|--------|-------------|
| 1-11 | Core System | Regime, Fractal, Adversarial, Timeline, Dynamic Regime, Order Flow, Sentiment, Correlation, Live Trading, Real-Time Adaptation, Compounding |
| 12-17 | Order Book | Imbalance, Iceberg, Spoofing, Arbitrage, Algo Hunter, Patterns |
| 18-19 | Financial | Fee-Aware Execution, Tax-Aware Optimization |
| 20-21 | Account Mgmt | Multi-Account Allocator, Withdrawal Optimizer |
| 22-25 | Visualization | Dashboard, Money Tree, Hydra Animation, Performance Cockpit |
| 26-31 | Harvesters | TradingView, Mutation, Operator Tuner, Diversity, Meta-Fitness, MTA |
| 32-34 | Extras | Reddit Results, Cosmic Harvester, Numerical Harmony |

## Running Paper Trading

### Start
```bash
cd /home/kenner/clawd/QuantCore
python run_paper_trading.py
```

### Monitor
```bash
tail -f paper_trading.log
```

### View Dashboard
```bash
# Start local server
python -m http.server 8080

# Open in browser
http://localhost:8080/dashboard.html
```

## Risk Management

### Circuit Breakers (engine/live_trading.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_daily_loss | 15% | Stop if daily loss exceeds |
| consecutive_losses | 10 | Stop after N consecutive losses |
| hourly_loss_limit | 2% | Hourly loss threshold |

### Adjusting Limits
```python
# In engine/live_trading.py
max_daily_loss: float = 0.15  # 15%
consecutive_losses: int = 10
```

## Visualizations

All visualizations available at:
- `dashboard.html` - Main dashboard
- `money_tree.html` - Compounding visualization
- `hydra.html` - 34-headed animation
- `performance_cockpit.html` - Analytics
- `tv_harvester.html` - TradingView strategies
- `mutation_harvester.html` - Mutation operators
- `operator_tuner.html` - Weight tuning
- `diversity_dashboard.html` - Population diversity
- `meta_fitness.html` - Fitness evolution
- `mta_harvester.html` - MTA strategies
- `cosmic_harvester.html` - Lunar/Delta
- `numerical_harmony.html` - Fibonacci/Gann

## Adding New Modules

### Steps
1. Create `engine/new_module.py`
2. Implement core logic
3. Add to `run_paper_trading.py`
4. Update this document
5. Commit to GitHub

### Template
```python
class NewModule:
    def __init__(self):
        self.name = "new_module"
    
    def process(self, data):
        # Your logic here
        return result
    
    def get_status(self):
        return {"status": "active"}
```

## Git Workflow

### Commit Changes
```bash
git add .
git commit -m "Add Head X: Description"
git push github master
```

### Version Tag
```bash
git tag -a vX.Y.Z -m "Version X.Y.Z"
git push github master --tags
```

## Performance Metrics

### Target Benchmarks
| Metric | Target | Current |
|--------|--------|---------|
| Win Rate | >50% | TBD |
| Profit Factor | >1.5 | TBD |
| Max Drawdown | <10% | TBD |
| Sharpe Ratio | >1.0 | TBD |

## Troubleshooting

### Circuit Breaker Triggering Too Often
- Increase `max_daily_loss`
- Increase `consecutive_losses`
- Check market conditions

### No Trades Executing
- Check API connectivity
- Verify symbol availability
- Review signal generation logic

### High Drawdown
- Reduce position size
- Tighten stop losses
- Enable more conservative modules

## Maintenance

### Daily
- Check paper trading log
- Review equity curve
- Monitor for errors

### Weekly
- Update best practices
- Backup configuration
- Review performance metrics

### Monthly
- Update dependencies
- Review and optimize modules
- Archive old logs

## Links

- GitHub: https://github.com/Kennethisaparadise/quantcore-evolution
- Documentation: See README.md

---

*Last Updated: 2026-02-23*
*Version: 2.7.0*
*Heads: 34*
