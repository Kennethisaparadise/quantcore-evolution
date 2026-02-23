"""
QuantCore - Non-Directional Strategy Engine v1.0
Head 40: Market-neutral, pairs, volatility, and event-driven strategies
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NonDirectionalStrategy:
    """A non-directional trading strategy."""
    name: str
    category: str  # pairs, volatility, event, mean_reversion, market_neutral
    description: str
    parameters: Dict


class NonDirectionalEngine:
    """
    Non-directional strategy engine for market-neutral profits.
    """
    
    # Known strategies
    STRATEGIES = [
        {
            "name": "Pairs Trading",
            "category": "pairs",
            "description": "Long undervalued, short overvalued correlated pair",
            "parameters": {
                "correlation_threshold": 0.8,
                "spread_threshold": 2.0,
                "lookback": 30
            }
        },
        {
            "name": "Statistical Arbitrage",
            "category": "arbitrage",
            "description": "Exploit price inefficiencies between similar assets",
            "parameters": {
                "z_score_threshold": 2.0,
                "lookback": 50,
                "exit_zscore": 0.5
            }
        },
        {
            "name": "Mean Reversion",
            "category": "mean_reversion",
            "description": "Buy at support, sell at resistance in range",
            "parameters": {
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_oversold": 30,
                "rsi_overbought": 70
            }
        },
        {
            "name": "Volatility Straddle",
            "category": "volatility",
            "description": "Buy straddle before major events (earnings, data)",
            "parameters": {
                "event_lookahead": 24,
                "implied_vol_threshold": 30,
                "profit_target_pct": 5.0
            }
        },
        {
            "name": "Market Neutral",
            "category": "market_neutral",
            "description": "Equal long/short to neutralize directional risk",
            "parameters": {
                "beta_target": 0.0,
                "rebalance_threshold": 0.1,
                "hedge_ratio": 1.0
            }
        },
        {
            "name": "Range Breakout",
            "category": "mean_reversion",
            "description": "Trade breakouts from consolidation",
            "parameters": {
                "consolidation_bars": 20,
                "breakout_threshold": 0.5,
                "stop_pct": 1.0
            }
        },
        {
            "name": "Event Gap",
            "category": "event",
            "description": "Trade gaps around news/events",
            "parameters": {
                "gap_threshold": 0.02,
                "fade_gap": True,
                "profit_target_pct": 2.0
            }
        },
        {
            "name": "VIX Spike",
            "category": "volatility",
            "description": "Trade VIX spikes (fear gauge)",
            "parameters": {
                "vix_threshold": 20,
                "vix_spike_pct": 15,
                "direction": "long"
            }
        },
    ]
    
    def __init__(self):
        self.strategies: List[NonDirectionalStrategy] = []
        self._load_strategies()
        
        # Performance tracking
        self.performance = {}
        for s in self.STRATEGIES:
            self.performance[s["name"]] = {
                "trades": 0,
                "wins": 0,
                "pnl": 0
            }
    
    def _load_strategies(self):
        """Load known strategies."""
        for s in self.STRATEGIES:
            strat = NonDirectionalStrategy(
                name=s["name"],
                category=s["category"],
                description=s["description"],
                parameters=s["parameters"]
            )
            self.strategies.append(strat)
    
    def get_strategies(self) -> List[Dict]:
        """Get all strategies."""
        return [
            {
                "name": s.name,
                "category": s.category,
                "description": s.description,
                "parameters": s.parameters
            }
            for s in self.strategies
        ]
    
    def get_strategies_by_category(self, category: str) -> List[str]:
        """Get strategy names by category."""
        return [s.name for s in self.strategies if s.category == category]
    
    def select_strategy(self, market_conditions: Dict) -> Optional[str]:
        """Select best strategy based on conditions."""
        regime = market_conditions.get("regime", "unknown")
        volatility = market_conditions.get("volatility", 0.03)
        has_events = market_conditions.get("has_events", False)
        
        # Select based on conditions
        if volatility > 0.05:
            # High volatility - volatility strategies
            candidates = self.get_strategies_by_category("volatility")
            if candidates:
                return random.choice(candidates)
        
        if regime == "sideways":
            # Range-bound - mean reversion
            candidates = self.get_strategies_by_category("mean_reversion")
            if candidates:
                return random.choice(candidates)
        
        if has_events:
            # Event-driven opportunities
            candidates = self.get_strategies_by_category("event")
            if candidates:
                return random.choice(candidates)
        
        # Default to market neutral
        candidates = self.get_strategies_by_category("market_neutral")
        if candidates:
            return candidates[0]
        
        return self.strategies[0].name
    
    def generate_signal(self, strategy_name: str, market_data: Dict) -> Dict:
        """Generate signal for a strategy."""
        # Simplified signal generation
        # In production, would use actual calculations
        
        return {
            "strategy": strategy_name,
            "direction": random.choice(["long", "short", "neutral"]),
            "confidence": random.uniform(0.5, 0.9),
            "parameters": {}
        }
    
    def record_trade(self, strategy_name: str, pnl: float, is_win: bool):
        """Record trade result."""
        if strategy_name in self.performance:
            self.performance[strategy_name]["trades"] += 1
            if is_win:
                self.performance[strategy_name]["wins"] += 1
            self.performance[strategy_name]["pnl"] += pnl
    
    def get_performance(self) -> Dict:
        """Get performance by strategy."""
        result = {}
        for name, stats in self.performance.items():
            if stats["trades"] > 0:
                result[name] = {
                    "trades": stats["trades"],
                    "wins": stats["wins"],
                    "win_rate": (stats["wins"] / stats["trades"]) * 100,
                    "pnl": stats["pnl"]
                }
        return result
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        strategies = self.get_strategies()
        perf = self.get_performance()
        
        # Group by category
        categories = {}
        for s in strategies:
            if s["category"] not in categories:
                categories[s["category"]] = []
            categories[s["category"]].append(s)
        
        cat_html = ""
        for cat, strats in categories.items():
            strat_rows = ""
            for s in strats:
                p = perf.get(s["name"], {"trades": 0, "win_rate": 0, "pnl": 0})
                strat_rows += f"""
                <tr>
                    <td>{s['name']}</td>
                    <td>{s['description']}</td>
                    <td>{p.get('trades', 0)}</td>
                    <td>{p.get('win_rate', 0):.1f}%</td>
                    <td style="color: {'#3fb950' if p.get('pnl', 0) > 0 else '#f85149'}">
                        ${p.get('pnl', 0):.2f}
                    </td>
                </tr>
                """
            
            cat_html += f"""
            <div class="category">
                <h3 style="color:{self._cat_color(cat)}">{cat.upper().replace('_', ' ')}</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Description</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>PnL</th>
                        </tr>
                    </thead>
                    <tbody>
                        {strat_rows}
                    </tbody>
                </table>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Non-Directional Engine - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #f778ba;
            margin-bottom: 20px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat {{
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }}
        .stat-value {{
            font-size: 28px;
            color: #f778ba;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
        }}
        .category {{
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #21262d;
            padding: 10px;
            text-align: left;
            color: #8b949e;
            font-size: 11px;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #21262d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">🎯 NON-DIRECTIONAL ENGINE</div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(strategies)}</div>
            <div class="stat-label">Strategies</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(categories)}</div>
            <div class="stat-label">Categories</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sum(p['trades'] for p in perf.values())}</div>
            <div class="stat-label">Total Trades</div>
        </div>
    </div>
    
    {cat_html}
</body>
</html>
"""
        return html
    
    def _cat_color(self, category: str) -> str:
        colors = {
            "pairs": "#58a6ff",
            "arbitrage": "#a371f7",
            "volatility": "#f7931a",
            "event": "#f85149",
            "mean_reversion": "#3fb950",
            "market_neutral": "#f778ba"
        }
        return colors.get(category, "#8b949e")


if __name__ == "__main__":
    print("=" * 50)
    print("NON-DIRECTIONAL ENGINE TEST")
    print("=" * 50)
    
    engine = NonDirectionalEngine()
    
    # Show strategies
    print(f"\nStrategies loaded: {len(engine.strategies)}")
    
    categories = {}
    for s in engine.strategies:
        if s.category not in categories:
            categories[s.category] = []
        categories[s.category].append(s.name)
    
    print("\nBy Category:")
    for cat, names in categories.items():
        print(f"  {cat}: {', '.join(names)}")
    
    # Test selection
    conditions = {"regime": "sideways", "volatility": 0.04, "has_events": False}
    selected = engine.select_strategy(conditions)
    print(f"\nSelected for sideways market: {selected}")
    
    # Save HTML
    html = engine.generate_html()
    with open("non_directional.html", "w") as f:
        f.write(html)
    
    print("\nSaved to non_directional.html")
    print("OK!")
