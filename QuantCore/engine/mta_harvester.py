"""
QuantCore - MTA Strategy Harvester v1.0
Head 31: Multi-Timeframe Analysis strategy harvester
"""

import random
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MTAStrategy:
    """Multi-Timeframe Analysis strategy."""
    name: str
    source: str
    htf: Dict  # Higher timeframe
    mtf: Dict  # Middle timeframe
    ltf: Dict  # Lower timeframe
    confluence_factors: List[str]
    risk_reward: str


class MTAStrategyHarvester:
    """Harvest and manage MTA strategies."""
    
    # Known MTA patterns
    KNOWN_STRATEGIES = [
        {
            "name": "Trend Continuation with Confluence",
            "source": "itrader.global",
            "htf": {
                "timeframe": "D1",
                "indicators": ["200 EMA", "market structure"],
                "condition": "Price above EMA = bullish"
            },
            "mtf": {
                "timeframe": "H1",
                "action": "correction to support zone",
                "tools": ["Fibonacci", "RSI divergence"]
            },
            "ltf": {
                "timeframe": "M15",
                "entry_triggers": ["pin bar", "engulfing", "liquidity sweep"]
            },
            "confluence_factors": ["HTF trend", "MTF zone", "LTF pattern", "volume"],
            "risk_reward": "1:3"
        },
        {
            "name": "Three-Layer Filter",
            "source": "SeputarForex",
            "htf": {
                "timeframe": "H4",
                "indicators": ["ADX", "200 SMA"],
                "action": "Trend detection"
            },
            "mtf": {
                "timeframe": "H1",
                "action": "Confirm trend direction"
            },
            "ltf": {
                "timeframe": "M15",
                "action": "Execute entry"
            },
            "confluence_factors": ["ADX > 25", "price above 200 SMA", "LTF pattern"],
            "risk_reward": "1:2"
        },
        {
            "name": "Factor of 4-6 Rule",
            "source": "TradingwithRayner",
            "htf": {
                "timeframe": "H4",
                "rule": "4-6x lower timeframe",
                "indicators": ["EMA 20", "EMA 50"]
            },
            "mtf": {
                "timeframe": "H1",
                "rule": "Confirms HTF direction"
            },
            "ltf": {
                "timeframe": "M15",
                "rule": "Entry execution"
            },
            "confluence_factors": ["EMA alignment", "candle pattern", "HTF direction"],
            "risk_reward": "1:2.5"
        },
        {
            "name": "ICT Killzone Entry",
            "source": "ICT Trading",
            "htf": {
                "timeframe": "D1",
                "action": "Identify market structure (HH/HL or LH/LL)"
            },
            "mtf": {
                "timeframe": "H1",
                "action": "Find order blocks, FVG, liquidity pools"
            },
            "ltf": {
                "timeframe": "M5",
                "action": "Execute at market structure break"
            },
            "confluence_factors": ["Order block", "FVG", "trend alignment", "Killzone time"],
            "risk_reward": "1:3"
        },
        {
            "name": " EMA Cross Scalper",
            "source": "LazyBear",
            "htf": {
                "timeframe": "H1",
                "indicators": ["EMA 200"],
                "action": "Trend direction"
            },
            "mtf": {
                "timeframe": "M15",
                "indicators": ["EMA 9", "EMA 21"],
                "action": "Cross signal"
            },
            "ltf": {
                "timeframe": "M5",
                "action": "Confirm and execute"
            },
            "confluence_factors": ["HTF trend", "EMA cross", "ADX filter"],
            "risk_reward": "1:2"
        },
        {
            "name": "Support Resistance Bounce",
            "source": "Amber Markets",
            "htf": {
                "timeframe": "D1",
                "action": "Identify major S/R levels"
            },
            "mtf": {
                "timeframe": "H4",
                "action": "Wait for price to approach S/R"
            },
            "ltf": {
                "timeframe": "M15",
                "action": "Bullish/bearish candle at level"
            },
            "confluence_factors": ["Major level", "candle pattern", "volume"],
            "risk_reward": "1:2"
        },
        {
            "name": "Volatility Expansion",
            "source": "DailyTrading",
            "htf": {
                "timeframe": "D1",
                "indicators": ["ATR", "BB width"],
                "action": "Identify high volatility"
            },
            "mtf": {
                "timeframe": "H1",
                "action": "Confirm momentum direction"
            },
            "ltf": {
                "timeframe": "M5",
                "action": "Enter on momentum"
            },
            "confluence_factors": ["ATR expansion", "momentum candle", "HTF direction"],
            "risk_reward": "1:2.5"
        },
        {
            "name": "News Reaction Trade",
            "source": "FundamentalTrader",
            "htf": {
                "timeframe": "D1",
                "action": "Pre-news positioning"
            },
            "mtf": {
                "timeframe": "H1",
                "action": "Post-news momentum"
            },
            "ltf": {
                "timeframe": "M5",
                "action": "Ride momentum"
            },
            "confluence_factors": ["News direction", "momentum", "HTF trend"],
            "risk_reward": "1:3"
        }
    ]
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        "W1": {"weight": 0, "hydra": "regime"},
        "D1": {"weight": 1, "hydra": "regime"},
        "H4": {"weight": 2, "hydra": "fractal"},
        "H1": {"weight": 3, "hydra": "fractal"},
        "M30": {"weight": 4, "hydra": "sentiment"},
        "M15": {"weight": 5, "hydra": "order_flow"},
        "M5": {"weight": 6, "hydra": "order_flow"},
        "M1": {"weight": 7, "hydra": "order_flow"},
    }
    
    def __init__(self):
        self.strategies: List[MTAStrategy] = []
        self._load_strategies()
    
    def _load_strategies(self):
        """Load known MTA strategies."""
        for s in self.KNOWN_STRATEGIES:
            strategy = MTAStrategy(
                name=s["name"],
                source=s["source"],
                htf=s["htf"],
                mtf=s["mtf"],
                ltf=s["ltf"],
                confluence_factors=s["confluence_factors"],
                risk_reward=s["risk_reward"]
            )
            self.strategies.append(strategy)
    
    def get_strategy(self, index: int) -> MTAStrategy:
        """Get strategy by index."""
        return self.strategies[index % len(self.strategies)]
    
    def get_all_strategies(self) -> List[Dict]:
        """Get all strategies as dicts."""
        return [
            {
                "name": s.name,
                "source": s.source,
                "htf": s.htf,
                "mtf": s.mtf,
                "ltf": s.ltf,
                "confluence": s.confluence_factors,
                "rr": s.risk_reward
            }
            for s in self.strategies
        ]
    
    def convert_to_seed(self, strategy: MTAStrategy) -> Dict:
        """Convert MTA strategy to Hydra seed."""
        # Map timeframes to Hydra modules
        htf_module = self._map_timeframe(strategy.htf.get("timeframe", "D1"))
        mtf_module = self._map_timeframe(strategy.mtf.get("timeframe", "H1"))
        ltf_module = self._map_timeframe(strategy.ltf.get("timeframe", "M15"))
        
        seed = {
            "name": f"MTA_{strategy.name[:20]}",
            "source": f"TradingView_{strategy.source}",
            "strategy_type": "mta",
            "htf": strategy.htf,
            "mtf": strategy.mtf,
            "ltf": strategy.ltf,
            "hydra_modules": [htf_module, mtf_module, ltf_module],
            "confluence_factors": strategy.confluence_factors,
            "risk_reward": strategy.risk_reward,
            "quality_score": random.uniform(0.6, 0.95)
        }
        
        return seed
    
    def _map_timeframe(self, tf: str) -> str:
        """Map timeframe to Hydra module."""
        mapped = self.TIMEFRAME_MAP.get(tf, {})
        return mapped.get("hydra", "regime")
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        strategies = self.get_all_strategies()
        
        strategy_rows = ""
        for i, s in enumerate(strategies):
            htf = s["htf"].get("timeframe", "N/A")
            mtf = s["mtf"].get("timeframe", "N/A")
            ltf = s["ltf"].get("timeframe", "N/A")
            
            strategy_rows += f"""
            <tr>
                <td>{s['name']}</td>
                <td>{s['source']}</td>
                <td>{htf} → {mtf} → {ltf}</td>
                <td>{len(s['confluence'])} factors</td>
                <td>{s['rr']}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>MTA Strategy Harvester - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #58a6ff;
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
            font-size: 32px;
            color: #58a6ff;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #21262d;
            padding: 12px;
            text-align: left;
            color: #8b949e;
            font-size: 11px;
            text-transform: uppercase;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #21262d;
            font-size: 12px;
        }}
        .timeframe-flow {{
            color: #3fb950;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="header">🌐 MTA STRATEGY HARVESTER</div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(strategies)}</div>
            <div class="stat-label">Strategies</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(set(s['source'] for s in strategies))}</div>
            <div class="stat-label">Sources</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sum(len(s['confluence']) for s in strategies)}</div>
            <div class="stat-label">Total Factors</div>
        </div>
    </div>
    
    <h3 style="color:#58a6ff">MTA Strategies</h3>
    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Source</th>
                <th>Timeframes</th>
                <th>Confluence</th>
                <th>R:R</th>
            </tr>
        </thead>
        <tbody>
            {strategy_rows}
        </tbody>
    </table>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("MTA STRATEGY HARVESTER TEST")
    print("=" * 50)
    
    harvester = MTAStrategyHarvester()
    
    print(f"\nStrategies loaded: {len(harvester.strategies)}")
    
    # Show strategies
    print("\nMTA Strategies:")
    for i, s in enumerate(harvester.strategies, 1):
        print(f"  {i}. {s.name}")
    
    # Convert to seed
    print("\nSeed conversion:")
    seed = harvester.convert_to_seed(harvester.strategies[0])
    print(f"  Name: {seed['name']}")
    print(f"  Modules: {seed['hydra_modules']}")
    print(f"  R:R: {seed['risk_reward']}")
    
    # Save HTML
    html = harvester.generate_html()
    with open("mta_harvester.html", "w") as f:
        f.write(html)
    
    print("\nSaved to mta_harvester.html")
    print("OK!")
