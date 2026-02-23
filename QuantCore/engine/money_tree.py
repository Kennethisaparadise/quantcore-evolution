"""
QuantCore - Money Tree Visualization v1.0
Head 23: The Compounding Money Tree
"""

import random
from datetime import datetime
from typing import Dict, List


class MoneyTree:
    """The Money Tree - visualizes compounding wealth."""
    
    def __init__(self, initial_equity: float = 100000):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.total_coins = 0
        self.leaves = []
        self.branches = [
            {"id": "regime", "name": "Regime", "color": "#f7931a", "trades": 0},
            {"id": "fractal", "name": "Fractal", "color": "#3fb950", "trades": 0},
            {"id": "sentiment", "name": "Sentiment", "color": "#a371f7", "trades": 0},
            {"id": "order_flow", "name": "Order Flow", "color": "#e3b341", "trades": 0},
            {"id": "correlation", "name": "Correlation", "color": "#58a6ff", "trades": 0},
            {"id": "compounding", "name": "Compounding", "color": "#f85149", "trades": 0},
        ]
    
    def update(self, new_equity: float):
        """Update with new equity."""
        self.equity = new_equity
    
    def add_trade(self, module: str, pnl: float):
        """Add a trade as a leaf."""
        is_profit = pnl > 0
        color = "#3fb950" if is_profit else "#f85149"
        
        self.leaves.append({
            "module": module,
            "pnl": pnl,
            "color": color
        })
        
        # Update branch
        for b in self.branches:
            if b["id"] == module:
                b["trades"] += 1
                break
        
        # Keep leaves bounded
        if len(self.leaves) > 100:
            self.leaves = self.leaves[-100:]
    
    def add_profit(self, amount: float):
        """Add profit coin."""
        self.total_coins += amount
    
    def get_state(self) -> Dict:
        """Get tree state."""
        growth = (self.equity / self.initial_equity - 1) * 100
        trunk_height = 100 + max(0, growth) * 3
        
        return {
            "equity": self.equity,
            "growth": growth,
            "trunk_height": trunk_height,
            "total_coins": self.total_coins,
            "leaves": len(self.leaves),
            "branches": self.branches
        }
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        s = self.get_state()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Money Tree - QuantCore</title>
    <style>
        body {{
            background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
            min-height: 100vh;
            font-family: monospace;
            color: #e6edf3;
            margin: 0;
            overflow: hidden;
        }}
        .container {{
            position: relative;
            width: 100%;
            height: 100vh;
        }}
        .stats {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(22, 27, 34, 0.9);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3fb950;
        }}
        .ground {{
            position: absolute;
            bottom: 0;
            width: 100%;
            height: 60px;
            background: linear-gradient(180deg, #238636 0%, #1a1f16 100%);
        }}
        .trunk {{
            position: absolute;
            bottom: 60px;
            left: 50%;
            transform: translateX(-50%);
            width: 30px;
            height: {s['trunk_height']}px;
            background: linear-gradient(90deg, #5d4037, #8d6e63);
            border-radius: 10px;
        }}
        .foliage {{
            position: absolute;
            bottom: {s['trunk_height'] + 50}px;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 150px;
            background: radial-gradient(ellipse, rgba(63,185,80,0.8), transparent);
            filter: blur(10px);
        }}
        .legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(22,27,34,0.9);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }}
        .branch-label {{
            position: absolute;
            color: #8b949e;
            font-size: 10px;
        }}
        .coin-pile {{
            position: absolute;
            bottom: 70px;
            right: 15%;
            display: flex;
            flex-wrap: wrap;
            max-width: 80px;
        }}
        .coin {{
            width: 15px;
            height: 15px;
            background: radial-gradient(circle at 30% 30%, #ffd700, #b8860b);
            border-radius: 50%;
            margin: 2px;
            box-shadow: 0 0 5px rgba(255,215,0,0.5);
        }}
        .growth {{
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            background: rgba(22,27,34,0.9);
            border-radius: 20px;
            border: 1px solid {'#3fb950' if s['growth'] >= 0 else '#f85149'};
            color: {'#3fb950' if s['growth'] >= 0 else '#f85149'};
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="growth">📈 {s['growth']:+.1f}% Growth</div>
        
        <div class="stats">
            <div class="stat-label">Equity</div>
            <div class="stat-value">${s['equity']:,.0f}</div>
            <div class="stat-label" style="margin-top:15px">Coins Collected</div>
            <div class="stat-value">${s['total_coins']:,.0f}</div>
            <div class="stat-label" style="margin-top:15px">Trades (Leaves)</div>
            <div class="stat-value">{s['leaves']}</div>
        </div>
        
        <div class="legend">
            <div style="color:#8b949e;font-size:11px;margin-bottom:10px;">STRATEGIES</div>
"""
        
        for b in s['branches']:
            html += f"""
            <div style="display:flex;align-items:center;gap:8px;margin:5px 0;">
                <div style="width:10px;height:10px;background:{b['color']};border-radius:50%;"></div>
                <span style="font-size:11px;">{b['name']} ({b['trades']})</span>
            </div>"""
        
        html += """
        </div>
        
        <div class="trunk"></div>
        <div class="foliage"></div>
        
        <div class="coin-pile">
"""
        
        # Add coins
        coin_count = min(30, int(s['total_coins'] / 500))
        for _ in range(coin_count):
            html += '<div class="coin"></div>'
        
        html += """
        </div>
        
        <div class="ground"></div>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("MONEY TREE TEST")
    print("=" * 50)
    
    tree = MoneyTree(100000)
    
    # Simulate trades
    for i in range(10):
        pnl = random.uniform(-100, 300)
        tree.add_trade("regime", pnl)
        if pnl > 50:
            tree.add_profit(pnl)
    
    tree.update(110000)
    
    state = tree.get_state()
    print(f"Equity: ${state['equity']:,.0f}")
    print(f"Growth: {state['growth']:+.1f}%")
    print(f"Coins: ${state['total_coins']:,.0f}")
    print(f"Leaves: {state['leaves']}")
    
    html = tree.generate_html()
    with open("money_tree.html", "w") as f:
        f.write(html)
    
    print("\nSaved to money_tree.html")
    print("OK!")
