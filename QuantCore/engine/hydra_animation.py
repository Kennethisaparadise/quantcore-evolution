"""
QuantCore - Hydra's Heads Animation v1.0
Head 24: Animated serpent heads for all 23 modules
"""

import random
from datetime import datetime


class HydraAnimation:
    """Animated hydra heads for all modules."""
    
    def __init__(self):
        self.modules = [
            {"id": "regime", "name": "Regime", "color": "#f7931a", "activity": 0, "trades": 0},
            {"id": "fractal", "name": "Fractal", "color": "#3fb950", "activity": 0, "trades": 0},
            {"id": "adversarial", "name": "Adversarial", "color": "#f85149", "activity": 0, "trades": 0},
            {"id": "timeline", "name": "Timeline", "color": "#58a6ff", "activity": 0, "trades": 0},
            {"id": "dynamic_regime", "name": "Dynamic Regime", "color": "#a371f7", "activity": 0, "trades": 0},
            {"id": "order_flow", "name": "Order Flow", "color": "#e3b341", "activity": 0, "trades": 0},
            {"id": "sentiment", "name": "Sentiment", "color": "#f778ba", "activity": 0, "trades": 0},
            {"id": "correlation", "name": "Correlation", "color": "#79c0ff", "activity": 0, "trades": 0},
            {"id": "live_trading", "name": "Live Trading", "color": "#3fb950", "activity": 0, "trades": 0},
            {"id": "realtime_adapt", "name": "Real-Time Adapt", "color": "#a371f7", "activity": 0, "trades": 0},
            {"id": "compounding", "name": "Compounding", "color": "#ffd700", "activity": 0, "trades": 0},
            {"id": "imbalance", "name": "Imbalance", "color": "#8b949e", "activity": 0, "trades": 0},
            {"id": "iceberg", "name": "Iceberg", "color": "#79c0ff", "activity": 0, "trades": 0},
            {"id": "spoofing", "name": "Spoofing", "color": "#f85149", "activity": 0, "trades": 0},
            {"id": "arbitrage", "name": "Arbitrage", "color": "#3fb950", "activity": 0, "trades": 0},
            {"id": "algo_hunter", "name": "Algo Hunter", "color": "#f7931a", "activity": 0, "trades": 0},
            {"id": "patterns", "name": "Patterns", "color": "#a371f7", "activity": 0, "trades": 0},
            {"id": "fee_aware", "name": "Fee Aware", "color": "#58a6ff", "activity": 0, "trades": 0},
            {"id": "tax_aware", "name": "Tax Aware", "color": "#3fb950", "activity": 0, "trades": 0},
            {"id": "multi_account", "name": "Multi-Account", "color": "#e3b341", "activity": 0, "trades": 0},
            {"id": "withdrawal", "name": "Withdrawal", "color": "#f778ba", "activity": 0, "trades": 0},
            {"id": "dashboard", "name": "Dashboard", "color": "#8b949e", "activity": 0, "trades": 0},
            {"id": "money_tree", "name": "Money Tree", "color": "#ffd700", "activity": 0, "trades": 0},
        ]
    
    def record_trade(self, module_id: str):
        """Record a trade for a module."""
        for m in self.modules:
            if m["id"] == module_id:
                m["trades"] += 1
                m["activity"] = 100  # Max activity
                break
    
    def decay_activity(self):
        """Decay activity over time."""
        for m in self.modules:
            m["activity"] = max(0, m["activity"] - 5)
    
    def get_state(self):
        """Get current state."""
        return self.modules
    
    def generate_html(self) -> str:
        """Generate animated HTML."""
        modules = self.modules
        n = len(modules)
        
        # Generate head elements
        heads = ""
        for i, m in enumerate(modules):
            angle = (360 / n) * i - 90
            radius = 180
            x = 250 + radius * (1 if angle < 0 else 1) * abs(angle / 180)
            y = 250 + radius * (1 if angle < 90 or angle > 270 else -1) * abs((angle + 180) % 360 - 180) / 180
            
            # Convert angle to radians for x,y
            import math
            rad = angle * math.pi / 180
            x = 250 + radius * math.cos(rad)
            y = 250 + radius * math.sin(rad)
            
            activity = m["activity"]
            scale = 0.5 + (activity / 100) * 0.5
            glow = f"drop-shadow(0 0 {activity/10}px {m['color']})"
            
            heads += f"""
            <g class="head" data-id="{m['id']}" data-trades="{m['trades']}" 
               transform="translate({x},{y}) scale({scale})">
                <ellipse cx="0" cy="0" rx="25" ry="15" fill="{m['color']}" 
                    filter="{glow}" opacity="{0.6 + activity/200}"/>
                <circle cx="10" cy="-5" r="5" fill="#fff"/>
                <circle cx="10" cy="-5" r="2" fill="#000"/>
                <polygon points="-5,5 0,15 5,5" fill="{m['color']}"/>
                <text y="25" text-anchor="middle" font-size="8" fill="#8b949e">{m['name']}</text>
            </g>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hydra - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            font-family: monospace;
            color: #e6edf3;
        }}
        .container {{
            text-align: center;
        }}
        svg {{
            width: 500px;
            height: 500px;
        }}
        .head {{
            cursor: pointer;
            transition: transform 0.3s;
        }}
        .head:hover {{
            transform: scale(1.2);
        }}
        .title {{
            font-size: 24px;
            margin-bottom: 10px;
            color: #58a6ff;
        }}
        .subtitle {{
            color: #8b949e;
            font-size: 12px;
            margin-bottom: 20px;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3fb950;
        }}
        .stat-label {{
            font-size: 10px;
            color: #8b949e;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">🐙 QUANTUM HYDRA</div>
        <div class="subtitle">23 Heads | {len([m for m in modules if m['trades'] > 0])} Active</div>
        
        <svg viewBox="0 0 500 500">
            <!-- Central Tree -->
            <circle cx="250" cy="250" r="40" fill="#238636" opacity="0.5">
                <animate attributeName="r" values="40;45;40" dur="2s" repeatCount="indefinite"/>
            </circle>
            <text x="250" y="255" text-anchor="middle" font-size="12" fill="#fff">TREE</text>
            
            <!-- Hydra Heads -->
            {heads}
        </svg>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{sum(m['trades'] for m in modules)}</div>
                <div class="stat-label">Total Trades</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len([m for m in modules if m['activity'] > 50])}</div>
                <div class="stat-label">Active Heads</div>
            </div>
        </div>
    </div>
    
    <script>
        // Animation loop
        function animate() {{
            const heads = document.querySelectorAll('.head');
            heads.forEach(head => {{
                const trades = parseInt(head.dataset.trades);
                if (trades > 0) {{
                    // Random snap animation
                    if (Math.random() < 0.02) {{
                        head.style.transform = head.style.transform + ' rotate(5deg)';
                        setTimeout(() => {{
                            head.style.transform = head.style.transform.replace(' rotate(5deg)', '');
                        }}, 100);
                    }}
                }}
            }});
            requestAnimationFrame(animate);
        }}
        animate();
    </script>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("HYDRA ANIMATION TEST")
    print("=" * 50)
    
    hydra = HydraAnimation()
    
    # Simulate trades
    for i in range(20):
        module = random.choice([m["id"] for m in hydra.modules])
        hydra.record_trade(module)
    
    # Decay
    hydra.decay_activity()
    
    state = hydra.get_state()
    active = len([m for m in state if m["trades"] > 0])
    total = sum(m["trades"] for m in state)
    
    print(f"Active heads: {active}")
    print(f"Total trades: {total}")
    
    html = hydra.generate_html()
    with open("hydra.html", "w") as f:
        f.write(html)
    
    print("\nSaved to hydra.html")
    print("OK!")
