"""
QuantCore - Numerical Harmony Engine v1.0
Head 34: Fibonacci, Gann, Lucas numbers and numerical patterns
"""

import math
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FibonacciLevel:
    """A Fibonacci retracement/extension level."""
    level: float
    price: float
    type: str  # retracement or extension


class NumericalHarmonyEngine:
    """
    Numerical harmony engine: Fibonacci, Gann, Lucas, binary switch.
    """
    
    # Fibonacci levels
    FIB_RETRACEMENTS = [0.236, 0.382, 0.5, 0.618, 0.786]
    FIB_EXTENSIONS = [1.272, 1.382, 1.618, 2.0, 2.618]
    
    # Lucas numbers
    LUCAS_SEQ = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199]
    
    # Gann fractions
    GANN_FRACTIONS = [
        0.125, 0.25, 0.333, 0.5, 0.667, 0.75, 0.875,
        1.0, 1.25, 1.333, 1.5, 1.618, 1.667, 1.75, 2.0
    ]
    
    # Significant digital roots
    DIGITAL_ROOTS = {
        0: "Completion",
        1: "New beginning",
        2: "Balance",
        3: "Creation",
        4: "Stability",
        5: "Change",
        6: "Harmony",
        7: "Spiritual",
        8: "Power",
        9: "Completion"
    }
    
    def __init__(self):
        self.current_swing_high = 0
        self.current_swing_low = 0
        self.binary_switch_state = "A"  # A or B
    
    def set_swing(self, high: float, low: float):
        """Set current swing high/low for Fibonacci calculations."""
        self.current_swing_high = high
        self.current_swing_low = low
    
    def calculate_fibonacci(self, high: float = None, low: float = None) -> Dict:
        """Calculate Fibonacci retracement and extension levels."""
        if high is None:
            high = self.current_swing_high
        if low is None:
            low = self.current_swing_low
        
        diff = high - low
        
        retracements = {}
        for level in self.FIB_RETRACEMENTS:
            retracements[level] = high - (diff * level)
        
        extensions = {}
        for level in self.FIB_EXTENSIONS:
            extensions[level] = low + (diff * level)
        
        return {
            "swing": {"high": high, "low": low},
            "retracements": retracements,
            "extensions": extensions,
            "golden_pocket": {
                "low": retracements.get(0.618),
                "high": retracements.get(0.65, retracements.get(0.618) * 1.02)
            }
        }
    
    def find_nearest_fib_level(self, price: float) -> Dict:
        """Find nearest Fibonacci level to current price."""
        fib = self.calculate_fibonacci()
        
        all_levels = []
        for level, price_level in fib["retracements"].items():
            all_levels.append((abs(price - price_level), level, price_level, "retracement"))
        for level, price_level in fib["extensions"].items():
            all_levels.append((abs(price - price_level), level, price_level, "extension"))
        
        # Sort by distance
        all_levels.sort(key=lambda x: x[0])
        
        if all_levels:
            nearest = all_levels[0]
            return {
                "distance": nearest[0],
                "level": nearest[1],
                "price": nearest[2],
                "type": nearest[3],
                "is_golden": nearest[1] in [0.618, 0.786, 1.272, 1.618]
            }
        
        return {}
    
    def calculate_lucas_levels(self, base: float) -> Dict:
        """Calculate Lucas number levels."""
        levels = {}
        for i, lucas_num in enumerate(self.LUCAS_SEQ):
            levels[f"lucas_{i}"] = base * (lucas_num / 100)
        return levels
    
    def calculate_gann_levels(self, high: float, low: float) -> Dict:
        """Calculate Gann fraction levels."""
        diff = high - low
        levels = {}
        
        for frac in self.GANN_FRACTIONS:
            levels[f"gann_{frac}"] = low + (diff * frac)
        
        return levels
    
    def get_binary_switch_signal(self, volatility: float, threshold: float = 0.02) -> Dict:
        """Binary switch between two strategies based on volatility."""
        # Switch logic
        if volatility > threshold and self.binary_switch_state == "A":
            self.binary_switch_state = "B"
        elif volatility < threshold and self.binary_switch_state == "B":
            self.binary_switch_state = "A"
        
        return {
            "state": self.binary_switch_state,
            "strategy": "momentum" if self.binary_switch_state == "A" else "mean_reversion",
            "volatility": volatility,
            "threshold": threshold
        }
    
    def calculate_digital_root(self, value: int = None) -> int:
        """Calculate digital root of a number."""
        if value is None:
            # Use current timestamp
            now = datetime.now()
            value = now.hour * 10000 + now.minute * 100 + now.second
        
        while value >= 10:
            value = sum(int(d) for d in str(value))
        
        return value
    
    def get_digital_root_timer(self) -> Dict:
        """Get current digital root with significance."""
        now = datetime.now()
        
        # Time-based root
        time_value = now.hour * 10000 + now.minute * 100 + now.second
        time_root = self.calculate_digital_root(time_value)
        
        # Date-based root
        date_value = now.year * 10000 + now.month * 100 + now.day
        date_root = self.calculate_digital_root(date_value)
        
        return {
            "time_root": time_root,
            "time_meaning": self.DIGITAL_ROOTS.get(time_root, ""),
            "date_root": date_root,
            "date_meaning": self.DIGITAL_ROOTS.get(date_root, ""),
            "significant": time_root in [3, 7, 9]
        }
    
    def calculate_confluence(self, price: float, indicators: Dict) -> Dict:
        """Calculate confluence score of multiple signals."""
        score = 0
        factors = []
        
        # Check Fibonacci proximity
        fib_level = self.find_nearest_fib_level(price)
        if fib_level.get("distance", 999) < 100:  # Within $100
            score += 30
            factors.append(f"Fib {fib_level.get('level')} nearby")
        
        # Check digital root significance
        dr = self.get_digital_root_timer()
        if dr["significant"]:
            score += 20
            factors.append(f"Digital root {dr['time_root']}")
        
        # Check binary switch state
        bs = self.get_binary_switch_signal(indicators.get("volatility", 0.01))
        if bs["state"] == "B":
            score += 20
            factors.append("Binary switch B")
        
        # Check if at golden pocket
        if fib_level.get("is_golden"):
            score += 30
            factors.append("Golden pocket")
        
        return {
            "score": min(100, score),
            "factors": factors,
            "interpretation": "high_confluence" if score >= 60 else "low_confluence"
        }
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        # Get current state
        fib = self.calculate_fibonacci(50000, 45000)  # Example
        dr = self.get_digital_root_timer()
        bs = self.get_binary_switch_signal(0.015)
        
        # Build Fibonacci table
        fib_rows = ""
        for level, price in fib["retracements"].items():
            is_golden = level in [0.618, 0.786]
            fib_rows += f"""
            <tr style="{('color:#f7931a' if is_golden else '')}">
                <td>{level}</td>
                <td>${price:,.0f}</td>
                <td>{"⭐" if is_golden else ""}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Numerical Harmony - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #f7931a;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .card {{
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }}
        .card-title {{
            color: #58a6ff;
            font-size: 14px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #21262d;
            padding: 8px;
            text-align: left;
            color: #8b949e;
            font-size: 10px;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #21262d;
            font-size: 12px;
        }}
        .binary-state {{
            font-size: 48px;
            text-align: center;
            padding: 20px;
        }}
        .state-a {{ color: #3fb950; }}
        .state-b {{ color: #f85149; }}
        .digital-root {{
            font-size: 36px;
            text-align: center;
            color: #a371f7;
        }}
        .significance {{
            color: #f7931a;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">🧮 NUMERICAL HARMONY ENGINE</div>
    
    <div class="grid">
        <div class="card">
            <div class="card-title">FIBONACCI LEVELS</div>
            <table>
                <tr><th>Level</th><th>Price</th><th></th></tr>
                {fib_rows}
            </table>
        </div>
        
        <div class="card">
            <div class="card-title">BINARY SWITCH</div>
            <div class="binary-state state-{bs['state'].lower()}">
                {bs['state']}
            </div>
            <div style="text-align:center;color:#8b949e;font-size:12px;">
                Strategy: {bs['strategy']}
            </div>
            <div style="text-align:center;color:#8b949e;font-size:11px;margin-top:10px;">
                Vol: {bs['volatility']:.2%} | Threshold: {bs['threshold']:.2%}
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">DIGITAL ROOT</div>
            <div class="digital-root">{dr['time_root']}</div>
            <div style="text-align:center;color:#8b949e;font-size:12px;">
                {dr['time_meaning']}
            </div>
            <div class="significance">
                {"⚠️ SIGNIFICANT" if dr['significant'] else "Not significant"}
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">KEY LEVELS</div>
            <table>
                <tr><td style="color:#f7931a">Golden Pocket</td><td>${fib['golden_pocket']['low']:,.0f} - ${fib['golden_pocket']['high']:,.0f}</td></tr>
                <tr><td>0.618 Retrace</td><td>${fib['retracements'].get(0.618, 0):,.0f}</td></tr>
                <tr><td>1.618 Extension</td><td>${fib['extensions'].get(1.618, 0):,.0f}</td></tr>
            </table>
        </div>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("NUMERICAL HARMONY ENGINE TEST")
    print("=" * 50)
    
    engine = NumericalHarmonyEngine()
    
    # Set swing
    engine.set_swing(50000, 45000)
    
    # Fibonacci
    fib = engine.calculate_fibonacci()
    print(f"\nFibonacci (swing: ${fib['swing']['high']:,.0f} - ${fib['swing']['low']:,.0f})")
    print(f"  0.618: ${fib['retracements'].get(0.618):,.0f}")
    print(f"  Golden Pocket: ${fib['golden_pocket']['low']:,.0f} - ${fib['golden_pocket']['high']:,.0f}")
    
    # Nearest level
    nearest = engine.find_nearest_fib_level(48500)
    print(f"\nNearest to $48,500: {nearest}")
    
    # Binary switch
    bs = engine.get_binary_switch_signal(0.025)
    print(f"\nBinary Switch: {bs['state']} ({bs['strategy']})")
    
    # Digital root
    dr = engine.get_digital_root_timer()
    print(f"\nDigital Root: {dr['time_root']} ({dr['time_meaning']})")
    
    # Confluence
    confluence = engine.calculate_confluence(48500, {"volatility": 0.02})
    print(f"\nConfluence at $48,500: {confluence['score']}/100")
    print(f"  Factors: {confluence['factors']}")
    
    # Save HTML
    html = engine.generate_html()
    with open("numerical_harmony.html", "w") as f:
        f.write(html)
    
    print("\nSaved to numerical_harmony.html")
    print("OK!")
