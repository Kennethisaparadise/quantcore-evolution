"""
QuantCore - Scalping Optimization Engine v1.0
Head 36: Ultra-short-term trading optimization
"""

import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ScalpTrade:
    """A scalping trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    duration_seconds: float
    spread_at_entry: float


class ScalpingEngine:
    """
    Scalping optimization engine for ultra-short-term trading.
    """
    
    def __init__(self):
        # Scalping parameters
        self.profit_target = 0.003  # 0.3% target
        self.stop_loss = 0.001      # 0.1% stop
        self.max_hold_time = 120    # 120 seconds max
        self.max_consecutive_losses = 5
        
        # State
        self.consecutive_losses = 0
        self.sessions: List[ScalpTrade] = []
        self.current_trade = None
        
        # Performance tracking
        self.total_scalps = 0
        self.winning_scalps = 0
        self.total_pnl = 0
    
    def should_enter(self, market_data: Dict) -> bool:
        """Determine if we should enter a scalp."""
        # Check for scalping opportunities
        spread = market_data.get("spread", 0)
        spread_pct = spread / market_data.get("price", 1) * 100
        
        # Tight spread required for scalping
        if spread_pct > 0.05:  # More than 0.05%
            return False
        
        # Check volume
        volume = market_data.get("volume", 0)
        if volume < 100:  # Low volume = risky
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        
        return True
    
    def calculate_position_size(self, capital: float, risk_pct: float = 0.02) -> float:
        """Calculate position size for scalp."""
        return capital * risk_pct
    
    def calculate_take_profit(self, entry_price: float, direction: int) -> float:
        """Calculate take profit level."""
        if direction == 1:  # Long
            return entry_price * (1 + self.profit_target)
        else:  # Short
            return entry_price * (1 - self.profit_target)
    
    def calculate_stop_loss(self, entry_price: float, direction: int) -> float:
        """Calculate stop loss level."""
        if direction == 1:  # Long
            return entry_price * (1 - self.stop_loss)
        else:  # Short
            return entry_price * (1 + self.stop_loss)
    
    def should_exit(self, current_price: float, entry_time: datetime, 
                    direction: int, entry_price: float) -> Dict:
        """Check if we should exit the current scalp."""
        now = datetime.now()
        duration = (now - entry_time).total_seconds()
        
        # Time-based exit
        if duration > self.max_hold_time:
            return {"should_exit": True, "reason": "time_limit", "pnl_pct": 0}
        
        # Profit target exit
        if direction == 1:  # Long
            if current_price >= self.calculate_take_profit(entry_price, 1):
                return {"should_exit": True, "reason": "take_profit", "pnl_pct": self.profit_target * 100}
            if current_price <= self.calculate_stop_loss(entry_price, 1):
                return {"should_exit": True, "reason": "stop_loss", "pnl_pct": -self.stop_loss * 100}
        else:  # Short
            if current_price <= self.calculate_take_profit(entry_price, -1):
                return {"should_exit": True, "reason": "take_profit", "pnl_pct": self.profit_target * 100}
            if current_price >= self.calculate_stop_loss(entry_price, -1):
                return {"should_exit": True, "reason": "stop_loss", "pnl_pct": -self.stop_loss * 100}
        
        return {"should_exit": False, "reason": None, "pnl_pct": 0}
    
    def record_trade(self, entry_time: datetime, exit_time: datetime, 
                    entry_price: float, exit_price: float, position_size: float):
        """Record a completed scalp trade."""
        pnl = (exit_price - entry_price) * position_size
        pnl_pct = (exit_price / entry_price - 1) * 100
        
        trade = ScalpTrade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_seconds=(exit_time - entry_time).total_seconds(),
            spread_at_entry=0  # Would be captured in real implementation
        )
        
        self.sessions.append(trade)
        self.total_scalps += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_scalps += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        return trade
    
    def get_performance(self) -> Dict:
        """Get scalping performance metrics."""
        if self.total_scalps == 0:
            return {
                "total_scalps": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "avg_duration": 0,
                "best_scalp": 0,
                "worst_scalp": 0
            }
        
        win_rate = (self.winning_scalps / self.total_scalps) * 100
        avg_pnl = self.total_pnl / self.total_scalps
        avg_duration = sum(t.duration_seconds for t in self.sessions) / len(self.sessions)
        best = max(t.pnl_pct for t in self.sessions) if self.sessions else 0
        worst = min(t.pnl_pct for t in self.sessions) if self.sessions else 0
        
        return {
            "total_scalps": self.total_scalps,
            "winning_scalps": self.winning_scalps,
            "losing_scalps": self.total_scalps - self.winning_scalps,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl": avg_pnl,
            "avg_duration": avg_duration,
            "best_scalp": best,
            "worst_scalp": worst,
            "consecutive_losses": self.consecutive_losses
        }
    
    def mutate_parameters(self):
        """Evolve scalping parameters."""
        # Mutate profit target
        if random.random() < 0.3:
            change = random.uniform(-0.001, 0.001)
            self.profit_target = max(0.001, min(0.01, self.profit_target + change))
        
        # Mutate stop loss
        if random.random() < 0.3:
            change = random.uniform(-0.0005, 0.0005)
            self.stop_loss = max(0.0005, min(0.005, self.stop_loss + change))
        
        # Mutate max hold time
        if random.random() < 0.2:
            change = random.randint(-10, 10)
            self.max_hold_time = max(30, min(300, self.max_hold_time + change))
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        perf = self.get_performance()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Scalping Engine - QuantCore</title>
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
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat {{
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #30363d;
            text-align: center;
        }}
        .stat-value {{
            font-size: 28px;
            color: #3fb950;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
            text-transform: uppercase;
        }}
        .params {{
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
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
        }}
    </style>
</head>
<body>
    <div class="header">⚡ SCALPING ENGINE</div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{perf['total_scalps']}</div>
            <div class="stat-label">Total Scalps</div>
        </div>
        <div class="stat">
            <div class="stat-value">{perf['win_rate']:.1f}%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="stat">
            <div class="stat-value" style="color: {'#3fb950' if perf['total_pnl'] > 0 else '#f85149'}">${perf['total_pnl']:.2f}</div>
            <div class="stat-label">Total PnL</div>
        </div>
        <div class="stat">
            <div class="stat-value">{perf['avg_duration']:.1f}s</div>
            <div class="stat-label">Avg Duration</div>
        </div>
    </div>
    
    <div class="params">
        <h3 style="color:#58a6ff">Current Parameters</h3>
        <table>
            <tr>
                <td>Profit Target</td>
                <td>{self.profit_target*100:.2f}%</td>
            </tr>
            <tr>
                <td>Stop Loss</td>
                <td>{self.stop_loss*100:.2f}%</td>
            </tr>
            <tr>
                <td>Max Hold Time</td>
                <td>{self.max_hold_time}s</td>
            </tr>
            <tr>
                <td>Max Consecutive Losses</td>
                <td>{self.max_consecutive_losses}</td>
            </tr>
        </table>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("SCALPING ENGINE TEST")
    print("=" * 50)
    
    engine = ScalpingEngine()
    
    # Simulate some trades
    entry_time = datetime.now()
    for i in range(10):
        exit_time = datetime.now()
        entry_price = 45000
        exit_price = entry_price * (1 + random.uniform(-0.002, 0.005))
        position_size = 0.1
        
        engine.record_trade(entry_time, exit_time, entry_price, exit_price, position_size)
        entry_time = datetime.now()
    
    perf = engine.get_performance()
    print(f"\nScalps: {perf['total_scalps']}")
    print(f"Win Rate: {perf['win_rate']:.1f}%")
    print(f"Total PnL: ${perf['total_pnl']:.2f}")
    print(f"Avg Duration: {perf['avg_duration']:.1f}s")
    print(f"Best: {perf['best_scalp']:.2f}%")
    print(f"Worst: {perf['worst_scalp']:.2f}%")
    
    # Mutate parameters
    engine.mutate_parameters()
    print(f"\nAfter mutation:")
    print(f"  Profit Target: {engine.profit_target*100:.2f}%")
    print(f"  Stop Loss: {engine.stop_loss*100:.2f}%")
    print(f"  Max Hold: {engine.max_hold_time}s")
    
    # Save HTML
    html = engine.generate_html()
    with open("scalping_engine.html", "w") as f:
        f.write(html)
    
    print("\nSaved to scalping_engine.html")
    print("OK!")
