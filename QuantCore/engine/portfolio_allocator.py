"""
QuantCore - Multi-Style Portfolio Allocator v1.0
Head 39: Dynamic capital allocation across trading modes
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ModePerformance:
    """Performance metrics for a trading mode."""
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0
    peak_equity: float = 0
    current_equity: float = 0
    max_drawdown: float = 0
    sharpe: float = 0
    last_trade_time: float = 0


class PortfolioAllocator:
    """
    Dynamic capital allocation across trading modes.
    """
    
    MODES = ["scalp", "intraday", "swing", "position"]
    
    def __init__(self):
        # Initial weights (equal)
        self.weights = {
            "scalp": 0.25,
            "intraday": 0.25,
            "swing": 0.25,
            "position": 0.25
        }
        
        # Performance tracking
        self.performance: Dict[str, ModePerformance] = {
            mode: ModePerformance() for mode in self.MODES
        }
        
        # Allocator settings
        self.rebalance_frequency = "daily"  # daily, weekly, on_trade
        self.rebalance_threshold = 0.05  # 5% shift triggers rebalance
        self.lookback_trades = 20
        self.max_drawdown_trigger = 0.15  # 15% drawdown triggers capital pull
        self.correlation_penalty = 0.1
        
        # Rebalancing history
        self.rebalance_history: List[Dict] = []
        
        # Evolvable parameters
        self.sharpe_weight = 0.5
        self.winrate_weight = 0.3
        self.drawdown_penalty_weight = 0.2
    
    def record_trade(self, mode: str, pnl: float, is_win: bool, equity: float):
        """Record a trade for a mode."""
        if mode not in self.performance:
            return
        
        perf = self.performance[mode]
        perf.trades += 1
        perf.total_pnl += pnl
        
        if is_win:
            perf.wins += 1
        
        perf.current_equity = equity
        
        # Update peak and drawdown
        if equity > perf.peak_equity:
            perf.peak_equity = equity
        
        if perf.peak_equity > 0:
            perf.max_drawdown = max(perf.max_drawdown, 
                (perf.peak_equity - equity) / perf.peak_equity)
        
        # Calculate rolling Sharpe (simplified)
        if perf.trades > 1:
            avg_pnl = perf.total_pnl / perf.trades
            # Simplified - would need std dev for real Sharpe
            perf.sharpe = avg_pnl * 10  # Placeholder
    
    def calculate_confidence(self, mode: str) -> float:
        """Calculate confidence score for a mode."""
        perf = self.performance[mode]
        
        if perf.trades == 0:
            return 0.5  # Neutral for no data
        
        # Win rate
        win_rate = perf.wins / perf.trades
        
        # Drawdown penalty
        dd_penalty = perf.max_drawdown * self.drawdown_penalty_weight
        
        # Combined confidence
        confidence = (
            self.sharpe_weight * min(perf.sharpe, 2.0) +
            self.winrate_weight * win_rate -
            dd_penalty
        )
        
        return max(0, min(1, confidence))
    
    def calculate_correlation_penalty(self) -> float:
        """Calculate penalty for correlated modes."""
        # Simplified correlation - in production would use actual returns
        return self.correlation_penalty
    
    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed."""
        # Check drawdown triggers
        for mode, perf in self.performance.items():
            if perf.max_drawdown > self.max_drawdown_trigger:
                return True
        
        # Check weight drift
        confidences = [self.calculate_confidence(m) for m in self.MODES]
        total = sum(confidences)
        
        if total == 0:
            return False
        
        target_weights = [c / total for c in confidences]
        
        for mode, target in zip(self.MODES, target_weights):
            if abs(self.weights[mode] - target) > self.rebalance_threshold:
                return True
        
        return False
    
    def rebalance(self) -> Dict[str, float]:
        """Rebalance portfolio weights."""
        # Calculate confidences
        confidences = {mode: self.calculate_confidence(mode) for mode in self.MODES}
        
        # Apply softmax-like transformation
        total = sum(confidences.values())
        if total == 0:
            new_weights = {mode: 0.25 for mode in self.MODES}
        else:
            # Apply correlation penalty
            corr_penalty = self.calculate_correlation_penalty()
            adjusted = {m: max(0, c - corr_penalty) for m, c in confidences.items()}
            
            total_adj = sum(adjusted.values())
            if total_adj == 0:
                new_weights = {mode: 0.25 for mode in self.MODES}
            else:
                new_weights = {m: c / total_adj for m, c in adjusted.items()}
        
        # Record rebalancing
        self.rebalance_history.append({
            "timestamp": "now",
            "old_weights": self.weights.copy(),
            "new_weights": new_weights.copy()
        })
        
        self.weights = new_weights
        return new_weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get current allocation weights."""
        return self.weights.copy()
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for all modes."""
        summary = {}
        for mode, perf in self.performance.items():
            summary[mode] = {
                "trades": perf.trades,
                "wins": perf.wins,
                "win_rate": (perf.wins / perf.trades * 100) if perf.trades > 0 else 0,
                "total_pnl": perf.total_pnl,
                "max_drawdown": perf.max_drawdown * 100,
                "sharpe": perf.sharpe,
                "confidence": self.calculate_confidence(mode),
                "weight": self.weights[mode]
            }
        return summary
    
    def get_allocated_capital(self, total_capital: float) -> Dict[str, float]:
        """Get allocated capital for each mode."""
        return {mode: total_capital * weight for mode, weight in self.weights.items()}
    
    def mutate_parameters(self):
        """Evolve allocator parameters."""
        if random.random() < 0.3:
            self.sharpe_weight = max(0.1, min(0.8, 
                self.sharpe_weight + random.uniform(-0.1, 0.1)))
        
        if random.random() < 0.3:
            self.winrate_weight = max(0.1, min(0.5, 
                self.winrate_weight + random.uniform(-0.05, 0.05)))
        
        if random.random() < 0.2:
            self.rebalance_threshold = max(0.02, min(0.2, 
                self.rebalance_threshold + random.uniform(-0.01, 0.01)))
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        summary = self.get_performance_summary()
        weights = self.get_weights()
        
        # Mode rows
        mode_rows = ""
        for mode in self.MODES:
            s = summary.get(mode, {})
            is_active = s.get("trades", 0) > 0
            
            mode_rows += f"""
            <tr class="{'active' if is_active else ''}">
                <td>{mode.upper()}</td>
                <td>{s.get('trades', 0)}</td>
                <td>{s.get('win_rate', 0):.1f}%</td>
                <td style="color: {'#3fb950' if s.get('total_pnl', 0) > 0 else '#f85149'}">
                    ${s.get('total_pnl', 0):.2f}
                </td>
                <td>{s.get('max_drawdown', 0):.1f}%</td>
                <td>{s.get('confidence', 0):.2f}</td>
                <td>
                    <div class="weight-bar">
                        <div class="weight-fill" style="width:{s.get('weight', 0)*100}%"></div>
                    </div>
                    {s.get('weight', 0)*100:.0f}%
                </td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Allocator - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #3fb950;
            margin-bottom: 20px;
        }}
        .current-weights {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .weight-card {{
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .weight-value {{
            font-size: 32px;
            color: #3fb950;
        }}
        .weight-label {{
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
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #21262d;
        }}
        tr.active {{
            background: rgba(63, 185, 80, 0.1);
        }}
        .weight-bar {{
            width: 100px;
            height: 10px;
            background: #21262d;
            border-radius: 5px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
        }}
        .weight-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3fb950, #58a6ff);
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">🌿 PORTFOLIO ALLOCATOR</div>
    
    <div class="current-weights">
"""
        
        for mode in self.MODES:
            html += f"""
        <div class="weight-card">
            <div class="weight-value">{weights[mode]*100:.0f}%</div>
            <div class="weight-label">{mode}</div>
        </div>
"""
        
        html += f"""
    </div>
    
    <h3 style="color:#58a6ff">Mode Performance</h3>
    <table>
        <thead>
            <tr>
                <th>Mode</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>PnL</th>
                <th>Max DD</th>
                <th>Confidence</th>
                <th>Weight</th>
            </tr>
        </thead>
        <tbody>
            {mode_rows}
        </tbody>
    </table>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("PORTFOLIO ALLOCATOR TEST")
    print("=" * 50)
    
    allocator = PortfolioAllocator()
    
    # Simulate trades
    for _ in range(10):
        mode = random.choice(allocator.MODES)
        pnl = random.uniform(-50, 150)
        is_win = pnl > 0
        equity = 1000 + random.uniform(-100, 200)
        allocator.record_trade(mode, pnl, is_win, equity)
    
    # Check should rebalance
    print(f"\nShould rebalance: {allocator.should_rebalance()}")
    
    # Get summary
    summary = allocator.get_performance_summary()
    print("\nPerformance:")
    for mode, s in summary.items():
        print(f"  {mode}: {s['trades']} trades, {s['win_rate']:.1f}% win, ${s['total_pnl']:.2f}")
    
    # Weights
    weights = allocator.get_weights()
    print("\nCurrent Weights:")
    for mode, w in weights.items():
        print(f"  {mode}: {w*100:.0f}%")
    
    # Rebalance
    if allocator.should_rebalance():
        new_weights = allocator.rebalance()
        print("\nAfter Rebalance:")
        for mode, w in new_weights.items():
            print(f"  {mode}: {w*100:.0f}%")
    
    # Save HTML
    html = allocator.generate_html()
    with open("portfolio_allocator.html", "w") as f:
        f.write(html)
    
    print("\nSaved to portfolio_allocator.html")
    print("OK!")
