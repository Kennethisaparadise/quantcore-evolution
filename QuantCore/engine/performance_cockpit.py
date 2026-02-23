"""
QuantCore - Performance Cockpit v1.0
Head 25: Deep analytics dashboard
"""

import random
from datetime import datetime
from typing import Dict, List


class PerformanceCockpit:
    """Deep analytics for all modules."""
    
    def __init__(self):
        self.trades = []
        self.module_stats = {}
    
    def add_trade(self, module: str, pnl: float, fees: float, tax: float, hold_time: int):
        """Add a trade."""
        self.trades.append({
            "module": module,
            "pnl": pnl,
            "fees": fees,
            "tax": tax,
            "hold_time": hold_time,
            "timestamp": datetime.now()
        })
        
        # Update module stats
        if module not in self.module_stats:
            self.module_stats[module] = {
                "trades": 0,
                "wins": 0,
                "total_pnl": 0,
                "total_fees": 0,
                "total_tax": 0,
                "hold_times": []
            }
        
        stats = self.module_stats[module]
        stats["trades"] += 1
        if pnl > 0:
            stats["wins"] += 1
        stats["total_pnl"] += pnl
        stats["total_fees"] += fees
        stats["total_tax"] += tax
        stats["hold_times"].append(hold_time)
    
    def get_summary(self) -> Dict:
        """Get overall summary."""
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_fees = sum(t["fees"] for t in self.trades)
        total_tax = sum(t["tax"] for t in self.trades)
        wins = len([t for t in self.trades if t["pnl"] > 0])
        
        return {
            "total_trades": len(self.trades),
            "wins": wins,
            "losses": len(self.trades) - wins,
            "win_rate": wins / max(1, len(self.trades)),
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "total_tax": total_tax,
            "net_pnl": total_pnl - total_fees - total_tax,
            "avg_hold_time": sum(t["hold_time"] for t in self.trades) / max(1, len(self.trades))
        }
    
    def get_module_stats(self) -> List[Dict]:
        """Get per-module stats."""
        results = []
        for module, stats in self.module_stats.items():
            win_rate = stats["wins"] / max(1, stats["trades"])
            avg_hold = sum(stats["hold_times"]) / max(1, len(stats["hold_times"]))
            
            results.append({
                "module": module,
                "trades": stats["trades"],
                "wins": stats["wins"],
                "win_rate": win_rate,
                "total_pnl": stats["total_pnl"],
                "total_fees": stats["total_fees"],
                "total_tax": stats["total_tax"],
                "net_pnl": stats["total_pnl"] - stats["total_fees"] - stats["total_tax"],
                "avg_hold_time": avg_hold
            })
        
        return sorted(results, key=lambda x: x["net_pnl"], reverse=True)
    
    def generate_html(self) -> str:
        """Generate HTML report."""
        summary = self.get_summary()
        modules = self.get_module_stats()
        
        # Module table rows
        module_rows = ""
        for m in modules:
            pnl_color = "#3fb950" if m["net_pnl"] > 0 else "#f85149"
            module_rows += f"""
            <tr>
                <td>{m['module']}</td>
                <td>{m['trades']}</td>
                <td>{m['win_rate']*100:.1f}%</td>
                <td style="color:{pnl_color}">${m['net_pnl']:,.2f}</td>
                <td>${m['total_fees']:,.2f}</td>
                <td>${m['total_tax']:,.2f}</td>
                <td>{m['avg_hold_time']:.1f}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Performance Cockpit - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #30363d;
        }}
        .title {{
            font-size: 28px;
            color: #58a6ff;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .card-label {{
            color: #8b949e;
            font-size: 11px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #161b22;
            border-radius: 10px;
            overflow: hidden;
        }}
        th {{
            background: #21262d;
            padding: 15px;
            text-align: left;
            font-size: 11px;
            text-transform: uppercase;
            color: #8b949e;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #21262d;
            font-size: 13px;
        }}
        tr:hover {{
            background: #1c2128;
        }}
        
        .section-title {{
            font-size: 18px;
            margin: 30px 0 15px;
            color: #58a6ff;
        }}
        
        .charts {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .chart {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            height: 200px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">📊 PERFORMANCE COCKPIT</div>
        <div style="color:#8b949e">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
    </div>
    
    <div class="summary">
        <div class="card">
            <div class="card-label">Total Trades</div>
            <div class="card-value">{summary['total_trades']}</div>
        </div>
        <div class="card">
            <div class="card-label">Win Rate</div>
            <div class="card-value">{summary['win_rate']*100:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-label">Net PnL</div>
            <div class="card-value {'positive' if summary['net_pnl'] > 0 else 'negative'}>
                ${summary['net_pnl']:,.2f}
            </div>
        </div>
        <div class="card">
            <div class="card-label">Total Fees + Tax</div>
            <div class="card-value negative">
                ${summary['total_fees'] + summary['total_tax']:,.2f}
            </div>
        </div>
    </div>
    
    <div class="section-title">📈 BY MODULE</div>
    <table>
        <thead>
            <tr>
                <th>Module</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>Net PnL</th>
                <th>Fees</th>
                <th>Tax</th>
                <th>Avg Hold</th>
            </tr>
        </thead>
        <tbody>
            {module_rows}
        </tbody>
    </table>
    
    <div class="charts">
        <div class="chart">
            <div style="color:#8b949e;font-size:11px;text-transform:uppercase;margin-bottom:10px;">Pnl by Module</div>
            <div style="height:150px;display:flex;align-items:flex-end;gap:5px;">
"""
        
        # Simple bar chart
        for m in modules[:6]:
            height = max(10, min(150, m["net_pnl"] / 10))
            color = "#3fb950" if m["net_pnl"] > 0 else "#f85149"
            html += f'<div style="flex:1;height:{height}px;background:{color};border-radius:3px 3px 0 0;"></div>'
        
        html += """
            </div>
        </div>
        <div class="chart">
            <div style="color:#8b949e;font-size:11px;text-transform:uppercase;margin-bottom:10px;">Win Rate by Module</div>
            <div style="height:150px;display:flex;align-items:flex-end;gap:5px;">
"""
        
        for m in modules[:6]:
            height = m["win_rate"] * 150
            html += f'<div style="flex:1;height:{height}px;background:#58a6ff;border-radius:3px 3px 0 0;"></div>'
        
        html += """
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("PERFORMANCE COCKPIT TEST")
    print("=" * 50)
    
    cockpit = PerformanceCockpit()
    
    # Simulate trades
    modules = ["regime", "fractal", "sentiment", "order_flow", "compounding"]
    for i in range(50):
        module = random.choice(modules)
        pnl = random.uniform(-200, 400)
        fees = abs(pnl) * 0.001
        tax = max(0, pnl * 0.25) if pnl > 0 else 0
        hold = random.randint(1, 100)
        cockpit.add_trade(module, pnl, fees, tax, hold)
    
    summary = cockpit.get_summary()
    print(f"\nTotal Trades: {summary['total_trades']}")
    print(f"Win Rate: {summary['win_rate']*100:.1f}%")
    print(f"Net PnL: ${summary['net_pnl']:.2f}")
    print(f"Fees + Tax: ${summary['total_fees'] + summary['total_tax']:.2f}")
    
    print("\nTop Module:")
    top = cockpit.get_module_stats()[0]
    print(f"  {top['module']}: ${top['net_pnl']:.2f}")
    
    html = cockpit.generate_html()
    with open("performance_cockpit.html", "w") as f:
        f.write(html)
    
    print("\nSaved to performance_cockpit.html")
    print("OK!")
