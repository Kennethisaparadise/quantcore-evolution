"""
QuantCore - Live Dashboard & Visualization Engine v1.0

Head 22: The Omniviz Dashboard - Real-time visualization of all modules.

Features:
1. Module status cards (all 21 modules)
2. Equity curve with annotations
3. Trade log with filtering
4. Regime timeline visualization
5. Performance metrics dashboard
6. Event feed
"""

import random
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================
class ModuleStatus(Enum):
    """Module status."""
    ACTIVE = "active"
    IDLE = "idle"
    ADAPTING = "adapting"
    ERROR = "error"


class TradeDirection(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class ModuleState:
    """State of a single module."""
    id: str
    name: str
    status: str
    icon: str
    metric: float
    metric_label: str
    trend: str  # up, down, stable
    last_update: str = ""
    last_update: str


@dataclass
class TradeRecord:
    """A trade for the log."""
    id: int
    timestamp: str
    module: str
    symbol: str
    direction: str
    quantity: float
    price: float
    pnl: float
    fees: float
    tax: float


@dataclass
class EquityPoint:
    """Equity curve point."""
    timestamp: str
    equity: float
    drawdown: float
    trades: int


@dataclass
class DashboardState:
    """Complete dashboard state."""
    # System
    version: str = "2.5.0"
    uptime_seconds: float = 0
    mode: str = "paper"  # paper or live
    
    # Equity
    initial_capital: float = 100000
    current_equity: float = 100000
    total_pnl: float = 0
    pnl_pct: float = 0
    max_drawdown: float = 0
    
    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0
    
    # Modules
    modules: List[ModuleState] = field(default_factory=list)
    
    # Recent trades
    recent_trades: List[TradeRecord] = field(default_factory=list)
    
    # Equity history
    equity_history: List[EquityPoint] = field(default_factory=list)
    
    # Events
    events: List[Dict] = field(default_factory=list)


# ============================================================
# MODULE DEFINITIONS
# ============================================================
def get_all_modules() -> List[ModuleState]:
    """Get all 21 module definitions."""
    return [
        ModuleState("regime", "Regime Switching", "active", "🎯", 0, "Bull", "up"),
        ModuleState("fractal", "Fractal Time Series", "active", "📊", 0, "1h TF", "stable"),
        ModuleState("adversarial", "Adversarial Validator", "idle", "🛡️", 0, "Ready", "stable"),
        ModuleState("timeline", "Timeline Visualization", "active", "📈", 0, "Running", "stable"),
        ModuleState("dynamic_regime", "Dynamic Regime Count", "active", "🔄", 3, "Regimes", "stable"),
        ModuleState("order_flow", "Order Flow Shadow", "active", "🐋", 0, "Delta: +0.3", "up"),
        ModuleState("sentiment", "Sentiment Divergence", "active", "💭", 65, "Greed: 65", "down"),
        ModuleState("correlation", "Correlation Pairs", "idle", "🔗", 0, "BTC/ETH", "stable"),
        ModuleState("live_trading", "Live Trading", "active", "⚡", 0, "Paper Mode", "stable"),
        ModuleState("realtime_adapt", "Real-Time Adaptation", "active", "🧠", 0, "Watching", "stable"),
        ModuleState("compounding", "Compounding Engine", "active", "💰", 0, "Reinvesting", "up"),
        ModuleState("imbalance", "Order Book Imbalance", "idle", "⚖️", 0, "Bid: 52%", "stable"),
        ModuleState("iceberg", "Iceberg Hunter", "idle", "🧊", 0, "Scanning", "stable"),
        ModuleState("spoofing", "Spoofing Detector", "idle", "🎭", 0, "Clean", "stable"),
        ModuleState("arbitrage", "Latency Arbitrage", "idle", "🚀", 0, "Opportunities: 0", "stable"),
        ModuleState("algo_hunter", "Adversarial Algo Hunter", "idle", "🤖", 0, "Hunting", "stable"),
        ModuleState("patterns", "Microstructure Patterns", "idle", "🔬", 0, "None", "stable"),
        ModuleState("fee_aware", "Fee-Aware Execution", "active", "💵", 0.001, "Fee: 0.1%", "stable"),
        ModuleState("tax_aware", "Tax-Aware Optimization", "active", "🏛️", 0.25, "Est: 25%", "stable"),
        ModuleState("multi_account", "Multi-Account Allocator", "idle", "🏦", 0, "3 Accounts", "stable"),
        ModuleState("withdrawal", "Withdrawal Optimizer", "idle", "💸", 0.04, "4% Rate", "stable"),
    ]


# ============================================================
# DASHBOARD ENGINE
# ============================================================
class LiveDashboard:
    """
    Real-time dashboard engine.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.state = DashboardState()
        self.state.initial_capital = initial_capital
        self.state.current_equity = initial_capital
        self.state.modules = get_all_modules()
        
        self.start_time = time.time()
        self.trade_counter = 0
        self._running = False
        self._thread = None
        
    def start(self):
        """Start dashboard updates."""
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("📊 Live Dashboard started")
    
    def stop(self):
        """Stop dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("📊 Live Dashboard stopped")
    
    def _update_loop(self):
        """Background update loop."""
        while self._running:
            self._update_state()
            time.sleep(1)  # Update every second
    
    def _update_state(self):
        """Update dashboard state."""
        # Update uptime
        self.state.uptime_seconds = time.time() - self.start_time
        
        # Update some module metrics randomly for demo
        for mod in self.state.modules:
            if random.random() < 0.05:  # 5% chance to change
                if mod.id == "sentiment":
                    mod.metric = random.randint(20, 80)
                    mod.metric_label = f"Greed: {mod.metric}"
                    mod.trend = "up" if mod.metric > 50 else "down"
                elif mod.id == "order_flow":
                    delta = random.uniform(-1, 1)
                    mod.metric = delta
                    mod.metric_label = f"Delta: {delta:.1f}"
                    mod.trend = "up" if delta > 0 else "down"
    
    def record_trade(self, symbol: str, direction: str, quantity: float, 
                    price: float, pnl: float, module: str = "strategy"):
        """Record a trade."""
        self.trade_counter += 1
        
        trade = TradeRecord(
            id=self.trade_counter,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            module=module,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            pnl=pnl,
            fees=price * quantity * 0.001,  # Estimate
            tax=max(0, pnl * 0.25)  # Estimate
        )
        
        self.state.recent_trades.insert(0, trade)
        if len(self.state.recent_trades) > 50:
            self.state.recent_trades = self.state.recent_trades[:50]
        
        # Update equity
        self.state.current_equity += pnl
        self.state.total_trades += 1
        
        if pnl > 0:
            self.state.winning_trades += 1
        
        if self.state.total_trades > 0:
            self.state.win_rate = self.state.winning_trades / self.state.total_trades
        
        # Update PnL
        self.state.total_pnl = self.state.current_equity - self.state.initial_capital
        self.state.pnl_pct = self.state.total_pnl / self.state.initial_capital
        
        # Record equity point
        point = EquityPoint(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            equity=self.state.current_equity,
            drawdown=0,  # Would calculate from peak
            trades=self.state.total_trades
        )
        self.state.equity_history.append(point)
        if len(self.state.equity_history) > 100:
            self.state.equity_history = self.state.equity_history[-100:]
        
        # Add event
        self.state.events.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "trade",
            "message": f"{direction.upper()} {quantity:.4f} {symbol} @ ${price:.2f} | PnL: ${pnl:.2f}"
        })
        if len(self.state.events) > 100:
            self.state.events = self.state.events[:100]
    
    def get_status(self) -> Dict:
        """Get dashboard status."""
        return {
            "version": self.state.version,
            "uptime": f"{int(self.state.uptime_seconds)}s",
            "mode": self.state.mode,
            "equity": {
                "current": self.state.current_equity,
                "initial": self.state.initial_capital,
                "pnl": self.state.total_pnl,
                "pnl_pct": self.state.pnl_pct * 100,
                "max_dd": self.state.max_drawdown * 100
            },
            "trades": {
                "total": self.state.total_trades,
                "wins": self.state.winning_trades,
                "win_rate": self.state.win_rate * 100
            },
            "modules": [
                {
                    "id": m.id,
                    "name": m.name,
                    "status": m.status,
                    "icon": m.icon,
                    "metric": m.metric,
                    "metric_label": m.metric_label,
                    "trend": m.trend
                }
                for m in self.state.modules
            ],
            "recent_trades": [
                {
                    "id": t.id,
                    "time": t.timestamp,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "price": t.price,
                    "pnl": t.pnl
                }
                for t in self.state.recent_trades[:10]
            ],
            "events": self.state.events[:20]
        }
    
    def get_html_dashboard(self) -> str:
        """Generate HTML dashboard."""
        s = self.state
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QuantCore Dashboard v{s.version}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            background: #0d1117; 
            color: #e6edf3; 
            font-family: 'SF Mono', 'Fira Code', monospace;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #30363d;
        }}
        .title {{
            font-size: 24px;
            font-weight: bold;
        }}
        .version {{
            color: #8b949e;
            font-size: 14px;
        }}
        .status {{
            display: flex;
            gap: 20px;
        }}
        .status-item {{
            text-align: center;
        }}
        .status-label {{
            font-size: 11px;
            color: #8b949e;
            text-transform: uppercase;
        }}
        .status-value {{
            font-size: 20px;
            font-weight: bold;
        }}
        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .module-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s;
        }}
        .module-card.active {{
            border-color: #3fb950;
            box-shadow: 0 0 10px rgba(63, 185, 80, 0.2);
        }}
        .module-card.idle {{
            opacity: 0.6;
        }}
        .module-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .module-icon {{
            font-size: 20px;
        }}
        .module-name {{
            font-size: 12px;
            font-weight: bold;
        }}
        .module-metric {{
            font-size: 16px;
            font-weight: bold;
            color: #58a6ff;
        }}
        .module-trend {{
            font-size: 11px;
        }}
        .trend-up {{ color: #3fb950; }}
        .trend-down {{ color: #f85149; }}
        .trend-stable {{ color: #8b949e; }}
        
        .equity-bar {{
            background: #21262d;
            height: 30px;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        .equity-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3fb950, #58a6ff);
            transition: width 0.5s;
        }}
        
        .log {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
        }}
        .log-title {{
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .log-entry {{
            font-size: 12px;
            padding: 4px 0;
            border-bottom: 1px solid #21262d;
            display: flex;
            gap: 10px;
        }}
        .log-time {{
            color: #8b949e;
            min-width: 60px;
        }}
        .log-buy {{ color: #3fb950; }}
        .log-sell {{ color: #f85149; }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-box {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-label {{
            font-size: 11px;
            color: #8b949e;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">🐙 QuantCore Dashboard</div>
            <div class="version">v{s.version} | {s.mode.upper()} Mode | Uptime: {int(s.uptime_seconds)}s</div>
        </div>
    </div>
    
    <div class="equity-bar">
        <div class="equity-fill" style="width: {min(100, (s.current_equity/s.initial_capital)*100)}%"></div>
    </div>
    
    <div class="grid">
"""
        
        # Module cards
        for m in s.modules:
            trend_class = f"trend-{m.trend}"
            active_class = "active" if m.status == "active" else "idle"
            html += f"""
        <div class="module-card {active_class}">
            <div class="module-header">
                <span class="module-icon">{m.icon}</span>
                <span class="module-name">{m.name}</span>
            </div>
            <div class="module-metric">{m.metric_label}</div>
            <div class="module-trend {trend_class}">{m.trend.upper()}</div>
        </div>
"""
        
        html += """
    </div>
    
    <div class="log">
        <div class="log-title">📜 Live Trade Log</div>
"""
        
        # Trade log
        for t in s.recent_trades[:15]:
            direction_class = f"log-{t.direction}"
            pnl_class = "positive" if t.pnl > 0 else "negative"
            html += f"""
        <div class="log-entry">
            <span class="log-time">{t.timestamp}</span>
            <span class="{direction_class}">{t.direction.upper()}</span>
            <span>{t.quantity:.4f} {t.symbol}</span>
            <span>@ ${t.price:.2f}</span>
            <span class="{pnl_class}">${t.pnl:.2f}</span>
        </div>
"""
        
        html += """
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-label">Total Equity</div>
            <div class="stat-value">${:,.0f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Total PnL</div>
            <div class="stat-value {:s}>${:,.0f} ({:.1f}%)</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Total Trades</div>
            <div class="stat-value">{:d}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Win Rate</div>
            <div class="stat-value">{:.1f}%</div>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 2 seconds
        setTimeout(() => location.reload(), 2000);
    </script>
</body>
</html>
""".format(
            s.current_equity,
            "positive" if s.total_pnl > 0 else "negative",
            s.total_pnl,
            s.pnl_pct * 100,
            s.total_trades,
            s.win_rate * 100
        )
        
        return html


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("QUANTCORE LIVE DASHBOARD TEST")
    print("=" * 50)
    
    # Create dashboard
    dashboard = LiveDashboard(initial_capital=100000)
    dashboard.start()
    
    # Simulate some trades
    for i in range(5):
        dashboard.record_trade(
            symbol="BTCUSDT",
            direction="buy" if i % 2 == 0 else "sell",
            quantity=0.1,
            price=45000 + random.randint(-500, 500),
            pnl=random.uniform(-100, 200),
            module="strategy"
        )
        time.sleep(0.1)
    
    # Get status
    status = dashboard.get_status()
    
    print("\n📊 Dashboard Status:")
    print(f"  Version: {status['version']}")
    print(f"  Uptime: {status['uptime']}")
    print(f"  Equity: ${status['equity']['current']:,.2f}")
    print(f"  PnL: ${status['equity']['pnl']:,.2f} ({status['equity']['pnl_pct']:.2f}%)")
    print(f"  Trades: {status['trades']['total']}")
    print(f"  Win Rate: {status['trades']['win_rate']:.1f}%")
    print(f"  Modules: {len(status['modules'])}")
    
    print("\n📜 Recent Trades:")
    for t in status['recent_trades'][:3]:
        print(f"  {t['time']} | {t['direction'].upper()} {t['quantity']:.4f} @ ${t['price']:.2f} | PnL: ${t['pnl']:.2f}")
    
    # Save HTML
    html = dashboard.get_html_dashboard()
    with open("dashboard.html", "w") as f:
        f.write(html)
    
    print("\n💾 HTML dashboard saved to dashboard.html")
    
    dashboard.stop()
    print("\n✅ Dashboard test complete!")
