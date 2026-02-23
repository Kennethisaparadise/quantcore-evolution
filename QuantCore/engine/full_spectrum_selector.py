"""
QuantCore - Full Spectrum Style Selector v1.0
Head 38: Four trading modes - Scalp, Intraday, Swing, Position
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class TradingMode(Enum):
    SCALP = "scalp"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"
    AUTO = "auto"


@dataclass
class ModeConfig:
    """Configuration for a trading mode."""
    name: str
    timeframe: str
    max_hold_seconds: int
    profit_target_pct: float
    stop_loss_pct: float
    position_size_pct: float
    min_capital: float
    description: str


class FullSpectrumSelector:
    """
    Full spectrum trading style selector with 4 modes.
    """
    
    # Mode configurations
    SCALP_CONFIG = ModeConfig(
        name="scalp",
        timeframe="tick-5m",
        max_hold_seconds=120,
        profit_target_pct=0.003,
        stop_loss_pct=0.001,
        position_size_pct=0.10,
        min_capital=1000,
        description="Extreme speed, high frequency, micro-profits"
    )
    
    INTRADAY_CONFIG = ModeConfig(
        name="intraday",
        timeframe="5m-1h",
        max_hold_seconds=14400,  # 4 hours
        profit_target_pct=0.015,
        stop_loss_pct=0.005,
        position_size_pct=0.20,
        min_capital=2500,
        description="Day trading, capture session moves"
    )
    
    SWING_CONFIG = ModeConfig(
        name="swing",
        timeframe="1h-4h",
        max_hold_seconds=259200,  # 3 days
        profit_target_pct=0.05,
        stop_loss_pct=0.02,
        position_size_pct=0.30,
        min_capital=5000,
        description="Multi-day trends, ride the wave"
    )
    
    POSITION_CONFIG = ModeConfig(
        name="position",
        timeframe="daily-weekly",
        max_hold_seconds=2592000,  # ~30 days
        profit_target_pct=0.15,
        stop_loss_pct=0.08,
        position_size_pct=0.40,
        min_capital=10000,
        description="Long-term trends, maximum swings"
    )
    
    def __init__(self):
        self.current_mode = TradingMode.AUTO
        self.manual_override = None
        self.mode_history: List[Dict] = []
        
        # Auto-switch thresholds
        self.volatility_low = 0.02
        self.volatility_high = 0.05
        self.trend_strong = 0.7
        self.trend_moderate = 0.4
        
        # Performance tracking
        self.performance = {
            "scalp": {"trades": 0, "pnl": 0, "wins": 0},
            "intraday": {"trades": 0, "pnl": 0, "wins": 0},
            "swing": {"trades": 0, "pnl": 0, "wins": 0},
            "position": {"trades": 0, "pnl": 0, "wins": 0}
        }
    
    def get_config(self, mode: str) -> ModeConfig:
        """Get config for a mode."""
        configs = {
            "scalp": self.SCALP_CONFIG,
            "intraday": self.INTRADAY_CONFIG,
            "swing": self.SWING_CONFIG,
            "position": self.POSITION_CONFIG
        }
        return configs.get(mode, self.SCALP_CONFIG)
    
    def set_mode(self, mode: str):
        """Manually set trading mode."""
        if mode in ["scalp", "intraday", "swing", "position", "auto"]:
            self.current_mode = TradingMode(mode)
            self.manual_override = mode if mode != "auto" else None
    
    def get_current_mode(self) -> str:
        """Get current active mode."""
        return self.current_mode.value
    
    def get_current_config(self) -> ModeConfig:
        """Get configuration for current mode."""
        if self.current_mode == TradingMode.AUTO:
            return self._determine_optimal_config()
        return self.get_config(self.current_mode.value)
    
    def _determine_optimal_config(self) -> ModeConfig:
        """Determine optimal mode based on market conditions."""
        # This would integrate with regime detection
        # Default to scalp for now
        return self.SCALP_CONFIG
    
    def determine_mode(self, regime: str, volatility: float, 
                     trend_strength: float, hour_of_day: int = None) -> str:
        """Determine best mode based on market conditions."""
        if hour_of_day is None:
            hour_of_day = 9  # Default to market hours
        
        # Position: strong trend, low volatility
        if trend_strength > self.trend_strong and volatility < self.volatility_low:
            return "position"
        
        # Swing: moderate-strong trend, not after-hours
        if trend_strength > self.trend_moderate and volatility < self.volatility_high:
            if 9 <= hour_of_day <= 16:
                return "swing"
        
        # Intraday: medium volatility, market hours
        if self.volatility_low <= volatility <= self.volatility_high:
            if 9 <= hour_of_day <= 16:
                return "intraday"
        
        # Default to scalp
        return "scalp"
    
    def switch_if_needed(self, regime: str, volatility: float, 
                        trend_strength: float, hour_of_day: int = None) -> str:
        """Switch mode if auto mode and conditions warrant."""
        if self.current_mode != TradingMode.AUTO:
            return self.current_mode.value
        
        new_mode = self.determine_mode(regime, volatility, trend_strength, hour_of_day)
        
        if new_mode != self.get_current_mode():
            self.current_mode = TradingMode(new_mode)
            self.mode_history.append({
                "timestamp": "now",
                "old_mode": self.get_current_mode(),
                "new_mode": new_mode,
                "regime": regime,
                "volatility": volatility,
                "trend_strength": trend_strength
            })
        
        return new_mode
    
    def record_trade(self, mode: str, pnl: float, is_win: bool):
        """Record trade performance by mode."""
        if mode in self.performance:
            self.performance[mode]["trades"] += 1
            self.performance[mode]["pnl"] += pnl
            if is_win:
                self.performance[mode]["wins"] += 1
    
    def get_performance(self) -> Dict:
        """Get performance metrics by mode."""
        result = {}
        for mode, stats in self.performance.items():
            if stats["trades"] > 0:
                result[mode] = {
                    "trades": stats["trades"],
                    "pnl": stats["pnl"],
                    "wins": stats["wins"],
                    "win_rate": (stats["wins"] / stats["trades"]) * 100,
                    "avg_pnl": stats["pnl"] / stats["trades"]
                }
            else:
                result[mode] = {"trades": 0, "pnl": 0, "wins": 0, "win_rate": 0, "avg_pnl": 0}
        return result
    
    def get_all_configs(self) -> List[Dict]:
        """Get all mode configurations."""
        configs = [
            self.SCALP_CONFIG,
            self.INTRADAY_CONFIG,
            self.SWING_CONFIG,
            self.POSITION_CONFIG
        ]
        return [
            {
                "name": c.name,
                "timeframe": c.timeframe,
                "max_hold": c.max_hold_seconds,
                "target": c.profit_target_pct * 100,
                "stop": c.stop_loss_pct * 100,
                "position": c.position_size_pct * 100,
                "min_capital": c.min_capital,
                "description": c.description
            }
            for c in configs
        ]
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        current_mode = self.get_current_mode()
        configs = self.get_all_configs()
        
        # Mode cards
        mode_cards = ""
        for cfg in configs:
            is_active = cfg["name"] == current_mode
            mode_cards += f"""
            <div class="mode-card {'active' if is_active else ''}">
                <div class="mode-header">
                    <span class="mode-name">{cfg['name'].upper()}</span>
                    {'✓' if is_active else ''}
                </div>
                <div class="mode-desc">{cfg['description']}</div>
                <div class="mode-params">
                    <div>📊 {cfg['timeframe']}</div>
                    <div>🎯 {cfg['target']:.1f}% target</div>
                    <div>🛡️ {cfg['stop']:.1f}% stop</div>
                    <div>💰 {cfg['position']:.0f}% position</div>
                </div>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Full Spectrum Selector - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #a371f7;
            margin-bottom: 20px;
        }}
        .current-mode {{
            background: linear-gradient(135deg, #6e40c9, #a371f7);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .current-mode h2 {{
            margin: 0;
            font-size: 28px;
        }}
        .current-mode p {{
            margin: 10px 0 0;
            color: #e6edf3;
        }}
        .modes-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .mode-card {{
            background: #161b22;
            border: 2px solid #30363d;
            border-radius: 12px;
            padding: 20px;
        }}
        .mode-card.active {{
            border-color: #a371f7;
            box-shadow: 0 0 25px rgba(163, 113, 247, 0.4);
        }}
        .mode-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .mode-name {{
            font-size: 18px;
            font-weight: bold;
            color: #58a6ff;
        }}
        .mode-desc {{
            color: #8b949e;
            font-size: 11px;
            margin-bottom: 15px;
        }}
        .mode-params {{
            background: #21262d;
            padding: 10px;
            border-radius: 5px;
            font-size: 11px;
            line-height: 1.8;
        }}
        .controls {{
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        button {{
            background: #238636;
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-family: monospace;
            font-size: 14px;
            margin: 5px;
        }}
        button:hover {{
            background: #2ea043;
        }}
    </style>
</head>
<body>
    <div class="header">🎛️ FULL SPECTRUM STYLE SELECTOR</div>
    
    <div class="current-mode">
        <h2>Current Mode: {current_mode.upper()}</h2>
        <p>{self.get_current_config().description}</p>
    </div>
    
    <div class="modes-grid">
        {mode_cards}
    </div>
    
    <div class="controls">
        <button onclick="setMode('scalp')">Scalp</button>
        <button onclick="setMode('intraday')">Intraday</button>
        <button onclick="setMode('swing')">Swing</button>
        <button onclick="setMode('position')">Position</button>
        <button onclick="setMode('auto')">Auto</button>
    </div>
    
    <script>
        function setMode(mode) {{
            fetch('/set_mode?mode=' + mode).then(r => location.reload());
        }}
    </script>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("FULL SPECTRUM SELECTOR TEST")
    print("=" * 50)
    
    selector = FullSpectrumSelector()
    
    # Test auto-determination
    print("\nAuto mode logic:")
    print(f"  Strong trend + low vol: {selector.determine_mode('bull', 0.015, 0.8, 10)}")
    print(f"  Moderate trend + med vol: {selector.determine_mode('bull', 0.035, 0.5, 12)}")
    print(f"  High vol + weak trend: {selector.determine_mode('chop', 0.07, 0.3, 15)}")
    
    # Test manual override
    selector.set_mode("position")
    print(f"\nManual override: {selector.get_current_mode()}")
    
    # Get all configs
    configs = selector.get_all_configs()
    print("\nMode Configurations:")
    for c in configs:
        print(f"  {c['name']}: {c['target']}% target, {c['stop']}% stop")
    
    # Save HTML
    html = selector.generate_html()
    with open("full_spectrum_selector.html", "w") as f:
        f.write(html)
    
    print("\nSaved to full_spectrum_selector.html")
    print("OK!")
