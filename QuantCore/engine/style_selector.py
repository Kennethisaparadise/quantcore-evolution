"""
QuantCore - Strategy Style Selector v1.0
Head 37: Switch between scalping and swing trading modes
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class TradingStyle(Enum):
    SCALP = "scalp"
    SWING = "swing"
    AUTO = "auto"


@dataclass
class StyleConfig:
    """Configuration for a trading style."""
    name: str
    timeframe: str
    max_hold_seconds: int
    profit_target_pct: float
    stop_loss_pct: float
    position_size_pct: float
    description: str


class StrategyStyleSelector:
    """
    Select and manage trading styles: scalp, swing, or auto.
    """
    
    # Style configurations
    SCALP_CONFIG = StyleConfig(
        name="scalp",
        timeframe="30s-5m",
        max_hold_seconds=120,
        profit_target_pct=0.003,  # 0.3%
        stop_loss_pct=0.001,     # 0.1%
        position_size_pct=0.1,    # 10% of capital
        description="High-frequency, low profit targets, tight stops"
    )
    
    SWING_CONFIG = StyleConfig(
        name="swing",
        timeframe="1h-1d",
        max_hold_seconds=259200,  # 3 days
        profit_target_pct=0.05,   # 5%
        stop_loss_pct=0.02,       # 2%
        position_size_pct=0.3,     # 30% of capital
        description="Lower frequency, larger targets, wider stops"
    )
    
    def __init__(self):
        self.current_style = TradingStyle.AUTO
        self.manual_override = None
        self.style_history: List[Dict] = []
        
        # Auto-switch parameters
        self.volatility_threshold = 0.03  # 3% - above this = scalp
        self.trend_threshold = 0.6  # Above this = swing
        
        # Performance tracking by style
        self.performance = {
            "scalp": {"trades": 0, "pnl": 0, "wins": 0},
            "swing": {"trades": 0, "pnl": 0, "wins": 0}
        }
    
    def set_style(self, style: str):
        """Manually set trading style."""
        if style in ["scalp", "swing", "auto"]:
            self.current_style = TradingStyle(style)
            self.manual_override = style if style != "auto" else None
    
    def get_current_style(self) -> str:
        """Get current active style."""
        if self.current_style == TradingStyle.AUTO:
            return "auto"
        return self.current_style.value
    
    def get_current_config(self) -> StyleConfig:
        """Get configuration for current style."""
        if self.current_style == TradingStyle.SCALP:
            return self.SCALP_CONFIG
        elif self.current_style == TradingStyle.SWING:
            return self.SWING_CONFIG
        else:
            # Auto - determine based on market conditions
            return self._determine_optimal_config()
    
    def _determine_optimal_config(self) -> StyleConfig:
        """Determine optimal style based on market conditions."""
        # This would integrate with regime detection
        # For now, default to scalp for high-frequency trading
        return self.SCALP_CONFIG
    
    def determine_style(self, regime: str, volatility: float, trend_strength: float) -> str:
        """Determine best style based on market conditions."""
        # High volatility + weak trend = scalp (avoid large moves)
        if volatility > self.volatility_threshold:
            return "scalp"
        
        # Strong trend + low volatility = swing (ride the wave)
        if trend_strength > self.trend_threshold and volatility < self.volatility_threshold * 0.5:
            return "swing"
        
        # Default to scalp
        return "scalp"
    
    def switch_if_needed(self, regime: str, volatility: float, trend_strength: float) -> str:
        """Switch style if auto mode and conditions warrant."""
        if self.current_style != TradingStyle.AUTO:
            return self.current_style.value
        
        new_style = self.determine_style(regime, volatility, trend_strength)
        
        if new_style != self.get_current_style():
            self.current_style = TradingStyle(new_style)
            self.style_history.append({
                "timestamp": "now",
                "old_style": self.get_current_style(),
                "new_style": new_style,
                "regime": regime,
                "volatility": volatility,
                "trend_strength": trend_strength
            })
        
        return new_style
    
    def record_trade(self, style: str, pnl: float, is_win: bool):
        """Record trade performance by style."""
        if style in self.performance:
            self.performance[style]["trades"] += 1
            self.performance[style]["pnl"] += pnl
            if is_win:
                self.performance[style]["wins"] += 1
    
    def get_performance(self) -> Dict:
        """Get performance metrics by style."""
        result = {}
        for style, stats in self.performance.items():
            if stats["trades"] > 0:
                result[style] = {
                    "trades": stats["trades"],
                    "pnl": stats["pnl"],
                    "wins": stats["wins"],
                    "win_rate": (stats["wins"] / stats["trades"]) * 100,
                    "avg_pnl": stats["pnl"] / stats["trades"]
                }
            else:
                result[style] = {"trades": 0, "pnl": 0, "wins": 0, "win_rate": 0, "avg_pnl": 0}
        return result
    
    def get_switch_history(self) -> List[Dict]:
        """Get style switch history."""
        return self.style_history
    
    def mutate_parameters(self):
        """Evolve auto-switch parameters."""
        if random.random() < 0.3:
            self.volatility_threshold = max(0.01, min(0.1, 
                self.volatility_threshold + random.uniform(-0.005, 0.005)))
        
        if random.random() < 0.3:
            self.trend_threshold = max(0.3, min(0.9, 
                self.trend_threshold + random.uniform(-0.05, 0.05)))
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        current_style = self.get_current_style()
        config = self.get_current_config()
        perf = self.get_performance()
        
        # Style cards
        style_cards = ""
        for style_name, style_config in [("scalp", self.SCALP_CONFIG), ("swing", self.SWING_CONFIG)]:
            is_active = style_name == current_style
            style_perf = perf.get(style_name, {})
            
            style_cards += f"""
            <div class="style-card {'active' if is_active else ''}">
                <div class="style-header">
                    <span class="style-name">{style_config.name.upper()}</span>
                    {'✓' if is_active else ''}
                </div>
                <div class="style-desc">{style_config.description}</div>
                <div class="style-params">
                    <div>Timeframe: {style_config.timeframe}</div>
                    <div>Target: {style_config.profit_target_pct*100:.1f}%</div>
                    <div>Stop: {style_config.stop_loss_pct*100:.2f}%</div>
                    <div>Position: {style_config.position_size_pct*100:.0f}%</div>
                </div>
                <div class="style-stats">
                    <div>Trades: {style_perf.get('trades', 0)}</div>
                    <div>Win Rate: {style_perf.get('win_rate', 0):.1f}%</div>
                </div>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Strategy Style Selector - QuantCore</title>
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
        .current-style {{
            background: linear-gradient(135deg, #238636, #2ea043);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .current-style h2 {{
            margin: 0;
            font-size: 24px;
        }}
        .current-style p {{
            margin: 10px 0 0;
            color: #8b949e;
        }}
        .styles-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .style-card {{
            background: #161b22;
            border: 2px solid #30363d;
            border-radius: 10px;
            padding: 20px;
        }}
        .style-card.active {{
            border-color: #3fb950;
            box-shadow: 0 0 20px rgba(63, 185, 80, 0.3);
        }}
        .style-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .style-name {{
            font-size: 18px;
            font-weight: bold;
            color: #58a6ff;
        }}
        .style-desc {{
            color: #8b949e;
            font-size: 12px;
            margin-bottom: 15px;
        }}
        .style-params {{
            background: #21262d;
            padding: 10px;
            border-radius: 5px;
            font-size: 11px;
            line-height: 1.8;
        }}
        .style-stats {{
            margin-top: 15px;
            display: flex;
            gap: 20px;
            font-size: 12px;
            color: #8b949e;
        }}
        .controls {{
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
        }}
        button {{
            background: #238636;
            color: #fff;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-family: monospace;
            margin-right: 10px;
        }}
        button:hover {{
            background: #2ea043;
        }}
        button.active {{
            background: #58a6ff;
        }}
    </style>
</head>
<body>
    <div class="header">⚙️ STRATEGY STYLE SELECTOR</div>
    
    <div class="current-style">
        <h2>Current Mode: {current_style.upper()}</h2>
        <p>{config.description}</p>
    </div>
    
    <div class="styles-grid">
        {style_cards}
    </div>
    
    <div class="controls">
        <h3 style="color:#58a6ff;margin-bottom:15px;">Manual Override</h3>
        <button onclick="setStyle('scalp')">Scalp</button>
        <button onclick="setStyle('swing')">Swing</button>
        <button onclick="setStyle('auto')">Auto</button>
    </div>
    
    <script>
        function setStyle(style) {{
            fetch('/set_style?style=' + style).then(r => location.reload());
        }}
    </script>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("STRATEGY STYLE SELECTOR TEST")
    print("=" * 50)
    
    selector = StrategyStyleSelector()
    
    # Test auto-switch
    print("\nAuto-switch logic:")
    print(f"  High vol + weak trend: {selector.determine_style('chop', 0.05, 0.3)}")
    print(f"  Low vol + strong trend: {selector.determine_style('bull', 0.02, 0.8)}")
    
    # Test manual override
    selector.set_style("swing")
    print(f"\nManual override: {selector.get_current_style()}")
    
    # Record some trades
    selector.record_trade("scalp", 50, True)
    selector.record_trade("scalp", -20, False)
    selector.record_trade("swing", 200, True)
    
    perf = selector.get_performance()
    print("\nPerformance by style:")
    for style, stats in perf.items():
        if stats["trades"] > 0:
            print(f"  {style}: {stats['trades']} trades, {stats['win_rate']:.1f}% win, ${stats['pnl']:.2f}")
    
    # Save HTML
    html = selector.generate_html()
    with open("style_selector.html", "w") as f:
        f.write(html)
    
    print("\nSaved to style_selector.html")
    print("OK!")
