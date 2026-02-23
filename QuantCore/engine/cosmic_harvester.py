"""
QuantCore - Cosmic Harvester v1.0
Head 33: Lunar phases and Delta phenomenon for market timing
"""

import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LunarPhase:
    """Lunar phase information."""
    phase: str  # new_moon, waxing, full_moon, waning
    illumination: float  # 0-100%
    days_from_new: float


class CosmicHarvester:
    """
    Harvest cosmic signals: lunar phases and Delta phenomenon.
    """
    
    # Synodic month (new moon to new moon)
    SYNODIC_MONTH = 29.530588853
    
    # Delta cycles (in days)
    DELTA_CYCLES = {
        "STD": 4,      # Short Term Delta - Earth rotation
        "ITD": 120,   # Intermediate Term Delta - lunar months
        "MTD": 365,   # Medium Term Delta - year
        "LTD": 1461,  # Long Term Delta - 4 years
        "SLTD": 6939, # Super Long Term Delta - 19 years
    }
    
    def __init__(self):
        self.last_new_moon = datetime(2026, 1, 29, 12, 0)  # Known new moon
        self.enable_lunar = True
        self.enable_delta = True
        
        # Historical patterns
        self.lunar_strategies = []
        self.delta_patterns = {}
    
    def get_lunar_phase(self, date: datetime = None) -> LunarPhase:
        """Calculate lunar phase for a date."""
        if date is None:
            date = datetime.now()
        
        # Days since known new moon
        days_since_new = (date - self.last_new_moon).total_seconds() / 86400
        days_in_cycle = days_since_new % self.SYNODIC_MONTH
        illumination = (1 - math.cos(2 * math.pi * days_in_cycle / self.SYNODIC_MONTH)) / 2 * 100
        
        # Determine phase
        if days_in_cycle < 1.85:
            phase = "new_moon"
        elif days_in_cycle < 7.38:
            phase = "waxing_crescent"
        elif days_in_cycle < 9.23:
            phase = "first_quarter"
        elif days_in_cycle < 14.77:
            phase = "waxing_gibbous"
        elif days_in_cycle < 16.61:
            phase = "full_moon"
        elif days_in_cycle < 22.15:
            phase = "waning_gibbous"
        elif days_in_cycle < 24.00:
            phase = "last_quarter"
        else:
            phase = "waning_crescent"
        
        return LunarPhase(
            phase=phase,
            illumination=illumination,
            days_from_new=days_in_cycle
        )
    
    def is_lunar_signal(self, date: datetime = None) -> Dict:
        """Check if today is a lunar signal day."""
        phase = self.get_lunar_phase(date)
        
        signals = {
            "new_moon": phase.days_from_new < 1,
            "full_moon": phase.days_from_new > 14.77 and phase.days_from_new < 16.61,
            "first_quarter": phase.days_from_new > 7.38 and phase.days_from_new < 9.23,
            "last_quarter": phase.days_from_new > 22.15 and phase.days_from_new < 24.00,
        }
        
        return {
            "phase": phase.phase,
            "illumination": phase.illumination,
            "days_from_new": phase.days_from_new,
            "signals": signals,
            "any_signal": any(signals.values())
        }
    
    def calculate_delta_cycles(self, date: datetime = None) -> Dict:
        """Calculate Delta cycle positions."""
        if date is None:
            date = datetime.now()
        
        # Reference date (can be adjusted per market)
        ref_date = datetime(2000, 1, 1)
        days_since_ref = (date - ref_date).total_seconds() / 86400
        
        positions = {}
        for name, cycle_days in self.DELTA_CYCLES.items():
            position = days_since_ref % cycle_days
            position_pct = (position / cycle_days) * 100
            positions[name] = {
                "position": position,
                "cycle_days": cycle_days,
                "progress_pct": position_pct,
                "turning_point": position < 3 or position > cycle_days - 3
            }
        
        return positions
    
    def get_delta_signal(self, date: datetime = None) -> Dict:
        """Get Delta-based signal."""
        cycles = self.calculate_delta_cycles(date)
        
        # Weight shorter cycles more for trading
        signal = 0
        for name, data in cycles.items():
            if data["turning_point"]:
                if name == "STD":
                    signal += 3
                elif name == "ITD":
                    signal += 2
                else:
                    signal += 1
        
        return {
            "signal_strength": signal,
            "is_turning_point": signal >= 3,
            "cycles": cycles
        }
    
    def get_cosmic_bias(self, date: datetime = None) -> Dict:
        """Get combined cosmic bias for trading."""
        lunar = self.is_lunar_signal(date)
        delta = self.get_delta_signal(date)
        
        # Calculate bias
        bias = 0
        reasons = []
        
        # Lunar bias
        if lunar["signals"]["new_moon"]:
            bias += 1
            reasons.append("New moon - historically bullish")
        elif lunar["signals"]["full_moon"]:
            bias -= 1
            reasons.append("Full moon - historically bearish")
        
        # Delta bias
        if delta["is_turning_point"]:
            bias += 1 if delta["signal_strength"] > 3 else 0
            reasons.append("Delta turning point")
        
        return {
            "lunar": lunar,
            "delta": delta,
            "bias": bias,
            "bias_direction": "bullish" if bias > 0 else ("bearish" if bias < 0 else "neutral"),
            "reasons": reasons
        }
    
    def add_lunar_strategy(self, name: str, entry_day: int, exit_day: int, description: str):
        """Add a lunar strategy to the pool."""
        self.lunar_strategies.append({
            "name": name,
            "entry_day": entry_day,
            "exit_day": exit_day,
            "description": description
        })
    
    def get_lunar_strategies(self) -> List[Dict]:
        """Get all lunar strategies."""
        return self.lunar_strategies
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        cosmic = self.get_cosmic_bias()
        lunar = cosmic["lunar"]
        delta = cosmic["delta"]
        
        # Phase indicator
        phase_icons = {
            "new_moon": "🌑",
            "waxing_crescent": "🌒",
            "first_quarter": "🌓",
            "waxing_gibbous": "🌔",
            "full_moon": "🌕",
            "waning_gibbous": "🌖",
            "last_quarter": "🌗",
            "waning_crescent": "🌘"
        }
        
        icon = phase_icons.get(lunar["phase"], "🌙")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cosmic Harvester - QuantCore</title>
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
        .phase-display {{
            text-align: center;
            padding: 30px;
            background: #161b22;
            border-radius: 20px;
            margin-bottom: 20px;
        }}
        .phase-icon {{
            font-size: 80px;
        }}
        .phase-name {{
            font-size: 24px;
            color: #58a6ff;
            margin-top: 10px;
        }}
        .illumination {{
            color: #8b949e;
            font-size: 14px;
        }}
        .bias {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .bias-card {{
            flex: 1;
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .bias-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .bullish {{ color: #3fb950; }}
        .bearish {{ color: #f85149; }}
        .neutral {{ color: #8b949e; }}
        
        .delta-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-top: 20px;
        }}
        .delta-card {{
            background: #161b22;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        .delta-name {{
            color: #8b949e;
            font-size: 11px;
        }}
        .delta-value {{
            color: #58a6ff;
            font-size: 18px;
            font-weight: bold;
        }}
        .turning-point {{
            color: #f7931a;
            font-size: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">🌙 COSMIC HARVESTER</div>
    
    <div class="phase-display">
        <div class="phase-icon">{icon}</div>
        <div class="phase-name">{lunar['phase'].replace('_', ' ').title()}</div>
        <div class="illumination">{lunar['illumination']:.1f}% illuminated</div>
    </div>
    
    <div class="bias">
        <div class="bias-card">
            <div class="bias-label" style="color:#8b949e;font-size:11px;">BIAS</div>
            <div class="bias-value {'bullish' if cosmic['bias_direction'] == 'bullish' else ('bearish' if cosmic['bias_direction'] == 'bearish' else 'neutral')}">{cosmic['bias_direction'].upper()}</div>
        </div>
        <div class="bias-card">
            <div class="bias-label" style="color:#8b949e;font-size:11px;">STRENGTH</div>
            <div class="bias-value">{abs(cosmic['bias'])}/3</div>
        </div>
        <div class="bias-card">
            <div class="bias-label" style="color:#8b949e;font-size:11px;">DELTA TP</div>
            <div class="bias-value {'bullish' if delta['is_turning_point'] else 'neutral'}">{'YES' if delta['is_turning_point'] else 'NO'}</div>
        </div>
    </div>
    
    <h3 style="color:#58a6ff">Delta Cycles</h3>
    <div class="delta-grid">
"""
        
        for name, data in delta["cycles"].items():
            turning = "⚠️ TP" if data["turning_point"] else ""
            html += f"""
        <div class="delta-card">
            <div class="delta-name">{name}</div>
            <div class="delta-value">{data['progress_pct']:.0f}%</div>
            <div class="turning-point">{turning}</div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("COSMIC HARVESTER TEST")
    print("=" * 50)
    
    cosmic = CosmicHarvester()
    
    # Get lunar phase
    lunar = cosmic.get_lunar_phase()
    print(f"\nCurrent Phase: {lunar.phase}")
    print(f"Illumination: {lunar.illumination:.1f}%")
    
    # Get cosmic bias
    bias = cosmic.get_cosmic_bias()
    print(f"\nBias: {bias['bias_direction']} ({bias['bias']})")
    print(f"Reasons: {', '.join(bias['reasons']) if bias['reasons'] else 'None'}")
    
    # Get Delta cycles
    delta = cosmic.get_delta_signal()
    print(f"\nDelta Turning Point: {'YES' if delta['is_turning_point'] else 'NO'}")
    print(f"Signal Strength: {delta['signal_strength']}")
    
    # Add some lunar strategies
    cosmic.add_lunar_strategy("New Moon Accumulator", 1, 15, "Buy at new moon, sell at full")
    cosmic.add_lunar_strategy("Full Moon Fader", 15, 29, "Short near full moon")
    
    strategies = cosmic.get_lunar_strategies()
    print(f"\nLunar Strategies: {len(strategies)}")
    
    # Save HTML
    html = cosmic.generate_html()
    with open("cosmic_harvester.html", "w") as f:
        f.write(html)
    
    print("\nSaved to cosmic_harvester.html")
    print("OK!")
