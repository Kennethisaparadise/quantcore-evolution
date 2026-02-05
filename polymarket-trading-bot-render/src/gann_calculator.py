"""
Gann Calculator
Implements W.D. Gann's geometric support/resistance methodology.

Gann Formula:
Degrees = (Price^.5 + Factor)^2

Where Factor corresponds to degree measurements:
┌─────────┬────────┐
│ Degrees │ Factor │
├─────────┼────────┤
│ 360°   │ 2.00   │
│ 270°   │ 1.50   │
│ 180°   │ 1.00   │
│ 135°   │ 0.75   │
│ 90°    │ 0.50   │
│ 45°    │ 0.25   │
│ 22.5°  │ 0.125  │
│ 11.25° │ 0.0625 │
└─────────┴────────┘

Example:
- Price: 44.40
- √44.40 = 6.663
- Add 0.50 (90°) = 7.163
- Square = 51.30 (90° resistance)
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class Degree(Enum):
    """Gann degree angles."""
    DEGREE_360 = 2.00
    DEGREE_315 = 1.75
    DEGREE_270 = 1.50
    DEGREE_225 = 1.25
    DEGREE_180 = 1.00
    DEGREE_135 = 0.75
    DEGREE_90 = 0.50
    DEGREE_45 = 0.25
    DEGREE_22_5 = 0.125
    DEGREE_11_25 = 0.0625


@dataclass
class GannLevel:
    """A single Gann support/resistance level."""
    price: float
    degree: float
    factor: float
    type: str  # "support" or "resistance"
    distance_from_price: float


@dataclass
class GannAnalysis:
    """Complete Gann analysis for a price."""
    current_price: float
    timestamp: datetime
    
    # Support levels (below price)
    support_levels: List[GannLevel]
    
    # Resistance levels (above price)
    resistance_levels: List[GannLevel]
    
    # Nearest support/resistance
    nearest_support: Optional[GannLevel]
    nearest_resistance: Optional[GannLevel]
    
    # Strongest level (closest to price)
    strongest_level: Optional[GannLevel]


class GannCalculator:
    """
    Calculator for Gann's geometric support/resistance.
    
    Uses the formula:
    Level = (Price^.5 + Factor)^2
    
    Where Factor depends on the degree angle.
    """
    
    # Degree factors
    DEGREE_FACTORS = {
        360.0: 2.00,
        315.0: 1.75,
        270.0: 1.50,
        225.0: 1.25,
        180.0: 1.00,
        135.0: 0.75,
        90.0: 0.50,
        45.0: 0.25,
        22.5: 0.125,
        11.25: 0.0625,
    }
    
    # Common degrees to check
    COMMON_DEGREES = [45, 90, 135, 180, 225, 270, 315, 360]
    
    def __init__(self):
        pass
    
    def calculate_level(self, price: float, degree: float) -> float:
        """
        Calculate a Gann level for a given price and degree.
        
        Formula:
        Level = (Price^.5 + Factor)^2
        
        Args:
            price: The base price
            degree: The degree angle
            
        Returns:
            The calculated support/resistance level
        """
        factor = self.DEGREE_FACTORS.get(degree, 0.50)  # Default to 90°
        
        sqrt_price = math.sqrt(price)
        level = (sqrt_price + factor) ** 2
        
        return round(level, 4)
    
    def calculate_support_resistance(self, price: float, 
                                   degrees: List[float] = None) -> GannAnalysis:
        """
        Calculate all support and resistance levels for a price.
        
        Args:
            price: Current price
            degrees: List of degrees to calculate (defaults to COMMON_DEGREES)
            
        Returns:
            GannAnalysis with all levels
        """
        if degrees is None:
            degrees = self.COMMON_DEGREES
        
        supports = []
        resistances = []
        
        for degree in degrees:
            factor = self.DEGREE_FACTORS.get(degree, 0.50)
            level = self.calculate_level(price, degree)
            
            gann_level = GannLevel(
                price=level,
                degree=degree,
                factor=factor,
                type="support" if level < price else "resistance",
                distance_from_price=abs(level - price)
            )
            
            if level < price:
                supports.append(gann_level)
            else:
                resistances.append(gann_level)
        
        # Sort by distance
        supports.sort(key=lambda x: x.distance_from_price)
        resistances.sort(key=lambda x: x.distance_from_price)
        
        # Find nearest
        nearest_support = supports[0] if supports else None
        nearest_resistance = resistances[0] if resistances else None
        
        # Find strongest (closest level regardless of direction)
        all_levels = supports + resistances
        all_levels.sort(key=lambda x: x.distance_from_price)
        strongest = all_levels[0] if all_levels else None
        
        return GannAnalysis(
            current_price=price,
            timestamp=datetime.now(),
            support_levels=supports,
            resistance_levels=resistances,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            strongest_level=strongest
        )
    
    def get_trading_levels(self, price: float) -> Dict:
        """
        Get actionable trading levels.
        
        Returns dict with stop loss, targets, and key levels.
        """
        analysis = self.calculate_support_resistance(price)
        
        # Key levels
        sl = analysis.nearest_support.price if analysis.nearest_support else price * 0.98
        tp = analysis.nearest_resistance.price if analysis.nearest_resistance else price * 1.02
        
        return {
            'current_price': price,
            'nearest_support': {
                'price': analysis.nearest_support.price if analysis.nearest_support else None,
                'degree': analysis.nearest_support.degree if analysis.nearest_support else None,
            },
            'nearest_resistance': {
                'price': analysis.nearest_resistance.price if analysis.nearest_resistance else None,
                'degree': analysis.nearest_resistance.degree if analysis.nearest_resistance else None,
            },
            'stop_loss': round(sl, 4),
            'take_profit': round(tp, 4),
            'risk_reward': round((tp - price) / (price - sl), 2) if sl < price else 0,
            'strongest_level': {
                'price': analysis.strongest_level.price if analysis.strongest_level else None,
                'degree': analysis.strongest_level.degree if analysis.strongest_level else None,
                'type': analysis.strongest_level.type if analysis.strongest_level else None,
            }
        }
    
    def calculate_degree_from_prices(self, low: float, high: float) -> float:
        """
        Calculate the degree of a move from low to high.
        
        Formula:
        Degree = (√High - √Low) / Factor * 180
        
        This tells you how many degrees a price move represents.
        """
        sqrt_high = math.sqrt(high)
        sqrt_low = math.sqrt(low)
        difference = sqrt_high - sqrt_low
        
        # Estimate degree (assuming 90° move = 0.50 factor)
        degree = (difference / 0.50) * 90
        
        return round(degree, 2)
    
    def find_cluster(self, prices: List[float], tolerance: float = 0.02) -> List[Dict]:
        """
        Find clustered Gann levels across multiple prices.
        
        When multiple prices hit the same Gann level,
        it creates a STRONG support/resistance zone.
        
        Args:
            prices: List of prices to analyze
            tolerance: How close levels must be to cluster (2% = 0.02)
            
        Returns:
            List of clustered levels with count of hits
        """
        all_levels = []
        
        for price in prices:
            analysis = self.calculate_support_resistance(price)
            if analysis.strongest_level:
                all_levels.append(analysis.strongest_level)
        
        # Find clusters
        clusters = []
        used = set()
        
        for i, level in enumerate(all_levels):
            if i in used:
                continue
            
            cluster = {
                'level': level.price,
                'degree': level.degree,
                'type': level.type,
                'count': 1,
                'prices_affected': [prices[i]]
            }
            
            for j, other in enumerate(all_levels):
                if j in used or j == i:
                    continue
                
                # Check if within tolerance
                diff = abs(other.price - level.price) / level.price
                if diff <= tolerance:
                    cluster['count'] += 1
                    cluster['prices_affected'].append(prices[j])
                    used.add(j)
            
            if cluster['count'] >= 2:  # Only clusters of 2+
                clusters.append(cluster)
        
        # Sort by count
        clusters.sort(key=lambda x: x['count'], reverse=True)
        
        return clusters


def gann_trading_example():
    """Example usage of Gann calculator."""
    
    print("=" * 60)
    print("GANN GEOMETRIC SUPPORT/RESISTANCE")
    print("=" * 60)
    
    # Example from the text: BKX bank index
    price = 44.40
    
    print(f"\nPrice: ${price}")
    print("-" * 60)
    
    calculator = GannCalculator()
    
    # Calculate levels
    analysis = calculator.calculate_support_resistance(price)
    
    print("\n📊 SUPPORT LEVELS (below price):")
    for level in analysis.support_levels[:5]:
        print(f"  ${level.price:>8} | {level.degree:>5.1f}° | Factor: {level.factor}")
    
    print("\n📊 RESISTANCE LEVELS (above price):")
    for level in analysis.resistance_levels[:5]:
        print(f"  ${level.price:>8} | {level.degree:>5.1f}° | Factor: {level.factor}")
    
    print(f"\n🎯 NEAREST SUPPORT: ${analysis.nearest_support.price:.4f} ({analysis.nearest_support.degree}°)")
    print(f"🎯 NEAREST RESISTANCE: ${analysis.nearest_resistance.price:.4f} ({analysis.nearest_resistance.degree}°)")
    print(f"💪 STRONGEST LEVEL: ${analysis.strongest_level.price:.4f} ({analysis.strongest_level.degree}°)")
    
    # Trading levels
    levels = calculator.get_trading_levels(price)
    print(f"\n📋 TRADING LEVELS:")
    print(f"  Entry: ${levels['current_price']}")
    print(f"  Stop Loss: ${levels['stop_loss']}")
    print(f"  Take Profit: ${levels['take_profit']}")
    print(f"  R:R Ratio: {levels['risk_reward']}:1")
    
    # Degree of move example
    print("\n" + "=" * 60)
    print("DEGREE OF MOVE CALCULATION")
    print("=" * 60)
    
    low = 44.40
    high = 51.30
    degree = calculator.calculate_degree_from_prices(low, high)
    print(f"\nMove from ${low} to ${high}")
    print(f"Represents: {degree}°")
    print("(Should be ~90° according to Gann)")


if __name__ == "__main__":
    gann_trading_example()
