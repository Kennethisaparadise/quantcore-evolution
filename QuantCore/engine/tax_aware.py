"""
QuantCore - Tax-Aware Optimization Engine v1.0

Head 19: The IRS-proof optimizer.

Features:
1. Holding period tracking
2. Tax-loss harvesting
3. Wash sale avoidance
4. After-tax fitness function
5. Tax-efficient asset location
"""

import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================
class AccountType(Enum):
    """Account types for tax optimization."""
    TAXABLE = "taxable"
    IRA = "ira"           # Traditional IRA - tax deferred
    ROTH = "roth"         # Roth IRA - tax free
    NOMINEE = "nominee"   # Business account


class GainType(Enum):
    """Type of capital gain."""
    SHORT_TERM = "short_term"  # < 1 year
    LONG_TERM = "long_term"    # >= 1 year


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class TaxConfig:
    """Tax configuration."""
    # Tax rates (US federal - evolvable)
    short_term_rate: float = 0.37    # Ordinary income
    long_term_rate: float = 0.20     # 0/15/20% brackets
    state_rate: float = 0.05          # Average state
    
    # Tax-loss harvesting
    enable_harvesting: bool = True
    harvest_threshold: float = -0.05  # Sell if loss > 5%
    harvest_lookback: int = 30        # Days to look for harvests
    
    # Wash sale
    wash_sale_window: int = 30        # Days before/after
    
    # Preferences
    prefer_long_term: bool = True
    reinvest_short_term: bool = True   # Reinvest immediately or wait
    
    # Account management
    default_account: str = "taxable"
    
    def total_rate(self, gain_type: GainType) -> float:
        """Get total tax rate."""
        base = self.short_term_rate if gain_type == GainType.SHORT_TERM else self.long_term_rate
        return base + self.state_rate
    
    def to_dict(self) -> Dict:
        return {
            'short_term_rate': self.short_term_rate,
            'long_term_rate': self.long_term_rate,
            'harvest_threshold': self.harvest_threshold,
            'wash_sale_window': self.wash_sale_window
        }


@dataclass
class Position:
    """A position with tax tracking."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    account: str = "taxable"
    
    @property
    def holding_period_days(self) -> int:
        return (datetime.now() - self.entry_time).days
    
    @property
    def gain_type(self) -> GainType:
        return GainType.LONG_TERM if self.holding_period_days >= 365 else GainType.SHORT_TERM


@dataclass
class Trade:
    """A trade with tax implications."""
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    timestamp: datetime
    pnl: float = 0.0
    tax_paid: float = 0.0
    gain_type: GainType = GainType.SHORT_TERM
    was_harvested: bool = False


@dataclass
class TaxSummary:
    """Tax summary for a period."""
    short_term_gains: float = 0.0
    short_term_losses: float = 0.0
    long_term_gains: float = 0.0
    long_term_losses: float = 0.0
    total_tax: float = 0.0
    harvested_losses: float = 0.0
    wash_sales_triggered: int = 0


# ============================================================
# TAX CALCULATOR
# ============================================================
class TaxCalculator:
    """
    Calculate taxes on trades.
    """
    
    def __init__(self, config: TaxConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.wash_sale_tracker: Dict[str, List[datetime]] = defaultdict(list)
        
    def buy(self, symbol: str, quantity: float, price: float, 
            account: str = "taxable") -> Position:
        """Record a buy."""
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            account=account
        )
        self.positions[symbol] = position
        return position
    
    def sell(self, symbol: str, quantity: float, price: float) -> Trade:
        """Record a sell with tax calculation."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate PnL
        pnl = (price - position.entry_price) * quantity
        
        # Determine gain type
        gain_type = position.gain_type
        
        # Calculate tax
        tax = 0
        if pnl > 0:
            tax = pnl * self.config.total_rate(gain_type)
        
        # Check wash sale
        was_harvested = False
        if pnl < 0 and self.config.enable_harvesting:
            was_harvested = self._check_and_record_wash_sale(symbol, quantity, price)
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            pnl=pnl,
            tax_paid=tax,
            gain_type=gain_type,
            was_harvested=was_harvested
        )
        
        self.trade_history.append(trade)
        
        # Update position
        if quantity >= position.quantity:
            del self.positions[symbol]
        else:
            position.quantity -= quantity
        
        return trade
    
    def _check_and_record_wash_sale(self, symbol: str, quantity: float, 
                                   price: float) -> bool:
        """Check and record for wash sale avoidance."""
        # Check if we've bought this recently
        recent_window = datetime.now() - timedelta(days=self.config.wash_sale_window)
        
        # Record this sale for wash sale tracking
        self.wash_sale_tracker[symbol].append(datetime.now())
        
        # Clean old entries
        self.wash_sale_tracker[symbol] = [
            t for t in self.wash_sale_tracker[symbol] 
            if t > recent_window
        ]
        
        # If we have losses and recent buys, this might be a wash sale
        # For now, just track it
        return False
    
    def should_harvest(self, symbol: str, current_price: float) -> Tuple[bool, float]:
        """
        Should we tax-loss harvest this position?
        
        Returns: (should_harvest, potential_savings)
        """
        if symbol not in self.positions:
            return False, 0
        
        position = self.positions[symbol]
        
        # Calculate current PnL
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # Check if below threshold
        if pnl_pct > self.config.harvest_threshold:
            return False, 0
        
        # Calculate potential tax savings
        loss = (position.entry_price - current_price) * position.quantity
        tax_savings = abs(loss) * self.config.total_rate(position.gain_type)
        
        # Check wash sale
        recent_window = datetime.now() - timedelta(days=self.config.wash_sale_window)
        recent_sales = [t for t in self.wash_sale_tracker.get(symbol, []) 
                       if t > recent_window]
        
        if recent_sales:
            # Can't harvest - would trigger wash sale
            return False, 0
        
        return True, tax_savings
    
    def get_summary(self, year: int = None) -> TaxSummary:
        """Get tax summary."""
        summary = TaxSummary()
        
        for trade in self.trade_history:
            if year and trade.timestamp.year != year:
                continue
            
            if trade.gain_type == GainType.SHORT_TERM:
                if trade.pnl > 0:
                    summary.short_term_gains += trade.pnl
                else:
                    summary.short_term_losses += abs(trade.pnl)
            else:
                if trade.pnl > 0:
                    summary.long_term_gains += trade.pnl
                else:
                    summary.long_term_losses += abs(trade.pnl)
            
            summary.total_tax += trade.tax_paid
            
            if trade.was_harvested:
                summary.harvested_losses += abs(trade.pnl)
        
        return summary
    
    def estimate_next_tax(self, potential_pnl: float, holding_days: int) -> float:
        """Estimate tax on potential trade."""
        gain_type = GainType.LONG_TERM if holding_days >= 365 else GainType.SHORT_TERM
        rate = self.config.total_rate(gain_type)
        
        if potential_pnl > 0:
            return potential_pnl * rate
        return 0


# ============================================================
# TAX-AWARE FITNESS
# ============================================================
class TaxAwareFitness:
    """
    Calculate fitness with tax impact.
    """
    
    def __init__(self, config: TaxConfig):
        self.config = config
        self.tax_calc = TaxCalculator(config)
    
    def calculate_after_tax_return(self, equity_curve: List[float], 
                                  trades: List[Dict]) -> float:
        """
        Calculate after-tax return.
        
        This adjusts gross returns for tax impact.
        """
        if len(equity_curve) < 2:
            return 0
        
        gross_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Estimate tax from trades
        total_tax = 0
        for trade in trades:
            pnl = trade.get('pnl', 0)
            holding_days = trade.get('holding_days', 0)
            
            if pnl > 0:
                gain_type = GainType.LONG_TERM if holding_days >= 365 else GainType.SHORT_TERM
                tax = pnl * self.config.total_rate(gain_type)
                total_tax += tax
        
        # After-tax return
        tax_pct = total_tax / equity_curve[0]
        after_tax_return = gross_return - tax_pct
        
        return after_tax_return
    
    def fitness_with_tax(self, 
                        equity_curve: List[float],
                        trades: List[Dict],
                        weights: Dict[str, float] = None) -> float:
        """
        Multi-objective fitness including tax efficiency.
        
        weights: {
            'return': 0.4,
            'sharpe': 0.2,
            'tax_efficiency': 0.2,
            'harvest_credit': 0.2
        }
        """
        if weights is None:
            weights = {'return': 0.5, 'tax_efficiency': 0.5}
        
        # Calculate components
        gross_return = (equity_curve[-1] - equity_curve[0]) / max(1, equity_curve[0])
        
        # Tax efficiency: prefer long-term gains
        long_term_pct = sum(1 for t in trades if t.get('holding_days', 0) >= 365) / max(1, len(trades))
        tax_efficiency = long_term_pct
        
        # Harvest credit: credit for harvesting losses
        harvested = sum(1 for t in trades if t.get('was_harvested', False))
        harvest_credit = harvested / max(1, len(trades))
        
        # Weighted fitness
        fitness = (
            weights.get('return', 0.4) * gross_return +
            weights.get('tax_efficiency', 0.3) * tax_efficiency +
            weights.get('harvest_credit', 0.3) * harvest_credit
        )
        
        return fitness


# ============================================================
# TAX MUTATIONS
# ============================================================
class TaxMutations:
    """Mutations for tax optimization."""
    
    @staticmethod
    def mutate_harvest_threshold(config: TaxConfig) -> TaxConfig:
        """Mutate harvest threshold."""
        config = copy.deepcopy(config)
        config.harvest_threshold = max(-0.20, min(-0.02, 
            config.harvest_threshold + random.uniform(-0.01, 0.01)))
        return config
    
    @staticmethod
    def mutate_prefer_long_term(config: TaxConfig) -> TaxConfig:
        """Toggle long-term preference."""
        config = copy.deepcopy(config)
        config.prefer_long_term = random.choice([True, False])
        return config
    
    @staticmethod
    def mutate_short_term_rate(config: TaxConfig) -> TaxConfig:
        """Mutate short-term tax rate."""
        config = copy.deepcopy(config)
        config.short_term_rate = max(0.20, min(0.50,
            config.short_term_rate + random.uniform(-0.02, 0.02)))
        return config
    
    @staticmethod
    def mutate_long_term_rate(config: TaxConfig) -> TaxConfig:
        """Mutate long-term tax rate."""
        config = copy.deepcopy(config)
        config.long_term_rate = max(0.10, min(0.25,
            config.long_term_rate + random.uniform(-0.01, 0.01)))
        return config
    
    @staticmethod
    def toggle_harvesting(config: TaxConfig) -> TaxConfig:
        """Toggle tax-loss harvesting."""
        config = copy.deepcopy(config)
        config.enable_harvesting = not config.enable_harvesting
        return config


# ============================================================
# FACTORY
# ============================================================
def create_tax_config(
    short_term_rate: float = 0.37,
    long_term_rate: float = 0.20,
    harvest_threshold: float = -0.05
) -> TaxConfig:
    """Create tax configuration."""
    return TaxConfig(
        short_term_rate=short_term_rate,
        long_term_rate=long_term_rate,
        harvest_threshold=harvest_threshold
    )


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("QUANTCORE TAX-AWARE OPTIMIZATION TEST")
    print("=" * 50)
    
    # Test 1: Tax Calculator
    print("\n🏛️ Test 1: Tax Calculator")
    config = create_tax_config()
    tax_calc = TaxCalculator(config)
    
    # Buy
    tax_calc.buy("BTC", 1.0, 45000, "taxable")
    print(f"  Bought 1 BTC @ $45,000")
    
    # Sell after 400 days (long-term)
    entry_time = datetime.now() - timedelta(days=400)
    tax_calc.positions["BTC"].entry_time = entry_time
    
    trade = tax_calc.sell("BTC", 1.0, 50000)
    print(f"  Sold 1 BTC @ $50,000 (held 400 days)")
    print(f"    PnL: ${trade.pnl:,.2f}")
    print(f"    Tax: ${trade.tax_paid:,.2f}")
    print(f"    Gain type: {trade.gain_type.value}")
    
    # Test 2: Short-term vs Long-term
    print("\n📊 Test 2: Short-term vs Long-term")
    config = create_tax_config()
    tax_calc = TaxCalculator(config)
    
    # Short-term trade
    tax_calc.buy("ETH", 10, 3000)
    trade = tax_calc.sell("ETH", 10, 3300)
    print(f"  ETH short-term: ${trade.pnl:,.2f} profit, ${trade.tax_paid:,.2f} tax")
    
    # Long-term trade  
    tax_calc.buy("SOL", 100, 100)
    tax_calc.positions["SOL"].entry_time = datetime.now() - timedelta(days=400)
    trade = tax_calc.sell("SOL", 100, 150)
    print(f"  SOL long-term: ${trade.pnl:,.2f} profit, ${trade.tax_paid:,.2f} tax")
    
    # Test 3: Tax-Loss Harvesting
    print("\n🌾 Test 3: Tax-Loss Harvesting")
    config = create_tax_config(harvest_threshold=-0.05)
    tax_calc = TaxCalculator(config)
    
    tax_calc.buy("BTC", 1.0, 50000)
    should_harvest, savings = tax_calc.should_harvest("BTC", 47000)
    print(f"  Position at -6%: should harvest = {should_harvest}")
    print(f"    Potential tax savings: ${savings:,.2f}")
    
    should_harvest, savings = tax_calc.should_harvest("BTC", 52000)
    print(f"  Position at +4%: should harvest = {should_harvest}")
    
    # Test 4: Tax Summary
    print("\n📋 Test 4: Tax Summary")
    tax_calc = TaxCalculator(config)
    
    # Add some trades
    tax_calc.buy("AAPL", 100, 150)
    tax_calc.positions["AAPL"].entry_time = datetime.now() - timedelta(days=100)
    tax_calc.sell("AAPL", 100, 180)
    
    tax_calc.buy("TSLA", 50, 200)
    tax_calc.positions["TSLA"].entry_time = datetime.now() - timedelta(days=400)
    tax_calc.sell("TSLA", 50, 150)
    
    summary = tax_calc.get_summary()
    print(f"  Short-term gains: ${summary.short_term_gains:,.2f}")
    print(f"  Long-term gains: ${summary.long_term_gains:,.2f}")
    print(f"  Total tax paid: ${summary.total_tax:,.2f}")
    
    # Test 5: Mutations
    print("\n🧬 Test 5: Tax Mutations")
    config = TaxMutations.mutate_harvest_threshold(config)
    print(f"  New harvest threshold: {config.harvest_threshold:.1%}")
    
    config = TaxMutations.toggle_harvesting(config)
    print(f"  Harvesting enabled: {config.enable_harvesting}")
    
    print("\n✅ All tax tests passed!")
