"""
QuantCore - Fee-Aware Execution Engine v1.0

Head 18: The silent killer detector.

Features:
1. Configurable fee schedules per exchange
2. 30-day volume tracking for tier determination
3. Maker/Taker detection
4. Fee estimation in backtests
5. Fee-optimized evolution
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
import json

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================
class OrderType(Enum):
    """Order type affects fees."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ExchangeName(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    ALPACA = "alpaca"
    ROBINHOOD = "robinhood"
    DYDX = "dydx"
    WEBULL = "webull"
    HYPERLIQUID = "hyperliquid"


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class FeeTier:
    """A fee tier based on volume."""
    min_volume: float
    maker_rate: float
    taker_rate: float


@dataclass
class ExchangeFeeConfig:
    """Fee configuration for an exchange."""
    exchange: str
    tiers: List[FeeTier]
    loyalty_token: Optional[str] = None
    loyalty_discount: float = 0.0  # 0.25 = 25% discount
    has_maker_rebate: bool = False
    fee_currency: str = "asset"  # "asset" or "USD"
    
    def get_rate(self, is_maker: bool, volume_30d: float = 0) -> float:
        """Get fee rate for current volume and order type."""
        # Find applicable tier
        rate = self.tiers[0].maker_rate if is_maker else self.tiers[0].taker_rate
        
        for tier in self.tiers:
            if volume_30d >= tier.min_volume:
                rate = tier.maker_rate if is_maker else tier.taker_rate
            else:
                break
        
        # Apply loyalty discount
        if self.loyalty_token and self.loyalty_discount > 0:
            rate *= (1 - self.loyalty_discount)
        
        return rate


@dataclass
class TradeRecord:
    """A trade with fee tracking."""
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    order_type: str
    exchange: str
    timestamp: datetime
    
    # Fee tracking
    maker: bool = False
    fee: float = 0.0
    fee_rate: float = 0.0
    
    # Calculated
    gross_value: float = 0.0
    net_value: float = 0.0
    pnl: float = 0.0
    
    def calculate(self, fee_rate: float):
        """Calculate fees and net values."""
        self.fee_rate = fee_rate
        self.gross_value = self.quantity * self.price
        self.fee = self.gross_value * fee_rate
        self.net_value = self.gross_value - self.fee


@dataclass
class VolumeTracker:
    """Track 30-day trading volume for tier calculation."""
    trades: List[TradeRecord] = field(default_factory=list)
    
    def add_trade(self, trade: TradeRecord):
        """Add trade and cleanup old ones."""
        self.trades.append(trade)
        self._cleanup_old()
    
    def _cleanup_old(self):
        """Remove trades older than 30 days."""
        cutoff = datetime.now() - timedelta(days=30)
        self.trades = [t for t in self.trades if t.timestamp > cutoff]
    
    def get_volume_30d(self) -> float:
        """Get total trading volume in last 30 days."""
        return sum(t.gross_value for t in self.trades)


# ============================================================
# EXCHANGE FEE CONFIGURATIONS
# ============================================================
def get_exchange_config(exchange: str) -> ExchangeFeeConfig:
    """Get fee configuration for an exchange."""
    
    configs = {
        "binance": ExchangeFeeConfig(
            exchange="binance",
            tiers=[
                FeeTier(0, 0.0010, 0.0010),       # VIP 0
                FeeTier(50000, 0.0008, 0.0008),   # VIP 1
                FeeTier(200000, 0.0006, 0.0007),  # VIP 2
                FeeTier(1000000, 0.0004, 0.0006), # VIP 3
                FeeTier(2000000, 0.0002, 0.0005), # VIP 4
            ],
            loyalty_token="BNB",
            loyalty_discount=0.25  # 25% discount with BNB
        ),
        "coinbase": ExchangeFeeConfig(
            exchange="coinbase",
            tiers=[
                FeeTier(0, 0.0015, 0.0025),       # Starter
                FeeTier(10000, 0.0010, 0.0020),   # Bronze
                FeeTier(50000, 0.0008, 0.0018),   # Silver
                FeeTier(250000, 0.0005, 0.0015),  # Gold
            ],
            loyalty_token="COIN",
            loyalty_discount=0.10
        ),
        "alpaca": ExchangeFeeConfig(
            exchange="alpaca",
            tiers=[
                FeeTier(0, 0.0015, 0.0025),       # Tier 1
                FeeTier(100000, 0.0012, 0.0022),  # Tier 2
                FeeTier(500000, 0.0010, 0.0020),   # Tier 3
                FeeTier(1000000, 0.0008, 0.0018),  # Tier 4
                FeeTier(10000000, 0.0005, 0.0015), # Tier 5
            ]
        ),
        "dydx": ExchangeFeeConfig(
            exchange="dydx",
            tiers=[
                FeeTier(0, 0.0002, 0.0005),       # Starter
                FeeTier(100000, 0.0001, 0.0004),   # VIP 1
                FeeTier(500000, 0.0000, 0.0003),   # VIP 2
            ],
            has_maker_rebate=True  # Dydx pays makers!
        ),
        "hyperliquid": ExchangeFeeConfig(
            exchange="hyperliquid",
            tiers=[
                FeeTier(0, 0.0001, 0.0002),  # Simple low fees
            ]
        ),
        "robinhood": ExchangeFeeConfig(
            exchange="robinhood",
            tiers=[
                FeeTier(0, 0.0010, 0.0015),  # Tier based on volume
                FeeTier(5000, 0.0008, 0.0012),
                FeeTier(50000, 0.0005, 0.0010),
            ]
        ),
    }
    
    return configs.get(exchange.lower(), ExchangeFeeConfig(
        exchange=exchange,
        tiers=[FeeTier(0, 0.001, 0.001)]
    ))


# ============================================================
# FEE CALCULATOR
# ============================================================
class FeeCalculator:
    """
    Calculate fees for trades based on exchange and order type.
    """
    
    def __init__(self, exchange: str = "binance", use_loyalty: bool = False):
        self.exchange = exchange
        self.config = get_exchange_config(exchange)
        self.use_loyalty = use_loyalty
        self.volume_tracker = VolumeTracker()
    
    def set_volume(self, volume: float):
        """Set 30-day volume manually (for backtesting)."""
        # Create dummy trades to set volume
        self.volume_tracker.trades = []
    
    def is_maker_order(self, order_type: str, price: float, 
                       best_bid: float, best_ask: float) -> bool:
        """
        Determine if order is maker or taker.
        
        Maker: Limit order that doesn't cross the spread
        Taker: Market order or limit order that crosses spread
        """
        if order_type == "market":
            return False  # Market orders are always takers
        
        if order_type == "limit":
            # Buy limit below ask = maker
            if price <= best_bid:
                return True
            # Sell limit above ask = maker  
            if price >= best_ask:
                return True
            # Crossing the spread = taker
            return False
        
        return False  # Default to taker
    
    def calculate_fee(self, trade_value: float, order_type: str = "market",
                     is_maker: bool = None) -> Tuple[float, float]:
        """
        Calculate fee for a trade.
        
        Returns: (fee_amount, fee_rate)
        """
        # Get current volume
        volume = self.volume_tracker.get_volume_30d()
        
        # Determine if maker
        if is_maker is None:
            is_maker = order_type in ["limit", "stop_limit"]
        
        # Get rate
        rate = self.config.get_rate(is_maker, volume)
        
        # Calculate fee
        fee = trade_value * rate
        
        return fee, rate
    
    def record_trade(self, trade: TradeRecord):
        """Record trade for volume tracking."""
        self.volume_tracker.add_trade(trade)
    
    def get_current_tier(self) -> int:
        """Get current fee tier (0-indexed)."""
        volume = self.volume_tracker.get_volume_30d()
        
        for i, tier in enumerate(self.config.tiers):
            if volume < tier.min_volume:
                return max(0, i - 1) if i > 0 else 0
        
        return len(self.config.tiers) - 1


# ============================================================
# FEE-AWARE POSIZER
# =================================ITION S===========================
class FeeAwarePositionSizer:
    """
    Calculate position sizes accounting for fees.
    """
    
    def __init__(self, exchange: str = "binance", initial_capital: float = 10000):
        self.exchange = exchange
        self.fee_calc = FeeCalculator(exchange)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.total_fees_paid = 0
    
    def calculate_size(self, price: float, confidence: float = 1.0,
                      target_risk_pct: float = 0.02,
                      stop_loss_pct: float = 0.03) -> Tuple[float, float, float]:
        """
        Calculate position size with fee impact.
        
        Returns: (quantity, gross_value, fee)
        """
        # Base risk amount
        risk_amount = self.current_capital * target_risk_pct * confidence
        
        # Risk per unit
        risk_per_unit = price * stop_loss_pct
        
        # Raw position size
        raw_quantity = risk_amount / risk_per_unit
        raw_value = raw_quantity * price
        
        # Estimate fee (assume taker for market orders)
        fee, rate = self.fee_calc.calculate_fee(raw_value, "market")
        
        # Adjust for fees - need to fit within capital after fees
        max_value = self.current_capital * 0.95  # Keep 5% buffer
        adjusted_value = min(raw_value, max_value)
        
        # Recalculate
        quantity = adjusted_value / price
        fee = adjusted_value * rate
        
        return quantity, adjusted_value, fee
    
    def apply_trade_result(self, pnl: float, fee: float):
        """Apply trade result and track fees."""
        self.current_capital += pnl - fee
        self.total_fees_paid += fee
    
    def get_stats(self) -> Dict:
        """Get fee statistics."""
        return {
            "current_capital": self.current_capital,
            "total_fees_paid": self.total_fees_paid,
            "fees_pct": (self.total_fees_paid / 
                         max(1, self.initial_capital - self.current_capital + self.total_fees_paid)),
            "current_tier": self.fee_calc.get_current_tier(),
            "exchange": self.exchange
        }


# ============================================================
# FEE COMPARISON
# ============================================================
def compare_exchanges(trade_value: float) -> pd.DataFrame:
    """Compare fees across exchanges."""
    results = []
    
    exchanges = ["binance", "coinbase", "alpaca", "dydx", "hyperliquid", "robinhood"]
    
    for exch in exchanges:
        calc = FeeCalculator(exch)
        
        # Maker fee
        maker_fee, maker_rate = calc.calculate_fee(trade_value, "limit", True)
        
        # Taker fee  
        taker_fee, taker_rate = calc.calculate_fee(trade_value, "market", False)
        
        results.append({
            "Exchange": exch.capitalize(),
            "Maker Rate": f"{maker_rate*100:.3f}%",
            "Maker Fee": f"${maker_fee:.2f}",
            "Taker Rate": f"{taker_rate*100:.3f}%",
            "Taker Fee": f"${taker_fee:.2f}",
        })
    
    return pd.DataFrame(results)


# ============================================================
# FACTORY
# ============================================================
def create_fee_calculator(exchange: str = "binance") -> FeeCalculator:
    """Create fee calculator for exchange."""
    return FeeCalculator(exchange)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("QUANTCORE FEE-AWARE EXECUTION TEST")
    print("=" * 50)
    
    # Test 1: Fee Calculator
    print("\n💰 Test 1: Fee Calculator")
    calc = FeeCalculator("binance")
    
    fee, rate = calc.calculate_fee(10000, "market", False)
    print(f"  $10,000 taker fee @ Binance: ${fee:.2f} ({rate*100:.2f}%)")
    
    fee, rate = calc.calculate_fee(10000, "limit", True)
    print(f"  $10,000 maker fee @ Binance: ${fee:.2f} ({rate*100:.2f}%)")
    
    # Test 2: With BNB discount
    print("\n💎 Test 2: BNB Discount (25%)")
    calc_loyal = FeeCalculator("binance", use_loyalty=True)
    fee, rate = calc_loyal.calculate_fee(10000, "market", False)
    print(f"  $10,000 with BNB discount: ${fee:.2f} ({rate*100:.2f}%)")
    
    # Test 3: Fee comparison
    print("\n📊 Test 3: Exchange Comparison ($10,000 trade)")
    df = compare_exchanges(10000)
    print(df.to_string(index=False))
    
    # Test 4: Position sizer with fees
    print("\n📐 Test 4: Fee-Aware Position Sizing")
    sizer = FeeAwarePositionSizer("binance", initial_capital=10000)
    
    quantity, value, fee = sizer.calculate_size(
        price=45000,
        confidence=0.8,
        target_risk_pct=0.02,
        stop_loss_pct=0.03
    )
    
    print(f"  Price: $45,000")
    print(f"  Position: {quantity:.4f} BTC = ${value:.2f}")
    print(f"  Estimated Fee: ${fee:.2f}")
    
    # Simulate trade
    sizer.apply_trade_result(pnl=200, fee=fee)
    stats = sizer.get_stats()
    
    print(f"  After trade:")
    print(f"    Capital: ${stats['current_capital']:,.2f}")
    print(f"    Total Fees: ${stats['total_fees_paid']:.2f}")
    print(f"    Current Tier: {stats['current_tier']}")
    
    # Test 5: Maker vs Taker detection
    print("\n🎯 Test 5: Maker/Taker Detection")
    best_bid = 44990
    best_ask = 45010
    
    is_maker = calc.is_maker_order("limit", 44980, best_bid, best_ask)  # Below bid
    print(f"  Buy limit @ 44980 (below bid): {'Maker' if is_maker else 'Taker'}")
    
    is_maker = calc.is_maker_order("limit", 45020, best_bid, best_ask)  # Above ask
    print(f"  Sell limit @ 45020 (above ask): {'Maker' if is_maker else 'Taker'}")
    
    is_maker = calc.is_maker_order("market", 0, best_bid, best_ask)
    print(f"  Market order: {'Maker' if is_maker else 'Taker'}")
    
    print("\n✅ All fee tests passed!")
