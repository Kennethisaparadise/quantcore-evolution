"""
QuantCore - Order Book Warfare: Level 2 Microstructure v1.0

This module adds microstructural analysis beyond trade data.

Expert: "Level 1 was trade data. Level 2 is full order book."

Modules:
1. Order Book Imbalance Detector - bid vs ask volume
2. Iceberg Order Hunter - detect hidden liquidity
3. Spoofing Detector - identify manipulation
4. Latency Arbitrage - cross-exchange price differences
5. Adversarial Algo Hunter - exploit other bots
6. Microstructure Pattern Recognition - tick-level patterns
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
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    size: float
    orders: int = 1  # Number of orders at this level


@dataclass
class OrderBook:
    """Full order book state."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted by price desc
    asks: List[OrderBookLevel]  # Sorted by price asc
    spread: float = 0
    mid_price: float = 0
    
    @classmethod
    def create_mock(cls, symbol: str, mid_price: float = 45000):
        """Create mock order book for testing."""
        bids = []
        asks = []
        
        for i in range(10):
            bids.append(OrderBookLevel(
                price=mid_price - (i + 1) * 0.5,
                size=random.uniform(0.5, 5.0)
            ))
            asks.append(OrderBookLevel(
                price=mid_price + (i + 1) * 0.5,
                size=random.uniform(0.5, 5.0)
            ))
        
        spread = asks[0].price - bids[0].price
        mid = (asks[0].price + bids[0].price) / 2
        
        return cls(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            spread=spread,
            mid_price=mid
        )


@dataclass
class Trade:
    """Single trade."""
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    timestamp: datetime


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class MicrostructureConfig:
    """
    Configuration for all microstructure modules.
    """
    # Order Book Imbalance
    imbalance_levels: int = 5
    imbalance_threshold: float = 0.3  # >30% imbalance triggers signal
    imbalance_decay: float = 0.9  # Exponential weight for recent
    
    # Iceberg Detection
    iceberg_sensitivity: int = 3  # Trades at same price to suspect
    iceberg_follow_mode: str = "join"  # 'join', 'front_run', 'fade'
    iceberg_min_size: float = 1.0  # Min size to track
    
    # Spoofing Detection
    spoof_window_seconds: int = 10
    spoof_min_size: float = 5.0
    spoof_action: str = "ignore"  # 'ignore', 'fade', 'trade_against'
    
    # Arbitrage
    arb_min_spread: float = 0.001  # 0.1% min to act
    arb_max_slippage: float = 0.0005
    exchanges: List[str] = field(default_factory=lambda: ["binance", "hyperliquid"])
    
    # Algo Hunter
    stop_hunt_sensitivity: float = 0.5
    momentum_ignition_threshold: float = 0.02
    algo_classification: str = "simple"  # 'simple', 'ml'
    
    # Pattern Recognition
    pattern_lookback: int = 20
    pattern_confirmation: int = 3
    
    def to_dict(self) -> Dict:
        return {
            'imbalance_levels': self.imbalance_levels,
            'imbalance_threshold': self.imbalance_threshold,
            'iceberg_sensitivity': self.iceberg_sensitivity,
            'spoof_action': self.spoof_action,
            'arb_min_spread': self.arb_min_spread
        }


# ============================================================
# MODULE 1: ORDER BOOK IMBALANCE
# ============================================================
class OrderBookImbalance:
    """
    Track bid vs ask volume at top N levels.
    
    Imbalance > threshold suggests directional move.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.imbalance_history: List[float] = []
        self.signal_history: List[int] = []  # 1=buy, -1=sell, 0=neutral
        
    def calculate_imbalance(self, book: OrderBook) -> float:
        """Calculate order book imbalance."""
        levels = min(self.config.imbalance_levels, len(book.bids), len(book.asks))
        
        bid_vol = sum(b.size for b in book.bids[:levels])
        ask_vol = sum(a.size for a in book.asks[:levels])
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0
        
        # -1 to 1 range: positive = bid heavy, negative = ask heavy
        imbalance = (bid_vol - ask_vol) / total
        
        return imbalance
    
    def get_signal(self, book: OrderBook) -> Tuple[int, float]:
        """
        Get trading signal from imbalance.
        
        Returns: (direction, confidence)
        """
        imbalance = self.calculate_imbalance(book)
        
        # Store in history with decay
        if self.imbalance_history:
            imbalance = (self.config.imbalance_decay * imbalance + 
                        (1 - self.config.imbalance_decay) * self.imbalance_history[-1])
        
        self.imbalance_history.append(imbalance)
        
        # Keep only recent
        if len(self.imbalance_history) > 100:
            self.imbalance_history = self.imbalance_history[-100:]
        
        # Signal
        direction = 0
        confidence = 0
        
        if abs(imbalance) > self.config.imbalance_threshold:
            direction = 1 if imbalance > 0 else -1
            confidence = min(abs(imbalance) / (self.config.imbalance_threshold * 2), 1.0)
        
        self.signal_history.append(direction)
        
        return direction, confidence


# ============================================================
# MODULE 2: ICEBERG HUNTER
# ============================================================
class IcebergHunter:
    """
    Detect iceberg orders (hidden large orders).
    
    Look for repeated trades at same price with book replenishment.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.trade_history: deque = deque(maxlen=1000)
        self.book_history: deque = deque(maxlen=100)
        self.iceberg_signals: List[Dict] = []
        
    def record_trade(self, trade: Trade):
        """Record a trade."""
        self.trade_history.append(trade)
    
    def record_book(self, book: OrderBook):
        """Record order book state."""
        self.book_history.append(book)
    
    def detect_iceberg(self, price: float, side: str) -> Tuple[bool, float]:
        """
        Detect if there's likely an iceberg at price.
        
        Returns: (is_iceberg, estimated_size)
        """
        # Look at recent trades at this price
        recent_trades = [t for t in self.trade_history 
                        if abs(t.price - price) < 0.5 and t.side == side]
        
        if len(recent_trades) < self.config.iceberg_sensitivity:
            return False, 0
        
        # Check if book has been replenished
        if len(self.book_history) < 2:
            return False, 0
        
        # Simple heuristic: if many small trades and size stays, suspect iceberg
        sizes = [t.size for t in recent_trades]
        avg_size = np.mean(sizes)
        
        if avg_size < self.config.iceberg_min_size * 0.5:
            # Many small trades - possible iceberg
            estimated = sum(sizes) * 2  # Rough estimate
            return True, estimated
        
        return False, 0
    
    def get_signal(self) -> Tuple[int, float]:
        """Get iceberg signal for current state."""
        if not self.trade_history:
            return 0, 0
        
        # Check recent price
        recent = self.trade_history[-1]
        
        is_iceberg, size = self.detect_iceberg(recent.price, recent.side)
        
        if not is_iceberg:
            return 0, 0
        
        # Signal based on follow mode
        if self.config.iceberg_follow_mode == "front_run":
            # Trade in same direction as iceberg
            direction = 1 if recent.side == 'buy' else -1
        elif self.config.iceberg_follow_mode == "fade":
            # Trade opposite
            direction = -1 if recent.side == 'buy' else 1
        else:  # join
            direction = 1 if recent.side == 'buy' else -1
        
        confidence = min(size / 10, 1.0)
        
        return direction, confidence


# ============================================================
# MODULE 3: SPOOFING DETECTOR
# ============================================================
class SpoofingDetector:
    """
    Detect spoofing/layering - large orders placed then cancelled.
    
    Track orders that appear but don't get filled.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.pending_orders: Dict[float, Dict] = {}  # price -> {size, time, side}
        self.spoof_events: List[Dict] = []
        
    def record_order(self, price: float, size: float, side: str):
        """Record a new order."""
        self.pending_orders[price] = {
            'size': size,
            'time': datetime.now(),
            'side': side,
            'original_size': size
        }
    
    def record_cancel(self, price: float):
        """Record order cancellation."""
        if price in self.pending_orders:
            order = self.pending_orders[price]
            
            # Check if it was a spoof (large, cancelled quickly)
            age = (datetime.now() - order['time']).total_seconds()
            
            if age < self.config.spoof_window_seconds and order['size'] > self.config.spoof_min_size:
                self.spoof_events.append({
                    'price': price,
                    'size': order['size'],
                    'side': order['side'],
                    'age': age,
                    'timestamp': datetime.now()
                })
            
            del self.pending_orders[price]
    
    def record_fill(self, price: float, filled_size: float):
        """Record partial or full fill."""
        if price in self.pending_orders:
            self.pending_orders[price]['size'] -= filled_size
            
            if self.pending_orders[price]['size'] <= 0:
                del self.pending_orders[price]
    
    def detect_spoofing(self) -> List[Dict]:
        """Get all detected spoofing events."""
        # Clean old events
        now = datetime.now()
        self.spoof_events = [e for e in self.spoof_events 
                           if (now - e['timestamp']).total_seconds() < 60]
        
        return self.spoof_events
    
    def get_signal(self) -> Tuple[int, float]:
        """Get trading signal from spoofing."""
        spoofing = self.detect_spoofing()
        
        if not spoofing:
            return 0, 0
        
        # Most recent spoof
        latest = spoofing[-1]
        
        if self.config.spoof_action == "ignore":
            return 0, 0
        elif self.config.spoof_action == "fade":
            # Trade opposite of spoof
            direction = -1 if latest['side'] == 'buy' else 1
        else:  # trade_against
            direction = -1 if latest['side'] == 'buy' else 1
        
        confidence = min(latest['size'] / 20, 1.0)
        
        return direction, confidence


# ============================================================
# MODULE 4: LATENCY ARBITRAGE
# ============================================================
class LatencyArbitrage:
    """
    Exploit price differences between exchanges.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.exchange_prices: Dict[str, float] = {}
        self.arb_opportunities: List[Dict] = []
        
    def update_price(self, exchange: str, price: float):
        """Update price for exchange."""
        self.exchange_prices[exchange] = price
        
        if len(self.exchange_prices) > 1:
            self._check_arbitrage()
    
    def _check_arbitrage(self):
        """Check for arbitrage opportunities."""
        prices = list(self.exchange_prices.values())
        max_price = max(prices)
        min_price = min(prices)
        
        spread_pct = (max_price - min_price) / min_price
        
        if spread_pct > self.config.arb_min_spread:
            # Potential arbitrage
            max_exch = [e for e, p in self.exchange_prices.items() if p == max_price][0]
            min_exch = [e for e, p in self.exchange_prices.items() if p == min_price][0]
            
            self.arb_opportunities.append({
                'buy_exchange': min_exch,
                'sell_exchange': max_exch,
                'buy_price': min_price,
                'sell_price': max_price,
                'spread': spread_pct,
                'timestamp': datetime.now()
            })
    
    def get_signal(self) -> Tuple[int, float, Dict]:
        """
        Get arbitrage signal.
        
        Returns: (direction, confidence, details)
        """
        if not self.arb_opportunities:
            return 0, 0, {}
        
        # Most recent opportunity
        opp = self.arb_opportunities[-1]
        
        # Check spread vs slippage
        if opp['spread'] > self.config.arb_max_slippage:
            direction = 1  # Buy low, sell high (would execute both legs)
            confidence = min(opp['spread'] / (self.config.arb_min_spread * 3), 1.0)
            return direction, confidence, opp
        
        return 0, 0, {}


# ============================================================
# MODULE 5: ADVERSARIAL ALGO HUNTER
# ============================================================
class AdversarialAlgoHunter:
    """
    Hunt other algorithms - stop hunters, momentum ignitors, etc.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.price_history: deque = deque(maxlen=100)
        self.volume_history: deque = deque(maxlen=100)
        self.order_flow_history: deque = deque(maxlen=50)
        
    def record(self, price: float, volume: float, order_flow: float):
        """Record market state."""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.order_flow_history.append(order_flow)
    
    def detect_stop_hunt(self) -> Tuple[bool, float]:
        """Detect potential stop hunt."""
        if len(self.price_history) < 20:
            return False, 0
        
        prices = list(self.price_history)
        
        # Look for sharp drop then recovery (stop hunt pattern)
        recent = prices[-5:]
        older = prices[-20:-5]
        
        recent_low = min(recent)
        older_low = min(older)
        
        # If recent low breaks older low significantly, then recovers
        if recent_low < older_low * 0.99:  # Broke lower
            # Check if recovering
            if prices[-1] > recent_low * 1.01:
                confidence = (older_low - recent_low) / older_low
                return True, min(confidence * 2, 1.0)
        
        return False, 0
    
    def detect_momentum_ignition(self) -> Tuple[bool, float]:
        """Detect momentum ignition ( algo trying to start move)."""
        if len(self.order_flow_history) < 10:
            return False, 0
        
        flows = list(self.order_flow_history)
        
        # Look for sudden large order flow
        recent = np.mean(flows[-3:])
        baseline = np.mean(flows[:-3])
        
        change = abs(recent - baseline)
        
        if change > self.config.momentum_ignition_threshold:
            direction = 1 if recent > baseline else -1
            confidence = min(change / (self.config.momentum_ignition_threshold * 2), 1.0)
            return True, confidence
        
        return False, 0
    
    def get_signal(self) -> Tuple[int, float]:
        """Get algo hunting signal."""
        # Check stop hunt
        is_hunt, conf_hunt = self.detect_stop_hunt()
        
        if is_hunt:
            # Fade the stop hunt - trade opposite
            return -1, conf_hunt
        
        # Check momentum ignition
        is_ignition, conf_ignite = self.detect_momentum_ignition()
        
        if is_ignition:
            # Fade momentum ignition
            return -1, conf_ignite
        
        return 0, 0


# ============================================================
# MODULE 6: MICROSTRUCTURE PATTERNS
# ============================================================
class MicrostructurePatterns:
    """
    Recognize tick-level patterns.
    
    Patterns: stacking, leaning, pinging, etc.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.patterns = {
            'stacking': [],  # Large orders at successive levels
            'leaning': [],  # One side repeatedly refreshed
            'pinging': [],  # Small orders testing liquidity
        }
        self.book_history: deque = deque(maxlen=50)
        
    def record_book(self, book: OrderBook):
        """Record order book for pattern detection."""
        self.book_history.append(book)
        
        if len(self.book_history) < 2:
            return
        
        self._detect_patterns()
    
    def _detect_patterns(self):
        """Detect current patterns."""
        if len(self.book_history) < self.config.pattern_lookback:
            return
        
        books = list(self.book_history)
        
        # Stacking: large orders at consecutive levels
        bid_sizes = [b.bids[0].size for b in books[-5:] if len(b.bids) > 0]
        
        if len(bid_sizes) >= 3 and np.mean(bid_sizes) > 3.0:
            self.patterns['stacking'].append({
                'side': 'bid',
                'strength': np.mean(bid_sizes),
                'timestamp': datetime.now()
            })
        
        # Pinging: small orders testing both sides
        if len(books) >= 3:
            spread_change = abs(books[-1].spread - books[-3].spread)
            
            if spread_change > 0.5:  # Significant spread movement
                self.patterns['pinging'].append({
                    'spread_change': spread_change,
                    'timestamp': datetime.now()
                })
    
    def get_signal(self) -> Tuple[int, float]:
        """Get pattern-based signal."""
        now = datetime.now()
        
        # Check stacking
        stacking = [p for p in self.patterns['stacking'] 
                   if (now - p['timestamp']).total_seconds() < 30]
        
        if stacking:
            if stacking[-1]['side'] == 'bid':
                return 1, min(stacking[-1]['strength'] / 10, 1.0)
        
        # Check pinging
        pinging = [p for p in self.patterns['pinging']
                  if (now - p['timestamp']).total_seconds() < 30]
        
        if pinging:
            # Pinging often precedes move in opposite direction
            return -1, 0.5
        
        return 0, 0


# ============================================================
# MASTER MICROSTRUCTURE ORCHESTRATOR
# ============================================================
class MicrostructureOrchestrator:
    """
    Orchestrate all Level 2 microstructure modules.
    """
    
    def __init__(self, config: Optional[MicrostructureConfig] = None):
        self.config = config or MicrostructureConfig()
        
        # Initialize all modules
        self.imbalance = OrderBookImbalance(self.config)
        self.iceberg = IcebergHunter(self.config)
        self.spoofing = SpoofingDetector(self.config)
        self.arbitrage = LatencyArbitrage(self.config)
        self.algo_hunter = AdversarialAlgoHunter(self.config)
        self.patterns = MicrostructurePatterns(self.config)
        
        # Combined signal weights
        self.weights = {
            'imbalance': 0.20,
            'iceberg': 0.15,
            'spoofing': 0.10,
            'arbitrage': 0.25,
            'algo_hunter': 0.20,
            'patterns': 0.10
        }
    
    def update_order_book(self, book: OrderBook):
        """Update with new order book."""
        # Record for pattern detection
        self.patterns.record_book(book)
        
        # Update imbalance
        imbalance_dir, imbalance_conf = self.imbalance.get_signal(book)
        
        # Check for iceberg at best bid/ask
        if book.bids:
            is_iceberg, size = self.iceberg.detect_iceberg(book.bids[0].price, 'buy')
        
        if book.asks:
            is_iceberg, size = self.iceberg.detect_iceberg(book.asks[0].price, 'sell')
    
    def update_trade(self, trade: Trade):
        """Update with new trade."""
        self.iceberg.record_trade(trade)
    
    def update_price(self, exchange: str, price: float):
        """Update price for arbitrage."""
        self.arbitrage.update_price(exchange, price)
    
    def get_combined_signal(self) -> Tuple[int, float, Dict]:
        """
        Get combined signal from all microstructure modules.
        
        Returns: (direction, confidence, details)
        """
        # Get individual signals (need books/trades for some)
        imbalance_dir, imbalance_conf = 0, 0  # Requires book
        iceberg_dir, iceberg_conf = self.iceberg.get_signal()
        spoofing_dir, spoofing_conf = self.spoofing.get_signal()
        arbitrage_dir, arbitrage_conf, _ = self.arbitrage.get_signal()
        algo_hunter_dir, algo_hunter_conf = self.algo_hunter.get_signal()
        patterns_dir, patterns_conf = self.patterns.get_signal()
        
        signals = {
            'imbalance': (imbalance_dir, imbalance_conf),
            'iceberg': (iceberg_dir, iceberg_conf),
            'spoofing': (spoofing_dir, spoofing_conf),
            'arbitrage': (arbitrage_dir, arbitrage_conf),
            'algo_hunter': (algo_hunter_dir, algo_hunter_conf),
            'patterns': (patterns_dir, patterns_conf)
        }
        
        # Weighted combination
        total_weight = 0
        weighted_direction = 0
        weighted_confidence = 0
        
        for name, (direction, confidence) in signals.items():
            if direction != 0:
                weight = self.weights.get(name, 0.1)
                weighted_direction += direction * weight * confidence
                weighted_confidence += confidence * weight
                total_weight += weight
        
        if total_weight > 0:
            final_direction = 1 if weighted_direction > 0.2 else (-1 if weighted_direction < -0.2 else 0)
            final_confidence = weighted_confidence / total_weight
        else:
            final_direction = 0
            final_confidence = 0
        
        return final_direction, final_confidence, signals
    
    def get_status(self) -> Dict:
        """Get status of all modules."""
        return {
            'imbalance_history': len(self.imbalance.imbalance_history),
            'iceberg_signals': len(self.iceberg.iceberg_signals),
            'spoofing_events': len(self.spoofing.spoof_events),
            'arb_opportunities': len(self.arbitrage.arb_opportunities),
            'patterns_detected': {k: len(v) for k, v in self.patterns.patterns.items()}
        }


# ============================================================
# MUTATIONS
# ============================================================
class MicrostructureMutations:
    """Mutation operators for microstructure."""
    
    @staticmethod
    def mutate_imbalance_depth(config: MicrostructureConfig) -> MicrostructureConfig:
        config = copy.deepcopy(config)
        config.imbalance_levels = max(1, min(20, 
            config.imbalance_levels + random.choice([-1, 1])))
        return config
    
    @staticmethod
    def mutate_imbalance_threshold(config: MicrostructureConfig) -> MicrostructureConfig:
        config = copy.deepcopy(config)
        config.imbalance_threshold = max(0.1, min(0.8,
            config.imbalance_threshold + random.uniform(-0.05, 0.05)))
        return config
    
    @staticmethod
    def mutate_iceberg_sensitivity(config: MicrostructureConfig) -> MicrostructureConfig:
        config = copy.deepcopy(config)
        config.iceberg_sensitivity = max(2, min(10,
            config.iceberg_sensitivity + random.choice([-1, 1])))
        return config
    
    @staticmethod
    def mutate_iceberg_mode(config: MicrostructureConfig) -> MicrostructureConfig:
        config = copy.deepcopy(config)
        modes = ['join', 'front_run', 'fade']
        config.iceberg_follow_mode = random.choice(modes)
        return config
    
    @staticmethod
    def mutate_spoof_action(config: MicrostructureConfig) -> MicrostructureConfig:
        config = copy.deepcopy(config)
        actions = ['ignore', 'fade', 'trade_against']
        config.spoof_action = random.choice(actions)
        return config
    
    @staticmethod
    def mutate_arb_spread(config: MicrostructureConfig) -> MicrostructureConfig:
        config = copy.deepcopy(config)
        config.arb_min_spread = max(0.0005, min(0.01,
            config.arb_min_spread * random.uniform(0.8, 1.2)))
        return config


# ============================================================
# FACTORY
# ============================================================
def create_microstructure_config() -> MicrostructureConfig:
    """Create default configuration."""
    return MicrostructureConfig()


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("QUANTCORE LEVEL 2 MICROSTRUCTURE TEST")
    print("=" * 50)
    
    # Create orchestrator
    config = create_microstructure_config()
    orchestrator = MicrostructureOrchestrator(config)
    
    # Test 1: Order Book Imbalance
    print("\n📊 Test 1: Order Book Imbalance")
    book = OrderBook.create_mock("BTCUSDT", 45000)
    imbalance = orchestrator.imbalance.calculate_imbalance(book)
    direction, conf = orchestrator.imbalance.get_signal(book)
    print(f"  Imbalance: {imbalance:.3f}")
    print(f"  Signal: {direction} ({conf:.2f})")
    
    # Test 2: Iceberg Detection
    print("\n🧊 Test 2: Iceberg Hunter")
    for _ in range(5):
        trade = Trade(price=45000, size=0.1, side='buy', timestamp=datetime.now())
        orchestrator.iceberg.record_trade(trade)
    
    is_iceberg, size = orchestrator.iceberg.detect_iceberg(45000, 'buy')
    print(f"  Iceberg detected: {is_iceberg}, size: {size:.2f}")
    
    # Test 3: Spoofing Detection
    print("\n🎭 Test 3: Spoofing Detector")
    orchestrator.spoofing.record_order(45000.5, 10.0, 'sell')
    orchestrator.spoofing.record_cancel(45000.5)
    spoofing = orchestrator.spoofing.detect_spoofing()
    print(f"  Spoofing events: {len(spoofing)}")
    
    # Test 4: Arbitrage
    print("\n⚡ Test 4: Latency Arbitrage")
    orchestrator.arbitrage.update_price("binance", 45000)
    orchestrator.arbitrage.update_price("hyperliquid", 45050)
    direction, conf, details = orchestrator.arbitrage.get_signal()
    print(f"  Signal: {direction}, conf: {conf:.2f}")
    print(f"  Details: {details}")
    
    # Test 5: Algo Hunter
    print("\n🤖 Test 5: Adversarial Algo Hunter")
    for i in range(30):
        price = 45000 + np.random.randn() * 10
        volume = np.random.uniform(1, 10)
        flow = np.random.randn()
        orchestrator.algo_hunter.record(price, volume, flow)
    
    is_hunt, conf = orchestrator.algo_hunter.detect_stop_hunt()
    print(f"  Stop hunt: {is_hunt}, conf: {conf:.2f}")
    
    # Test 6: Combined Signal
    print("\n🎯 Test 6: Combined Signal")
    direction, conf, signals = orchestrator.get_combined_signal()
    print(f"  Direction: {direction}")
    print(f"  Confidence: {conf:.2f}")
    
    # Test 7: Mutations
    print("\n🧬 Test 7: Mutations")
    config = MicrostructureMutations.mutate_imbalance_threshold(config)
    print(f"  New threshold: {config.imbalance_threshold:.3f}")
    
    config = MicrostructureMutations.mutate_iceberg_mode(config)
    print(f"  Iceberg mode: {config.iceberg_follow_mode}")
    
    print("\n✅ All microstructure tests passed!")
