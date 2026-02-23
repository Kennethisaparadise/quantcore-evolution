"""
QuantCore - Order Flow Shadow Mutation v1.0

This module adds microstructure/order flow intelligence to the evolution engine.

Expert: "Order flow is the closest you can get to reading the market's mind 
without a neural implant. It's the difference between trading with the 
whales and being their lunch."

Features:
1. Simulated Order Flow from 1-min bars (tick rule)
2. Order Flow Metrics (cumulative delta, divergence, imbalance)
3. Order Flow Mutation Operators
4. Flow-Regime Hybridization
5. Large Trade Detection & Stop Hunt识别
"""

import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# ============================================================
# ORDER FLOW TYPES
# ============================================================
class FlowType(Enum):
    """Types of order flow analysis."""
    CUMULATIVE_DELTA = "cumulative_delta"
    DELTA_DIVERGENCE = "delta_divergence"
    BID_ASK_IMBALANCE = "bid_ask_imbalance"
    LARGE_TRADE = "large_trade"
    STOP_HUNT = "stop_hunt"
    VOLUME_PROFILE = "volume_profile"


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class FlowConfig:
    """
    Evolvable order flow configuration.
    
    The GA evolves:
    - Which flow metrics to use
    - Thresholds for each metric
    - Time windows for calculation
    """
    # Enabled flow types
    use_cumulative_delta: bool = True
    use_delta_divergence: bool = True
    use_bid_ask_imbalance: bool = False
    use_large_trade: bool = True
    use_stop_hunt: bool = False
    
    # Delta settings
    delta_window: int = 20           # Bars to calculate delta over
    delta_smoothing: int = 5         # Smoothing window
    delta_threshold: float = 0.1     # Min delta ratio for signal
    
    # Divergence settings
    divergence_lookback: int = 50    # Bars to check for divergence
    divergence_threshold: float = 0.3 # Price/delta divergence threshold
    
    # Large trade settings
    large_trade_std: float = 2.0    # Std devs for "large" trade
    large_trade_window: int = 20    # Window for std calculation
    large_trade_confirm: bool = True  # Require large trade for entry
    
    # Imbalance settings
    imbalance_window: int = 10       # Window for imbalance calc
    imbalance_threshold: float = 0.6 # Required imbalance ratio
    
    # Stop hunt settings
    stop_hunt_lookback: int = 10    # Bars to check for stop hunt
    stop_hunt_threshold: float = 0.05 # Price movement threshold
    
    # Regime-specific flow (evolvable)
    flow_per_regime: Dict[str, dict] = field(default_factory=lambda: {
        'bull': {'delta_threshold': 0.1, 'require_large_trade': True},
        'bear': {'delta_threshold': 0.1, 'require_large_trade': True},
        'sideways': {'delta_threshold': 0.05, 'require_large_trade': False},
        'high_vol': {'delta_threshold': 0.15, 'require_large_trade': True},
        'low_vol': {'delta_threshold': 0.05, 'require_large_trade': False}
    })
    
    def to_dict(self) -> Dict:
        return {
            'use_cumulative_delta': self.use_cumulative_delta,
            'use_delta_divergence': self.use_delta_divergence,
            'use_bid_ask_imbalance': self.use_bid_ask_imbalance,
            'use_large_trade': self.use_large_trade,
            'use_stop_hunt': self.use_stop_hunt,
            'delta_window': self.delta_window,
            'delta_smoothing': self.delta_smoothing,
            'delta_threshold': self.delta_threshold,
            'divergence_lookback': self.divergence_lookback,
            'divergence_threshold': self.divergence_threshold,
            'large_trade_std': self.large_trade_std,
            'large_trade_window': self.large_trade_window,
            'large_trade_confirm': self.large_trade_confirm,
            'imbalance_window': self.imbalance_window,
            'imbalance_threshold': self.imbalance_threshold,
            'stop_hunt_lookback': self.stop_hunt_lookback,
            'stop_hunt_threshold': self.stop_hunt_threshold,
            'flow_per_regime': self.flow_per_regime
        }


@dataclass
class FlowSignal:
    """Signal from order flow analysis."""
    direction: int           # -1 (sell), 0 (neutral), 1 (buy)
    confidence: float       # 0-1
    flow_type: FlowType
    metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================
# ORDER FLOW CALCULATOR
# ============================================================
class OrderFlowCalculator:
    """
    Calculate order flow metrics from OHLCV data.
    
    Uses tick rule: if close > open, volume is buying; else selling.
    """
    
    def __init__(self, config: FlowConfig):
        self.config = config
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all order flow metrics.
        
        Returns dict with:
        - cumulative_delta
        - delta_divergence  
        - bid_ask_imbalance
        - large_trade_flag
        - stop_hunt_flag
        """
        close = data['close'].values
        open_prices = data['open'].values if 'open' in data.columns else close
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
        
        # Calculate buy/sell volume using tick rule
        buy_volume = np.zeros(len(close))
        sell_volume = np.zeros(len(close))
        
        for i in range(len(close)):
            if close[i] > open_prices[i]:
                buy_volume[i] = volume[i]
            elif close[i] < open_prices[i]:
                sell_volume[i] = volume[i]
            else:
                # Close == open: split 50/50
                buy_volume[i] = volume[i] * 0.5
                sell_volume[i] = volume[i] * 0.5
        
        # Net delta
        net_delta = buy_volume - sell_volume
        
        # Cumulative delta
        cumulative_delta = np.cumsum(net_delta)
        
        # Delta metrics
        delta_window = min(self.config.delta_window, len(close) - 1)
        recent_delta = net_delta[-delta_window:]
        avg_delta = np.mean(recent_delta)
        delta_ratio = avg_delta / (np.mean(volume[-delta_window:]) + 1)
        
        # Delta divergence
        price_lookback = min(self.config.divergence_lookback, len(close) - 1)
        recent_prices = close[-price_lookback:]
        recent_deltas = net_delta[-price_lookback:]
        
        # Price: new high?
        price_new_high = recent_prices[-1] > np.max(recent_prices[:-1]) if len(recent_prices) > 1 else False
        # Delta: also new high?
        delta_new_high = recent_deltas[-1] > np.max(recent_deltas[:-1]) if len(recent_deltas) > 1 else False
        
        divergence = 0.0
        if price_new_high and not delta_new_high:
            divergence = -1  # Bearish divergence
        elif not price_new_high and delta_new_high:
            divergence = 1   # Bullish divergence
        
        # Bid-Ask imbalance
        imbalance_window = min(self.config.imbalance_window, len(close) - 1)
        recent_buy = np.sum(buy_volume[-imbalance_window:])
        recent_sell = np.sum(sell_volume[-imbalance_window:])
        total = recent_buy + recent_sell
        imbalance = (recent_buy - recent_sell) / (total + 1)
        
        # Large trade detection
        vol_std = np.std(volume[-self.config.large_trade_window:])
        avg_vol = np.mean(volume[-self.config.large_trade_window:])
        current_vol = volume[-1]
        large_trade = current_vol > (avg_vol + self.config.large_trade_std * vol_std)
        
        # Stop hunt detection
        # Rapid price move on low volume = likely stop hunt
        stop_window = min(self.config.stop_hunt_lookback, len(close) - 1)
        price_moves = np.abs(np.diff(close[-stop_window-1:]))
        avg_move = np.mean(price_moves)
        last_move = np.abs(close[-1] - close[-stop_window-1])
        avg_vol_recent = np.mean(volume[-stop_window:])
        
        # Stop hunt = big move on below-average volume
        stop_hunt = (last_move > avg_move * 2) and (volume[-1] < avg_vol_recent * 0.8)
        
        return {
            'cumulative_delta': cumulative_delta[-1],
            'delta_ratio': delta_ratio,
            'delta_direction': 1 if delta_ratio > self.config.delta_threshold else (-1 if delta_ratio < -self.config.delta_threshold else 0),
            'divergence': divergence,
            'divergence_signal': 1 if divergence > self.config.divergence_threshold else (-1 if divergence < -self.config.divergence_threshold else 0),
            'imbalance': imbalance,
            'imbalance_signal': 1 if imbalance > self.config.imbalance_threshold else (-1 if imbalance < -self.config.imbalance_threshold else 0),
            'large_trade': large_trade,
            'stop_hunt': stop_hunt,
            'buy_volume': buy_volume[-1],
            'sell_volume': sell_volume[-1],
            'volume': volume[-1]
        }
    
    def generate_signal(self, data: pd.DataFrame) -> FlowSignal:
        """Generate trading signal from order flow."""
        metrics = self.calculate(data)
        
        signals = []
        confidences = []
        
        # Delta signal
        if self.config.use_cumulative_delta:
            if metrics['delta_direction'] != 0:
                signals.append(metrics['delta_direction'])
                confidences.append(min(abs(metrics['delta_ratio']) * 2, 1.0))
        
        # Divergence signal
        if self.config.use_delta_divergence:
            if metrics['divergence_signal'] != 0:
                signals.append(metrics['divergence_signal'])
                confidences.append(abs(metrics['divergence']))
        
        # Imbalance signal
        if self.config.use_bid_ask_imbalance:
            if metrics['imbalance_signal'] != 0:
                signals.append(metrics['imbalance_signal'])
                confidences.append(abs(metrics['imbalance']))
        
        # Large trade requirement
        if self.config.use_large_trade and self.config.large_trade_confirm:
            if not metrics['large_trade']:
                # No large trade = reduce confidence or block
                if signals:
                    confidences = [c * 0.5 for c in confidences]
        
        # Combine signals
        if not signals:
            return FlowSignal(0, 0.0, FlowType.CUMULATIVE_DELTA, metrics)
        
        # Weighted average
        direction = 1 if sum(signals) > 0 else (-1 if sum(signals) < 0 else 0)
        confidence = np.mean(confidences) if confidences else 0.0
        
        return FlowSignal(
            direction=direction,
            confidence=confidence,
            flow_type=FlowType.CUMULATIVE_DELTA,
            metrics=metrics
        )


# ============================================================
# FLOW MUTATION OPERATORS
# ============================================================
class FlowMutationOperators:
    """
    Mutation operators for order flow configuration.
    
    These let the GA evolve:
    - Which flow metrics to use
    - Thresholds for each
    - Time windows
    """
    
    @staticmethod
    def enable_delta_filter(config: FlowConfig) -> FlowConfig:
        """Enable cumulative delta filtering."""
        config = copy.deepcopy(config)
        config.use_cumulative_delta = random.choice([True, False])
        return config
    
    @staticmethod
    def mutate_delta_threshold(config: FlowConfig, delta: float = 0.02) -> FlowConfig:
        """Mutate delta threshold."""
        config = copy.deepcopy(config)
        config.delta_threshold = max(0.01, min(0.5, 
            config.delta_threshold + random.uniform(-delta, delta)))
        return config
    
    @staticmethod
    def mutate_delta_window(config: FlowConfig, delta: int = 5) -> FlowConfig:
        """Mutate delta calculation window."""
        config = copy.deepcopy(config)
        config.delta_window = max(5, min(100, 
            config.delta_window + random.choice([-delta, delta])))
        return config
    
    @staticmethod
    def enable_divergence(config: FlowConfig) -> FlowConfig:
        """Enable delta divergence detection."""
        config = copy.deepcopy(config)
        config.use_delta_divergence = random.choice([True, False])
        return config
    
    @staticmethod
    def mutate_divergence_threshold(config: FlowConfig, delta: float = 0.05) -> FlowConfig:
        """Mutate divergence threshold."""
        config = copy.deepcopy(config)
        config.divergence_threshold = max(0.1, min(0.8, 
            config.divergence_threshold + random.uniform(-delta, delta)))
        return config
    
    @staticmethod
    def enable_large_trade(config: FlowConfig) -> FlowConfig:
        """Enable large trade detection."""
        config = copy.deepcopy(config)
        config.use_large_trade = random.choice([True, False])
        return config
    
    @staticmethod
    def mutate_large_trade_std(config: FlowConfig, delta: float = 0.25) -> FlowConfig:
        """Mutate large trade standard deviation threshold."""
        config = copy.deepcopy(config)
        config.large_trade_std = max(1.0, min(4.0, 
            config.large_trade_std + random.uniform(-delta, delta)))
        return config
    
    @staticmethod
    def mutate_large_trade_confirm(config: FlowConfig) -> FlowConfig:
        """Toggle large trade confirmation requirement."""
        config = copy.deepcopy(config)
        config.large_trade_confirm = not config.large_trade_confirm
        return config
    
    @staticmethod
    def enable_imbalance(config: FlowConfig) -> FlowConfig:
        """Enable bid-ask imbalance."""
        config = copy.deepcopy(config)
        config.use_bid_ask_imbalance = random.choice([True, False])
        return config
    
    @staticmethod
    def enable_stop_hunt(config: FlowConfig) -> FlowConfig:
        """Enable stop hunt detection."""
        config = copy.deepcopy(config)
        config.use_stop_hunt = random.choice([True, False])
        return config
    
    @staticmethod
    def mutate_regime_flow(config: FlowConfig, regime: str = None) -> FlowConfig:
        """Mutate flow parameters for a specific regime."""
        config = copy.deepcopy(config)
        
        regimes = list(config.flow_per_regime.keys())
        if regime is None:
            regime = random.choice(regimes)
        
        if regime in config.flow_per_regime:
            current = config.flow_per_regime[regime]
            # Mutate
            if 'delta_threshold' in current:
                current['delta_threshold'] = max(0.01, min(0.5, 
                    current['delta_threshold'] + random.uniform(-0.02, 0.02)))
            if 'require_large_trade' in current:
                current['require_large_trade'] = random.choice([True, False])
        
        return config
    
    @staticmethod
    def relax_flow_for_regime(config: FlowConfig, regime: str) -> FlowConfig:
        """Relax flow requirements for a specific regime."""
        config = copy.deepcopy(config)
        
        if regime in config.flow_per_regime:
            config.flow_per_regime[regime]['delta_threshold'] *= 0.8
            config.flow_per_regime[regime]['require_large_trade'] = False
        
        return config
    
    @staticmethod
    def tighten_flow_for_regime(config: FlowConfig, regime: str) -> FlowConfig:
        """Tighten flow requirements for a specific regime."""
        config = copy.deepcopy(config)
        
        if regime in config.flow_per_regime:
            config.flow_per_regime[regime]['delta_threshold'] *= 1.2
            config.flow_per_regime[regime]['require_large_trade'] = True
        
        return config


# ============================================================
# FLOW-REGIME HYBRIDIZER
# ============================================================
class FlowRegimeHybridizer:
    """
    Expert: "Learn, for each regime, the optimal order-flow configuration."
    
    When regime changes, automatically adjust flow parameters.
    """
    
    def __init__(self, flow_config: FlowConfig):
        self.flow_config = flow_config
        
    def get_flow_for_regime(self, regime: str) -> FlowConfig:
        """Get optimized flow config for current regime."""
        config = copy.deepcopy(self.flow_config)
        
        regime_params = self.flow_config.flow_per_regime.get(regime, {})
        
        if 'delta_threshold' in regime_params:
            config.delta_threshold = regime_params['delta_threshold']
        
        if 'require_large_trade' in regime_params:
            config.large_trade_confirm = regime_params['require_large_trade']
        
        # Adjust based on regime characteristics
        if regime == 'high_vol':
            # In high vol, require stronger confirmation
            config.delta_threshold *= 1.3
            config.large_trade_confirm = True
        elif regime == 'low_vol':
            # In low vol, relax requirements
            config.delta_threshold *= 0.7
            config.large_trade_confirm = False
        elif regime == 'sideways':
            # In sideways, use tighter imbalance
            config.use_bid_ask_imbalance = True
            config.imbalance_threshold = 0.4
        
        return config


# ============================================================
# FACTORY FUNCTIONS
# ============================================================
def create_default_flow_config() -> FlowConfig:
    """Create default order flow configuration."""
    return FlowConfig(
        use_cumulative_delta=True,
        use_delta_divergence=True,
        use_bid_ask_imbalance=False,
        use_large_trade=True,
        use_stop_hunt=False,
        delta_window=20,
        delta_smoothing=5,
        delta_threshold=0.1,
        divergence_lookback=50,
        divergence_threshold=0.3,
        large_trade_std=2.0,
        large_trade_window=20,
        large_trade_confirm=True,
        imbalance_window=10,
        imbalance_threshold=0.6,
        stop_hunt_lookback=10,
        stop_hunt_threshold=0.05,
        flow_per_regime={
            'bull': {'delta_threshold': 0.1, 'require_large_trade': True},
            'bear': {'delta_threshold': 0.1, 'require_large_trade': True},
            'sideways': {'delta_threshold': 0.05, 'require_large_trade': False},
            'high_vol': {'delta_threshold': 0.15, 'require_large_trade': True},
            'low_vol': {'delta_threshold': 0.05, 'require_large_trade': False}
        }
    )


# ============================================================
# CLI / TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data with some microstructure
    np.random.seed(42)
    n = 100
    
    # Create realistic 1-min style data
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    open_prices = close + np.random.randn(n) * 0.2
    high = np.maximum(close, open_prices) + np.abs(np.random.randn(n) * 0.3)
    low = np.minimum(close, open_prices) - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n)
    
    # Inject some buy/sell pressure
    # Make last 20 bars clearly bullish
    close[-20:] = 100 + np.cumsum(np.random.randn(20) * 0.8)
    volume[-20:] = volume[-20:] * 2  # Higher volume
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Test Order Flow Calculator
    print("Testing Order Flow Calculator...")
    config = create_default_flow_config()
    calc = OrderFlowCalculator(config)
    
    metrics = calc.calculate(data)
    print(f"  Cumulative Delta: {metrics['cumulative_delta']:.0f}")
    print(f"  Delta Ratio: {metrics['delta_ratio']:.3f}")
    print(f"  Delta Direction: {metrics['delta_direction']}")
    print(f"  Divergence: {metrics['divergence']}")
    print(f"  Large Trade: {metrics['large_trade']}")
    print(f"  Imbalance: {metrics['imbalance']:.3f}")
    
    signal = calc.generate_signal(data)
    print(f"  Signal: {signal.direction}, Confidence: {signal.confidence:.2f}")
    
    # Test Mutations
    print("\nTesting Flow Mutations...")
    config = FlowMutationOperators.mutate_delta_threshold(config)
    print(f"  After mutate_delta_threshold: {config.delta_threshold:.3f}")
    
    config = FlowMutationOperators.mutate_large_trade_std(config)
    print(f"  After mutate_large_trade_std: {config.large_trade_std:.2f}")
    
    config = FlowMutationOperators.enable_divergence(config)
    print(f"  After enable_divergence: {config.use_delta_divergence}")
    
    # Test Regime Hybridizer
    print("\nTesting Flow-Regime Hybridizer...")
    hybridizer = FlowRegimeHybridizer(config)
    
    for regime in ['bull', 'bear', 'sideways', 'high_vol', 'low_vol']:
        regime_config = hybridizer.get_flow_for_regime(regime)
        print(f"  {regime}: delta_thresh={regime_config.delta_threshold:.3f}, large_confirm={regime_config.large_trade_confirm}")
    
    print("\n✅ All order flow tests passed!")
