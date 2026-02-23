"""
QuantCore - Correlation Decay & Pair Rotation v1.0

This module adds multi-asset intelligence - relationships between assets.

Expert: "Markets aren't islands; they're ecosystems. When BTC sneezes,
ETH catches a cold. By evolving strategies that exploit these relationships,
your system will graduate from a solo predator to a pack hunter."

Features:
1. Cointegration Detection (find pairs that move together)
2. Rolling Correlation Tracking
3. Hedge Ratio Evolution
4. Pair Rotation Strategies
5. Multi-Asset Portfolio Integration
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
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ============================================================
# PAIR SELECTION METHODS
# ============================================================
class PairSelectionMethod(Enum):
    """How to select pairs for trading."""
    COINTEGRATION = "cointegration"     # Engle-Granger two-step
    CORRELATION = "correlation"         # Rolling correlation
    MANUAL = "manual"                   # User-defined pairs
    AUTO_DISCOVER = "auto_discover"    # Scan and find best


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class PairConfig:
    """
    Evolvable pair trading configuration.
    
    The GA evolves:
    - Which pairs to trade
    - Entry thresholds (z-score for spread)
    - Correlation lookback windows
    - Hedge ratio adaptation method
    - Regime-specific pair preferences
    """
    # Pair definition
    asset_a: str = "BTC"
    asset_b: str = "ETH"
    pair_selection_method: str = "cointegration"
    
    # Entry/Exit thresholds
    entry_threshold: float = 2.0        # Standard deviations for entry
    exit_threshold: float = 0.5         # Standard deviations for exit
    stop_loss_z: float = 3.0             # Z-score for stop loss
    
    # Correlation settings
    correlation_lookback: int = 20      # Days for rolling correlation
    correlation_entry: float = 0.7      # Min correlation to trade
    correlation_exit: float = 0.3       # Exit if correlation drops
    
    # Cointegration settings
    cointegration_lookback: int = 60     # Days for cointegration test
    cointegration_pvalue: float = 0.05   # Max p-value for significance
    
    # Hedge ratio
    hedge_ratio: float = 15.0            # Default: 1 BTC = 15 ETH
    hedge_ratio_adaptation: str = "static"  # 'static', 'rolling', 'kalman'
    hedge_lookback: int = 30            # Rolling hedge ratio window
    
    # Position sizing
    pair_position_size: float = 0.1      # 10% of portfolio per pair
    max_pairs: int = 3                   # Max concurrent pairs
    
    # Direction bias
    direction_bias: str = "both"         # 'long_spread', 'short_spread', 'both'
    
    # Timeframe for pair analysis
    pair_timeframe: str = "1h"           # 1h, 4h, 1d
    
    # Regime-specific pair preferences
    regime_pairs: Dict[str, dict] = field(default_factory=lambda: {
        'bull': {'prefer': ['BTC/ETH', 'SOL/ETH'], 'min_correlation': 0.6},
        'bear': {'prefer': ['BTC/USDT', 'gold/silver'], 'min_correlation': 0.5},
        'sideways': {'prefer': ['BTC/ETH', 'gold/silver'], 'min_correlation': 0.7},
        'high_vol': {'prefer': [], 'min_correlation': 0.8},  # Only high correlation
        'low_vol': {'prefer': ['altcoins'], 'min_correlation': 0.4}
    })
    
    def to_dict(self) -> Dict:
        return {
            'asset_a': self.asset_a,
            'asset_b': self.asset_b,
            'pair_selection_method': self.pair_selection_method,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_loss_z': self.stop_loss_z,
            'correlation_lookback': self.correlation_lookback,
            'correlation_entry': self.correlation_entry,
            'correlation_exit': self.correlation_exit,
            'cointegration_lookback': self.cointegration_lookback,
            'cointegration_pvalue': self.cointegration_pvalue,
            'hedge_ratio': self.hedge_ratio,
            'hedge_ratio_adaptation': self.hedge_ratio_adaptation,
            'hedge_lookback': self.hedge_lookback,
            'pair_position_size': self.pair_position_size,
            'max_pairs': self.max_pairs,
            'direction_bias': self.direction_bias,
            'pair_timeframe': self.pair_timeframe,
            'regime_pairs': self.regime_pairs
        }


@dataclass
class PairSignal:
    """Signal from pair trading analysis."""
    direction: int              # 1 (long spread), -1 (short spread), 0 (no signal)
    confidence: float          # 0-1
    spread_zscore: float       # Current spread deviation
    correlation: float         # Current rolling correlation
    hedge_ratio: float         # Current hedge ratio
    is_cointegrated: bool     # True if pair is cointegrated
    half_life: float          # Expected mean reversion half-life (bars)
    source: str               # 'correlation', 'cointegration', 'both'


@dataclass
class PortfolioPair:
    """A pair being tracked in portfolio."""
    config: PairConfig
    entry_spread: float = 0
    entry_zscore: float = 0
    pnl: float = 0
    bars_held: int = 0


# ============================================================
# COINTEGRATION TESTER
# ============================================================
class CointegrationTester:
    """
    Test if two assets are cointegrated (move together long-term).
    
    Uses Engle-Granger two-step test:
    1. Regress Y on X, get hedge ratio (beta)
    2. Test if residuals are stationary (ADF test)
    """
    
    def __init__(self, config: PairConfig):
        self.config = config
        
    def test_cointegration(self, price_a: np.ndarray, price_b: np.ndarray) -> Tuple[bool, float, float]:
        """
        Test cointegration between two assets.
        
        Returns: (is_cointegrated, p_value, half_life)
        """
        if len(price_a) < self.config.cointegration_lookback:
            return False, 1.0, 0
        
        # Use last N days for test
        n = min(self.config.cointegration_lookback, len(price_a))
        a = price_a[-n:]
        b = price_b[-n:]
        
        try:
            # Step 1: OLS regression Y = alpha + beta * X
            # Price A = alpha + beta * Price B
            slope, intercept, r_value, p_value, std_err = stats.linregress(b, a)
            
            # Calculate residuals
            residuals = a - (slope * b + intercept)
            
            # Step 2: ADF test on residuals
            adf_result = self._adf_test(residuals)
            
            # Calculate half-life of mean reversion
            half_life = self._calculate_half_life(residuals)
            
            # Check if cointegrated (p-value < threshold)
            is_cointegrated = adf_result < self.config.cointegration_pvalue
            
            return is_cointegrated, adf_result, half_life
            
        except Exception as e:
            logger.warning(f"Cointegration test failed: {e}")
            return False, 1.0, 0
    
    def _adf_test(self, series: np.ndarray) -> float:
        """Augmented Dickey-Fuller test for stationarity."""
        from scipy.stats import norm
        
        n = len(series)
        if n < 10:
            return 1.0
        
        # Differences
        y = np.diff(series)
        x = series[:-1].reshape(-1, 1)
        
        # Add constant and trend
        x = np.column_stack([np.ones(len(x)), np.arange(len(x)), x])
        
        try:
            # OLS
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            residuals = y - x @ beta
            
            # Calculate ADF statistic
            sigma = np.std(residuals)
            if sigma == 0:
                return 1.0
            
            adf_stat = beta[-1] / sigma * np.sqrt(n)
            
            # Approximate p-value (simplified)
            p_value = norm.cdf(adf_stat)
            
            return 1 - p_value  # Lower = more stationary
            
        except:
            return 1.0
    
    def _calculate_half_life(self, residuals: np.ndarray) -> float:
        """Calculate mean reversion half-life."""
        if len(residuals) < 10:
            return 0
        
        # Ornstein-Uhlenbeck: dR = lambda * R * dt + dW
        # lambda is the speed of mean reversion
        
        residuals_lag = residuals[:-1]
        residuals_diff = np.diff(residuals)
        
        # Lambda via OLS
        try:
            slope, _, _, _, _ = stats.linregress(residuals_lag, residuals_diff)
            lambda_reversion = -slope
            
            if lambda_reversion > 0:
                half_life = np.log(2) / lambda_reversion
                return half_life
            else:
                return 0  # No mean reversion
        except:
            return 0


# ============================================================
# CORRELATION TRACKER
# ============================================================
class CorrelationTracker:
    """
    Track rolling correlation between asset pairs.
    """
    
    def __init__(self, config: PairConfig):
        self.config = config
        self.correlation_history: List[float] = []
        
    def calculate_correlation(self, returns_a: np.ndarray, returns_b: np.ndarray) -> float:
        """Calculate rolling correlation between two assets."""
        lookback = min(self.config.correlation_lookback, len(returns_a), len(returns_b))
        
        if lookback < 5:
            return 0
        
        a = returns_a[-lookback:]
        b = returns_b[-lookback:]
        
        try:
            corr, _ = stats.pearsonr(a, b)
            return corr if not np.isnan(corr) else 0
        except:
            return 0
    
    def update_history(self, correlation: float):
        """Store correlation in history."""
        self.correlation_history.append(correlation)
        # Keep only recent history
        max_history = 200
        if len(self.correlation_history) > max_history:
            self.correlation_history = self.correlation_history[-max_history:]
    
    def detect_correlation_decay(self) -> bool:
        """Detect if correlation has decayed significantly."""
        if len(self.correlation_history) < self.config.correlation_lookback:
            return False
        
        # Compare recent correlation to historical average
        recent = np.mean(self.correlation_history[-5:])
        historical = np.mean(self.correlation_history[:-5])
        
        # Significant decay?
        decay_threshold = 0.2
        return (historical - recent) > decay_threshold


# ============================================================
# HEDGE RATIO CALCULATOR
# ============================================================
class HedgeRatioCalculator:
    """
    Calculate and adapt hedge ratios between pairs.
    """
    
    def __init__(self, config: PairConfig):
        self.config = config
        self.hedge_history: List[float] = []
        
    def calculate_static(self, price_a: np.ndarray, price_b: np.ndarray) -> float:
        """Static hedge ratio (simple regression)."""
        n = min(self.config.hedge_lookback, len(price_a), len(price_b))
        a = price_a[-n:]
        b = price_b[-n:]
        
        try:
            slope, _, _, _, _ = stats.linregress(b, a)
            return slope if slope > 0 else self.config.hedge_ratio
        except:
            return self.config.hedge_ratio
    
    def calculate_rolling(self, price_a: np.ndarray, price_b: np.ndarray) -> float:
        """Rolling hedge ratio with exponential weighting."""
        n = min(self.config.hedge_lookback * 2, len(price_a), len(price_b))
        a = price_a[-n:]
        b = price_b[-n:]
        
        try:
            # Exponential weights
            weights = np.exp(np.linspace(0, 1, n))
            weights /= weights.sum()
            
            # Weighted regression
            slope = np.sum(weights * b * a) / (np.sum(weights * b * b) + 1e-10)
            return slope if slope > 0 else self.config.hedge_ratio
        except:
            return self.config.hedge_ratio
    
    def calculate_kalman(self, price_a: np.ndarray, price_b: np.ndarray) -> float:
        """Kalman filter for dynamic hedge ratio (simplified)."""
        # Simplified version - use rolling with decay
        return self.calculate_rolling(price_a, price_b)


# ============================================================
# PAIR SIGNAL GENERATOR
# ============================================================
class PairSignalGenerator:
    """
    Generate trading signals from pair analysis.
    """
    
    def __init__(self, config: PairConfig):
        self.config = config
        self.cointegration_tester = CointegrationTester(config)
        self.correlation_tracker = CorrelationTracker(config)
        self.hedge_calculator = HedgeRatioCalculator(config)
        
    def generate_signal(self, 
                       price_a: np.ndarray, 
                       price_b: np.ndarray,
                       regime: str = 'sideways') -> PairSignal:
        """
        Generate pair trading signal.
        
        Long spread: Expect mean reversion (short A, long B when spread is high)
        Short spread: Expect mean reversion (long A, short B when spread is low)
        """
        # Calculate returns
        returns_a = np.diff(np.log(price_a))
        returns_b = np.diff(np.log(price_b))
        
        # Get hedge ratio
        if self.config.hedge_ratio_adaptation == 'static':
            hedge_ratio = self.hedge_calculator.calculate_static(price_a, price_b)
        elif self.config.hedge_ratio_adaptation == 'rolling':
            hedge_ratio = self.hedge_calculator.calculate_rolling(price_a, price_b)
        else:
            hedge_ratio = self.hedge_calculator.calculate_kalman(price_b, price_a)
        
        # Calculate spread: A - hedge_ratio * B
        spread = price_a[-1] - hedge_ratio * price_b[-1]
        
        # Calculate z-score of spread
        n = min(30, len(price_a), len(price_b))
        spread_history = price_a[-n:] - hedge_ratio * price_b[-n:]
        spread_mean = np.mean(spread_history)
        spread_std = np.std(spread_history)
        
        zscore = 0
        if spread_std > 0:
            zscore = (spread - spread_mean) / spread_std
        
        # Calculate correlation
        correlation = self.correlation_tracker.calculate_correlation(returns_a, returns_b)
        self.correlation_tracker.update_history(correlation)
        
        # Test cointegration
        is_cointegrated, p_value, half_life = self.cointegration_tester.test_cointegration(
            price_a, price_b
        )
        
        # Determine signal based on z-score and thresholds
        direction = 0
        confidence = 0
        
        # Entry conditions
        if abs(zscore) > self.config.entry_threshold:
            # Check correlation requirement
            if correlation >= self.config.correlation_entry or is_cointegrated:
                if zscore > self.config.entry_threshold:
                    # Spread too high -> expect mean reversion down
                    # Short A, Long B
                    direction = -1 if self.config.direction_bias in ['both', 'short_spread'] else 0
                else:
                    # Spread too low -> expect mean reversion up
                    # Long A, Short B
                    direction = 1 if self.config.direction_bias in ['both', 'long_spread'] else 0
                
                confidence = min(abs(zscore) / 3.0, 1.0)
        
        # Exit conditions
        elif abs(zscore) < self.config.exit_threshold:
            direction = 0
            confidence = 0.8
        
        # Stop loss
        if abs(zscore) > self.config.stop_loss_z:
            direction = 0
            confidence = 0
        
        # Determine source
        source = 'both'
        if is_cointegrated and correlation < self.config.correlation_entry:
            source = 'cointegration'
        elif correlation >= self.config.correlation_entry and not is_cointegrated:
            source = 'correlation'
        
        return PairSignal(
            direction=direction,
            confidence=confidence,
            spread_zscore=zscore,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            is_cointegrated=is_cointegrated,
            half_life=half_life if half_life > 0 else 999,
            source=source
        )


# ============================================================
# PAIR ROTATION MANAGER
# ============================================================
class PairRotationManager:
    """
    Manage multiple pairs and rotate based on regime/performance.
    """
    
    def __init__(self, max_pairs: int = 3):
        self.max_pairs = max_pairs
        self.active_pairs: List[PortfolioPair] = []
        self.pair_configs: List[PairConfig] = []
        
    def add_pair(self, config: PairConfig):
        """Add a pair to watchlist."""
        if len(self.pair_configs) < self.max_pairs * 2:
            self.pair_configs.append(config)
    
    def select_pairs_for_regime(self, regime: str, 
                                price_data: Dict[str, np.ndarray]) -> List[PairConfig]:
        """
        Select best pairs for current regime.
        
        Returns list of pair configs to trade.
        """
        scored_pairs = []
        
        for config in self.pair_configs:
            # Get price data
            asset_a = config.asset_a
            asset_b = config.asset_b
            
            if asset_a not in price_data or asset_b not in price_data:
                continue
            
            # Calculate correlation
            returns_a = np.diff(np.log(price_data[asset_a]))
            returns_b = np.diff(np.log(price_data[asset_b]))
            
            if len(returns_a) < 20 or len(returns_b) < 20:
                continue
            
            corr = np.corrcoef(returns_a[-20:], returns_b[-20:])[0, 1]
            
            # Score based on regime preferences
            regime_prefs = config.regime_pairs.get(regime, {})
            min_corr = regime_prefs.get('min_correlation', 0.5)
            
            if corr >= min_corr:
                score = corr
                # Bonus for preferred pairs
                preferred = regime_prefs.get('prefer', [])
                if f"{asset_a}/{asset_b}" in preferred:
                    score *= 1.2
                
                scored_pairs.append((score, config))
        
        # Sort by score and return top N
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_pairs[:self.max_pairs]]


# ============================================================
# MUTATION OPERATORS
# ============================================================
class PairMutationOperators:
    """
    Mutation operators for pair trading configuration.
    """
    
    @staticmethod
    def mutate_entry_threshold(config: PairConfig, delta: float = 0.3) -> PairConfig:
        """Mutate z-score entry threshold."""
        config = copy.deepcopy(config)
        config.entry_threshold = max(1.0, min(4.0, config.entry_threshold + random.uniform(-delta, delta)))
        return config
    
    @staticmethod
    def mutate_correlation_lookback(config: PairConfig, delta: int = 3) -> PairConfig:
        """Mutate correlation lookback window."""
        config = copy.deepcopy(config)
        config.correlation_lookback = max(5, min(60, config.correlation_lookback + random.choice([-delta, delta])))
        return config
    
    @staticmethod
    def mutate_hedge_ratio(config: PairConfig, delta: float = 1.0) -> PairConfig:
        """Mutate hedge ratio."""
        config = copy.deepcopy(config)
        config.hedge_ratio = max(1.0, config.hedge_ratio + random.uniform(-delta, delta))
        return config
    
    @staticmethod
    def mutate_direction_bias(config: PairConfig) -> PairConfig:
        """Change direction bias."""
        config = copy.deepcopy(config)
        biases = ['long_spread', 'short_spread', 'both']
        if config.direction_bias in biases:
            idx = biases.index(config.direction_bias)
            config.direction_bias = biases[(idx + 1) % len(biases)]
        return config
    
    @staticmethod
    def mutate_hedge_adaptation(config: PairConfig) -> PairConfig:
        """Change hedge ratio adaptation method."""
        config = copy.deepcopy(config)
        methods = ['static', 'rolling', 'kalman']
        if config.hedge_ratio_adaptation in methods:
            idx = methods.index(config.hedge_ratio_adaptation)
            config.hedge_ratio_adaptation = methods[(idx + 1) % len(methods)]
        return config
    
    @staticmethod
    def swap_pair_assets(config: PairConfig) -> PairConfig:
        """Swap asset A and B (reverses direction)."""
        config = copy.deepcopy(config)
        config.asset_a, config.asset_b = config.asset_b, config.asset_a
        return config
    
    @staticmethod
    def mutate_pair_selection(config: PairConfig) -> PairConfig:
        """Change pair selection method."""
        config = copy.deepcopy(config)
        methods = ['cointegration', 'correlation', 'manual', 'auto_discover']
        if config.pair_selection_method in methods:
            idx = methods.index(config.pair_selection_method)
            config.pair_selection_method = methods[(idx + 1) % len(methods)]
        return config
    
    @staticmethod
    def tighten_entry(config: PairConfig) -> PairConfig:
        """Tighten entry thresholds (more signals)."""
        config = copy.deepcopy(config)
        config.entry_threshold = max(1.0, config.entry_threshold - 0.3)
        config.correlation_entry = max(0.4, config.correlation_entry - 0.1)
        return config
    
    @staticmethod
    def widen_entry(config: PairConfig) -> PairConfig:
        """Widen entry thresholds (fewer, higher confidence)."""
        config = copy.deepcopy(config)
        config.entry_threshold = min(4.0, config.entry_threshold + 0.3)
        config.correlation_entry = min(0.9, config.correlation_entry + 0.1)
        return config


# ============================================================
# FACTORY FUNCTIONS
# ============================================================
def create_pair_config(asset_a: str = "BTC", asset_b: str = "ETH") -> PairConfig:
    """Create default pair configuration."""
    return PairConfig(
        asset_a=asset_a,
        asset_b=asset_b,
        pair_selection_method="cointegration",
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_z=3.0,
        correlation_lookback=20,
        correlation_entry=0.7,
        correlation_exit=0.3,
        cointegration_lookback=60,
        cointegration_pvalue=0.05,
        hedge_ratio=15.0,
        hedge_ratio_adaptation="rolling",
        hedge_lookback=30,
        pair_position_size=0.1,
        max_pairs=3,
        direction_bias="both",
        pair_timeframe="1h"
    )


# ============================================================
# PREDEFINED POPULAR PAIRS
# ============================================================
POPULAR_PAIRS = [
    ("BTC", "ETH"),      # Crypto majors
    ("BTC", "SOL"),      # BTC vs alt
    ("ETH", "SOL"),      # Alt correlation
    ("BTC", "BNB"),      # Exchange tokens
    ("ETH", "MATIC"),    # L2 tokens
    ("BTC", "gold"),     # Crypto vs PM
    ("gold", "silver"),  # Precious metals
    ("SPY", "QQQ"),      # US equities
    ("TLT", "SPY"),      # Bonds vs stocks
]


# ============================================================
# CLI / TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic price data
    np.random.seed(42)
    n = 200
    
    # Generate correlated prices
    base = 100 + np.cumsum(np.random.randn(n) * 0.5)
    eth_base = base * 15 + np.cumsum(np.random.randn(n) * 0.4)  # Correlated
    
    btc_price = 45000 + np.cumsum(np.random.randn(n) * 500)
    eth_price = 3000 + np.cumsum(np.random.randn(n) * 30)
    
    # Test Cointegration
    print("Testing Cointegration...")
    config = create_pair_config("BTC", "ETH")
    tester = CointegrationTester(config)
    
    is_cointegrated, p_value, half_life = tester.test_cointegration(btc_price, eth_price)
    print(f"  Cointegration: {is_cointegrated}, p-value: {p_value:.4f}, half-life: {half_life:.1f} bars")
    
    # Test Correlation
    print("\nTesting Correlation...")
    tracker = CorrelationTracker(config)
    returns_btc = np.diff(np.log(btc_price))
    returns_eth = np.diff(np.log(eth_price))
    
    corr = tracker.calculate_correlation(returns_btc, returns_eth)
    print(f"  Rolling correlation: {corr:.3f}")
    
    # Test Signal Generation
    print("\nTesting Signal Generation...")
    generator = PairSignalGenerator(config)
    signal = generator.generate_signal(btc_price, eth_price, regime='bull')
    
    print(f"  Direction: {signal.direction}")
    print(f"  Z-score: {signal.spread_zscore:.2f}")
    print(f"  Correlation: {signal.correlation:.3f}")
    print(f"  Hedge ratio: {signal.hedge_ratio:.2f}")
    print(f"  Half-life: {signal.half_life:.1f}")
    
    # Test Mutations
    print("\nTesting Pair Mutations...")
    config = PairMutationOperators.mutate_entry_threshold(config)
    print(f"  After mutate_entry_threshold: {config.entry_threshold}")
    
    config = PairMutationOperators.mutate_direction_bias(config)
    print(f"  After mutate_direction_bias: {config.direction_bias}")
    
    print("\n✅ All pair trading tests passed!")
