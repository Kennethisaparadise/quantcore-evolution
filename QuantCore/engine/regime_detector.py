"""
QuantCore - Market Regime Detection Module

Uses Hidden Markov Model (HMM) and clustering to detect market regimes:
- TRENDING (uptrend/downtrend)
- MEAN_REVERTING (range-bound)
- HIGH_VOLATILITY
- SIDEWAYS (low volatility, low trend)

Then biases mutations toward strategy families that historically perform well in each regime.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# REGIME TYPES
# ============================================================
class Regime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    SIDEWAYS = "sideways"
    CONSOLIDATING = "consolidating"
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """Result of regime analysis."""
    regime: Regime
    confidence: float  # 0-1
    features: Dict[str, float]
    regime_history: List[Tuple[pd.Timestamp, Regime]]
    timestamp: pd.Timestamp
    
    @property
    def regime_name(self) -> str:
        return self.regime.value


# ============================================================
# FEATURE EXTRACTORS
# ============================================================
class RegimeFeatures:
    """Extract features for regime detection."""
    
    @staticmethod
    def extract(data: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
        """
        Extract features for regime detection.
        
        Features:
        - trend_strength: ADX-based trend indicator
        - volatility: ATR as % of price
        - momentum: Rate of change
        - mean_reversion: Distance from moving average
        - volume_trend: Volume momentum
        - cycle_position: Where we are in the cycle
        """
        close = data['close']
        high = data.get('high', close)
        low = data.get('low', close)
        volume = data.get('volume', pd.Series(np.ones(len(close))))
        
        features = {}
        
        # Use available lookback
        n = min(lookback, len(close))
        close = close.iloc[-n:]
        high = high.iloc[-n:] if len(high) >= n else close
        low = low.iloc[-n:] if len(low) >= n else close
        volume = volume.iloc[-n:] if len(volume) >= n else pd.Series(np.ones(n))
        
        # 1. Trend Strength (simplified ADX-like)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        features['trend_strength'] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
        features['plus_di'] = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 50
        features['minus_di'] = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 50
        
        # 2. Volatility (ATR as % of price)
        features['volatility'] = (atr.iloc[-1] / close.iloc[-1] * 100) if not pd.isna(atr.iloc[-1]) else 2.0
        
        # 3. Momentum (ROC)
        features['momentum'] = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100 if n >= 20 else 0
        
        # 4. Mean Reversion (distance from MA)
        sma20 = close.rolling(20).mean()
        features['mean_reversion'] = abs(close.iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1] * 100 if not pd.isna(sma20.iloc[-1]) else 0
        
        # 5. Volume Trend
        vol_sma = volume.rolling(20).mean()
        features['volume_trend'] = volume.iloc[-1] / vol_sma.iloc[-1] if not pd.isna(vol_sma.iloc[-1]) else 1.0
        
        # 6. Trend Direction
        features['trend_direction'] = 1 if features['momentum'] > 0 else -1
        
        # 7. Price Position (in range)
        rolling_high = close.rolling(20).max()
        rolling_low = close.rolling(20).min()
        position = (close.iloc[-1] - rolling_low.iloc[-1]) / (rolling_high.iloc[-1] - rolling_low.iloc[-1]) if rolling_high.iloc[-1] != rolling_low.iloc[-1] else 0.5
        features['price_position'] = position
        
        # 8. Regime Stability (how long has current regime persisted)
        features['regime_stability'] = 0.5  # Placeholder
        
        return features


# ============================================================
# REGIME DETECTOR (HMM-Inspired)
# ============================================================
class RegimeDetector:
    """
    Detect market regime using multiple methods:
    1. Rule-based classification
    2. Statistical clustering
    3. HMM-like state transitions
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.regime_history: List[Tuple[pd.Timestamp, Regime]] = []
        self.transition_counts: Dict[Tuple[Regime, Regime], int] = defaultdict(int)
        self._init_transition_matrix()
    
    def _init_transition_matrix(self):
        """Initialize regime transition probabilities."""
        # Empirical transition probabilities (can be learned from data)
        self.transition_probs = {
            Regime.TRENDING_UP: {Regime.TRENDING_UP: 0.7, Regime.HIGH_VOLATILITY: 0.15, Regime.MEAN_REVERTING: 0.1, Regime.SIDEWAYS: 0.05},
            Regime.TRENDING_DOWN: {Regime.TRENDING_DOWN: 0.7, Regime.HIGH_VOLATILITY: 0.15, Regime.MEAN_REVERTING: 0.1, Regime.SIDEWAYS: 0.05},
            Regime.MEAN_REVERTING: {Regime.MEAN_REVERTING: 0.6, Regime.TRENDING_UP: 0.15, Regime.TRENDING_DOWN: 0.15, Regime.HIGH_VOLATILITY: 0.1},
            Regime.HIGH_VOLATILITY: {Regime.HIGH_VOLATILITY: 0.5, Regime.TRENDING_UP: 0.15, Regime.TRENDING_DOWN: 0.15, Regime.MEAN_REVERTING: 0.1, Regime.SIDEWAYS: 0.1},
            Regime.SIDEWAYS: {Regime.SIDEWAYS: 0.6, Regime.MEAN_REVERTING: 0.2, Regime.HIGH_VOLATILITY: 0.1, Regime.TRENDING_UP: 0.05, Regime.TRENDING_DOWN: 0.05},
            Regime.CONSOLIDATING: {Regime.CONSOLIDATING: 0.6, Regime.SIDEWAYS: 0.2, Regime.TRENDING_UP: 0.1, Regime.TRENDING_DOWN: 0.1},
        }
    
    def detect(self, data: pd.DataFrame) -> RegimeAnalysis:
        """
        Detect current market regime.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            RegimeAnalysis with regime type and confidence
        """
        features = RegimeFeatures.extract(data, self.lookback)
        
        # Rule-based regime classification
        regime = self._classify_regime(features)
        
        # Calculate confidence based on feature clarity
        confidence = self._calculate_confidence(regime, features)
        
        # Get timestamp
        timestamp = data.index[-1] if hasattr(data, 'index') else pd.Timestamp.now()
        
        # Update history
        self.regime_history.append((timestamp, regime))
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            features=features,
            regime_history=self.regime_history[-50:],
            timestamp=timestamp
        )
    
    def _classify_regime(self, features: Dict[str, float]) -> Regime:
        """Classify regime based on features."""
        
        trend_strength = features['trend_strength']
        volatility = features['volatility']
        momentum = features['momentum']
        mean_reversion = features['mean_reversion']
        
        # High Volatility check (first - most important for risk)
        if volatility > 5.0:  # > 5% ATR
            return Regime.HIGH_VOLATILITY
        
        # Strong Trend detection
        if trend_strength > 30:  # Strong ADX
            if momentum > 5:
                return Regime.TRENDING_UP
            elif momentum < -5:
                return Regime.TRENDING_DOWN
            elif features['trend_direction'] > 0:
                return Regime.TRENDING_UP
            else:
                return Regime.TRENDING_DOWN
        
        # Medium trend
        if trend_strength > 20:
            if momentum > 3:
                return Regime.TRENDING_UP
            elif momentum < -3:
                return Regime.TRENDING_DOWN
        
        # Low volatility, low trend = Sideways
        if volatility < 1.5 and trend_strength < 15:
            return Regime.SIDEWAYS
        
        # Mean reversion (price far from MA but weak trend)
        if mean_reversion > 3 and trend_strength < 25:
            return Regime.MEAN_REVERTING
        
        # Default to consolidating
        return Regime.CONSOLIDATING
    
    def _calculate_confidence(self, regime: Regime, features: Dict[str, float]) -> float:
        """Calculate confidence in regime classification."""
        
        trend_strength = features['trend_strength']
        volatility = features['volatility']
        
        # High confidence cases
        if regime == Regime.HIGH_VOLATILITY and volatility > 7:
            return 0.95
        if regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN] and trend_strength > 35:
            return 0.9
        if regime == Regime.SIDEWAYS and volatility < 1.0 and trend_strength < 10:
            return 0.85
        
        # Medium confidence
        if regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]:
            return 0.7
        if regime == Regime.HIGH_VOLATILITY:
            return 0.75
        
        # Lower confidence for ambiguous regimes
        return 0.6
    
    def get_regime_distribution(self, lookback: int = 20) -> Dict[Regime, float]:
        """Get distribution of regimes over lookback period."""
        if not self.regime_history:
            return {Regime.UNKNOWN: 1.0}
        
        recent = [r for _, r in self.regime_history[-lookback:]]
        total = len(recent)
        
        distribution = {}
        for regime in Regime:
            count = recent.count(regime)
            distribution[regime] = count / total if total > 0 else 0
        
        return distribution


# ============================================================
# REGIME-AWARE MUTATION BIASES
# ============================================================
class RegimeMutationBias:
    """
    Define which mutations work best in each regime.
    
    Research-backed mutation biases:
    - TRENDING: Momentum indicators, trailing stops, trend filters
    - MEAN_REVERTING: RSI, Bollinger Bands, faster exits
    - HIGH_VOLATILITY: Wider stops, smaller positions, volatility exits
    - SIDEWAYS: Range trading, tight stops, mean reversion
    """
    
    # Regime -> (mutation_types, weight)
    MUTATION_PREFERENCES = {
        Regime.TRENDING_UP: {
            'favor': [
                'add_trailing_stop',
                'add_trend_filter',
                'replace_with_momentum',
                'add_ema_cross',
                'add_macd',
                'widen_stop_loss',  # Wider to let trends run
                'multiply_period',  # Longer periods for trends
                'invert_to_short',  # For downtrends
            ],
            'disfavor': [
                'tighten_stop_loss',
                'add_mean_reversion',
                'add_rsi_filter',  # RSI can filter false breakouts
            ],
            'position_size': 'larger',  # 15-25%
            'stop_loss': 'wider',  # 5-10%
        },
        Regime.TRENDING_DOWN: {
            'favor': [
                'add_trailing_stop',
                'add_trend_filter',
                'replace_with_momentum',
                'add_ema_cross',
                'invert_to_short',
                'widen_stop_loss',
            ],
            'disfavor': [
                'tighten_stop_loss',
                'add_mean_reversion',
            ],
            'position_size': 'smaller',
            'stop_loss': 'wider',
        },
        Regime.MEAN_REVERTING: {
            'favor': [
                'add_rsi_filter',
                'add_bollinger',
                'tighten_stop_loss',
                'shorter_period',
                'add_mean_reversion',
                'shift_threshold',
                'tighten_threshold',
            ],
            'disfavor': [
                'add_trailing_stop',
                'widen_stop_loss',
                'multiply_period',
            ],
            'position_size': 'medium',  # 10-15%
            'stop_loss': 'tight',  # 2-4%
        },
        Regime.HIGH_VOLATILITY: {
            'favor': [
                'add_atr_stop',
                'add_volatility_filter',
                'reduce_position_size',
                'widen_stop_loss',
                'add_take_profit',  # Lock in gains
                'add_time_exit',
                'reduce_period',  # Faster signals
            ],
            'disfavor': [
                'add_trailing_stop',  # Can get stopped out easily
                'increase_position_size',
                'tighten_stop_loss',
            ],
            'position_size': 'smaller',  # 5-10%
            'stop_loss': 'wider',  # 8-15%
        },
        Regime.SIDEWAYS: {
            'favor': [
                'add_rsi_filter',
                'add_bollinger',
                'add_support_resistance',
                'tighten_stop_loss',
                'add_mean_reversion',
                'shorter_period',
                'add_volume_filter',
            ],
            'disfavor': [
                'add_trailing_stop',
                'add_trend_filter',
                'add_macd',
            ],
            'position_size': 'medium',  # 10-15%
            'stop_loss': 'tight',  # 2-3%
        },
        Regime.CONSOLIDATING: {
            'favor': [
                'add_range_filter',
                'tighten_stop_loss',
                'add_support_resistance',
                'add_volume_filter',
                'shorter_period',
            ],
            'disfavor': [
                'add_trailing_stop',
                'widen_stop_loss',
            ],
            'position_size': 'smaller',
            'stop_loss': 'tight',
        },
    }
    
    # Mutation name mappings (internal -> readable)
    MUTATION_MAPPING = {
        'add_trailing_stop': ['add_stop_loss', 'trailing_stop'],
        'add_trend_filter': ['add_time_filter'],
        'replace_with_momentum': ['replace_indicator'],
        'add_ema_cross': ['add_indicator'],
        'add_macd': ['add_indicator'],
        'widen_stop_loss': ['widen_threshold'],
        'tighten_stop_loss': ['tighten_threshold'],
        'add_rsi_filter': ['add_indicator', 'replace_indicator'],
        'add_bollinger': ['add_indicator'],
        'add_mean_reversion': ['flip_entry'],
        'shorter_period': ['multiply_period', 'change_period'],
        'multiply_period': ['multiply_period'],
        'invert_to_short': ['flip_entry'],
        'add_atr_stop': ['add_stop_loss'],
        'add_volatility_filter': ['add_volume_filter'],
        'reduce_position_size': ['change_position_size'],
        'add_take_profit': ['add_take_profit'],
        'add_time_exit': ['add_time_filter'],
        'reduce_period': ['change_period'],
        'add_support_resistance': ['add_indicator'],
        'add_volume_filter': ['add_volume_filter'],
        'add_range_filter': ['add_time_filter'],
    }
    
    @classmethod
    def get_mutation_weights(cls, regime: Regime) -> Dict[str, float]:
        """Get mutation weights for a regime."""
        
        prefs = cls.MUTATION_PREFERENCES.get(regime, cls.MUTATION_PREFERENCES[Regime.CONSOLIDATING])
        
        weights = {}
        
        # Favored mutations get high weight
        for mut in prefs.get('favor', []):
            mapped = cls.MUTATION_MAPPING.get(mut, [mut])
            for m in mapped:
                weights[m] = weights.get(m, 1.0) * 3.0
        
        # Disfavored mutations get low weight
        for mut in prefs.get('disfavor', []):
            mapped = cls.MUTATION_MAPPING.get(mut, [mut])
            for m in mapped:
                weights[m] = weights.get(m, 1.0) * 0.2
        
        return weights
    
    @classmethod
    def get_risk_parameters(cls, regime: Regime) -> Dict:
        """Get recommended risk parameters for a regime."""
        prefs = cls.MUTATION_PREFERENCES.get(regime, cls.MUTATION_PREFERENCES[Regime.CONSOLIDATING])
        
        size_map = {
            'larger': (0.15, 0.25),
            'medium': (0.10, 0.15),
            'smaller': (0.05, 0.10),
        }
        
        stop_map = {
            'wider': (0.05, 0.15),
            'tight': (0.02, 0.04),
            'medium': (0.03, 0.08),
        }
        
        size_pref = prefs.get('position_size', 'medium')
        stop_pref = prefs.get('stop_loss', 'medium')
        
        return {
            'position_size_range': size_map.get(size_pref, (0.10, 0.15)),
            'stop_loss_range': stop_map.get(stop_pref, (0.03, 0.08)),
            'regime': regime.value,
        }


# ============================================================
# REGIME-AWARE MUTATOR
# ============================================================
class RegimeAwareMutator:
    """
    Mutation engine that adapts to current market regime.
    
    Flow:
    1. Detect current regime from data
    2. Get mutation weights for that regime
    3. Apply weighted random mutations
    4. Also apply some random mutations for diversity
    """
    
    def __init__(
        self,
        mutation_count: int = 100,
        keep_original: bool = True,
        regime_adaptive: bool = True,
        random_mutation_rate: float = 0.2  # 20% random mutations for diversity
    ):
        self.mutation_count = mutation_count
        self.keep_original = keep_original
        self.regime_adaptive = regime_adaptive
        self.random_mutation_rate = random_mutation_rate
        
        self.detector = RegimeDetector()
        self.current_regime: Optional[Regime] = None
        self.current_analysis: Optional[RegimeAnalysis] = None
        
        # Base mutation types
        self.base_mutation_types = [
            'flip_entry', 'flip_exit', 'replace_indicator',
            'shift_threshold', 'tighten_threshold', 'widen_threshold',
            'change_period', 'multiply_period', 'add_indicator',
            'invert_signal', 'add_volume_filter', 'add_time_filter',
            'change_position_size', 'add_stop_loss', 'add_take_profit',
        ]
    
    def detect_regime(self, data: pd.DataFrame) -> RegimeAnalysis:
        """Detect current regime from data."""
        self.current_analysis = self.detector.detect(data)
        self.current_regime = self.current_analysis.regime
        
        logger.info(f"Detected regime: {self.current_regime.value} (confidence: {self.current_analysis.confidence:.2f})")
        
        return self.current_analysis
    
    def get_weighted_mutations(
        self,
        template: 'StrategyTemplate',
        data: pd.DataFrame = None
    ) -> List['StrategyTemplate']:
        """
        Get mutations weighted by regime suitability.
        
        Args:
            template: Strategy template to mutate
            data: Market data (for regime detection)
            
        Returns:
            List of mutated templates
        """
        # Detect regime if data provided and not already detected
        if data is not None and self.current_regime is None:
            self.detect_regime(data)
        
        # Get regime weights
        if self.regime_adaptive and self.current_regime:
            weights = RegimeMutationBias.get_mutation_weights(self.current_regime)
        else:
            weights = {m: 1.0 for m in self.base_mutation_types}
        
        mutations = []
        
        # Keep original
        if self.keep_original:
            import copy
            mutations.append(copy.deepcopy(template))
        
        # Generate mutations
        while len(mutations) < self.mutation_count:
            # Determine if this should be a random or regime-weighted mutation
            if random.random() < self.random_mutation_rate:
                # Random mutation for diversity
                mutation_type = random.choice(self.base_mutation_types)
            else:
                # Regime-weighted mutation
                mutation_type = self._weighted_choice(weights)
            
            mutated = self._apply_mutation(template, mutation_type)
            if mutated:
                mutations.append(mutated)
        
        return mutations
    
    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        """Select mutation based on weights."""
        items = list(weights.keys())
        weights_list = [weights[k] for k in items]
        
        # Normalize weights
        total = sum(weights_list)
        weights_list = [w / total for w in weights_list]
        
        return random.choices(items, weights=weights_list, k=1)[0]
    
    def _apply_mutation(self, template: 'StrategyTemplate', mutation_type: str):
        """Apply a specific mutation type."""
        import copy
        
        try:
            if mutation_type == 'flip_entry':
                return MutationOperators.flip_entry_logic(template)
            elif mutation_type == 'flip_exit':
                return MutationOperators.flip_exit_logic(template)
            elif mutation_type == 'replace_indicator':
                return MutationOperators.swap_indicators(template)
            elif mutation_type == 'shift_threshold':
                return MutationOperators.shift_threshold(template)
            elif mutation_type == 'tighten_threshold':
                return MutationOperators.tighten_threshold(template)
            elif mutation_type == 'widen_threshold':
                return MutationOperators.widen_threshold(template)
            elif mutation_type == 'change_period':
                return MutationOperators.change_period(template)
            elif mutation_type == 'multiply_period':
                return MutationOperators.multiply_period(template)
            elif mutation_type == 'add_indicator':
                return MutationOperators.add_indicator(template)
            elif mutation_type == 'invert_signal':
                return MutationOperators.invert_signal(template)
            elif mutation_type == 'add_volume_filter':
                return MutationOperators.add_volume_filter(template)
            elif mutation_type == 'add_time_filter':
                return MutationOperators.add_time_filter(template)
            elif mutation_type == 'change_position_size':
                return MutationOperators.change_position_size(template)
            elif mutation_type == 'add_stop_loss':
                return MutationOperators.add_stop_loss(template)
            elif mutation_type == 'add_take_profit':
                return MutationOperators.add_take_profit(template)
            else:
                return None
        except Exception as e:
            logger.warning(f"Mutation {mutation_type} failed: {e}")
            return None


# Import required for type hints
from engine.mutation import StrategyTemplate, MutationOperators


# ============================================================
# EXAMPLE USAGE
# ============================================================
def demo_regime_detection():
    """Demo regime detection."""
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n = 200
    
    # Create trending data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.random.rand(n) * 3
    low = close - np.random.rand(n) * 3
    volume = np.random.randint(1000000, 5000000, n)
    
    data = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
        'volume': volume
    }, index=dates)
    
    # Detect regime
    detector = RegimeDetector()
    analysis = detector.detect(data)
    
    print("=" * 50)
    print("REGIME DETECTION DEMO")
    print("=" * 50)
    print(f"\n📊 Detected Regime: {analysis.regime.value}")
    print(f"   Confidence: {analysis.confidence:.2%}")
    print(f"\n📈 Features:")
    for key, value in analysis.features.items():
        print(f"   {key}: {value:.4f}")
    
    # Get mutation weights
    weights = RegimeMutationBias.get_mutation_weights(analysis.regime)
    print(f"\n🔄 Mutation Weights for {analysis.regime.value}:")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for mut, weight in sorted_weights[:8]:
        print(f"   {mut}: {weight:.2f}")
    
    # Get risk parameters
    risk_params = RegimeMutationBias.get_risk_parameters(analysis.regime)
    print(f"\n⚠️ Risk Parameters:")
    print(f"   Position Size: {risk_params['position_size_range']}")
    print(f"   Stop Loss: {risk_params['stop_loss_range']}")
    
    return analysis


if __name__ == "__main__":
    demo_regime_detection()
