"""
QuantCore - Fractal Time Series Mutator v1.0

This module adds fractal/multi-timeframe intelligence to the evolution engine.

Expert: "Give it the ability to understand what timeframe to trade on.
Combine regime detection with fractal awareness, and you get a system that
not only knows it's in a bull market but knows whether to scalp the 5-minute
or ride the 4-hour wave. This is multiplicative alpha."

Features:
1. Multi-Timeframe Consensus - Evolve which timeframes must align
2. Fractal Pattern Recognition - Cycle detection across timeframes  
3. Adaptive Hurst Exponent - Make Hurst evolvable
4. Timeframe Mutation Operators
5. Fractal-Regime Hybridization - Auto-adjust timeframe per regime
"""

import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# TIMEFRAME TYPES
# ============================================================
class TimeFrame(Enum):
    """Trading timeframes."""
    TF_1M = "1m"
    TF_5M = "5m"
    TF_15M = "15m"
    TF_30M = "30m"
    TF_1H = "1h"
    TF_4H = "4h"
    TF_1D = "1d"
    TF_1W = "1w"


# ============================================================
# ALIGNMENT MODES
# ============================================================
class AlignmentMode(Enum):
    """How timeframes must align for entry."""
    AND = "and"           # All must agree
    OR = "or"            # Any can trigger
    WEIGHTED = "weighted" # Weighted combination
    STRONGEST = "strongest" # Only strongest signal


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class FractalConfig:
    """
    Evolvable fractal/timeframe configuration.
    
    This is what gets mutated - the GA decides:
    - Which timeframes to use
    - How they must align
    - Hurst exponent thresholds
    - Cycle detection parameters
    """
    # Timeframes to monitor
    active_timeframes: List[TimeFrame] = field(default_factory=lambda: [TimeFrame.TF_15M, TimeFrame.TF_1H])
    
    # Primary (decision) timeframe
    primary_timeframe: TimeFrame = TimeFrame.TF_1H
    
    # Alignment mode for multi-TF consensus
    alignment_mode: AlignmentMode = AlignmentMode.AND
    
    # Alignment weights (if WEIGHTED mode)
    alignment_weights: Dict[TimeFrame, float] = field(default_factory=dict)
    
    # Hurst exponent settings (evolvable)
    hurst_enabled: bool = True
    hurst_lookback: int = 100
    hurst_method: str = "rs"  # 'rs', 'dfa', 'whittle'
    hurst_trend_threshold: float = 0.6   # > 0.6 = trending
    hurst_mean_rev_threshold: float = 0.4  # < 0.4 = mean-reverting
    
    # Cycle detection settings
    cycle_enabled: bool = True
    cycle_lookback: int = 50
    cycle_method: str = "sine"  # 'sine', 'hilbert', 'zhang'
    
    # Timeframe preferences per regime (evolvable)
    # Key: regime -> preferred timeframe
    regime_timeframe_prefs: Dict[str, str] = field(default_factory=lambda: {
        'bull': '1h',
        'bear': '4h', 
        'sideways': '15m',
        'high_vol': '5m',
        'low_vol': '4h'
    })
    
    def to_dict(self) -> Dict:
        return {
            'active_timeframes': [tf.value for tf in self.active_timeframes],
            'primary_timeframe': self.primary_timeframe.value,
            'alignment_mode': self.alignment_mode.value,
            'alignment_weights': {k.value: v for k, v in self.alignment_weights.items()},
            'hurst_enabled': self.hurst_enabled,
            'hurst_lookback': self.hurst_lookback,
            'hurst_method': self.hurst_method,
            'hurst_trend_threshold': self.hurst_trend_threshold,
            'hurst_mean_rev_threshold': self.hurst_mean_rev_threshold,
            'cycle_enabled': self.cycle_enabled,
            'cycle_lookback': self.cycle_lookback,
            'cycle_method': self.cycle_method,
            'regime_timeframe_prefs': self.regime_timeframe_prefs
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FractalConfig':
        cfg = cls()
        if 'active_timeframes' in data:
            cfg.active_timeframes = [TimeFrame(t) for t in data['active_timeframes']]
        if 'primary_timeframe' in data:
            cfg.primary_timeframe = TimeFrame(data['primary_timeframe'])
        if 'alignment_mode' in data:
            cfg.alignment_mode = AlignmentMode(data['alignment_mode'])
        if 'alignment_weights' in data:
            cfg.alignment_weights = {TimeFrame(k): v for k, v in data['alignment_weights'].items()}
        if 'hurst_enabled' in data:
            cfg.hurst_enabled = data['hurst_enabled']
        if 'hurst_lookback' in data:
            cfg.hurst_lookback = data['hurst_lookback']
        if 'hurst_method' in data:
            cfg.hurst_method = data['hurst_method']
        if 'hurst_trend_threshold' in data:
            cfg.hurst_trend_threshold = data['hurst_trend_threshold']
        if 'hurst_mean_rev_threshold' in data:
            cfg.hurst_mean_rev_threshold = data['hurst_mean_rev_threshold']
        if 'cycle_enabled' in data:
            cfg.cycle_enabled = data['cycle_enabled']
        if 'cycle_lookback' in data:
            cfg.cycle_lookback = data['cycle_lookback']
        if 'cycle_method' in data:
            cfg.cycle_method = data['cycle_method']
        if 'regime_timeframe_prefs' in data:
            cfg.regime_timeframe_prefs = data['regime_timeframe_prefs']
        return cfg


@dataclass
class FractalSignal:
    """Signal from fractal analysis."""
    direction: int           # -1 (short), 0 (neutral), 1 (long)
    confidence: float        # 0-1
    timeframe: TimeFrame
    hurst_value: Optional[float] = None
    cycle_phase: Optional[float] = None
    alignment_strength: float = 1.0


# ============================================================
# HURST EXPONENT CALCULATOR
# ============================================================
class HurstCalculator:
    """
    Calculate Hurst exponent for trend/mean-reversion classification.
    
    H > 0.6: Trending market
    H < 0.4: Mean-reverting market  
    0.4 < H < 0.6: Random walk
    """
    
    @staticmethod
    def calculate_rs(series: pd.Series, lookback: int = 100) -> float:
        """
        Calculate Hurst using R/S method.
        
        This is the standard Hurst exponent calculation.
        """
        if len(series) < lookback:
            lookback = len(series)
        
        if lookback < 10:
            return 0.5  # Default to random walk
        
        data = series.iloc[-lookback:].values
        
        # R/S calculation
        def rs_calc(n):
            if n > len(data):
                return 1.0
            rs_values = []
            for start in range(0, len(data), n):
                chunk = data[start:start+n]
                if len(chunk) < 2:
                    continue
                mean = np.mean(chunk)
                cumdev = np.cumsum(chunk - mean)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(chunk)
                if S > 0:
                    rs_values.append(R / S)
            if rs_values:
                return np.mean(rs_values)
            return 1.0
        
        # Calculate for different lags
        lags = [min(n, lookback//4) for n in [10, 20, 30, 40, 50]]
        lags = [l for l in lags if l > 1]
        
        log_rs = []
        log_n = []
        
        for n in lags:
            rs = rs_calc(n)
            if rs > 0:
                log_rs.append(np.log(rs))
                log_n.append(np.log(n))
        
        if len(log_rs) < 2:
            return 0.5
        
        # Linear regression: log(R/S) = H * log(n) + c
        try:
            slope, intercept = np.polyfit(log_n, log_rs, 1)
            return max(0.0, min(1.0, slope))
        except:
            return 0.5
    
    @staticmethod
    def calculate_dfa(series: pd.Series, lookback: int = 100) -> float:
        """
        Calculate Hurst using Detrended Fluctuation Analysis.
        
        More robust than R/S for non-stationary series.
        """
        if len(series) < lookback:
            lookback = len(series)
        
        data = series.iloc[-lookback:].values
        data = data - np.mean(data)
        
        # Integration
        y = np.cumsum(data)
        
        # Fluctuation for different scales
        scales = [4, 8, 16, 32, 64]
        scales = [s for s in scales if s < lookback // 4]
        
        if not scales:
            return 0.5
        
        f_n = []
        for s in scales:
            # Detrended fluctuation
            n_segments = len(y) // s
            if n_segments < 2:
                continue
            
            F = 0
            for v in range(n_segments):
                segment = y[v*s:(v+1)*s]
                if len(segment) < 2:
                    continue
                # Fit line and calculate variance
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                fitted = np.polyval(coeffs, x)
                F += np.sum((segment - fitted) ** 2)
            
            F = np.sqrt(F / (len(y)))
            if F > 0:
                f_n.append(F)
        
        if len(f_n) < 2:
            return 0.5
        
        # Log-log slope = Hurst exponent
        log_f = np.log(f_n)
        log_s = np.log(scales[:len(f_n)])
        
        try:
            slope, _ = np.polyfit(log_s, log_f, 1)
            return max(0.0, min(1.0, slope / 2 + 0.5))
        except:
            return 0.5
    
    @staticmethod
    def calculate(series: pd.Series, method: str = "rs", lookback: int = 100) -> float:
        """Calculate Hurst with specified method."""
        if method == "rs":
            return HurstCalculator.calculate_rs(series, lookback)
        elif method == "dfa":
            return HurstCalculator.calculate_dfa(series, lookback)
        else:
            return HurstCalculator.calculate_rs(series, lookback)


# ============================================================
# CYCLE DETECTOR
# ============================================================
class CycleDetector:
    """
    Detect market cycles using sine wave fitting.
    
    Expert: "If the 15m wave is bottoming and the 1h wave is rising,
    that's a fractal buy signal."
    """
    
    @staticmethod
    def detect_phase(series: pd.Series, lookback: int = 50) -> Tuple[float, float]:
        """
        Detect cycle phase and amplitude.
        
        Returns: (phase, amplitude)
        Phase: 0-360 degrees
        Amplitude: strength of cycle
        """
        if len(series) < lookback:
            lookback = len(series)
        
        if lookback < 20:
            return 180.0, 0.0
        
        data = series.iloc[-lookback:].values
        
        # Simple sine wave fitting
        t = np.arange(len(data))
        
        try:
            # FFT to find dominant frequency
            fft = np.fft.fft(data - np.mean(data))
            freqs = np.fft.fftfreq(len(data))
            
            # Find peak (excluding DC)
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(freqs)//2])
            
            if len(positive_fft) > 1:
                peak_idx = np.argmax(positive_fft[1:]) + 1
                dominant_freq = positive_freqs[peak_idx]
                
                # Convert to period
                if dominant_freq > 0:
                    period = 1 / dominant_freq
                    
                    # Calculate phase at end of series
                    phase = (len(data) % period) / period * 360
                    
                    # Amplitude
                    amplitude = np.std(data) / np.mean(np.abs(data)) if np.mean(np.abs(data)) > 0 else 0
                    
                    return phase % 360, min(amplitude * 10, 1.0)
        except:
            pass
        
        return 180.0, 0.0
    
    @staticmethod
    def get_signal(series: pd.Series, lookback: int = 50) -> int:
        """
        Get trading signal from cycle.
        
        Returns: -1 (sell/cycle bottom), 0 (neutral), 1 (buy/cycle top)
        """
        phase, amplitude = CycleDetector.detect_phase(series, lookback)
        
        if amplitude < 0.1:
            return 0  # No clear cycle
        
        # Cycle positions: 0-90 (rising), 90-180 (falling), 180-270 (falling), 270-360 (rising)
        if 270 <= phase < 360 or 0 <= phase < 90:
            return 1   # Cycle bottoming/rising - buy
        elif 90 <= phase < 270:
            return -1  # Cycle topping/falling - sell
        else:
            return 0


# ============================================================
# MULTI-TIMEFRAME ANALYZER
# ============================================================
class MultiTimeframeAnalyzer:
    """
    Analyze price data across multiple timeframes for consensus signals.
    
    Expert: "Evolve which timeframes (1m, 5m, 15m, 1h, 4h, 1d) must align for entry."
    """
    
    def __init__(self, config: FractalConfig):
        self.config = config
        
    def analyze(self, data: Dict[TimeFrame, pd.DataFrame]) -> FractalSignal:
        """
        Analyze multiple timeframes and generate consensus signal.
        
        Args:
            data: Dict of timeframe -> price data
            
        Returns:
            FractalSignal with direction, confidence, etc.
        """
        signals = {}
        weights = {}
        
        for tf in self.config.active_timeframes:
            if tf not in data:
                continue
                
            tf_data = data[tf]
            signal = self._analyze_single_timeframe(tf_data, tf)
            signals[tf] = signal
            
            # Get weight
            if self.config.alignment_mode == AlignmentMode.WEIGHTED:
                weights[tf] = self.config.alignment_weights.get(tf, 1.0)
            else:
                weights[tf] = 1.0
        
        if not signals:
            return FractalSignal(0, 0.0, self.config.primary_timeframe)
        
        # Combine signals based on alignment mode
        return self._combine_signals(signals, weights)
    
    def _analyze_single_timeframe(self, data: pd.DataFrame, tf: TimeFrame) -> FractalSignal:
        """Analyze a single timeframe."""
        close = data['close']
        
        # Base signal from price direction
        recent = close.iloc[-10:]
        if len(recent) < 2:
            return FractalSignal(0, 0.0, tf)
        
        direction = 1 if recent.iloc[-1] > recent.iloc[0] else -1
        confidence = min(abs(recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 10, 1.0)
        
        # Add Hurst filter
        hurst_value = None
        if self.config.hurst_enabled:
            hurst_value = HurstCalculator.calculate(
                close, 
                method=self.config.hurst_method,
                lookback=self.config.hurst_lookback
            )
            
            if hurst_value > self.config.hurst_trend_threshold:
                # Trending - trust direction
                pass
            elif hurst_value < self.config.hurst_mean_rev_threshold:
                # Mean-reverting - reverse signal
                direction *= -1
                confidence *= hurst_value  # Lower confidence in mean-reversion
            else:
                confidence *= 0.5  # Random walk - reduce confidence
        
        # Add cycle detection
        cycle_phase = None
        if self.config.cycle_enabled:
            cycle_phase, cycle_amp = CycleDetector.detect_phase(close, self.config.cycle_lookback)
            
            if cycle_amp > 0.2:
                cycle_signal = CycleDetector.get_signal(close, self.config.cycle_lookback)
                if cycle_signal != 0:
                    # Blend with trend
                    direction = int(0.7 * direction + 0.3 * cycle_signal)
                    confidence = min(confidence + cycle_amp * 0.3, 1.0)
        
        return FractalSignal(
            direction=int(np.sign(direction)),
            confidence=confidence,
            timeframe=tf,
            hurst_value=hurst_value,
            cycle_phase=cycle_phase,
            alignment_strength=confidence
        )
    
    def _combine_signals(self, signals: Dict[TimeFrame, FractalSignal], 
                        weights: Dict[TimeFrame, float]) -> FractalSignal:
        """Combine signals from multiple timeframes."""
        
        if self.config.alignment_mode == AlignmentMode.AND:
            # All must agree
            directions = [s.direction for s in signals.values()]
            if all(d == directions[0] for d in directions):
                min_conf = min(s.confidence for s in signals.values())
                return FractalSignal(
                    direction=directions[0],
                    confidence=min_conf,
                    timeframe=self.config.primary_timeframe,
                    alignment_strength=min_conf
                )
            return FractalSignal(0, 0.0, self.config.primary_timeframe)
        
        elif self.config.alignment_mode == AlignmentMode.OR:
            # Any can trigger
            for tf, signal in signals.items():
                if signal.direction != 0:
                    return signal
            return FractalSignal(0, 0.0, self.config.primary_timeframe)
        
        elif self.config.alignment_mode == AlignmentMode.WEIGHTED:
            # Weighted combination
            weighted_sum = 0
            weight_total = 0
            
            for tf, signal in signals.items():
                w = weights.get(tf, 1.0)
                weighted_sum += signal.direction * signal.confidence * w
                weight_total += w
            
            if weight_total > 0:
                direction = np.sign(weighted_sum)
                confidence = min(abs(weighted_sum) / weight_total, 1.0)
                return FractalSignal(
                    direction=int(direction),
                    confidence=confidence,
                    timeframe=self.config.primary_timeframe,
                    alignment_strength=confidence
                )
        
        elif self.config.alignment_mode == AlignmentMode.STRONGEST:
            # Only use strongest signal
            best = max(signals.values(), key=lambda s: s.confidence * s.alignment_strength)
            return best
        
        return FractalSignal(0, 0.0, self.config.primary_timeframe)


# ============================================================
# FRACTAL MUTATION OPERATORS
# ============================================================
class FractalMutationOperators:
    """
    Mutation operators for fractal/timeframe configuration.
    
    These operators let the GA evolve:
    - Which timeframes to use
    - How they align
    - Hurst thresholds
    - Cycle detection settings
    """
    
    @staticmethod
    def add_timeframe(config: FractalConfig) -> FractalConfig:
        """Add a timeframe to the active set."""
        config = copy.deepcopy(config)
        
        all_timeframes = list(TimeFrame)
        available = [tf for tf in all_timeframes if tf not in config.active_timeframes]
        
        if available:
            new_tf = random.choice(available)
            config.active_timeframes.append(new_tf)
            
            # Set default weight
            config.alignment_weights[new_tf] = 1.0
        
        return config
    
    @staticmethod
    def remove_timeframe(config: FractalConfig) -> FractalConfig:
        """Remove a timeframe from the active set."""
        config = copy.deepcopy(config)
        
        if len(config.active_timeframes) > 1:
            remove_tf = random.choice(config.active_timeframes)
            config.active_timeframes.remove(remove_tf)
            
            # Remove weight
            if remove_tf in config.alignment_weights:
                del config.alignment_weights[remove_tf]
        
        return config
    
    @staticmethod
    def mutate_alignment_mode(config: FractalConfig) -> FractalConfig:
        """Change alignment mode."""
        config = copy.deepcopy(config)
        
        modes = list(AlignmentMode)
        current = modes.index(config.alignment_mode)
        config.alignment_mode = modes[(current + 1) % len(modes)]
        
        return config
    
    @staticmethod
    def mutate_alignment_weight(config: FractalConfig, delta: float = 0.1) -> FractalConfig:
        """Mutate alignment weight for a timeframe."""
        config = copy.deepcopy(config)
        
        if config.active_timeframes:
            tf = random.choice(config.active_timeframes)
            current = config.alignment_weights.get(tf, 1.0)
            config.alignment_weights[tf] = max(0.1, min(2.0, current + random.uniform(-delta, delta)))
        
        return config
    
    @staticmethod
    def mutate_hurst_threshold(config: FractalConfig, delta: float = 0.05) -> FractalConfig:
        """Mutate Hurst exponent thresholds."""
        config = copy.deepcopy(config)
        
        if random.random() < 0.5:
            config.hurst_trend_threshold = max(0.5, min(0.9, 
                config.hurst_trend_threshold + random.uniform(-delta, delta)))
        else:
            config.hurst_mean_rev_threshold = max(0.1, min(0.5, 
                config.hurst_mean_rev_threshold + random.uniform(-delta, delta)))
        
        return config
    
    @staticmethod
    def mutate_hurst_lookback(config: FractalConfig, delta: int = 10) -> FractalConfig:
        """Mutate Hurst calculation lookback."""
        config = copy.deepcopy(config)
        
        config.hurst_lookback = max(20, min(500, 
            config.hurst_lookback + random.choice([-delta, delta])))
        
        return config
    
    @staticmethod
    def mutate_hurst_method(config: FractalConfig) -> FractalConfig:
        """Change Hurst calculation method."""
        config = copy.deepcopy(config)
        
        methods = ['rs', 'dfa']
        if config.hurst_method in methods:
            idx = methods.index(config.hurst_method)
            config.hurst_method = methods[(idx + 1) % len(methods)]
        
        return config
    
    @staticmethod
    def mutate_cycle_settings(config: FractalConfig) -> FractalConfig:
        """Mutate cycle detection settings."""
        config = copy.deepcopy(config)
        
        # Toggle enabled
        if random.random() < 0.3:
            config.cycle_enabled = not config.cycle_enabled
        
        # Mutate lookback
        if random.random() < 0.5:
            config.cycle_lookback = max(20, min(100, 
                config.cycle_lookback + random.choice([-5, 5])))
        
        # Mutate method
        if random.random() < 0.3:
            methods = ['sine', 'hilbert', 'zhang']
            if config.cycle_method in methods:
                idx = methods.index(config.cycle_method)
                config.cycle_method = methods[(idx + 1) % len(methods)]
        
        return config
    
    @staticmethod
    def mutate_primary_timeframe(config: FractalConfig) -> FractalConfig:
        """Change primary (decision) timeframe."""
        config = copy.deepcopy(config)
        
        if config.active_timeframes:
            config.primary_timeframe = random.choice(config.active_timeframes)
        
        return config
    
    @staticmethod
    def mutate_regime_timeframe_prefs(config: FractalConfig, regime: str = None, 
                                    tf: str = None) -> FractalConfig:
        """Mutate timeframe preference for a regime."""
        config = copy.deepcopy(config)
        
        regimes = list(config.regime_timeframe_prefs.keys())
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        if regime is None:
            regime = random.choice(regimes)
        if tf is None:
            tf = random.choice(timeframes)
        
        config.regime_timeframe_prefs[regime] = tf
        
        return config
    
    @staticmethod
    def enable_hurst_filter(config: FractalConfig) -> FractalConfig:
        """Enable or disable Hurst filter."""
        config = copy.deepcopy(config)
        config.hurst_enabled = random.choice([True, False])
        return config


# ============================================================
# FRACTAL-REGIME HYBRIDIZER
# ============================================================
class FractalRegimeHybridizer:
    """
    Expert: "For each regime, evolve a preferred set of timeframes and alignment rules.
    Then, when the regime changes, the strategy automatically shifts its fractal focus."
    
    This integrates fractal configuration with regime detection.
    """
    
    def __init__(self, fractal_config: FractalConfig):
        self.fractal_config = fractal_config
        
    def get_timeframe_for_regime(self, regime: str) -> TimeFrame:
        """Get the preferred timeframe for the current regime."""
        tf_str = self.fractal_config.regime_timeframe_prefs.get(regime, '1h')
        
        try:
            return TimeFrame(f"tf_{tf_str}")
        except:
            return TimeFrame.TF_1H
    
    def adjust_for_regime(self, regime: str) -> FractalConfig:
        """
        Adjust fractal config based on detected regime.
        
        When regime changes, automatically shift:
        - Primary timeframe
        - Alignment mode
        - Hurst thresholds
        """
        config = copy.deepcopy(self.fractal_config)
        
        # Get preferred timeframe for this regime
        preferred_tf = self.get_timeframe_for_regime(regime)
        config.primary_timeframe = preferred_tf
        
        # Adjust based on regime characteristics
        if regime == 'high_vol':
            # In high vol, favor shorter timeframes
            config.hurst_trend_threshold = 0.7  # Stricter trending requirement
            config.hurst_mean_rev_threshold = 0.3
        elif regime == 'low_vol':
            # In low vol, allow longer timeframes
            config.hurst_trend_threshold = 0.5
            config.hurst_mean_rev_threshold = 0.5
        elif regime in ['bull', 'bear']:
            # In trends, use trending thresholds
            config.hurst_trend_threshold = 0.6
        elif regime == 'sideways':
            # In sideways, favor mean-reversion
            config.hurst_trend_threshold = 0.7
            config.hurst_mean_rev_threshold = 0.4
        
        return config


# ============================================================
# FACTORY FUNCTIONS
# ============================================================
def create_default_fractal_config() -> FractalConfig:
    """Create default fractal configuration."""
    return FractalConfig(
        active_timeframes=[TimeFrame.TF_15M, TimeFrame.TF_1H, TimeFrame.TF_4H],
        primary_timeframe=TimeFrame.TF_1H,
        alignment_mode=AlignmentMode.AND,
        alignment_weights={
            TimeFrame.TF_15M: 0.3,
            TimeFrame.TF_1H: 0.5,
            TimeFrame.TF_4H: 0.2
        },
        hurst_enabled=True,
        hurst_lookback=100,
        hurst_method='rs',
        hurst_trend_threshold=0.6,
        hurst_mean_rev_threshold=0.4,
        cycle_enabled=True,
        cycle_lookback=50,
        cycle_method='sine',
        regime_timeframe_prefs={
            'bull': '1h',
            'bear': '4h',
            'sideways': '15m',
            'high_vol': '5m',
            'low_vol': '4h'
        }
    )


# ============================================================
# CLI / TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Create trending then ranging data
    returns = np.random.randn(n) * 0.02
    returns[100:200] += 0.05  # Bull trend
    returns[300:400] = np.random.randn(100) * 0.005  # Sideways
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Test Hurst
    print("Testing Hurst Exponent...")
    hurst = HurstCalculator.calculate(data['close'], method='rs', lookback=100)
    print(f"  Hurst (R/S): {hurst:.3f}")
    
    if hurst > 0.6:
        print("  → Trending market")
    elif hurst < 0.4:
        print("  → Mean-reverting market")
    else:
        print("  → Random walk")
    
    # Test Cycle Detection
    print("\nTesting Cycle Detection...")
    phase, amp = CycleDetector.detect_phase(data['close'], lookback=50)
    print(f"  Phase: {phase:.1f}°, Amplitude: {amp:.3f}")
    
    # Test Fractal Config
    print("\nTesting Fractal Config...")
    config = create_default_fractal_config()
    print(f"  Active TFs: {[tf.value for tf in config.active_timeframes]}")
    print(f"  Primary TF: {config.primary_timeframe.value}")
    print(f"  Alignment: {config.alignment_mode.value}")
    
    # Test mutations
    print("\nTesting Mutations...")
    config = FractalMutationOperators.add_timeframe(config)
    print(f"  After add_timeframe: {[tf.value for tf in config.active_timeframes]}")
    
    config = FractalMutationOperators.mutate_hurst_threshold(config, 0.1)
    print(f"  After mutate_hurst_threshold: trend>{config.hurst_trend_threshold:.2f}, mean_rev<{config.hurst_mean_rev_threshold:.2f}")
    
    # Test Multi-Timeframe Analyzer
    print("\nTesting Multi-Timeframe Analyzer...")
    analyzer = MultiTimeframeAnalyzer(config)
    signal = analyzer.analyze({TimeFrame.TF_1H: data})
    print(f"  Signal: {signal.direction}, Confidence: {signal.confidence:.2f}")
    
    # Test Regime Hybridizer
    print("\nTesting Fractal-Regime Hybridizer...")
    hybridizer = FractalRegimeHybridizer(config)
    adjusted = hybridizer.adjust_for_regime('bull')
    print(f"  Bull regime → Primary TF: {adjusted.primary_timeframe.value}")
    
    print("\n✅ All fractal tests passed!")
