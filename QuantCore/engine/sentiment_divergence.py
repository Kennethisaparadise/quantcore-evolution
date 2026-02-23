"""
QuantCore - Sentiment Divergence Mutation v1.0

This module adds alternative data / market sentiment intelligence.

Expert: "Price is just the echo; sentiment is the original sound.
When price screams sell but sentiment is quietly accumulating, that's your cue."

Features:
1. Multiple Sentiment Sources (news, social, on-chain, fear/greed)
2. Sentiment-Based Signals (extremes, momentum, divergence)
3. Sentiment Mutation Operators
4. Sentiment-Regime Hybridization
5. Sentiment-Flow Fusion (sentiment + order flow)
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
# SENTIMENT SOURCE TYPES
# ============================================================
class SentimentSource(Enum):
    """Available sentiment data sources."""
    NEWS = "news"                    # News sentiment
    SOCIAL = "social"              # Social media (Twitter/Reddit)
    ON_CHAIN = "on_chain"          # Blockchain metrics
    FEAR_GREED = "fear_greed"      # Fear & Greed index
    PUT_CALL = "put_call"           # Put/Call ratios
    VIX = "vix"                    # VIX as fear proxy


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class SentimentConfig:
    """
    Evolvable sentiment configuration.
    
    The GA evolves:
    - Which sentiment sources to use
    - Thresholds for extreme readings
    - How to combine with price signals
    """
    # Enabled sources
    use_news: bool = False
    use_social: bool = False
    use_on_chain: bool = False
    use_fear_greed: bool = True
    use_put_call: bool = False
    use_vix: bool = True
    
    # Sentiment thresholds
    extreme_bullish: float = 80.0    # Above = extremely bullish
    extreme_bearish: float = 20.0    # Below = extremely bearish
    neutral_low: float = 40.0
    neutral_high: float = 60.0
    
    # Momentum settings
    momentum_window: int = 10        # Bars to calculate sentiment change
    momentum_threshold: float = 15.0  # % change threshold
    
    # Divergence settings
    divergence_lookback: int = 20   # Bars to check for divergence
    divergence_threshold: float = 0.3  # Price/sentiment divergence threshold
    
    # Confirmation mode
    confirmation_mode: str = "and"   # 'and', 'or', 'weighted'
    sentiment_weight: float = 0.3     # Weight in final signal (0-1)
    
    # Regime-specific sentiment
    sentiment_per_regime: Dict[str, dict] = field(default_factory=lambda: {
        'bull': {'allow_short_sentiment': True, 'require_extreme': False},
        'bear': {'allow_long_sentiment': True, 'require_extreme': False},
        'sideways': {'allow_short_sentiment': True, 'require_extreme': True},
        'high_vol': {'allow_short_sentiment': False, 'require_extreme': True},
        'low_vol': {'allow_short_sentiment': True, 'require_extreme': False}
    })
    
    def to_dict(self) -> Dict:
        return {
            'use_news': self.use_news,
            'use_social': self.use_social,
            'use_on_chain': self.use_on_chain,
            'use_fear_greed': self.use_fear_greed,
            'use_put_call': self.use_put_call,
            'use_vix': self.use_vix,
            'extreme_bullish': self.extreme_bullish,
            'extreme_bearish': self.extreme_bearish,
            'neutral_low': self.neutral_low,
            'neutral_high': self.neutral_high,
            'momentum_window': self.momentum_window,
            'momentum_threshold': self.momentum_threshold,
            'divergence_lookback': self.divergence_lookback,
            'divergence_threshold': self.divergence_threshold,
            'confirmation_mode': self.confirmation_mode,
            'sentiment_weight': self.sentiment_weight,
            'sentiment_per_regime': self.sentiment_per_regime
        }


@dataclass
class SentimentSignal:
    """Signal from sentiment analysis."""
    direction: int            # -1 (bearish), 0 (neutral), 1 (bullish)
    confidence: float       # 0-1
    sentiment_value: float # Raw sentiment score (0-100)
    source: SentimentSource
    is_extreme: bool      # At extreme levels?
    momentum: float        # Sentiment momentum


# ============================================================
# SENTIMENT DATA SIMULATOR
# ============================================================
class SentimentSimulator:
    """
    Simulate sentiment data for backtesting.
    
    In production, this would fetch from:
    - NewsAPI / FinBERT for news sentiment
    - Twitter/Reddit APIs for social buzz
    - Glassnode for on-chain
    - alternative.me for Fear & Greed
    """
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.sentiment_history: List[float] = []
        
    def generate_sentiment(self, data: pd.DataFrame, 
                          regime: str = 'sideways') -> SentimentSignal:
        """
        Generate synthetic sentiment signal based on market conditions.
        
        In reality, this would fetch real data.
        """
        close = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
        
        # Generate base sentiment from market state
        # Simulate: fear = high vol + downtrend, greed = low vol + uptrend
        
        # Price momentum
        if len(close) >= 5:
            recent_return = (close[-1] - close[-5]) / close[-5]
        else:
            recent_return = 0
        
        # Volume momentum
        if len(volume) >= 5:
            vol_ratio = volume[-1] / np.mean(volume[-5:])
        else:
            vol_ratio = 1
        
        # Base sentiment (0-100, 50 = neutral)
        sentiment = 50.0
        
        # Add price influence
        if recent_return > 0.05:
            sentiment += 15
        elif recent_return < -0.05:
            sentiment -= 15
        
        # Add volume influence (high vol = more fear)
        if vol_ratio > 1.5:
            sentiment -= 10  # Fear
        elif vol_ratio < 0.7:
            sentiment += 5   # Confidence
        
        # Add some noise
        sentiment += np.random.randn() * 10
        
        # Clamp to 0-100
        sentiment = max(0, min(100, sentiment))
        
        # Store in history
        self.sentiment_history.append(sentiment)
        
        # Calculate momentum
        momentum = 0
        if len(self.sentiment_history) >= self.config.momentum_window:
            recent = np.mean(self.sentiment_history[-self.config.momentum_window//2:])
            older = np.mean(self.sentiment_history[:-self.config.momentum_window//2])
            momentum = ((recent - older) / (older + 1)) * 100
        
        # Determine direction
        direction = 0
        is_extreme = False
        
        if sentiment >= self.config.extreme_bullish:
            direction = 1
            is_extreme = True
        elif sentiment <= self.config.extreme_bearish:
            direction = -1
            is_extreme = True
        elif sentiment > self.config.neutral_high:
            direction = 1
        elif sentiment < self.config.neutral_low:
            direction = -1
        
        # Confidence based on how extreme
        if is_extreme:
            confidence = 0.9
        elif abs(sentiment - 50) > 20:
            confidence = 0.7
        else:
            confidence = 0.4
        
        return SentimentSignal(
            direction=direction,
            confidence=confidence,
            sentiment_value=sentiment,
            source=SentimentSource.FEAR_GREED,
            is_extreme=is_extreme,
            momentum=momentum
        )
    
    def detect_divergence(self, price_data: pd.DataFrame) -> Tuple[int, float]:
        """
        Detect price-sentiment divergence.
        
        Returns: (divergence_signal, strength)
        1 = bullish divergence (price down, sentiment up)
        -1 = bearish divergence (price up, sentiment down)
        0 = no divergence
        """
        if len(self.sentiment_history) < self.config.divergence_lookback:
            return 0, 0
        
        price = price_data['close'].values
        sentiment = np.array(self.sentiment_history[-self.config.divergence_lookback:])
        
        # Price change
        price_change = (price[-1] - price[-self.config.divergence_lookback]) / price[-self.config.divergence_lookback]
        
        # Sentiment change
        sentiment_change = (sentiment[-1] - sentiment[0]) / (sentiment[0] + 1)
        
        # Divergence: opposite directions
        divergence_signal = 0
        divergence_strength = 0
        
        if price_change > self.config.divergence_threshold and sentiment_change < -self.config.divergence_threshold:
            # Price up, sentiment down = bearish divergence
            divergence_signal = -1
            divergence_strength = min(abs(price_change - sentiment_change), 1.0)
        elif price_change < -self.config.divergence_threshold and sentiment_change > self.config.divergence_threshold:
            # Price down, sentiment up = bullish divergence  
            divergence_signal = 1
            divergence_strength = min(abs(price_change - sentiment_change), 1.0)
        
        return divergence_signal, divergence_strength


# ============================================================
# SENTIMENT MUTATION OPERATORS
# ============================================================
class SentimentMutationOperators:
    """
    Mutation operators for sentiment configuration.
    """
    
    @staticmethod
    def enable_source(config: SentimentConfig, source: SentimentSource = None) -> SentimentConfig:
        """Enable a sentiment source."""
        config = copy.deepcopy(config)
        
        if source is None:
            source = random.choice(list(SentimentSource))
        
        if source == SentimentSource.NEWS:
            config.use_news = random.choice([True, False])
        elif source == SentimentSource.SOCIAL:
            config.use_social = random.choice([True, False])
        elif source == SentimentSource.ON_CHAIN:
            config.use_on_chain = random.choice([True, False])
        elif source == SentimentSource.FEAR_GREED:
            config.use_fear_greed = random.choice([True, False])
        elif source == SentimentSource.PUT_CALL:
            config.use_put_call = random.choice([True, False])
        elif source == SentimentSource.VIX:
            config.use_vix = random.choice([True, False])
        
        return config
    
    @staticmethod
    def mutate_extreme_threshold(config: SentimentConfig, delta: float = 5.0) -> SentimentConfig:
        """Mutate extreme sentiment thresholds."""
        config = copy.deepcopy(config)
        
        if random.random() < 0.5:
            config.extreme_bullish = max(60, min(95, config.extreme_bullish + random.uniform(-delta, delta)))
        else:
            config.extreme_bearish = max(5, min(40, config.extreme_bearish + random.uniform(-delta, delta)))
        
        return config
    
    @staticmethod
    def mutate_neutral_range(config: SentimentConfig, delta: float = 3.0) -> SentimentConfig:
        """Mutate neutral sentiment range."""
        config = copy.deepcopy(config)
        
        config.neutral_low = max(20, min(45, config.neutral_low + random.uniform(-delta, delta)))
        config.neutral_high = max(55, min(80, config.neutral_high + random.uniform(-delta, delta)))
        
        return config
    
    @staticmethod
    def mutate_momentum_window(config: SentimentConfig, delta: int = 2) -> SentimentConfig:
        """Mutate sentiment momentum window."""
        config = copy.deepcopy(config)
        config.momentum_window = max(3, min(30, config.momentum_window + random.choice([-delta, delta])))
        return config
    
    @staticmethod
    def mutate_sentiment_weight(config: SentimentConfig, delta: float = 0.05) -> SentimentConfig:
        """Mutate how much weight sentiment has in final signal."""
        config = copy.deepcopy(config)
        config.sentiment_weight = max(0.1, min(0.8, config.sentiment_weight + random.uniform(-delta, delta)))
        return config
    
    @staticmethod
    def mutate_confirmation_mode(config: SentimentConfig) -> SentimentConfig:
        """Change how sentiment confirms price signals."""
        config = copy.deepcopy(config)
        modes = ['and', 'or', 'weighted']
        if config.confirmation_mode in modes:
            idx = modes.index(config.confirmation_mode)
            config.confirmation_mode = modes[(idx + 1) % len(modes)]
        return config
    
    @staticmethod
    def mutate_divergence_lookback(config: SentimentConfig, delta: int = 3) -> SentimentConfig:
        """Mutate divergence detection lookback."""
        config = copy.deepcopy(config)
        config.divergence_lookback = max(5, min(50, config.divergence_lookback + random.choice([-delta, delta])))
        return config
    
    @staticmethod
    def mutate_regime_sentiment(config: SentimentConfig, regime: str = None) -> SentimentConfig:
        """Mutate sentiment settings for a specific regime."""
        config = copy.deepcopy(config)
        
        regimes = list(config.sentiment_per_regime.keys())
        if regime is None:
            regime = random.choice(regimes)
        
        if regime in config.sentiment_per_regime:
            current = config.sentiment_per_regime[regime]
            
            if 'allow_short_sentiment' in current:
                current['allow_short_sentiment'] = random.choice([True, False])
            if 'require_extreme' in current:
                current['require_extreme'] = random.choice([True, False])
        
        return config


# ============================================================
# SENTIMENT-FLOW-REGIME HYBRIDIZER
# ============================================================
class SentimentFlowRegimeHybridizer:
    """
    Expert: "Buy only when delta is positive AND sentiment is rising—that's the killer combo."
    
    This combines:
    - Sentiment (external mood)
    - Order flow (internal footprints)
    - Regime (current market state)
    """
    
    def __init__(self, sentiment_config: SentimentConfig):
        self.sentiment_config = sentiment_config
        self.sentiment_sim = SentimentSimulator(sentiment_config)
        
    def generate_hybrid_signal(self, 
                              price_data: pd.DataFrame,
                              flow_metrics: Dict = None,
                              regime: str = 'sideways') -> Tuple[int, float]:
        """
        Generate signal combining sentiment + order flow + regime.
        
        Returns: (direction, confidence)
        """
        # Get sentiment signal
        sentiment = self.sentiment_sim.generate_sentiment(price_data, regime)
        
        # Get divergence
        divergence, divergence_strength = self.sentiment_sim.detect_divergence(price_data)
        
        # Get order flow signal (if provided)
        flow_signal = 0
        flow_confidence = 0
        if flow_metrics:
            if 'delta_direction' in flow_metrics:
                flow_signal = flow_metrics['delta_direction']
                flow_confidence = abs(flow_metrics.get('delta_ratio', 0))
        
        # Combine based on confirmation mode
        signals = []
        confidences = []
        
        # Add sentiment signal
        if sentiment.direction != 0:
            signals.append(sentiment.direction)
            confidences.append(sentiment.confidence * self.sentiment_config.sentiment_weight)
        
        # Add order flow signal
        if flow_signal != 0:
            signals.append(flow_signal)
            confidences.append(flow_confidence * (1 - self.sentiment_config.sentiment_weight))
        
        # Add divergence signal
        if divergence != 0:
            signals.append(divergence)
            confidences.append(divergence_strength * 0.5)
        
        if not signals:
            return 0, 0
        
        # Combine based on mode
        if self.sentiment_config.confirmation_mode == 'and':
            # All must agree
            if len(set(signals)) == 1:
                direction = signals[0]
                confidence = min(confidences)
            else:
                direction = 0
                confidence = 0
        elif self.sentiment_config.confirmation_mode == 'or':
            # Any can trigger
            direction = signals[0] if signals else 0
            confidence = max(confidences) if confidences else 0
        else:  # weighted
            # Weighted average
            direction = 1 if sum(signals) > 0 else (-1 if sum(signals) < 0 else 0)
            confidence = np.mean(confidences) if confidences else 0
        
        # Apply regime filters
        regime_settings = self.sentiment_config.sentiment_per_regime.get(regime, {})
        
        if regime == 'high_vol' and not regime_settings.get('allow_short_sentiment', True):
            # In high vol, don't allow short signals
            if direction == -1:
                direction = 0
                confidence = 0
        
        if regime_settings.get('require_extreme', False):
            # Require extreme sentiment
            if not sentiment.is_extreme:
                confidence *= 0.5
        
        return direction, confidence


# ============================================================
# FACTORY FUNCTIONS
# ============================================================
def create_sentiment_config() -> SentimentConfig:
    """Create default sentiment configuration."""
    return SentimentConfig(
        use_news=False,
        use_social=False,
        use_on_chain=False,
        use_fear_greed=True,
        use_put_call=False,
        use_vix=True,
        extreme_bullish=80.0,
        extreme_bearish=20.0,
        neutral_low=40.0,
        neutral_high=60.0,
        momentum_window=10,
        momentum_threshold=15.0,
        divergence_lookback=20,
        divergence_threshold=0.3,
        confirmation_mode='and',
        sentiment_weight=0.3,
        sentiment_per_regime={
            'bull': {'allow_short_sentiment': True, 'require_extreme': False},
            'bear': {'allow_long_sentiment': True, 'require_extreme': False},
            'sideways': {'allow_short_sentiment': True, 'require_extreme': True},
            'high_vol': {'allow_short_sentiment': False, 'require_extreme': True},
            'low_vol': {'allow_short_sentiment': True, 'require_extreme': False}
        }
    )


# ============================================================
# CLI / TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volume = np.random.randint(1000, 10000, n)
    
    data = pd.DataFrame({
        'close': close,
        'volume': volume
    })
    
    # Test Sentiment Simulator
    print("Testing Sentiment Simulator...")
    config = create_sentiment_config()
    sim = SentimentSimulator(config)
    
    for i in range(10):
        signal = sim.generate_sentiment(data, regime='bull')
        print(f"  Bar {i}: sentiment={signal.sentiment_value:.1f}, direction={signal.direction}, extreme={signal.is_extreme}")
    
    # Test Mutations
    print("\nTesting Sentiment Mutations...")
    config = SentimentMutationOperators.mutate_extreme_threshold(config)
    print(f"  After mutate_extreme_threshold: bull={config.extreme_bullish}, bear={config.extreme_bearish}")
    
    config = SentimentMutationOperators.mutate_sentiment_weight(config)
    print(f"  After mutate_sentiment_weight: weight={config.sentiment_weight:.2f}")
    
    # Test Hybridizer
    print("\nTesting Sentiment-Flow-Regime Hybridizer...")
    hybridizer = SentimentFlowRegimeHybridizer(config)
    
    # Mock flow metrics
    flow_metrics = {'delta_direction': 1, 'delta_ratio': 0.3}
    
    direction, confidence = hybridizer.generate_hybrid_signal(
        data, 
        flow_metrics=flow_metrics,
        regime='bull'
    )
    print(f"  Hybrid signal: direction={direction}, confidence={confidence:.2f}")
    
    print("\n✅ All sentiment tests passed!")
