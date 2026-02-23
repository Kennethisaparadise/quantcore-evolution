"""
QuantCore - Regime-Switching Meta-Mutation Engine v1.0

This module implements the Tier 1 feature: Regime-Switching Meta-Mutation.

Key Components:
1. MetaStrategy: Container for multiple strategies + regime mapping rules
2. RegimeMapping: Which strategy to use per detected regime
3. SwitchConfig: Parameters controlling regime transitions
4. MetaEvolution: GA that evolves meta-strategies (not just single strategies)

The GA now evolves:
- Which strategies to include in the pool
- Which regime each strategy is assigned to
- When to switch (confidence thresholds, hold times)
- How to switch (hard, smooth, blend)
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

# Import existing regime detection
import sys
sys.path.insert(0, '/home/kenner/clawd/QuantCore')
from engine.regime_detector import RegimeDetector, Regime, RegimeAnalysis

logger = logging.getLogger(__name__)


# ============================================================
# REGIME TYPES (Extended)
# ============================================================
class MetaRegime(Enum):
    """Regime types for meta-strategy evolution."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    TRENDING = "trending"
    RANGING = "ranging"


# ============================================================
# TRANSITION MODES
# ============================================================
class TransitionMode(Enum):
    """How to transition between strategies."""
    HARD = "hard"           # Immediate switch
    SMOOTH = "smooth"       # Blend over N bars
    BLEND = "blend"         # Weighted combination


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class SwitchConfig:
    """Configuration for regime switching."""
    min_hold_time: int = 5           # Min bars to hold before switching
    reentry_cooldown: int = 3         # Bars to wait before re-entering
    transition_mode: TransitionMode = TransitionMode.HARD
    hysteresis: float = 0.1          # Buffer to prevent flip-flopping
    confidence_threshold: float = 0.6  # Min confidence to switch
    
    def to_dict(self) -> Dict:
        return {
            'min_hold_time': self.min_hold_time,
            'reentry_cooldown': self.reentry_cooldown,
            'transition_mode': self.transition_mode.value,
            'hysteresis': self.hysteresis,
            'confidence_threshold': self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SwitchConfig':
        return cls(
            min_hold_time=data.get('min_hold_time', 5),
            reentry_cooldown=data.get('reentry_cooldown', 3),
            transition_mode=TransitionMode(data.get('transition_mode', 'hard')),
            hysteresis=data.get('hysteresis', 0.1),
            confidence_threshold=data.get('confidence_threshold', 0.6)
        )


@dataclass
class RegimeMapping:
    """Maps a regime to a specific strategy."""
    regime: MetaRegime
    strategy_id: str
    confidence_threshold: float = 0.6
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'regime': self.regime.value,
            'strategy_id': self.strategy_id,
            'confidence_threshold': self.confidence_threshold,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RegimeMapping':
        return cls(
            regime=MetaRegime(data.get('regime', 'sideways')),
            strategy_id=data.get('strategy_id', ''),
            confidence_threshold=data.get('confidence_threshold', 0.6),
            enabled=data.get('enabled', True)
        )


@dataclass 
class StrategyGene:
    """A single strategy in the meta-strategy pool."""
    id: str
    name: str
    strategy_type: str           # 'trend', 'mean_revert', 'momentum', 'breakout'
    params: Dict[str, Any]
    fitness: float = 0.0
    regime_performance: Dict[MetaRegime, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'strategy_type': self.strategy_type,
            'params': self.params,
            'fitness': self.fitness,
            'regime_performance': {k.value: v for k, v in self.regime_performance.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyGene':
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            strategy_type=data.get('strategy_type', 'trend'),
            params=data.get('params', {}),
            fitness=data.get('fitness', 0.0),
            regime_performance={MetaRegime(k): v for k, v in data.get('regime_performance', {}).items()}
        )


# ============================================================
# EVOLVABLE REGIME DETECTION CONFIG (Must be before MetaStrategy)
# ============================================================
@dataclass
class EvolvableRegimeConfig:
    """Evolved config for regime detection."""
    n_regimes: int = 4
    method: str = "hmm"
    volatility_weight: float = 1.0
    trend_weight: float = 1.0
    volume_weight: float = 0.5
    correlation_weight: float = 0.5
    momentum_weight: float = 0.8
    vol_high_threshold: float = 0.8
    vol_low_threshold: float = 0.2
    trend_strong_threshold: float = 0.7
    smoothing_window: int = 20
    
    def to_dict(self) -> Dict:
        return {
            'n_regimes': self.n_regimes, 'method': self.method,
            'volatility_weight': self.volatility_weight, 'trend_weight': self.trend_weight,
            'volume_weight': self.volume_weight, 'correlation_weight': self.correlation_weight,
            'momentum_weight': self.momentum_weight, 'vol_high_threshold': self.vol_high_threshold,
            'vol_low_threshold': self.vol_low_threshold, 'trend_strong_threshold': self.trend_strong_threshold,
            'smoothing_window': self.smoothing_window
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvolvableRegimeConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TransitionCostMatrix:
    """Evolved transition costs between regimes."""
    bull_to_bear: float = 0.15
    bull_to_sideways: float = 0.05
    bear_to_bull: float = 0.15
    bear_to_sideways: float = 0.05
    sideways_to_bull: float = 0.03
    sideways_to_bear: float = 0.03
    high_vol_penalty: float = 0.10
    same_regime_bonus: float = -0.02
    
    def get_cost(self, from_regime: MetaRegime, to_regime: MetaRegime) -> float:
        if from_regime == to_regime:
            return self.same_regime_bonus
        key = f"{from_regime.value}_to_{to_regime.value}"
        return getattr(self, key, 0.05)
    
    def to_dict(self) -> Dict:
        return {'bull_to_bear': self.bull_to_bear, 'bull_to_sideways': self.bull_to_sideways,
            'bear_to_bull': self.bear_to_bull, 'bear_to_sideways': self.bear_to_sideways,
            'sideways_to_bull': self.sideways_to_bull, 'sideways_to_bear': self.sideways_to_bear,
            'high_vol_penalty': self.high_vol_penalty, 'same_regime_bonus': self.same_regime_bonus}


@dataclass
class MetaStrategy:
    """
    A meta-strategy that contains multiple strategies and regime switching rules.
    This is the primary unit of evolution in the meta-mutation system.
    
    Now includes EVOLVABLE REGIME DETECTION - the GA evolves not just strategies,
    but HOW it detects regimes.
    """
    id: str
    name: str
    strategies: List[StrategyGene] = field(default_factory=list)
    regime_mappings: List[RegimeMapping] = field(default_factory=list)
    switch_config: SwitchConfig = field(default_factory=SwitchConfig)
    
    # NEW: Evolvable regime detection (meta-evolution)
    regime_config: EvolvableRegimeConfig = field(default_factory=EvolvableRegimeConfig)
    transition_costs: TransitionCostMatrix = field(default_factory=TransitionCostMatrix)
    
    # Fitness metrics
    overall_fitness: float = 0.0
    regime_fitness: Dict[MetaRegime, float] = field(default_factory=dict)
    switch_count: int = 0
    transition_penalty: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    generation: int = 0
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'strategies': [s.to_dict() for s in self.strategies],
            'regime_mappings': [m.to_dict() for m in self.regime_mappings],
            'switch_config': self.switch_config.to_dict(),
            'regime_config': self.regime_config.to_dict() if self.regime_config else EvolvableRegimeConfig().to_dict(),
            'transition_costs': self.transition_costs.to_dict() if self.transition_costs else TransitionCostMatrix().to_dict(),
            'overall_fitness': self.overall_fitness,
            'regime_fitness': {k.value: v for k, v in self.regime_fitness.items()},
            'switch_count': self.switch_count,
            'transition_penalty': self.transition_penalty,
            'created_at': self.created_at.isoformat(),
            'generation': self.generation,
            'parent_id': self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MetaStrategy':
        # Handle both old and new format
        regime_config = data.get('regime_config', {})
        transition_costs = data.get('transition_costs', {})
        
        ms = cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            strategies=[StrategyGene.from_dict(s) for s in data.get('strategies', [])],
            regime_mappings=[RegimeMapping.from_dict(m) for m in data.get('regime_mappings', [])],
            switch_config=SwitchConfig.from_dict(data.get('switch_config', {})),
            regime_config=EvolvableRegimeConfig.from_dict(regime_config) if regime_config else EvolvableRegimeConfig(),
            transition_costs=TransitionCostMatrix() if not transition_costs else TransitionCostMatrix(**{
                k: v for k, v in transition_costs.items() 
                if k in TransitionCostMatrix.__dataclass_fields__
            }),
            overall_fitness=data.get('overall_fitness', 0.0),
            regime_fitness={MetaRegime(k): v for k, v in data.get('regime_fitness', {}).items()},
            switch_count=data.get('switch_count', 0),
            transition_penalty=data.get('transition_penalty', 0.0),
            generation=data.get('generation', 0),
            parent_id=data.get('parent_id')
        )
        if 'created_at' in data:
            ms.created_at = datetime.fromisoformat(data['created_at'])
        return ms
    
    def get_strategy_for_regime(self, regime: MetaRegime) -> Optional[StrategyGene]:
        """Get the active strategy for a given regime."""
        for mapping in self.regime_mappings:
            if mapping.regime == regime and mapping.enabled:
                for strategy in self.strategies:
                    if strategy.id == mapping.strategy_id:
                        return strategy
        # Fallback: return first strategy
        return self.strategies[0] if self.strategies else None


# ============================================================
# META-MUTATION OPERATORS
# ============================================================
class MetaMutationOperators:
    """Mutation operators that modify meta-strategies."""
    
    @staticmethod
    def assign_strategy_to_regime(meta: MetaStrategy, regime: MetaRegime, strategy_id: str) -> MetaStrategy:
        """Change which strategy is assigned to a regime."""
        meta = copy.deepcopy(meta)
        
        # Update or add mapping
        found = False
        for mapping in meta.regime_mappings:
            if mapping.regime == regime:
                mapping.strategy_id = strategy_id
                found = True
                break
        
        if not found:
            meta.regime_mappings.append(RegimeMapping(regime=regime, strategy_id=strategy_id))
        
        return meta
    
    @staticmethod
    def swap_strategies_between_regimes(meta: MetaStrategy, regime_a: MetaRegime, regime_b: MetaRegime) -> MetaStrategy:
        """Swap strategies between two regimes."""
        meta = copy.deepcopy(meta)
        
        strategy_a = None
        strategy_b = None
        
        for mapping in meta.regime_mappings:
            if mapping.regime == regime_a:
                strategy_a = mapping.strategy_id
            elif mapping.regime == regime_b:
                strategy_b = mapping.strategy_id
        
        if strategy_a and strategy_b:
            for mapping in meta.regime_mappings:
                if mapping.regime == regime_a:
                    mapping.strategy_id = strategy_b
                elif mapping.regime == regime_b:
                    mapping.strategy_id = strategy_a
        
        return meta
    
    @staticmethod
    def mutate_confidence_threshold(meta: MetaStrategy, regime: Optional[MetaRegime] = None, delta: float = 0.1) -> MetaStrategy:
        """Adjust confidence threshold for regime switching."""
        meta = copy.deepcopy(meta)
        
        if regime:
            for mapping in meta.regime_mappings:
                if mapping.regime == regime:
                    mapping.confidence_threshold = max(0.1, min(0.95, mapping.confidence_threshold + delta))
        else:
            meta.switch_config.confidence_threshold = max(0.1, min(0.95, 
                meta.switch_config.confidence_threshold + delta))
        
        return meta
    
    @staticmethod
    def mutate_min_hold_time(meta: MetaStrategy, delta: int = 1) -> MetaStrategy:
        """Adjust minimum hold time before switching."""
        meta = copy.deepcopy(meta)
        meta.switch_config.min_hold_time = max(1, min(20, meta.switch_config.min_hold_time + delta))
        return meta
    
    @staticmethod
    def mutate_hysteresis(meta: MetaStrategy, delta: float = 0.05) -> MetaStrategy:
        """Adjust hysteresis to prevent flip-flopping."""
        meta = copy.deepcopy(meta)
        meta.switch_config.hysteresis = max(0.0, min(0.5, meta.switch_config.hysteresis + delta))
        return meta
    
    @staticmethod
    def mutate_transition_mode(meta: MetaStrategy) -> MetaStrategy:
        """Switch between hard/smooth/blend transition modes."""
        meta = copy.deepcopy(meta)
        
        modes = list(TransitionMode)
        current_idx = modes.index(meta.switch_config.transition_mode)
        next_idx = (current_idx + 1) % len(modes)
        meta.switch_config.transition_mode = modes[next_idx]
        
        return meta
    
    @staticmethod
    def mutate_cooldown(meta: MetaStrategy, delta: int = 1) -> MetaStrategy:
        """Adjust reentry cooldown."""
        meta = copy.deepcopy(meta)
        meta.switch_config.reentry_cooldown = max(0, min(10, meta.switch_config.reentry_cooldown + delta))
        return meta
    
    @staticmethod
    def add_strategy_to_pool(meta: MetaStrategy, strategy: StrategyGene) -> MetaStrategy:
        """Add a new strategy to the meta-strategy pool."""
        meta = copy.deepcopy(meta)
        
        # Check if strategy already exists
        if not any(s.id == strategy.id for s in meta.strategies):
            meta.strategies.append(strategy)
            
            # Auto-create regime mapping for random regime
            if meta.regime_mappings:
                random_regime = random.choice(list(MetaRegime))
                meta.regime_mappings.append(RegimeMapping(
                    regime=random_regime,
                    strategy_id=strategy.id
                ))
        
        return meta
    
    @staticmethod
    def remove_strategy(meta: MetaStrategy, strategy_id: str) -> MetaStrategy:
        """Remove a strategy from the pool."""
        meta = copy.deepcopy(meta)
        
        # Don't remove if it's the only strategy
        if len(meta.strategies) <= 1:
            return meta
        
        # Remove strategy
        meta.strategies = [s for s in meta.strategies if s.id != strategy_id]
        
        # Remove mappings for this strategy
        meta.regime_mappings = [m for m in meta.regime_mappings if m.strategy_id != strategy_id]
        
        # Reassign orphaned regimes
        if meta.strategies and not meta.regime_mappings:
            for regime in MetaRegime:
                meta.regime_mappings.append(RegimeMapping(
                    regime=regime,
                    strategy_id=meta.strategies[0].id
                ))
        
        return meta
    
    @staticmethod
    def crossover_regime_mappings(meta_a: MetaStrategy, meta_b: MetaStrategy) -> Tuple[MetaStrategy, MetaStrategy]:
        """Crossover regime mappings between two meta-strategies."""
        meta_a = copy.deepcopy(meta_a)
        meta_b = copy.deepcopy(meta_b)
        
        # Randomly swap half the mappings
        n_swap = len(meta_a.regime_mappings) // 2
        
        for i in range(n_swap):
            if i < len(meta_a.regime_mappings) and i < len(meta_b.regime_mappings):
                # Swap the strategy_id assignments
                temp = meta_a.regime_mappings[i].strategy_id
                meta_a.regime_mappings[i].strategy_id = meta_b.regime_mappings[i].strategy_id
                meta_b.regime_mappings[i].strategy_id = temp
        
        return meta_a, meta_b
    
    @staticmethod
    def crossover_switch_config(meta_a: MetaStrategy, meta_b: MetaStrategy) -> Tuple[MetaStrategy, MetaStrategy]:
        """Crossover switch configurations."""
        meta_a = copy.deepcopy(meta_a)
        meta_b = copy.deepcopy(meta_b)
        
        # Swap random config values
        if random.random() < 0.5:
            meta_a.switch_config.min_hold_time, meta_b.switch_config.min_hold_time = \
                meta_b.switch_config.min_hold_time, meta_a.switch_config.min_hold_time
        
        if random.random() < 0.5:
            meta_a.switch_config.confidence_threshold, meta_b.switch_config.confidence_threshold = \
                meta_b.switch_config.confidence_threshold, meta_a.switch_config.confidence_threshold
        
        if random.random() < 0.5:
            meta_a.switch_config.transition_mode, meta_b.switch_config.transition_mode = \
                meta_b.switch_config.transition_mode, meta_a.switch_config.transition_mode
        
        return meta_a, meta_b
    
    @staticmethod
    def crossover_regime_configs(meta_a: MetaStrategy, meta_b: MetaStrategy) -> Tuple[MetaStrategy, MetaStrategy]:
        """Crossover regime detection configs between two meta-strategies.
        
        This is how you breed super-adapters: take the regime detection 
        config from one and mix with the other.
        """
        meta_a = copy.deepcopy(meta_a)
        meta_b = copy.deepcopy(meta_b)
        
        # Swap n_regimes
        if random.random() < 0.5:
            meta_a.regime_config.n_regimes, meta_b.regime_config.n_regimes = \
                meta_b.regime_config.n_regimes, meta_a.regime_config.n_regimes
        
        # Swap detection method
        if random.random() < 0.5:
            meta_a.regime_config.method, meta_b.regime_config.method = \
                meta_b.regime_config.method, meta_a.regime_config.method
        
        # Swap feature weights
        if random.random() < 0.5:
            meta_a.regime_config.volatility_weight, meta_b.regime_config.volatility_weight = \
                meta_b.regime_config.volatility_weight, meta_a.regime_config.volatility_weight
            meta_a.regime_config.trend_weight, meta_b.regime_config.trend_weight = \
                meta_b.regime_config.trend_weight, meta_a.regime_config.trend_weight
        
        return meta_a, meta_b
    
    @staticmethod
    def crossover_transition_costs(meta_a: MetaStrategy, meta_b: MetaStrategy) -> Tuple[MetaStrategy, MetaStrategy]:
        """Crossover transition cost matrices."""
        meta_a = copy.deepcopy(meta_a)
        meta_b = copy.deepcopy(meta_b)
        
        # Swap individual transition costs
        cost_keys = ['bull_to_bear', 'bull_to_sideways', 'bear_to_bull', 
                     'bear_to_sideways', 'sideways_to_bull', 'sideways_to_bear',
                     'high_vol_penalty', 'same_regime_bonus']
        
        for key in cost_keys:
            if random.random() < 0.3:  # 30% chance per cost
                val_a = getattr(meta_a.transition_costs, key)
                val_b = getattr(meta_b.transition_costs, key)
                setattr(meta_a.transition_costs, key, val_b)
                setattr(meta_b.transition_costs, key, val_a)
        
        return meta_a, meta_b
    
    @staticmethod
    def crossover_strategy_pools(meta_a: MetaStrategy, meta_b: MetaStrategy) -> Tuple[MetaStrategy, MetaStrategy]:
        """Crossover entire strategy pools between meta-strategies.
        
        Takes half the strategies from each parent to create children.
        """
        meta_a = copy.deepcopy(meta_a)
        meta_b = copy.deepcopy(meta_b)
        
        # Get half from each
        n_a = len(meta_a.strategies) // 2
        n_b = len(meta_b.strategies) // 2
        
        # Create new pools
        new_a_strategies = meta_a.strategies[:n_a] + meta_b.strategies[:n_b]
        new_b_strategies = meta_b.strategies[n_b:] + meta_a.strategies[n_a:]
        
        meta_a.strategies = new_a_strategies
        meta_b.strategies = new_b_strategies
        
        return meta_a, meta_b
    
    @staticmethod
    def full_crossover(meta_a: MetaStrategy, meta_b: MetaStrategy) -> Tuple[MetaStrategy, MetaStrategy]:
        """Full crossover between two meta-strategies - breeds super-adapters.
        
        Recombines:
        - Strategy pools
        - Regime mappings  
        - Switch configs
        - Regime detection configs
        - Transition costs
        """
        # Apply all crossover types
        meta_a, meta_b = MetaMutationOperators.crossover_strategy_pools(meta_a, meta_b)
        meta_a, meta_b = MetaMutationOperators.crossover_regime_mappings(meta_a, meta_b)
        meta_a, meta_b = MetaMutationOperators.crossover_switch_config(meta_a, meta_b)
        meta_a, meta_b = MetaMutationOperators.crossover_regime_configs(meta_a, meta_b)
        meta_a, meta_b = MetaMutationOperators.crossover_transition_costs(meta_a, meta_b)
        
        return meta_a, meta_b
    
    # ============================================================
    # REGIME DETECTION EVOLUTION MUTATIONS (Meta-Evolution)
    # ============================================================
    
    @staticmethod
    def mutate_n_regimes(meta: MetaStrategy, delta: int = 1) -> MetaStrategy:
        """Mutate the number of regimes (2-6).
        
        When regimes increase: split an existing regime
        When regimes decrease: merge similar regimes
        """
        meta = copy.deepcopy(meta)
        new_n = meta.regime_config.n_regimes + delta
        new_n = max(2, min(6, new_n))
        
        if new_n != meta.regime_config.n_regimes:
            if new_n > meta.regime_config.n_regimes:
                # SPLIT: Add a new regime by copying an existing one
                meta = MetaMutationOperators._split_regime(meta)
            else:
                # MERGE: Remove a regime and reassign its strategies
                meta = MetaMutationOperators._merge_regimes(meta)
            
            meta.regime_config.n_regimes = new_n
        
        return meta
    
    @staticmethod
    def _split_regime(meta: MetaStrategy) -> MetaStrategy:
        """Split an existing regime into two sub-regimes."""
        # Find regime with most strategies
        regime_counts = {}
        for mapping in meta.regime_mappings:
            regime_counts[mapping.regime] = regime_counts.get(mapping.regime, 0) + 1
        
        if not regime_counts:
            return meta
        
        # Pick regime to split (one with most mappings)
        source_regime = max(regime_counts.keys(), key=lambda r: regime_counts[r])
        
        # Create new regime
        all_regimes = list(MetaRegime)
        used_regimes = set(m.regime for m in meta.regime_mappings)
        available = [r for r in all_regimes if r not in used_regimes]
        
        if not available:
            return meta
        
        new_regime = available[0]
        
        # Add mapping for new regime
        source_strategy = None
        for mapping in meta.regime_mappings:
            if mapping.regime == source_regime:
                source_strategy = mapping.strategy_id
                break
        
        if source_strategy:
            meta.regime_mappings.append(RegimeMapping(
                regime=new_regime,
                strategy_id=source_strategy,
                confidence_threshold=0.5
            ))
        
        return meta
    
    @staticmethod
    def _merge_regimes(meta: MetaStrategy) -> MetaStrategy:
        """Merge two similar regimes into one."""
        if len(meta.regime_mappings) <= 2:
            return meta  # Can't merge below 2
        
        # Find two regimes with same strategy assignment
        strategy_to_regimes = {}
        for mapping in meta.regime_mappings:
            if mapping.strategy_id not in strategy_to_regimes:
                strategy_to_regimes[mapping.strategy_id] = []
            strategy_to_regimes[mapping.strategy_id].append(mapping.regime)
        
        # Find pair to merge
        for strategy_id, regimes in strategy_to_regimes.items():
            if len(regimes) >= 2:
                # Keep first, remove second
                keep_regime = regimes[0]
                remove_regime = regimes[1]
                
                meta.regime_mappings = [
                    m for m in meta.regime_mappings 
                    if m.regime != remove_regime
                ]
                break
        
        return meta
    
    @staticmethod
    def mutate_detection_method(meta: MetaStrategy) -> MetaStrategy:
        """Switch between detection methods."""
        meta = copy.deepcopy(meta)
        methods = ['hmm', 'kmeans', 'rule', 'ensemble']
        current = methods.index(meta.regime_config.method) if meta.regime_config.method in methods else 0
        meta.regime_config.method = methods[(current + 1) % len(methods)]
        return meta
    
    @staticmethod
    def mutate_feature_weight(meta: MetaStrategy, feature: str = None, delta: float = 0.1) -> MetaStrategy:
        """Mutate feature weights for regime detection."""
        meta = copy.deepcopy(meta)
        
        features = ['volatility_weight', 'trend_weight', 'volume_weight', 
                   'correlation_weight', 'momentum_weight']
        
        if feature is None:
            feature = random.choice(features)
        
        if hasattr(meta.regime_config, feature):
            current = getattr(meta.regime_config, feature)
            setattr(meta.regime_config, feature, max(0.1, min(2.0, current + delta)))
        
        return meta
    
    @staticmethod
    def mutate_vol_threshold(meta: MetaStrategy, delta: float = 0.05) -> MetaStrategy:
        """Mutate volatility thresholds."""
        meta = copy.deepcopy(meta)
        
        if random.random() < 0.5:
            meta.regime_config.vol_high_threshold = max(0.5, min(1.0, 
                meta.regime_config.vol_high_threshold + delta))
        else:
            meta.regime_config.vol_low_threshold = max(0.1, min(0.5, 
                meta.regime_config.vol_low_threshold - delta))
        
        return meta
    
    @staticmethod
    def mutate_smoothing_window(meta: MetaStrategy, delta: int = 2) -> MetaStrategy:
        """Mutate smoothing window for regime detection."""
        meta = copy.deepcopy(meta)
        meta.regime_config.smoothing_window = max(5, min(50, 
            meta.regime_config.smoothing_window + delta))
        return meta
    
    # ============================================================
    # TRANSITION COST MUTATIONS
    # ============================================================
    
    @staticmethod
    def mutate_transition_cost(meta: MetaStrategy, from_regime: MetaRegime = None, 
                             to_regime: MetaRegime = None, delta: float = 0.02) -> MetaStrategy:
        """Mutate transition cost between regimes."""
        meta = copy.deepcopy(meta)
        
        if from_regime is None or to_regime is None:
            # Random pair
            regimes = list(MetaRegime)
            from_regime, to_regime = random.sample(regimes, 2)
        
        key = f"{from_regime.value}_to_{to_regime.value}"
        
        if hasattr(meta.transition_costs, key):
            current = getattr(meta.transition_costs, key)
            setattr(meta.transition_costs, key, max(0.0, min(0.5, current + delta)))
        
        return meta
    
    @staticmethod
    def mutate_high_vol_penalty(meta: MetaStrategy, delta: float = 0.01) -> MetaStrategy:
        """Mutate penalty for transitioning to high volatility."""
        meta = copy.deepcopy(meta)
        meta.transition_costs.high_vol_penalty = max(0.0, min(0.3, 
            meta.transition_costs.high_vol_penalty + delta))
        return meta


# ============================================================
# EVOLVABLE REGIME DETECTION CONFIG
# ============================================================
# REGIME DETECTOR (Wrapper)
# ============================================================
class MetaRegimeDetector:
    """Wrapper around existing regime detector for meta-evolution."""
    
    def __init__(self):
        self.detector = RegimeDetector()
        
    def detect_sequence(self, data: pd.DataFrame) -> List[Tuple[int, MetaRegime, float]]:
        """
        Detect regime for each bar in the data.
        Returns: [(bar_index, regime, confidence), ...]
        """
        results = []
        
        for i in range(50, len(data)):  # Need lookback
            window = data.iloc[:i+1]
            analysis = self.detector.detect(window)
            
            # Map existing regime to MetaRegime
            meta_regime = self._map_regime_to_meta(analysis.regime)
            confidence = analysis.confidence
            
            results.append((i, meta_regime, confidence))
        
        return results
    
    def _map_regime_to_meta(self, regime: Regime) -> MetaRegime:
        """Map RegimeDetector output to MetaRegime."""
        regime_map = {
            Regime.TRENDING_UP: MetaRegime.BULL,
            Regime.TRENDING_DOWN: MetaRegime.BEAR,
            Regime.MEAN_REVERTING: MetaRegime.RANGING,
            Regime.HIGH_VOLATILITY: MetaRegime.HIGH_VOL,
            Regime.SIDEWAYS: MetaRegime.SIDEWAYS,
            Regime.CONSOLIDATING: MetaRegime.RANGING,
            Regime.UNKNOWN: MetaRegime.SIDEWAYS,
        }
        
        return regime_map.get(regime, MetaRegime.SIDEWAYS)


# ============================================================
# BAYESIAN REGIME COUNT OPTIMIZER
# ============================================================
class BayesianRegimeOptimizer:
    """
    Uses BIC (Bayesian Information Criterion) to find optimal regime count.
    
    Expert: "Use BIC or Dirichlet process to allow the number of 
    clusters to vary during evolution."
    
    BIC = k * log(n) - 2 * log(L)
    where k = # parameters, n = # observations, L = likelihood
    
    Lower BIC = better (more efficient model)
    """
    
    def __init__(self, min_regimes=2, max_regimes=6):
        self.min_regimes = min_regimes
        self.max_regimes = max_regimes
        
    def calculate_bic(self, data: pd.DataFrame, n_regimes: int) -> float:
        """
        Calculate BIC for a given number of regimes.
        
        Uses within-regime variance as likelihood proxy.
        Falls back to simple heuristic if sklearn unavailable.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            # Fallback: use simple variance heuristic
            return self._calculate_bic_simple(data, n_regimes)
        
        # Extract features
        returns = data['close'].pct_change().dropna().values[-100:]
        if len(returns) < 20:
            return float('inf')
        
        # Reshape for sklearn
        X = returns.reshape(-1, 1)
        
        try:
            # Fit k-means
            kmeans = KMeans(n_clusters=n_regimes, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Calculate likelihood (within-cluster variance)
            log_likelihood = 0
            for i in range(n_regimes):
                cluster_points = X[labels == i]
                if len(cluster_points) > 1:
                    variance = np.var(cluster_points)
                    if variance > 0:
                        log_likelihood += np.sum(np.log(variance))
            
            # Number of parameters: k-1 (means) + k (variances)
            k = 2 * n_regimes - 1
            n = len(returns)
            
            # BIC
            bic = k * np.log(n) - 2 * log_likelihood
            
            return bic
            
        except Exception as e:
            return self._calculate_bic_simple(data, n_regimes)
    
    def _calculate_bic_simple(self, data: pd.DataFrame, n_regimes: int) -> float:
        """Fallback BIC calculation without sklearn."""
        returns = data['close'].pct_change().dropna().values[-100:]
        
        if len(returns) < 20:
            return float('inf')
        
        # Simple heuristic: variance reduction from using more clusters
        # Use quantiles to approximate clusters
        quantiles = np.linspace(0, 1, n_regimes + 1)
        bounds = np.quantile(returns, quantiles)
        
        total_variance = np.var(returns)
        
        # Within-cluster variance
        within_var = 0
        for i in range(n_regimes):
            cluster = returns[(returns >= bounds[i]) & (returns < bounds[i+1])]
            if len(cluster) > 1:
                within_var += np.var(cluster) * len(cluster)
        
        within_var /= len(returns)
        
        # BIC approximation
        k = 2 * n_regimes - 1
        n = len(returns)
        
        if within_var > 0:
            log_likelihood = n * np.log(within_var)
        else:
            log_likelihood = -1000
        
        bic = k * np.log(n) - 2 * log_likelihood
        
        return bic
    
    def find_optimal_regimes(self, data: pd.DataFrame) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of regimes using BIC.
        
        Returns: (optimal_count, bic_scores)
        """
        bic_scores = {}
        
        for n in range(self.min_regimes, self.max_regimes + 1):
            bic = self.calculate_bic(data, n)
            bic_scores[n] = bic
        
        # Find minimum BIC
        optimal = min(bic_scores.keys(), key=lambda x: bic_scores[x])
        
        return optimal, bic_scores
    
    def evaluate_regime_count_fitness(self, meta: MetaStrategy, 
                                     data: pd.DataFrame,
                                     base_fitness: float) -> float:
        """
        Evaluate fitness with BIC penalty for too many regimes.
        
        Penalizes unnecessary complexity while rewarding proper fit.
        """
        current_n = meta.regime_config.n_regimes
        
        # Get BIC score for current n
        bic = self.calculate_bic(data, current_n)
        
        # Normalize BIC to 0-1 range
        all_bic = [self.calculate_bic(data, n) for n in range(self.min_regimes, self.max_regimes + 1)]
        valid_bic = [b for b in all_bic if b != float('inf')]
        
        if valid_bic:
            min_bic = min(valid_bic)
            max_bic = max(valid_bic)
            if max_bic > min_bic:
                bic_normalized = (bic - min_bic) / (max_bic - min_bic)
            else:
                bic_normalized = 0.5
        else:
            bic_normalized = 0.5
        
        # Fitness = base - complexity_penalty
        # More regimes = higher complexity penalty
        complexity_penalty = 0.1 * (current_n - self.min_regimes)
        
        # Also add BIC-based penalty
        bic_penalty = bic_normalized * 0.05
        
        return base_fitness - complexity_penalty - bic_penalty


# ============================================================
# META-EVOLUTION ENGINE
# ============================================================
class MetaEvolutionEngine:
    """
    Genetic Algorithm for evolving meta-strategies.
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 elite_count: int = 2,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.2):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population: List[MetaStrategy] = []
        self.generation = 0
        self.regime_detector = MetaRegimeDetector()
        
        # Mutation operators list
        self.mutation_operators = [
            # Strategy ↔ Regime mapping mutations
            ('assign_strategy_to_regime', self._mutate_assign),
            ('swap_strategies', self._mutate_swap),
            ('mutate_confidence', self._mutate_confidence),
            ('mutate_hold_time', self._mutate_hold_time),
            ('mutate_hysteresis', self._mutate_hysteresis),
            ('mutate_transition', self._mutate_transition),
            ('mutate_cooldown', self._mutate_cooldown),
            # NEW: Regime detection evolution (meta-evolution)
            ('mutate_n_regimes', self._mutate_n_regimes),
            ('mutate_detection_method', self._mutate_detection_method),
            ('mutate_feature_weight', self._mutate_feature_weight),
            ('mutate_vol_threshold', self._mutate_vol_threshold),
            ('mutate_smoothing', self._mutate_smoothing),
            # NEW: Transition cost evolution
            ('mutate_transition_cost', self._mutate_transition_cost),
            ('mutate_high_vol_penalty', self._mutate_high_vol_penalty),
        ]
    
    def initialize_population(self, seed_strategies: List[StrategyGene], 
                             regime_types: List[MetaRegime] = None) -> None:
        """Initialize population with random meta-strategies."""
        if regime_types is None:
            regime_types = list(MetaRegime)
        
        self.population = []
        
        for i in range(self.population_size):
            # Create meta-strategy with 2-4 strategies
            n_strategies = random.randint(2, min(4, len(seed_strategies)))
            selected_strategies = random.sample(seed_strategies, n_strategies)
            
            # Assign strategies to regimes
            mappings = []
            for j, regime in enumerate(regime_types[:n_strategies]):
                mappings.append(RegimeMapping(
                    regime=regime,
                    strategy_id=selected_strategies[j % len(selected_strategies)].id,
                    confidence_threshold=random.uniform(0.4, 0.8)
                ))
            
            # Random switch config
            switch_config = SwitchConfig(
                min_hold_time=random.randint(2, 10),
                reentry_cooldown=random.randint(1, 5),
                transition_mode=random.choice(list(TransitionMode)),
                hysteresis=random.uniform(0.05, 0.2),
                confidence_threshold=random.uniform(0.5, 0.8)
            )
            
            meta = MetaStrategy(
                id=f"meta_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                name=f"Meta-Strategy-{i+1}",
                strategies=selected_strategies,
                regime_mappings=mappings,
                switch_config=switch_config,
                generation=0
            )
            
            self.population.append(meta)
        
        logger.info(f"Initialized population with {len(self.population)} meta-strategies")
    
    def evaluate(self, market_data: pd.DataFrame, 
                 fitness_func: Callable) -> None:
        """Evaluate fitness for all meta-strategies."""
        # Get regime sequence
        regime_sequence = self.regime_detector.detect_sequence(market_data)
        
        for meta in self.population:
            fitness, regime_fitness, switch_count = self._evaluate_meta_strategy(
                meta, market_data, regime_sequence, fitness_func
            )
            
            meta.overall_fitness = fitness
            meta.regime_fitness = regime_fitness
            meta.switch_count = switch_count
    
    def _evaluate_meta_strategy(self, meta: MetaStrategy, data: pd.DataFrame,
                                 regime_sequence: List[Tuple[int, MetaRegime, float]],
                                 fitness_func: Callable) -> Tuple[float, Dict[MetaRegime, float], int]:
        """Evaluate a single meta-strategy."""
        
        # Track returns per regime
        regime_returns: Dict[MetaRegime, List[float]] = defaultdict(list)
        current_strategy = None
        last_regime = None
        switch_count = 0
        hold_bars = 0
        
        for bar_idx, regime, confidence in regime_sequence:
            # Check if we should switch
            should_switch = False
            
            if current_strategy is None:
                should_switch = True
            elif regime != last_regime and confidence >= meta.switch_config.confidence_threshold:
                # Apply hysteresis
                regime_change_magnitude = abs(list(MetaRegime).index(regime) - list(MetaRegime).index(last_regime))
                if regime_change_magnitude > meta.switch_config.hysteresis * 10:
                    if hold_bars >= meta.switch_config.min_hold_time:
                        should_switch = True
            
            if should_switch:
                current_strategy = meta.get_strategy_for_regime(regime)
                if last_regime != regime:
                    switch_count += 1
                hold_bars = 0
                last_regime = regime
            else:
                hold_bars += 1
            
            # Skip if no strategy
            if current_strategy is None:
                continue
            
            # Get signal from current strategy (simplified)
            # In real implementation, this would call the strategy's signal generation
            signal = random.choice([-1, 0, 1])  # Simplified
            
            # Calculate return
            if bar_idx < len(data) - 1:
                price_change = (data['close'].iloc[bar_idx + 1] - data['close'].iloc[bar_idx]) / data['close'].iloc[bar_idx]
                pnl = signal * price_change
                regime_returns[regime].append(pnl)
        
        # Calculate fitness per regime
        regime_fitness = {}
        for regime, returns in regime_returns.items():
            if len(returns) >= 5:
                # Sharpe-like fitness
                mean_ret = np.mean(returns)
                std_ret = np.std(returns) + 1e-6
                sharpe = mean_ret / std_ret * np.sqrt(252)
                regime_fitness[regime] = sharpe
            else:
                regime_fitness[regime] = 0.0
        
        # Overall fitness = weighted average across regimes
        # Apply evolved transition costs (not just flat penalty)
        switch_penalty = 0.0
        
        # Track regime transitions for cost calculation
        transition_history = []
        prev_regime = None
        
        for bar_idx, regime, confidence in regime_sequence:
            if prev_regime is not None and regime != prev_regime:
                transition_history.append((prev_regime, regime))
            prev_regime = regime
        
        # Calculate transition cost based on evolved matrix
        for from_reg, to_reg in transition_history:
            cost = meta.transition_costs.get_cost(from_reg, to_reg)
            switch_penalty += cost
        
        # Add high volatility penalty if we're in HIGH_VOL
        high_vol_bars = sum(1 for _, r, _ in regime_sequence if r == MetaRegime.HIGH_VOL)
        if high_vol_bars > len(regime_sequence) * 0.2:  # >20% in high vol
            switch_penalty += meta.transition_costs.high_vol_penalty
        
        meta.transition_penalty = switch_penalty
        
        overall_fitness = np.mean(list(regime_fitness.values())) - switch_penalty if regime_fitness else 0.0
        
        return overall_fitness, regime_fitness, switch_count
    
    def select(self) -> List[MetaStrategy]:
        """Tournament selection."""
        selected = []
        
        for _ in range(self.population_size - self.elite_count):
            # Tournament of 3
            tournament = random.sample(self.population, min(3, len(self.population)))
            winner = max(tournament, key=lambda x: x.overall_fitness)
            selected.append(copy.deepcopy(winner))
        
        # Keep elites
        elites = sorted(self.population, key=lambda x: x.overall_fitness, reverse=True)[:self.elite_count]
        selected.extend(elites)
        
        return selected
    
    def mutate(self, meta: MetaStrategy) -> MetaStrategy:
        """Apply random mutation to meta-strategy."""
        if random.random() > self.mutation_rate:
            return meta
        
        # Choose random mutation operator
        op_name, op_func = random.choice(self.mutation_operators)
        
        try:
            meta = op_func(meta)
        except Exception as e:
            logger.warning(f"Mutation {op_name} failed: {e}")
        
        return meta
    
    def _mutate_assign(self, meta: MetaStrategy) -> MetaStrategy:
        """Assign strategy to regime."""
        if not meta.strategies or not meta.regime_mappings:
            return meta
        
        regime = random.choice(list(MetaRegime))
        strategy_id = random.choice(meta.strategies).id
        return MetaMutationOperators.assign_strategy_to_regime(meta, regime, strategy_id)
    
    def _mutate_swap(self, meta: MetaStrategy) -> MetaStrategy:
        """Swap strategies between regimes."""
        if len(meta.regime_mappings) < 2:
            return meta
        
        regime_a, regime_b = random.sample(list(MetaRegime), 2)
        return MetaMutationOperators.swap_strategies_between_regimes(meta, regime_a, regime_b)
    
    def _mutate_confidence(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate confidence threshold."""
        delta = random.uniform(-0.1, 0.1)
        return MetaMutationOperators.mutate_confidence_threshold(meta, delta=delta)
    
    def _mutate_hold_time(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate hold time."""
        delta = random.choice([-1, 1])
        return MetaMutationOperators.mutate_min_hold_time(meta, delta)
    
    def _mutate_hysteresis(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate hysteresis."""
        delta = random.uniform(-0.05, 0.05)
        return MetaMutationOperators.mutate_hysteresis(meta, delta)
    
    def _mutate_transition(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate transition mode."""
        return MetaMutationOperators.mutate_transition_mode(meta)
    
    def _mutate_cooldown(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate cooldown."""
        delta = random.choice([-1, 1])
        return MetaMutationOperators.mutate_cooldown(meta, delta)
    
    # NEW: Regime detection evolution mutations
    def _mutate_n_regimes(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate number of regimes."""
        delta = random.choice([-1, 1])
        return MetaMutationOperators.mutate_n_regimes(meta, delta)
    
    def _mutate_detection_method(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate detection method."""
        return MetaMutationOperators.mutate_detection_method(meta)
    
    def _mutate_feature_weight(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate feature weights."""
        delta = random.uniform(-0.1, 0.1)
        return MetaMutationOperators.mutate_feature_weight(meta, delta=delta)
    
    def _mutate_vol_threshold(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate volatility thresholds."""
        delta = random.uniform(0.02, 0.08)
        return MetaMutationOperators.mutate_vol_threshold(meta, delta)
    
    def _mutate_smoothing(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate smoothing window."""
        delta = random.choice([-2, 2])
        return MetaMutationOperators.mutate_smoothing_window(meta, delta)
    
    def _mutate_transition_cost(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate transition costs."""
        delta = random.uniform(-0.02, 0.02)
        return MetaMutationOperators.mutate_transition_cost(meta, delta=delta)
    
    def _mutate_high_vol_penalty(self, meta: MetaStrategy) -> MetaStrategy:
        """Mutate high volatility penalty."""
        delta = random.uniform(-0.01, 0.01)
        return MetaMutationOperators.mutate_high_vol_penalty(meta, delta)
    
    def evolve(self, market_data: pd.DataFrame, 
               fitness_func: Callable, 
               n_generations: int = 10) -> MetaStrategy:
        """Run the evolution for N generations."""
        
        logger.info(f"Starting meta-evolution for {n_generations} generations")
        
        for gen in range(n_generations):
            self.generation = gen + 1
            
            # Evaluate
            self.evaluate(market_data, fitness_func)
            
            # Log best
            best = max(self.population, key=lambda x: x.overall_fitness)
            logger.info(f"Gen {self.generation}: Best fitness = {best.overall_fitness:.4f} ({best.name})")
            
            # Select
            selected = self.select()
            
            # Create new population
            new_population = []
            
            # Apply mutations
            for meta in selected:
                mutated = self.mutate(meta)
                mutated.generation = self.generation
                if mutated.id != meta.id:
                    mutated.id = f"{mutated.id}_gen{self.generation}"
                new_population.append(mutated)
            
            # Fill remaining with random
            while len(new_population) < self.population_size:
                new_population.append(random.choice(selected))
            
            self.population = new_population
        
        # Final evaluation
        self.evaluate(market_data, fitness_func)
        
        # Return best
        best = max(self.population, key=lambda x: x.overall_fitness)
        logger.info(f"Evolution complete. Best: {best.name} (fitness: {best.overall_fitness:.4f})")
        
        return best
    
    def get_best(self) -> Optional[MetaStrategy]:
        """Get the best meta-strategy from population."""
        if not self.population:
            return None
        return max(self.population, key=lambda x: x.overall_fitness)


# ============================================================
# ENSEMBLE REGIME DETECTOR
# ============================================================
class EnsembleRegimeDetector:
    """
    Ensemble of multiple regime detectors.
    
    Expert insight: "Use an ensemble of detectors and let the GA evolve 
    which one to trust when. This is like having a committee of market 
    thermometers and letting the algorithm decide which one to read."
    
    Detectors:
    - GARCH-based: volatility clustering
    - Trend-based: ADX + Hurst exponent
    - Volume-based: volume profile anomalies
    - Correlation-based: pairwise correlation structure
    """
    
    def __init__(self):
        self.detectors = {
            'garch': self._garch_detect,
            'trend': self._trend_detect,
            'volume': self._volume_detect,
            'pattern': self._pattern_detect
        }
        # Evolvable weights (learned by GA)
        self.detector_weights = {
            'garch': 0.25,
            'trend': 0.25,
            'volume': 0.25,
            'pattern': 0.25
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """Set detector weights (evolved by GA)."""
        self.detector_weights.update(weights)
    
    def detect(self, data: pd.DataFrame) -> Tuple[MetaRegime, float, Dict[str, float]]:
        """
        Ensemble detection - combines multiple signals.
        
        Returns: (dominant_regime, confidence, detector_votes)
        """
        votes = {}
        
        for name, detector in self.detectors.items():
            regime, confidence = detector(data)
            weight = self.detector_weights.get(name, 0.25)
            votes[regime.value] = votes.get(regime.value, 0) + confidence * weight
        
        # Get winner
        dominant = max(votes.items(), key=lambda x: x[1])
        confidence = dominant[1]
        
        # Convert back to MetaRegime
        regime = MetaRegime(dominant[0])
        
        return regime, min(confidence, 1.0), votes
    
    def _garch_detect(self, data: pd.DataFrame) -> Tuple[MetaRegime, float]:
        """Detect based on volatility clustering (GARCH-like)."""
        returns = data['close'].pct_change().dropna()
        vol = returns.rolling(20).std() * np.sqrt(252)
        current_vol = vol.iloc[-1] if not pd.isna(vol.iloc[-1]) else 0.3
        
        if current_vol > 0.6:
            return MetaRegime.HIGH_VOL, 0.8
        elif current_vol < 0.2:
            return MetaRegime.LOW_VOL, 0.7
        else:
            # Check for trend
            price = data['close']
            slope = (price.iloc[-1] - price.iloc[-20]) / price.iloc[-20]
            if slope > 0.05:
                return MetaRegime.BULL, 0.7
            elif slope < -0.05:
                return MetaRegime.BEAR, 0.7
            return MetaRegime.SIDEWAYS, 0.6
    
    def _trend_detect(self, data: pd.DataFrame) -> Tuple[MetaRegime, float]:
        """Detect based on trend strength (ADX-like + Hurst)."""
        close = data['close']
        
        # Simple trend calculation
        ma_fast = close.rolling(10).mean()
        ma_slow = close.rolling(30).mean()
        
        trend_strength = abs(ma_fast.iloc[-1] - ma_slow.iloc[-1]) / ma_slow.iloc[-1]
        
        if trend_strength > 0.05:
            if ma_fast.iloc[-1] > ma_slow.iloc[-1]:
                return MetaRegime.BULL, 0.75
            else:
                return MetaRegime.BEAR, 0.75
        elif trend_strength < 0.01:
            return MetaRegime.SIDEWAYS, 0.7
        else:
            return MetaRegime.TRENDING, 0.6
    
    def _volume_detect(self, data: pd.DataFrame) -> Tuple[MetaRegime, float]:
        """Detect based on volume profile."""
        volume = data.get('volume', pd.Series([1]*len(data)))
        
        avg_vol = volume.rolling(20).mean()
        current_vol = volume.iloc[-1]
        vol_ratio = current_vol / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1
        
        if vol_ratio > 2.0:
            return MetaRegime.HIGH_VOL, 0.7
        elif vol_ratio < 0.5:
            return MetaRegime.LOW_VOL, 0.6
        else:
            # Check trend
            return self._trend_detect(data)
    
    def _pattern_detect(self, data: pd.DataFrame) -> Tuple[MetaRegime, float]:
        """Detect based on pattern recognition (sine wave / cycle)."""
        close = data['close']
        
        # Simple cycle detection
        recent = close.iloc[-20:]
        if len(recent) < 10:
            return MetaRegime.SIDEWAYS, 0.5
        
        # Check for clear direction
        first_half = recent.iloc[:10].mean()
        second_half = recent.iloc[10:].mean()
        
        if second_half > first_half * 1.05:
            return MetaRegime.BULL, 0.65
        elif second_half < first_half * 0.95:
            return MetaRegime.BEAR, 0.65
        else:
            return MetaRegime.RANGING, 0.6


# ============================================================
# REGIME HALL OF FAME
# ============================================================
class RegimeHallOfFame:
    """
    Store the best strategies for each detected regime across all runs.
    
    Expert insight: "Extend it to regimeHallOfFame—store the best strategies 
    for each detected regime across all runs. Then, when the current regime 
    matches a past one, you can seed the population with those proven champs."
    """
    
    def __init__(self, max_per_regime: int = 10):
        self.max_per_regime = max_per_regime
        # {regime: [(meta_strategy, fitness), ...]}
        self.hall_of_fame: Dict[MetaRegime, List[Tuple[MetaStrategy, float]]] = {
            r: [] for r in MetaRegime
        }
    
    def add(self, meta: MetaStrategy):
        """Add a meta-strategy to the Hall of Fame based on its regime fitness."""
        for regime, fitness in meta.regime_fitness.items():
            if fitness > 0:  # Only positive performers
                self._add_to_regime(meta, regime, fitness)
    
    def _add_to_regime(self, meta: MetaStrategy, regime: MetaRegime, fitness: float):
        """Add to specific regime."""
        entries = self.hall_of_fame[regime]
        
        # Check if already exists
        for i, (existing, _) in enumerate(entries):
            if existing.id == meta.id:
                # Update if better
                if fitness > entries[i][1]:
                    entries[i] = (meta, fitness)
                return
        
        # Add new
        entries.append((meta, fitness))
        
        # Sort by fitness descending
        entries.sort(key=lambda x: x[1], reverse=True)
        
        # Trim
        self.hall_of_fame[regime] = entries[:self.max_per_regime]
    
    def get_champs_for_regime(self, regime: MetaRegime, n: int = 3) -> List[MetaStrategy]:
        """Get the top N champions for a specific regime."""
        entries = self.hall_of_fame.get(regime, [])
        return [e[0] for e in entries[:n]]
    
    def get_all_champs(self) -> List[MetaStrategy]:
        """Get all Hall of Fame strategies."""
        champs = []
        for entries in self.hall_of_fame.values():
            for meta, _ in entries:
                if meta not in champs:
                    champs.append(meta)
        return champs
    
    def seed_population(self, engine: 'MetaEvolutionEngine', regime: MetaRegime) -> List[MetaStrategy]:
        """Seed the population with Hall of Fame champions for current regime."""
        champs = self.get_champs_for_regime(regime, n=3)
        
        # Clone champions and add to population
        seeded = []
        for champ in champs:
            cloned = copy.deepcopy(champ)
            cloned.id = f"{cloned.id}_seeded_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            cloned.generation = 0
            seeded.append(cloned)
        
        return seeded
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            regime.value: [
                (meta.to_dict(), fitness) 
                for meta, fitness in entries
            ]
            for regime, entries in self.hall_of_fame.items()
        }


# ============================================================
# FACTORY FUNCTIONS
# ============================================================
def create_seed_strategies() -> List[StrategyGene]:
    """Create seed strategies for meta-evolution."""
    seeds = [
        StrategyGene(
            id="trend_1",
            name="EMA Trend Follower",
            strategy_type="trend",
            params={"fast_ema": 12, "slow_ema": 26, "stop_loss": 0.02}
        ),
        StrategyGene(
            id="mean_rev_1", 
            name="RSI Mean Reversion",
            strategy_type="mean_revert",
            params={"rsi_period": 14, "oversold": 30, "overbought": 70}
        ),
        StrategyGene(
            id="momentum_1",
            name="Momentum Burst",
            strategy_type="momentum",
            params={"lookback": 20, "threshold": 0.05}
        ),
        StrategyGene(
            id="breakout_1",
            name="Volatility Breakout",
            strategy_type="breakout",
            params={"lookback": 20, "atr_multiplier": 2.0}
        ),
        StrategyGene(
            id="vol_adapt_1",
            name="Volatility Adaptive",
            strategy_type="vol_adaptive",
            params={"atr_period": 14, "vol_threshold": 0.03}
        ),
    ]
    return seeds


def create_default_meta_strategy() -> MetaStrategy:
    """Create a default meta-strategy for testing."""
    seeds = create_seed_strategies()
    
    return MetaStrategy(
        id="default_meta",
        name="Default Surfer+Sniper",
        strategies=seeds[:3],
        regime_mappings=[
            RegimeMapping(regime=MetaRegime.BULL, strategy_id="trend_1"),
            RegimeMapping(regime=MetaRegime.BEAR, strategy_id="vol_adapt_1"),
            RegimeMapping(regime=MetaRegime.SIDEWAYS, strategy_id="mean_rev_1"),
            RegimeMapping(regime=MetaRegime.HIGH_VOL, strategy_id="vol_adapt_1"),
            RegimeMapping(regime=MetaRegime.LOW_VOL, strategy_id="mean_rev_1"),
        ],
        switch_config=SwitchConfig(
            min_hold_time=5,
            reentry_cooldown=3,
            transition_mode=TransitionMode.SMOOTH,
            hysteresis=0.1,
            confidence_threshold=0.6
        )
    )


# ============================================================
# CLI / TEST
# ============================================================
if __name__ == "__main__":
    # Test the meta-evolution engine
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Create price data with regime changes
    returns = np.random.randn(n) * 0.02
    # Inject regime changes
    returns[100:150] *= 3  # High volatility
    returns[200:250] *= 0.3  # Low volatility / sideways
    returns[300:400] = np.cumsum(np.random.randn(100) * 0.01)  # Trend
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Create engine
    engine = MetaEvolutionEngine(population_size=10, mutation_rate=0.4)
    
    # Initialize with seed strategies
    seeds = create_seed_strategies()
    engine.initialize_population(seeds)
    
    # Evolve
    best = engine.evolve(data, lambda s, d: 0.0, n_generations=5)
    
    print(f"\n=== BEST META-STRATEGY ===")
    print(f"Name: {best.name}")
    print(f"ID: {best.id}")
    print(f"Fitness: {best.overall_fitness:.4f}")
    print(f"Strategies: {len(best.strategies)}")
    print(f"Switch Config: {best.switch_config.to_dict()}")
    print(f"Regime Mappings:")
    for m in best.regime_mappings:
        print(f"  {m.regime.value} -> {m.strategy_id}")
    
    # Save to JSON
    with open('/home/kenner/clawd/meta_strategy_result.json', 'w') as f:
        json.dump(best.to_dict(), f, indent=2)
    
    print("\nSaved to meta_strategy_result.json")
