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


@dataclass
class MetaStrategy:
    """
    A meta-strategy that contains multiple strategies and regime switching rules.
    This is the primary unit of evolution in the meta-mutation system.
    """
    id: str
    name: str
    strategies: List[StrategyGene] = field(default_factory=list)
    regime_mappings: List[RegimeMapping] = field(default_factory=list)
    switch_config: SwitchConfig = field(default_factory=SwitchConfig)
    
    # Fitness metrics
    overall_fitness: float = 0.0
    regime_fitness: Dict[MetaRegime, float] = field(default_factory=dict)
    switch_count: int = 0
    
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
            'overall_fitness': self.overall_fitness,
            'regime_fitness': {k.value: v for k, v in self.regime_fitness.items()},
            'switch_count': self.switch_count,
            'created_at': self.created_at.isoformat(),
            'generation': self.generation,
            'parent_id': self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MetaStrategy':
        ms = cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            strategies=[StrategyGene.from_dict(s) for s in data.get('strategies', [])],
            regime_mappings=[RegimeMapping.from_dict(m) for m in data.get('regime_mappings', [])],
            switch_config=SwitchConfig.from_dict(data.get('switch_config', {})),
            overall_fitness=data.get('overall_fitness', 0.0),
            regime_fitness={MetaRegime(k): v for k, v in data.get('regime_fitness', {}).items()},
            switch_count=data.get('switch_count', 0),
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
            ('assign_strategy_to_regime', self._mutate_assign),
            ('swap_strategies', self._mutate_swap),
            ('mutate_confidence', self._mutate_confidence),
            ('mutate_hold_time', self._mutate_hold_time),
            ('mutate_hysteresis', self._mutate_hysteresis),
            ('mutate_transition', self._mutate_transition),
            ('mutate_cooldown', self._mutate_cooldown),
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
        # Bonus for lower switch count
        switch_penalty = switch_count * 0.05
        
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
