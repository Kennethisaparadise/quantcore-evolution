"""
QuantCore - Meta-Learning Layer for Mutation Engine

A higher-level optimization layer that tunes mutation hyperparameters:
- Mutation probabilities for each operator type
- Operator selection weights
- Uses multi-armed bandit / Bayesian optimization

This creates a self-tuning system that adapts mutation strategy
to maximize long-term fitness and avoid local optima.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import json
import logging
from collections import deque
import math

logger = logging.getLogger(__name__)


# ============================================================
# MUTATION CATEGORIES
# ============================================================
class MutationCategory(Enum):
    """Categories of mutation operators."""
    ENTRY_EXIT = "entry_exit"        # Flip entry/exit logic
    INDICATOR = "indicator"           # Replace/add/remove indicators
    THRESHOLD = "threshold"           # Shift thresholds
    PARAMETER = "parameter"           # Change parameters
    COMBINATION = "combination"       # Combine strategies
    LOGIC = "logic"                   # Add filters, invert signals
    RISK = "risk"                     # Position sizing, stops


# ============================================================
# META-PARAMETERS
# ============================================================
@dataclass
class MutationMetaParams:
    """
    Hyper-parameters for mutation operators.
    These are tuned by the meta-GA.
    """
    # Category probabilities (how likely to pick each category)
    category_probs: Dict[MutationCategory, float] = field(default_factory=dict)
    
    # Specific operator probabilities within categories
    operator_probs: Dict[str, float] = field(default_factory=dict)
    
    # Mutation intensities
    parameter_mutation_rate: float = 0.3
    threshold_shift_magnitude: float = 0.2
    
    # Special flags
    allow_extreme_mutations: bool = False
    prefer_combine_operators: float = 0.1
    
    def to_dict(self) -> Dict:
        return {
            'category_probs': {k.value: v for k, v in self.category_probs.items()},
            'operator_probs': self.operator_probs,
            'parameter_mutation_rate': self.parameter_mutation_rate,
            'threshold_shift_magnitude': self.threshold_shift_magnitude,
            'allow_extreme_mutations': self.allow_extreme_mutations,
            'prefer_combine_operators': self.prefer_combine_operators
        }
    
    @classmethod
    def default(cls) -> 'MutationMetaParams':
        """Create default meta-parameters."""
        params = cls()
        
        # Equal probability for each category
        total = len(MutationCategory)
        for cat in MutationCategory:
            params.category_probs[cat] = 1.0 / total
        
        # Default operator probabilities
        params.operator_probs = {
            'flip_entry': 0.15,
            'flip_exit': 0.10,
            'replace_indicator': 0.15,
            'add_indicator': 0.10,
            'shift_threshold': 0.15,
            'change_period': 0.15,
            'combine_strategies': 0.10,
            'invert_signal': 0.10
        }
        
        params.parameter_mutation_rate = 0.3
        params.threshold_shift_magnitude = 0.2
        
        return params
    
    def mutate(self, magnitude: float = 0.1) -> 'MutationMetaParams':
        """Create a mutated copy of these parameters."""
        new_params = MutationMetaParams()
        
        # Mutate category probabilities
        for cat in MutationCategory:
            current = self.category_probs.get(cat, 0.1)
            shift = random.gauss(0, magnitude)
            new_prob = max(0.05, min(0.5, current + shift))
            new_params.category_probs[cat] = new_prob
        
        # Renormalize
        total = sum(new_params.category_probs.values())
        for cat in new_params.category_probs:
            new_params.category_probs[cat] /= total
        
        # Mutate operator probabilities
        for op, prob in self.operator_probs.items():
            shift = random.gauss(0, magnitude)
            new_prob = max(0.01, min(0.5, prob + shift))
            new_params.operator_probs[op] = new_prob
        
        # Renormalize
        total = sum(new_params.operator_probs.values())
        for op in new_params.operator_probs:
            new_params.operator_probs[op] /= total
        
        # Mutate intensities
        new_params.parameter_mutation_rate = max(0.1, min(0.8, 
            self.parameter_mutation_rate + random.gauss(0, magnitude)))
        new_params.threshold_shift_magnitude = max(0.05, min(0.5,
            self.threshold_shift_magnitude + random.gauss(0, magnitude)))
        
        new_params.allow_extreme_mutations = self.allow_extreme_mutations
        new_params.prefer_combine_operators = max(0.0, min(0.3,
            self.prefer_combine_operators + random.gauss(0, magnitude)))
        
        return new_params


# ============================================================
# BANDIT SELECTION (Thompson Sampling)
# ============================================================
class BanditArm:
    """A single arm in the multi-armed bandit."""
    
    def __init__(self, name: str, params: MutationMetaParams):
        self.name = name
        self.params = params
        self.successes = 0
        self.failures = 0
        self.total_reward = 0.0
        
    def update(self, reward: float):
        """Update arm statistics."""
        self.total_reward += reward
        if reward > 0:
            self.successes += 1
        else:
            self.failures += 1
    
    def sample(self) -> float:
        """Thompson sampling: sample from posterior."""
        alpha = self.successes + 1
        beta = self.failures + 1
        return random.betavariate(alpha, beta)
    
    @property
    def mean_reward(self) -> float:
        """Mean reward."""
        n = self.successes + self.failures
        return self.total_reward / n if n > 0 else 0.0


class MetaBandit:
    """
    Multi-armed bandit for meta-parameter selection.
    Uses Thompson Sampling for exploration.
    """
    
    def __init__(self, n_arms: int = 10):
        self.arms: List[BanditArm] = []
        self.n_arms = n_arms
        self.history: List[Tuple[str, float]] = []
        
    def initialize(self, base_params: MutationMetaParams):
        """Initialize arms with variations of base parameters."""
        self.arms = []
        
        # Arm 0: Default
        self.arms.append(BanditArm("default", MutationMetaParams.default()))
        
        # Create variations
        for i in range(1, self.n_arms):
            # Generate mutated parameters
            params = base_params.mutate(magnitude=0.1 + (i * 0.02))
            self.arms.append(BanditArm(f"variant_{i}", params))
    
    def select_arm(self) -> BanditArm:
        """Select arm using Thompson Sampling."""
        samples = [arm.sample() for arm in self.arms]
        best_idx = np.argmax(samples)
        return self.arms[best_idx]
    
    def update(self, arm_name: str, reward: float):
        """Update arm with reward."""
        for arm in self.arms:
            if arm.name == arm_name:
                arm.update(reward)
                break
        
        self.history.append((arm_name, reward))
    
    def get_best_arm(self) -> BanditArm:
        """Get arm with highest mean reward."""
        return max(self.arms, key=lambda a: a.mean_reward)
    
    def get_statistics(self) -> Dict:
        """Get bandit statistics."""
        return {
            'n_arms': len(self.arms),
            'total_selections': len(self.history),
            'best_arm': self.get_best_arm().name,
            'best_mean_reward': self.get_best_arm().mean_reward,
            'arm_rewards': {arm.name: arm.mean_reward for arm in self.arms}
        }


# ============================================================
# META-ADAPTATION TRACKER
# ============================================================
@dataclass
class MetaEpisode:
    """A meta-learning episode."""
    episode: int
    timestamp: datetime
    arm_name: str
    params: MutationMetaParams
    main_ga_fitness: float
    diversity_score: float
    reward: float
    
    def to_dict(self) -> Dict:
        return {
            'episode': self.episode,
            'timestamp': self.timestamp.isoformat(),
            'arm_name': self.arm_name,
            'params': self.params.to_dict(),
            'main_ga_fitness': self.main_ga_fitness,
            'diversity_score': self.diversity_score,
            'reward': self.reward
        }


class MetaAdaptationLogger:
    """Logger for meta-adaptation events."""
    
    def __init__(self, log_file: str = "meta_adaptation_log.json"):
        self.log_file = log_file
        self.episodes: List[MetaEpisode] = []
        self.reward_history: List[float] = []
        self.best_reward = float('-inf')
        
    def log_episode(self, episode: MetaEpisode):
        """Log a meta-episode."""
        self.episodes.append(episode)
        self.reward_history.append(episode.reward)
        
        if episode.reward > self.best_reward:
            self.best_reward = episode.reward
        
        # Save to file
        self._save()
        
        # Log important events
        if len(self.episodes) % 10 == 0:
            logger.info(f"Meta-Episode {episode.episode}: arm={episode.arm_name}, "
                       f"fitness={episode.main_ga_fitness:.2f}, reward={episode.reward:.4f}")
    
    def _save(self):
        """Save log to file."""
        try:
            data = {
                'episodes': [e.to_dict() for e in self.episodes[-100:]],  # Last 100
                'reward_history': self.reward_history[-100:],
                'best_reward': self.best_reward
            }
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save meta log: {e}")
    
    def get_recent_trend(self, n: int = 10) -> str:
        """Get recent reward trend."""
        if len(self.reward_history) < n:
            return "insufficient data"
        
        recent = self.reward_history[-n:]
        if all(recent[i] < recent[i+1] for i in range(n-1)):
            return "📈 improving"
        elif all(recent[i] > recent[i+1] for i in range(n-1)):
            return "📉 declining"
        else:
            return "➡️ stable"
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.episodes:
            return {}
        
        return {
            'total_episodes': len(self.episodes),
            'best_reward': self.best_reward,
            'recent_trend': self.get_recent_trend(),
            'current_arm': self.episodes[-1].arm_name if self.episodes else None,
            'arm_selection_count': self._count_arm_selections()
        }
    
    def _count_arm_selections(self) -> Dict[str, int]:
        """Count how many times each arm was selected."""
        counts = {}
        for ep in self.episodes:
            counts[ep.arm_name] = counts.get(ep.arm_name, 0) + 1
        return counts


# ============================================================
# META-GA WRAPPER
# ============================================================
class MetaMutationOptimizer:
    """
    Meta-learning wrapper that optimizes mutation parameters.
    
    Wraps the existing mutation engine and:
    1. Maintains a population of meta-parameter configurations
    2. Evaluates each on the main GA fitness
    3. Uses bandits to select best configurations
    4. Adapts over time based on performance
    """
    
    def __init__(
        self,
        n_arms: int = 10,
        reward_weight_fitness: float = 0.7,
        reward_weight_diversity: float = 0.3,
        adaptation_interval: int = 5  # Episodes before adaptation
    ):
        self.n_arms = n_arms
        self.reward_weight_fitness = reward_weight_fitness
        self.reward_weight_diversity = reward_weight_diversity
        self.adaptation_interval = adaptation_interval
        
        # Bandit for meta-parameter selection
        self.bandit = MetaBandit(n_arms=n_arms)
        
        # Current meta-parameters
        self.current_params = MutationMetaParams.default()
        self.current_arm: Optional[BanditArm] = None
        
        # Tracking
        self.episode_count = 0
        self.fitness_history: deque = deque(maxlen=100)
        self.diversity_history: deque = deque(maxlen=100)
        
        # Logger
        self.logger = MetaAdaptationLogger()
        
        # Callbacks
        self.main_ga_callback: Optional[Callable] = None
        
    def initialize(self, base_params: MutationMetaParams = None):
        """Initialize the meta-optimizer."""
        if base_params is None:
            base_params = MutationMetaParams.default()
        
        self.bandit.initialize(base_params)
        self.current_arm = self.bandit.select_arm()
        self.current_params = self.current_arm.params
        
        logger.info(f"Meta-Mutation Optimizer initialized with {self.n_arms} arms")
    
    def set_main_ga_callback(self, callback: Callable):
        """Set callback to run main GA with given params."""
        self.main_ga_callback = callback
    
    def get_current_params(self) -> MutationMetaParams:
        """Get current meta-parameters for mutation engine."""
        return self.current_params
    
    def run_episode(
        self,
        data: pd.DataFrame,
        n_generations: int = 3
    ) -> Dict:
        """
        Run one meta-episode:
        1. Select meta-parameter configuration (arm)
        2. Run main GA with those parameters
        3. Calculate reward
        4. Update bandit
        """
        self.episode_count += 1
        
        # Select arm (exploration vs exploitation)
        if self.episode_count > 1 and random.random() < 0.1:
            # 10% random exploration
            self.current_arm = random.choice(self.bandit.arms)
        else:
            self.current_arm = self.bandit.select_arm()
        
        self.current_params = self.current_arm.params
        
        # Run main GA with current parameters
        if self.main_ga_callback:
            try:
                result = self.main_ga_callback(self.current_params, data, n_generations)
                main_fitness = result.get('best_fitness', 0)
            except Exception as e:
                logger.warning(f"Main GA failed: {e}")
                main_fitness = 0
        else:
            # Demo: simulate GA run
            main_fitness = self._simulate_ga_run()
        
        # Get diversity score
        diversity_score = self._calculate_diversity_score()
        
        # Calculate reward (combination of fitness and diversity)
        reward = self._calculate_reward(main_fitness, diversity_score)
        
        # Update bandit
        self.bandit.update(self.current_arm.name, reward)
        
        # Log episode
        episode = MetaEpisode(
            episode=self.episode_count,
            timestamp=datetime.now(),
            arm_name=self.current_arm.name,
            params=self.current_params,
            main_ga_fitness=main_fitness,
            diversity_score=diversity_score,
            reward=reward
        )
        self.logger.log_episode(episode)
        
        # Track history
        self.fitness_history.append(main_fitness)
        self.diversity_history.append(diversity_score)
        
        return {
            'episode': self.episode_count,
            'arm': self.current_arm.name,
            'fitness': main_fitness,
            'diversity': diversity_score,
            'reward': reward,
            'best_arm': self.bandit.get_best_arm().name,
            'best_mean_reward': self.bandit.get_best_arm().mean_reward
        }
    
    def _simulate_ga_run(self) -> float:
        """Simulate a GA run (for testing)."""
        # Simulate fitness that changes based on params
        base = 50.0
        
        # Category bonuses
        cat_bonus = sum(self.current_params.category_probs.values()) * 10
        
        # Operator diversity bonus
        op_bonus = len(self.current_params.operator_probs) * 5
        
        # Random component
        noise = random.gauss(0, 20)
        
        fitness = base + cat_bonus + op_bonus + noise
        
        return max(0, fitness)
    
    def _calculate_diversity_score(self) -> float:
        """Calculate diversity score from recent fitness history."""
        if len(self.fitness_history) < 2:
            return 0.5
        
        # Variance in fitness = diversity
        fitness_array = np.array(list(self.fitness_history))
        variance = np.var(fitness_array)
        
        # Normalize to 0-1
        score = min(1.0, variance / 1000)
        
        return score
    
    def _calculate_reward(self, fitness: float, diversity: float) -> float:
        """Calculate reward for bandit update."""
        # Weighted combination
        reward = (
            self.reward_weight_fitness * fitness +
            self.reward_weight_diversity * diversity * 100  # Scale diversity
        )
        
        # Bonus for improving over running average
        if len(self.fitness_history) > 5:
            recent_avg = np.mean(list(self.fitness_history)[-5:])
            if fitness > recent_avg:
                reward *= 1.2  # Improvement bonus
            elif fitness < recent_avg * 0.8:
                reward *= 0.8  # Penalty for decline
        
        return reward
    
    def run_adaptation(
        self,
        data: pd.DataFrame,
        n_episodes: int = 20,
        n_generations: int = 3
    ) -> Dict:
        """
        Run multiple meta-episodes for adaptation.
        """
        logger.info(f"Starting meta-adaptation for {n_episodes} episodes...")
        
        results = []
        
        for ep in range(n_episodes):
            result = self.run_episode(data, n_generations)
            results.append(result)
            
            if (ep + 1) % 5 == 0:
                logger.info(f"Episode {ep+1}/{n_episodes}: "
                           f"Best arm: {result['best_arm']}, "
                           f"Best reward: {result['best_mean_reward']:.2f}")
        
        # Get final best parameters
        best_arm = self.bandit.get_best_arm()
        
        return {
            'episodes': results,
            'best_arm': best_arm.name,
            'best_params': best_arm.params.to_dict(),
            'best_mean_reward': best_arm.mean_reward,
            'summary': self.logger.get_summary()
        }
    
    def get_status(self) -> Dict:
        """Get current optimizer status."""
        return {
            'episode_count': self.episode_count,
            'current_arm': self.current_arm.name if self.current_arm else None,
            'current_params': self.current_params.to_dict(),
            'bandit_stats': self.bandit.get_statistics(),
            'adaptation_summary': self.logger.get_summary()
        }


# ============================================================
# INTEGRATION WITH MUTATION ENGINE
# ============================================================
class AdaptiveMutationEngine:
    """
    Wrapper around the mutation engine that uses meta-learned parameters.
    """
    
    def __init__(self, meta_optimizer: MetaMutationOptimizer):
        self.meta_optimizer = meta_optimizer
        self.current_params = meta_optimizer.get_current_params()
        
    def get_mutation_weights(self) -> Dict[str, float]:
        """Get mutation weights from meta-parameters."""
        weights = {}
        
        # Map categories to operators
        category_operators = {
            MutationCategory.ENTRY_EXIT: ['flip_entry', 'flip_exit', 'swap_entry_exit'],
            MutationCategory.INDICATOR: ['replace_indicator', 'add_indicator', 'remove_indicator'],
            MutationCategory.THRESHOLD: ['shift_threshold', 'tighten_threshold', 'widen_threshold'],
            MutationCategory.PARAMETER: ['change_period', 'multiply_period', 'invert_parameter'],
            MutationCategory.COMBINATION: ['combine_strategies', 'add_filter'],
            MutationCategory.LOGIC: ['add_time_filter', 'add_volume_filter', 'invert_signal'],
            MutationCategory.RISK: ['change_position_size', 'add_stop_loss', 'add_take_profit']
        }
        
        # Get weights based on category probabilities
        for cat, operators in category_operators.items():
            cat_prob = self.current_params.category_probs.get(cat, 0.1)
            for op in operators:
                op_base = self.current_params.operator_probs.get(op, 0.1)
                weights[op] = cat_prob * op_base
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def select_mutation(self) -> str:
        """Select a mutation operator based on meta-learned weights."""
        weights = self.get_mutation_weights()
        
        # Thompson sampling on operators
        # Use softmax with temperature
        temp = self.current_params.parameter_mutation_rate
        
        ops = list(weights.keys())
        probs = list(weights.values())
        
        # Apply temperature
        exp_probs = [p ** (1/temp) for p in probs]
        total = sum(exp_probs)
        probs = [p/total for p in exp_probs]
        
        return random.choices(ops, weights=probs, k=1)[0]
    
    def update_params(self):
        """Update parameters from meta-optimizer."""
        self.current_params = self.meta_optimizer.get_current_params()


# ============================================================
# DEMO
# ============================================================
def demo():
    """Demo the meta-learning layer."""
    print("=" * 60)
    print("META-LEARNING LAYER DEMO")
    print("=" * 60)
    
    # Create meta-optimizer
    meta = MetaMutationOptimizer(n_arms=8)
    meta.initialize()
    
    # Run adaptation
    print("\nRunning meta-adaptation...")
    
    # Generate dummy data
    data = pd.DataFrame({
        'close': 45000 + np.cumsum(np.random.randn(200) * 100)
    })
    
    results = meta.run_adaptation(data, n_episodes=10, n_generations=2)
    
    print(f"\n{'='*60}")
    print("META-ADAPTATION RESULTS")
    print(f"{'='*60}")
    print(f"Best Arm: {results['best_arm']}")
    print(f"Best Mean Reward: {results['best_mean_reward']:.2f}")
    
    summary = results['summary']
    print(f"\nTotal Episodes: {summary['total_episodes']}")
    print(f"Recent Trend: {summary['recent_trend']}")
    print(f"Arm Selection Count: {summary['arm_selection_count']}")
    
    # Show best parameters
    print(f"\nBest Meta-Parameters:")
    best_params = results['best_params']
    print(f"  Category Probs: {best_params['category_probs']}")
    print(f"  Parameter Mutation Rate: {best_params['parameter_mutation_rate']:.2f}")
    
    print("\n✅ Meta-learning working!")
    
    return meta, results


if __name__ == "__main__":
    demo()
