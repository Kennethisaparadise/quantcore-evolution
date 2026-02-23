"""
QuantCore - Strategy Mutation Engine

Genetic evolution for trading strategies:
1. Take a strategy
2. Mutate 100+ ways (flip logic, swap indicators, combine, threshold shifts)
3. Backtest every mutation
4. Keep top performers
5. Mutate again (evolutionary loop)
"""

import random
import copy
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import itertools
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================
# MUTATION TYPES
# ============================================================
class MutationType:
    """Types of mutations to apply to strategies."""
    
    # Entry/Exit flips
    FLIP_ENTRY_LOGIC = "flip_entry"
    FLIP_EXIT_LOGIC = "flip_exit"
    SWAP_ENTRY_EXIT = "swap_entry_exit"
    
    # Indicator swaps
    REPLACE_INDICATOR = "replace_indicator"
    ADD_INDICATOR = "add_indicator"
    REMOVE_INDICATOR = "remove_indicator"
    
    # Threshold shifts
    SHIFT_THRESHOLD = "shift_threshold"
    TIGHTEN_THRESHOLD = "tighten_threshold"
    WIDEN_THRESHOLD = "widen_threshold"
    
    # Parameter mutations
    CHANGE_PERIOD = "change_period"
    MULTIPLY_PERIOD = "multiply_period"
    INVERT_PARAMETER = "invert_parameter"
    
    # Combination mutations
    COMBINE_WITH_OTHER = "combine_strategies"
    ADD_FILTER = "add_filter"
    ADD_EXIT_FILTER = "add_exit_filter"
    
    # Logic mutations
    ADD_TIME_FILTER = "add_time_filter"
    ADD_VOLUME_FILTER = "add_volume_filter"
    INVERT_SIGNAL = "invert_signal"
    
    # Risk mutations
    CHANGE_POSITION_SIZE = "change_position_size"
    ADD_STOP_LOSS = "add_stop_loss"
    ADD_TAKE_PROFIT = "add_take_profit"


# ============================================================
# INDICATOR REPLACEMENTS
# ============================================================
INDICATOR_REPLACEMENTS = {
    'sma': ['ema', 'wma', 'hma', 'tema'],
    'ema': ['sma', 'wma', 'hma', 'tema'],
    'rsi': ['stoch_rsi', 'mfi', 'cci', 'williams_r'],
    'stoch_rsi': ['rsi', 'mfi', 'cci'],
    'macd': ['rsi', 'stoch', 'adx'],
    'bollinger': ['keltner', 'donchian', 'atr_bands'],
    'atr': ['stddev', 'historical_volatility'],
    'adx': ['rsi', 'macd', 'trix'],
    'volume': ['obv', 'adl', 'cmf'],
    'obv': ['volume', 'adl', 'cmf'],
}


# ============================================================
# STRATEGY TEMPLATES
# ============================================================
class StrategyTemplate:
    """Template for strategy mutations."""
    
    def __init__(self, name: str, params: Dict[str, Any], entry_logic: str, exit_logic: str):
        self.name = name
        self.params = params
        self.entry_logic = entry_logic
        self.exit_logic = exit_logic
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'params': self.params,
            'entry_logic': self.entry_logic,
            'exit_logic': self.exit_logic
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'StrategyTemplate':
        return cls(d['name'], d['params'], d['entry_logic'], d['exit_logic'])


# ============================================================
# MUTATION OPERATORS
# ============================================================
class MutationOperators:
    """Operators for mutating strategies."""
    
    @staticmethod
    def flip_entry_logic(template: StrategyTemplate) -> StrategyTemplate:
        """Flip entry conditions (buy <-> sell)."""
        new_template = copy.deepcopy(template)
        
        # Flip comparison operators
        new_template.entry_logic = template.entry_logic.replace('<', '>_temp').replace('>', '<')
        new_template.entry_logic = new_template.entry_logic.replace('>_temp', '>')
        
        # Swap buy/sell keywords
        new_template.entry_logic = new_template.entry_logic.replace('buy', '_BUY_')
        new_template.entry_logic = new_template.entry_logic.replace('sell', 'buy')
        new_template.entry_logic = new_template.entry_logic.replace('_BUY_', 'sell')
        
        new_template.name = f"{template.name}_FLIP"
        
        return new_template
    
    @staticmethod
    def flip_exit_logic(template: StrategyTemplate) -> StrategyTemplate:
        """Flip exit conditions."""
        new_template = copy.deepcopy(template)
        new_template.exit_logic = template.exit_logic.replace('<', '>_temp').replace('>', '<')
        new_template.exit_logic = new_template.exit_logic.replace('>_temp', '>')
        new_template.name = f"{template.name}_EXIT_FLIP"
        return new_template
    
    @staticmethod
    def swap_indicators(template: StrategyTemplate) -> StrategyTemplate:
        """Replace one indicator with another."""
        new_template = copy.deepcopy(template)
        
        # Find and replace indicators in params
        for key, value in new_template.params.items():
            if isinstance(value, str) and value in INDICATOR_REPLACEMENTS:
                replacements = INDICATOR_REPLACEMENTS[value]
                new_value = random.choice(replacements)
                new_template.params[key] = new_value
                new_template.name = f"{template.name}_{value.upper()}_TO_{new_value.upper()}"
                break
        
        return new_template
    
    @staticmethod
    def shift_threshold(template: StrategyTemplate) -> StrategyTemplate:
        """Shift threshold values."""
        new_template = copy.deepcopy(template)
        
        threshold_keys = ['oversold', 'overbought', 'threshold', 'rsi_period', 'rsi_oversold', 'rsi_overbought']
        
        for key in threshold_keys:
            if key in new_template.params:
                old_val = new_template.params[key]
                if isinstance(old_val, (int, float)):
                    # Shift by random amount (-50% to +50%)
                    shift = random.uniform(0.5, 1.5)
                    new_template.params[key] = int(old_val * shift) if isinstance(old_val, int) else old_val * shift
                    new_template.name = f"{template.name}_THRESH_SHIFT"
        
        return new_template
    
    @staticmethod
    def tighten_threshold(template: StrategyTemplate) -> StrategyTemplate:
        """Make thresholds tighter (more selective)."""
        new_template = copy.deepcopy(template)
        
        threshold_keys = ['oversold', 'overbought', 'threshold']
        
        for key in threshold_keys:
            if key in new_template.params:
                old_val = new_template.params[key]
                if isinstance(old_val, (int, float)):
                    if 'overbought' in key or key == 'threshold':
                        new_template.params[key] = old_val * 0.8  # Tighter
                    elif 'oversold' in key:
                        new_template.params[key] = old_val * 1.2  # Tighter (closer to center)
                    new_template.name = f"{template.name}_TIGHTEN"
        
        return new_template
    
    @staticmethod
    def widen_threshold(template: StrategyTemplate) -> StrategyTemplate:
        """Make thresholds wider (less selective)."""
        new_template = copy.deepcopy(template)
        
        threshold_keys = ['oversold', 'overbought', 'threshold']
        
        for key in threshold_keys:
            if key in new_template.params:
                old_val = new_template.params[key]
                if isinstance(old_val, (int, float)):
                    if 'overbought' in key or key == 'threshold':
                        new_template.params[key] = old_val * 1.2
                    elif 'oversold' in key:
                        new_template.params[key] = old_val * 0.8
                    new_template.name = f"{template.name}_WIDEN"
        
        return new_template
    
    @staticmethod
    def change_period(template: StrategyTemplate) -> StrategyTemplate:
        """Change period parameters."""
        new_template = copy.deepcopy(template)
        
        period_keys = [k for k in template.params.keys() if 'period' in k.lower()]
        
        if period_keys:
            key = random.choice(period_keys)
            old_val = template.params[key]
            
            # Change by random amount
            change = random.choice([-3, -2, -1, 1, 2, 3, 2, 0.5, 0.75, 1.25, 1.5])
            
            if isinstance(old_val, int):
                new_val = max(1, int(old_val * change) if change < 1 else old_val + change)
            else:
                new_val = max(1, old_val * change)
            
            new_template.params[key] = new_val
            new_template.name = f"{template.name}_PERIOD_{old_val}_TO_{new_val}"
        
        return new_template
    
    @staticmethod
    def multiply_period(template: StrategyTemplate) -> StrategyTemplate:
        """Multiply all periods by a factor."""
        new_template = copy.deepcopy(template)
        
        period_keys = [k for k in template.params.keys() if 'period' in k.lower()]
        
        if period_keys:
            factor = random.choice([0.5, 0.75, 1.25, 1.5, 2.0])
            
            for key in period_keys:
                old_val = template.params[key]
                if isinstance(old_val, int):
                    new_template.params[key] = max(1, int(old_val * factor))
            
            new_template.name = f"{template.name}_PERIOD_X{factor}"
        
        return new_template
    
    @staticmethod
    def add_indicator(template: StrategyTemplate) -> StrategyTemplate:
        """Add a new indicator parameter."""
        new_template = copy.deepcopy(template)
        
        new_indicators = {
            'ema_period': 20,
            'atr_period': 14,
            'volume_filter': True,
            'volatility_filter': True,
            'trend_filter': True,
        }
        
        key = random.choice(list(new_indicators.keys()))
        if key not in new_template.params:
            new_template.params[key] = new_indicators[key]
            new_template.name = f"{template.name}_ADD_{key}"
        
        return new_template
    
    @staticmethod
    def invert_signal(template: StrategyTemplate) -> StrategyTemplate:
        """Invert the entire signal logic."""
        new_template = copy.deepcopy(template)
        
        # Wrap entry logic in NOT
        new_template.entry_logic = f"NOT({template.entry_logic})"
        
        # Swap entry/exit
        new_template.entry_logic, new_template.exit_logic = template.exit_logic, template.entry_logic
        
        new_template.name = f"{template.name}_INVERT"
        
        return new_template
    
    @staticmethod
    def add_volume_filter(template: StrategyTemplate) -> StrategyTemplate:
        """Add volume confirmation filter."""
        new_template = copy.deepcopy(template)
        
        new_template.params['volume_filter'] = True
        new_template.params['volume_threshold'] = random.uniform(0.5, 1.5)
        
        new_template.name = f"{template.name}_VOL_FILTER"
        
        return new_template
    
    @staticmethod
    def add_time_filter(template: StrategyTemplate) -> StrategyTemplate:
        """Add time-based filter."""
        new_template = copy.deepcopy(template)
        
        # Add session filter
        sessions = ['asian', 'london', 'new_york', 'overlap']
        new_template.params['session_filter'] = random.choice(sessions)
        
        new_template.name = f"{template.name}_TIME_FILTER"
        
        return new_template
    
    @staticmethod
    def change_position_size(template: StrategyTemplate) -> StrategyTemplate:
        """Change position sizing method."""
        new_template = copy.deepcopy(template)
        
        sizing_methods = ['fixed', 'kelly', 'atr', 'volatility']
        new_template.params['sizing_method'] = random.choice(sizing_methods)
        
        if 'position_size' in new_template.params:
            new_template.params['position_size'] = random.uniform(0.05, 0.25)
        
        new_template.name = f"{template.name}_SIZE_{new_template.params['sizing_method']}"
        
        return new_template
    
    @staticmethod
    def add_stop_loss(template: StrategyTemplate) -> StrategyTemplate:
        """Add or modify stop loss."""
        new_template = copy.deepcopy(template)
        
        new_template.params['stop_loss_pct'] = random.uniform(0.02, 0.10)
        new_template.name = f"{template.name}_SL_{int(new_template.params['stop_loss_pct']*100)}%"
        
        return new_template
    
    @staticmethod
    def add_take_profit(template: StrategyTemplate) -> StrategyTemplate:
        """Add or modify take profit."""
        new_template = copy.deepcopy(template)
        
        new_template.params['take_profit_pct'] = random.uniform(0.05, 0.20)
        new_template.name = f"{template.name}_TP_{int(new_template.params['take_profit_pct']*100)}%"
        
        return new_template


# ============================================================
# STRATEGY MUTATOR
# ============================================================
class StrategyMutator:
    """
    Main mutation engine.
    
    Takes strategies, applies mutations, returns mutated versions.
    """
    
    def __init__(
        self,
        mutation_count: int = 100,
        keep_original: bool = True,
        mutation_types: List[str] = None
    ):
        self.mutation_count = mutation_count
        self.keep_original = keep_original
        self.mutation_types = mutation_types or [
            MutationType.FLIP_ENTRY_LOGIC,
            MutationType.FLIP_EXIT_LOGIC,
            MutationType.REPLACE_INDICATOR,
            MutationType.SHIFT_THRESHOLD,
            MutationType.TIGHTEN_THRESHOLD,
            MutationType.WIDEN_THRESHOLD,
            MutationType.CHANGE_PERIOD,
            MutationType.MULTIPLY_PERIOD,
            MutationType.ADD_INDICATOR,
            MutationType.INVERT_SIGNAL,
            MutationType.ADD_VOLUME_FILTER,
            MutationType.ADD_TIME_FILTER,
            MutationType.CHANGE_POSITION_SIZE,
            MutationType.ADD_STOP_LOSS,
            MutationType.ADD_TAKE_PROFIT,
        ]
        
        self.operators = MutationOperators()
    
    def mutate(self, template: StrategyTemplate) -> List[StrategyTemplate]:
        """Generate N mutated versions of a strategy."""
        mutations = []
        
        # Keep original
        if self.keep_original:
            mutations.append(copy.deepcopy(template))
        
        # Apply random mutations
        while len(mutations) < self.mutation_count:
            mutated = self._apply_random_mutation(template)
            mutations.append(mutated)
        
        return mutations
    
    def _apply_random_mutation(self, template: StrategyTemplate) -> StrategyTemplate:
        """Apply a random mutation to a template."""
        mutation_type = random.choice(self.mutation_types)
        
        try:
            if mutation_type == MutationType.FLIP_ENTRY_LOGIC:
                return self.operators.flip_entry_logic(template)
            elif mutation_type == MutationType.FLIP_EXIT_LOGIC:
                return self.operators.flip_exit_logic(template)
            elif mutation_type == MutationType.REPLACE_INDICATOR:
                return self.operators.swap_indicators(template)
            elif mutation_type == MutationType.SHIFT_THRESHOLD:
                return self.operators.shift_threshold(template)
            elif mutation_type == MutationType.TIGHTEN_THRESHOLD:
                return self.operators.tighten_threshold(template)
            elif mutation_type == MutationType.WIDEN_THRESHOLD:
                return self.operators.widen_threshold(template)
            elif mutation_type == MutationType.CHANGE_PERIOD:
                return self.operators.change_period(template)
            elif mutation_type == MutationType.MULTIPLY_PERIOD:
                return self.operators.multiply_period(template)
            elif mutation_type == MutationType.ADD_INDICATOR:
                return self.operators.add_indicator(template)
            elif mutation_type == MutationType.INVERT_SIGNAL:
                return self.operators.invert_signal(template)
            elif mutation_type == MutationType.ADD_VOLUME_FILTER:
                return self.operators.add_volume_filter(template)
            elif mutation_type == MutationType.ADD_TIME_FILTER:
                return self.operators.add_time_filter(template)
            elif mutation_type == MutationType.CHANGE_POSITION_SIZE:
                return self.operators.change_position_size(template)
            elif mutation_type == MutationType.ADD_STOP_LOSS:
                return self.operators.add_stop_loss(template)
            elif mutation_type == MutationType.ADD_TAKE_PROFIT:
                return self.operators.add_take_profit(template)
            else:
                return copy.deepcopy(template)
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            return copy.deepcopy(template)


# ============================================================
# GENETIC EVOLUTION ENGINE
# ============================================================
@dataclass
class BacktestResult:
    """Result of a backtest."""
    template: StrategyTemplate
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    profit_factor: float
    fitness_score: float = 0.0
    
    def __post_init__(self):
        # Calculate fitness score (can be customized)
        # Higher is better: combines return, Sharpe, drawdown
        self.fitness_score = (
            self.total_return * 0.4 +
            self.sharpe_ratio * 20 * 0.3 -
            self.max_drawdown * 0.3
        )


class EvolutionEngine:
    """
    Genetic evolution for trading strategies.
    
    Flow:
    1. Start with seed strategy
    2. Generate N mutations
    3. Backtest each mutation
    4. Select top performers
    5. Mutate top performers again
    6. Repeat for N generations
    """
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 5,
        top_percent: float = 0.2,
        mutation_count: int = 100,
        backtest_function: Callable = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.top_percent = top_percent
        self.mutation_count = mutation_count
        self.backtest_function = backtest_function
        self.mutator = StrategyMutator(mutation_count=mutation_count)
        
        self.best_strategies: List[BacktestResult] = []
        self.all_results: List[BacktestResult] = []
    
    def evolve(
        self,
        seed_template: StrategyTemplate,
        data: Any = None,
        verbose: bool = True
    ) -> List[BacktestResult]:
        """
        Run evolutionary optimization.
        
        Args:
            seed_template: Starting strategy template
            data: Market data for backtesting
            verbose: Print progress
        
        Returns:
            List of best strategies after evolution
        """
        current_population = [seed_template]
        
        for gen in range(self.generations):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Generation {gen + 1}/{self.generations}")
                print(f"{'='*50}")
            
            # Generate mutations
            all_mutation_templates = []
            for template in current_population:
                mutations = self.mutator.mutate(template)
                all_mutation_templates.extend(mutations)
            
            if verbose:
                print(f"Testing {len(all_mutation_templates)} strategies...")
            
            # Backtest all mutations
            generation_results = []
            for i, template in enumerate(all_mutation_templates):
                try:
                    if self.backtest_function:
                        result = self.backtest_function(template, data)
                    else:
                        # Demo backtest (replace with real backtest)
                        result = self._demo_backtest(template)
                    
                    generation_results.append(result)
                    
                    if verbose and (i + 1) % 20 == 0:
                        print(f"  Tested {i + 1}/{len(all_mutation_templates)}...")
                        
                except Exception as e:
                    logger.warning(f"Backtest failed for {template.name}: {e}")
            
            # Sort by fitness
            generation_results.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Store all results
            self.all_results.extend(generation_results)
            
            # Select top performers
            top_count = max(1, int(len(generation_results) * self.top_percent))
            top_performers = generation_results[:top_count]
            
            if verbose:
                print(f"\nTop {top_count} strategies:")
                for i, res in enumerate(top_performers[:5]):
                    print(f"  {i+1}. {res.template.name}: "
                          f"Return={res.total_return:.2f}%, "
                          f"Sharpe={res.sharpe_ratio:.2f}, "
                          f"DD={res.max_drawdown:.2f}%, "
                          f"Fitness={res.fitness_score:.3f}")
            
            # Set top performers as next generation
            current_population = [r.template for r in top_performers]
        
        # Final results
        self.best_strategies = sorted(self.all_results, key=lambda x: x.fitness_score, reverse=True)
        
        return self.best_strategies
    
    def evolve_with_diversity(
        self,
        seed_template: StrategyTemplate,
        data: Any = None,
        diversity_threshold: float = 0.15,
        verbose: bool = True
    ) -> List[BacktestResult]:
        """
        Run evolutionary optimization WITH diversity monitoring.
        
        Prevents premature convergence by:
        1. Tracking population diversity
        2. Triggering mutagen events when diversity drops
        3. Injecting random strategies
        
        Args:
            seed_template: Starting strategy template
            data: Market data for backtesting
            diversity_threshold: Trigger mutagen below this
            verbose: Print progress
        
        Returns:
            List of best strategies after evolution
        """
        from .diversity_monitor import DiversityMonitor, StrategyGenome
        
        # Initialize diversity monitor
        diversity_monitor = DiversityMonitor(threshold=diversity_threshold)
        
        # Create initial population from seed
        current_population = [seed_template]
        
        for gen in range(self.generations):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Generation {gen + 1}/{self.generations}")
                print(f"{'='*50}")
            
            # Generate mutations
            all_mutation_templates = []
            for template in current_population:
                mutations = self.mutator.mutate(template)
                all_mutation_templates.extend(mutations)
            
            if verbose:
                print(f"Testing {len(all_mutation_templates)} strategies...")
            
            # Backtest all mutations
            generation_results = []
            for i, template in enumerate(all_mutation_templates):
                try:
                    if self.backtest_function:
                        result = self.backtest_function(template, data)
                    else:
                        result = self._demo_backtest(template)
                    
                    generation_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Backtest failed: {e}")
            
            # === DIVERSITY CHECK ===
            # Convert to genomes for diversity analysis
            population_for_diversity = [
                {'id': f'gen{gen}_strat{i}', 'params': r.template.params, 'fitness': r.fitness_score}
                for i, r in enumerate(generation_results)
            ]
            
            stats = diversity_monitor.update(population_for_diversity, generation=gen)
            
            if verbose:
                print(f"\n📊 Diversity: AvgDist={stats.avg_pairwise_distance:.3f}, "
                      f"Species={stats.species_count}, Conv={stats.convergence_score:.1%}")
            
            # === MUTAGEN TRIGGER ===
            if stats.mutagen_triggered:
                if verbose:
                    print("\n🔥 MUTAGEN TRIGGERED! Injecting diversity...")
                
                # Trigger mutagen
                diversity_monitor.analyzer.trigger_mutagen(population_for_diversity)
                
                # Add some completely random strategies
                random_strategies = self._create_random_strategies(10)
                for rs in random_strategies:
                    try:
                        result = self._demo_backtest(rs)
                        generation_results.append(result)
                    except:
                        pass
            
            # Sort by fitness
            generation_results.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Store all results
            self.all_results.extend(generation_results)
            
            # Select top performers
            top_count = max(1, int(len(generation_results) * self.top_percent))
            top_performers = generation_results[:top_count]
            
            if verbose:
                print(f"\nTop {top_count} strategies:")
                for i, res in enumerate(top_performers[:5]):
                    print(f"  {i+1}. {res.template.name}: "
                          f"Return={res.total_return:.2f}%, "
                          f"Sharpe={res.sharpe_ratio:.2f}, "
                          f"Fitness={res.fitness_score:.3f}")
            
            # Set top performers as next generation
            current_population = [r.template for r in top_performers]
        
        # Final results
        self.best_strategies = sorted(self.all_results, key=lambda x: x.fitness_score, reverse=True)
        
        if verbose:
            print(f"\n🧬 Total mutagen events: {diversity_monitor.analyzer.mutagen_count}")
        
        return self.best_strategies
    
    def _create_random_strategies(self, n: int) -> List[StrategyTemplate]:
        """Create random strategy templates for diversity injection."""
        import random
        
        templates = []
        indicators = ['rsi', 'macd', 'bollinger', 'sma', 'ema']
        
        for i in range(n):
            ind = random.choice(indicators)
            template = StrategyTemplate(
                name=f"random_{i}_{ind}",
                params={
                    'indicator': ind,
                    'period': random.choice([7, 10, 14, 20, 30]),
                    'oversold': random.randint(15, 35),
                    'overbought': random.randint(65, 85),
                    'position_size': random.uniform(0.05, 0.25),
                },
                entry_logic=f"{ind}_signal",
                exit_logic="profit_target"
            )
            templates.append(template)
        
        return templates
        
        if verbose:
            print(f"\n{'='*50}")
            print("EVOLUTION COMPLETE")
            print(f"{'='*50}")
            print(f"Total strategies tested: {len(self.all_results)}")
            print(f"\n🏆 TOP 10 STRATEGIES:")
            for i, res in enumerate(self.best_strategies[:10]):
                print(f"  {i+1}. {res.template.name}")
                print(f"      Return: {res.total_return:.2f}% | Sharpe: {res.sharpe_ratio:.2f} | DD: {res.max_drawdown:.2f}%")
        
        return self.best_strategies
    
    def _demo_backtest(self, template: StrategyTemplate) -> BacktestResult:
        """Demo backtest with random results (replace with real backtest)."""
        import numpy as np
        
        # Generate deterministic results based on template
        seed = hash(template.name) % 10000
        np.random.seed(seed)
        
        # Parameters affect outcomes
        period_factor = template.params.get('period', 20) / 20
        threshold_factor = template.params.get('threshold', 50) / 50
        
        # Base metrics
        base_return = np.random.normal(10, 20) * period_factor
        base_sharpe = np.random.normal(1.0, 0.8)
        base_dd = abs(np.random.normal(8, 5))
        base_win = np.random.normal(0.55, 0.15)
        
        # Apply threshold effects
        if 'TIGHTEN' in template.name:
            base_return *= 1.1
            base_sharpe *= 1.1
            base_dd *= 0.9
        elif 'WIDEN' in template.name:
            base_return *= 0.9
            base_dd *= 1.1
        
        trade_count = int(np.random.randint(20, 200))
        avg_trade = base_return / 100 * 100000 / trade_count if trade_count > 0 else 0
        profit_factor = np.random.uniform(1.0, 2.5)
        
        return BacktestResult(
            template=template,
            total_return=base_return,
            sharpe_ratio=base_sharpe,
            max_drawdown=base_dd,
            win_rate=base_win,
            trade_count=trade_count,
            profit_factor=profit_factor
        )
    
    def save_results(self, filepath: str = "evolution_results.json"):
        """Save evolution results to JSON file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'generations': self.generations,
            'total_tested': len(self.all_results),
            'population_size': self.population_size,
            'mutation_rate': getattr(self, 'mutation_rate', 0.3),
            'best_strategies': [
                {
                    'name': r.template.name,
                    'params': r.template.params,
                    'entry_logic': r.template.entry_logic,
                    'exit_logic': r.template.exit_logic,
                    'total_return': r.total_return,
                    'sharpe_ratio': r.sharpe_ratio,
                    'max_drawdown': r.max_drawdown,
                    'win_rate': r.win_rate,
                    'trade_count': r.trade_count,
                    'fitness_score': r.fitness_score,
                    'profit_factor': getattr(r, 'profit_factor', 0),
                    'avg_trade': getattr(r, 'avg_trade', 0),
                    'avg_win': getattr(r, 'avg_win', 0),
                    'avg_loss': getattr(r, 'avg_loss', 0),
                    'consecutive_wins': getattr(r, 'consecutive_wins', 0),
                    'consecutive_losses': getattr(r, 'consecutive_losses', 0),
                }
                for r in self.best_strategies[:100]
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
        return filepath
    
    def export_to_csv(self, filepath: str = "strategies.csv") -> str:
        """
        Export best strategies to CSV with ALL parameters.
        
        Includes:
        - Strategy metadata (name, generation)
        - Entry/Exit logic
        - Performance metrics (return, sharpe, drawdown, win rate)
        - ALL strategy parameters (100+ fields)
        - Trade statistics
        """
        import csv
        
        # Flatten all parameters
        rows = []
        
        for r in self.best_strategies[:100]:
            params = r.template.params if hasattr(r.template, 'params') else {}
            
            # Base row
            row = {
                # Metadata
                'strategy_name': r.template.name,
                'generation': getattr(r, 'generation', 0),
                'entry_logic': r.template.entry_logic,
                'exit_logic': r.template.exit_logic,
                
                # Performance
                'total_return_pct': r.total_return,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown_pct': r.max_drawdown,
                'win_rate': r.win_rate,
                'trade_count': r.trade_count,
                'fitness_score': r.fitness_score,
                'profit_factor': getattr(r, 'profit_factor', 0),
                
                # Trade stats
                'avg_trade_pct': getattr(r, 'avg_trade', 0),
                'avg_win_pct': getattr(r, 'avg_win', 0),
                'avg_loss_pct': getattr(r, 'avg_loss', 0),
                'consecutive_wins': getattr(r, 'consecutive_wins', 0),
                'consecutive_losses': getattr(r, 'consecutive_losses', 0),
                
                # Risk metrics
                'risk_adjusted_return': r.total_return / (r.max_drawdown + 0.001) if r.max_drawdown > 0 else 0,
                'calmar_ratio': r.total_return / (abs(r.max_drawdown) + 0.001) if r.max_drawdown != 0 else 0,
            }
            
            # Add ALL parameters (100+ possible)
            # Entry parameters
            row['rsi_period'] = params.get('rsi_period', '')
            row['rsi_oversold'] = params.get('oversold', '')
            row['rsi_overbought'] = params.get('overbought', '')
            row['macd_fast'] = params.get('macd_fast', '')
            row['macd_slow'] = params.get('macd_slow', '')
            row['macd_signal'] = params.get('macd_signal', '')
            row['sma_period'] = params.get('sma_period', '')
            row['ema_period'] = params.get('ema_period', '')
            row['bb_period'] = params.get('bb_period', '')
            row['bb_std'] = params.get('bb_std', '')
            row['atr_period'] = params.get('atr_period', '')
            row['adx_period'] = params.get('adx_period', '')
            row['stoch_period'] = params.get('stoch_period', '')
            row['cci_period'] = params.get('cci_period', '')
            
            # Position management
            row['position_size'] = params.get('position_size', '')
            row['max_position_pct'] = params.get('max_position_pct', '')
            row['use_kelly'] = params.get('use_kelly', False)
            row['kelly_fraction'] = params.get('kelly_fraction', '')
            
            # Stop loss / Take profit
            row['stop_loss_pct'] = params.get('stop_loss', '')
            row['take_profit_pct'] = params.get('take_profit', '')
            row['trailing_stop_pct'] = params.get('trailing_stop', '')
            row['use_trailing_stop'] = params.get('use_trailing_stop', False)
            row['atr_multiplier'] = params.get('atr_multiplier', '')
            
            # Filters
            row['volume_filter'] = params.get('volume_filter', False)
            row['time_filter'] = params.get('time_filter', False)
            row['min_volume'] = params.get('min_volume', '')
            row['session_filter'] = params.get('session_filter', '')
            
            # Signal modifiers
            row['invert_signal'] = params.get('invert', False)
            row['confirm_indicators'] = params.get('confirm_indicators', '')
            row['require_multiple_signals'] = params.get('require_multiple_signals', False)
            
            # Time exits
            row['max_bars'] = params.get('max_bars', '')
            row['time_exit_bars'] = params.get('time_exit_bars', '')
            
            # Advanced
            row['regime_filter'] = params.get('regime_filter', '')
            row['volatility_adjust'] = params.get('volatility_adjust', False)
            row['use_correlation'] = params.get('use_correlation', False)
            
            # Regime-specific
            row['bull_params'] = str(params.get('bull_params', {}))
            row['bear_params'] = str(params.get('bear_params', {}))
            row['sideways_params'] = str(params.get('sideways_params', {}))
            
            rows.append(row)
        
        if not rows:
            logger.warning("No strategies to export")
            return ""
        
        # Get all unique keys for CSV header
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(list(all_keys)))
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"CSV exported to {filepath}")
        
        return filepath
    
    def get_results_summary(self) -> Dict:
        """Get comprehensive results summary."""
        if not self.best_strategies:
            return {}
        
        returns = [r.total_return for r in self.best_strategies[:20]]
        sharpes = [r.sharpe_ratio for r in self.best_strategies[:20]]
        dds = [r.max_drawdown for r in self.best_strategies[:20]]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'generations': self.generations,
            'total_tested': len(self.all_results),
            'best_return': max(returns) if returns else 0,
            'avg_return': np.mean(returns) if returns else 0,
            'best_sharpe': max(sharpes) if sharpes else 0,
            'avg_sharpe': np.mean(sharpes) if sharpes else 0,
            'best_drawdown': min(dds) if dds else 0,
            'avg_drawdown': np.mean(dds) if dds else 0,
            'top_strategies': [
                {
                    'name': r.template.name,
                    'return': r.total_return,
                    'sharpe': r.sharpe_ratio,
                    'drawdown': r.max_drawdown,
                    'fitness': r.fitness_score
                }
                for r in self.best_strategies[:5]
            ]
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================
def demo():
    """Demo the mutation engine."""
    print("🚀 Strategy Mutation Engine Demo")
    print("=" * 50)
    
    # Seed strategy
    seed = StrategyTemplate(
        name="RSI_MeanReversion",
        params={
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'position_size': 0.10
        },
        entry_logic="rsi < oversold -> buy",
        exit_logic="rsi > overbought -> sell"
    )
    
    print(f"\n📊 Original Strategy: {seed.name}")
    print(f"   Params: {seed.params}")
    
    # Create mutator
    mutator = StrategyMutator(mutation_count=20)
    
    # Generate mutations
    print(f"\n🔄 Generating {20} mutations...")
    mutations = mutator.mutate(seed)
    
    print("\n📋 Sample Mutations:")
    for i, m in enumerate(mutations[:10]):
        print(f"  {i+1}. {m.name}")
        print(f"      Params: {m.params}")
    
    # Run evolution
    print("\n" + "=" * 50)
    print("🧬 Running Evolution (3 generations)...")
    
    engine = EvolutionEngine(
        population_size=20,
        generations=3,
        top_percent=0.3,
        mutation_count=20
    )
    
    results = engine.evolve(seed, verbose=True)
    
    # Save results
    engine.save_results("evolution_results.json")
    print("\n💾 Results saved to evolution_results.json")
    
    return results


if __name__ == "__main__":
    demo()
