"""
QuantCore - Genetic Programming Strategy Evolution

A GP approach where strategies are trees of operators:
- Indicators (SMA, EMA, RSI, MACD, etc.)
- Logical gates (AND, OR, NOT, >, <, ==)
- Arithmetic (+, -, *, /)
- Terminals (constants, price fields)

This allows discovering novel strategies no human imagined!
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import random
import copy
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================
# STRONGLY-TYPED GP NODES
# ============================================================
class GPType:
    """Types for strongly-typed GP."""
    PRICE = "price"           # Scalar from price series
    SERIES = "series"         # Time series
    BOOLEAN = "bool"          # True/False series
    CONSTANT = "const"        # Constant value


@dataclass
class GPNode:
    """A node in the GP tree."""
    name: str
    node_type: str
    value: Any = None
    children: List['GPNode'] = field(default_factory=list)
    arity: int = 0
    
    def __repr__(self):
        if self.node_type == GPType.CONSTANT:
            return f"{self.name}({self.value})"
        elif self.children:
            return f"({self.name} {' '.join(map(str, self.children))})"
        return self.name
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.node_type,
            'value': self.value,
            'children': [c.to_dict() for c in self.children]
        }


# ============================================================
# PRIMITIVE LIBRARY
# ============================================================
class PrimitiveLibrary:
    """Library of GP primitives."""
    
    CONSTANTS = [
        ('const_5', GPType.CONSTANT, 5),
        ('const_10', GPType.CONSTANT, 10),
        ('const_14', GPType.CONSTANT, 14),
        ('const_20', GPType.CONSTANT, 20),
        ('const_30', GPType.CONSTANT, 30),
        ('const_50', GPType.CONSTANT, 50),
    ]
    
    PRICE_TERMINALS = [
        ('close', GPType.SERIES),
        ('open', GPType.SERIES),
        ('high', GPType.SERIES),
        ('low', GPType.SERIES),
        ('volume', GPType.SERIES),
    ]
    
    INDICATORS = [
        ('sma', GPType.SERIES, 1),
        ('ema', GPType.SERIES, 1),
        ('rsi', GPType.SERIES, 1),
        ('macd', GPType.SERIES, 0),
        ('bb_upper', GPType.SERIES, 0),
        ('bb_lower', GPType.SERIES, 0),
        ('atr', GPType.SERIES, 0),
        ('momentum', GPType.SERIES, 1),
    ]
    
    ARITHMETIC = [
        ('add', GPType.SERIES, 2),
        ('sub', GPType.SERIES, 2),
        ('mul', GPType.SERIES, 2),
        ('div', GPType.SERIES, 2),
    ]
    
    COMPARISONS = [
        ('gt', GPType.BOOLEAN, 2),
        ('lt', GPType.BOOLEAN, 2),
        ('gte', GPType.BOOLEAN, 2),
        ('lte', GPType.BOOLEAN, 2),
    ]
    
    LOGICAL = [
        ('and', GPType.BOOLEAN, 2),
        ('or', GPType.BOOLEAN, 2),
    ]
    
    CROSSES = [
        ('crosses_above', GPType.BOOLEAN, 2),
        ('crosses_below', GPType.BOOLEAN, 2),
    ]


# ============================================================
# GP TREE BUILDER
# ============================================================
class GPTreeBuilder:
    """Builds valid GP trees with strongly-typed constraints."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        
    def create_random_tree(self, return_type: str, depth: int = 0) -> GPNode:
        """Create a random tree of the specified type."""
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.3):
            return self._create_terminal(return_type)
        
        functions = self._get_functions(return_type)
        
        if not functions:
            return self._create_terminal(return_type)
        
        func = random.choice(functions)
        
        node = GPNode(name=func[0], node_type=func[1], arity=func[2])
        
        # Create children
        for _ in range(node.arity):
            child_type = GPType.SERIES if return_type == GPType.SERIES else return_type
            if return_type == GPType.BOOLEAN and random.random() < 0.3:
                child_type = GPType.SERIES
            child = self.create_random_tree(child_type, depth + 1)
            node.children.append(child)
        
        return node
    
    def _create_terminal(self, return_type: str) -> GPNode:
        """Create a terminal node."""
        if return_type == GPType.CONSTANT:
            p = random.choice(PrimitiveLibrary.CONSTANTS)
            return GPNode(name=p[0], node_type=p[1], value=p[2])
        else:
            p = random.choice(PrimitiveLibrary.PRICE_TERMINALS)
            return GPNode(name=p[0], node_type=p[1])
    
    def _get_functions(self, return_type: str) -> List:
        functions = []
        if return_type == GPType.SERIES:
            functions.extend(PrimitiveLibrary.INDICATORS)
            functions.extend(PrimitiveLibrary.ARITHMETIC)
        elif return_type == GPType.BOOLEAN:
            functions.extend(PrimitiveLibrary.COMPARISONS)
            functions.extend(PrimitiveLibrary.LOGICAL)
            functions.extend(PrimitiveLibrary.CROSSES)
        return functions


# ============================================================
# GP TREE MUTATORS
# ============================================================
class GPMutator:
    """Mutation operators for GP trees."""
    
    def __init__(self, tree_builder: GPTreeBuilder):
        self.builder = tree_builder
    
    def mutate(self, node: GPNode, probability: float = 0.3) -> GPNode:
        """Apply random mutation."""
        if random.random() > probability:
            return node
        
        new_node = copy.deepcopy(node)
        mutation_type = random.choice(['subtree', 'node', 'grow'])
        
        if mutation_type == 'subtree':
            return self._mutate_subtree(new_node)
        elif mutation_type == 'grow':
            return self._mutate_grow(new_node)
        return new_node
    
    def _mutate_subtree(self, node: GPNode) -> GPNode:
        """Replace a random subtree."""
        if not node.children:
            return self.builder.create_random_tree(node.node_type)
        
        idx = random.randint(0, len(node.children) - 1)
        node.children[idx] = self.builder.create_random_tree(node.children[idx].node_type)
        
        return node
    
    def _mutate_grow(self, node: GPNode) -> GPNode:
        """Grow a new branch."""
        terminals = []
        self._find_terminals(node, terminals)
        
        if terminals:
            target = random.choice(terminals)
            parent, idx = self._find_parent(node, target)
            if parent:
                parent.children[idx] = self.builder.create_random_tree(target.node_type, depth=2)
        
        return node
    
    def _find_terminals(self, node: GPNode, terminals: List):
        if not node.children:
            terminals.append(node)
        for child in node.children:
            self._find_terminals(child, terminals)
    
    def _find_parent(self, node: GPNode, target: GPNode) -> Tuple:
        for i, child in enumerate(node.children):
            if child is target:
                return node, i
            parent, idx = self._find_parent(child, target)
            if parent:
                return parent, idx
        return None, -1


# ============================================================
# TREE EVALUATOR
# ============================================================
class GPTreeEvaluator:
    """Evaluates GP trees on market data."""
    
    def evaluate(self, node: GPNode, data: pd.DataFrame) -> np.ndarray:
        """Evaluate tree."""
        if node.node_type == GPType.CONSTANT:
            return np.full(len(data), node.value)
        
        elif node.node_type == GPType.SERIES:
            return self._eval_series(node, data)
        
        elif node.node_type == GPType.BOOLEAN:
            return self._eval_bool(node, data)
        
        return np.zeros(len(data))
    
    def _eval_series(self, node: GPNode, data: pd.DataFrame) -> np.ndarray:
        close = data['close'].values
        
        # Terminals
        if node.name == 'close':
            return close
        elif node.name == 'open':
            return data['open'].values
        elif node.name == 'high':
            return data['high'].values
        elif node.name == 'low':
            return data['low'].values
        elif node.name == 'volume':
            return data['volume'].values
        
        # Indicators
        period = 14
        if node.children and node.children[0].node_type == GPType.CONSTANT:
            period = int(node.children[0].value)
        
        if node.name == 'sma':
            return pd.Series(close).rolling(period).mean().fillna(0).values
        elif node.name == 'ema':
            return pd.Series(close).ewm(span=period).mean().fillna(0).values
        elif node.name == 'rsi':
            return self._rsi(close, period)
        elif node.name == 'macd':
            ema12 = pd.Series(close).ewm(span=12).mean()
            ema26 = pd.Series(close).ewm(span=26).mean()
            return (ema12 - ema26).fillna(0).values
        elif node.name == 'bb_upper':
            sma = pd.Series(close).rolling(20).mean()
            std = pd.Series(close).rolling(20).std()
            return (sma + 2*std).fillna(0).values
        elif node.name == 'bb_lower':
            sma = pd.Series(close).rolling(20).mean()
            std = pd.Series(close).rolling(20).std()
            return (sma - 2*std).fillna(0).values
        elif node.name == 'momentum':
            return pd.Series(close).diff(period).fillna(0).values
        
        # Arithmetic
        elif node.name == 'add':
            a = self._eval_series(node.children[0], data)
            b = self._eval_series(node.children[1], data)
            return a + b
        elif node.name == 'sub':
            a = self._eval_series(node.children[0], data)
            b = self._eval_series(node.children[1], data)
            return a - b
        elif node.name == 'mul':
            a = self._eval_series(node.children[0], data)
            b = self._eval_series(node.children[1], data)
            return a * b
        elif node.name == 'div':
            a = self._eval_series(node.children[0], data)
            b = self._eval_series(node.children[1], data)
            return a / (b + 1e-10)
        
        return np.zeros(len(data))
    
    def _eval_bool(self, node: GPNode, data: pd.DataFrame) -> np.ndarray:
        if node.name in ['gt', 'lt', 'gte', 'lte']:
            a = self._eval_series(node.children[0], data)
            b = self._eval_series(node.children[1], data)
            if node.name == 'gt':
                return a > b
            elif node.name == 'lt':
                return a < b
            elif node.name == 'gte':
                return a >= b
            else:
                return a <= b
        
        elif node.name in ['and', 'or']:
            a = self._eval_bool(node.children[0], data)
            b = self._eval_bool(node.children[1], data)
            if node.name == 'and':
                return np.logical_and(a, b)
            else:
                return np.logical_or(a, b)
        
        elif node.name == 'crosses_above':
            a = self._eval_series(node.children[0], data)
            b = self._eval_series(node.children[1], data)
            shifted_a = np.roll(a, 1)
            shifted_b = np.roll(b, 1)
            return (a > b) & (shifted_a <= shifted_b)
        
        elif node.name == 'crosses_below':
            a = self._eval_series(node.children[0], data)
            b = self._eval_series(node.children[1], data)
            shifted_a = np.roll(a, 1)
            shifted_b = np.roll(b, 1)
            return (a < b) & (shifted_a >= shifted_b)
        
        return np.zeros(len(data), dtype=bool)
    
    def _rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        delta = pd.Series(data).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return (100 - (100 / (1 + rs))).fillna(50).values


# ============================================================
# GP STRATEGY
# ============================================================
@dataclass
class GPStrategy:
    id: str
    tree: GPNode
    fitness: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0


# ============================================================
# GP EVOLUTION
# ============================================================
class GPEvolution:
    """Genetic Programming for trading strategies."""
    
    def __init__(
        self,
        population_size: int = 100,
        max_depth: int = 5,
        mutation_rate: float = 0.3,
        elite_percent: float = 0.1
    ):
        self.population_size = population_size
        self.max_depth = max_depth
        self.mutation_rate = mutation_rate
        self.elite_percent = elite_percent
        
        self.builder = GPTreeBuilder(max_depth=max_depth)
        self.mutator = GPMutator(self.builder)
        self.evaluator = GPTreeEvaluator()
        
        self.population: List[GPStrategy] = []
        self.generation = 0
        self.best_strategy: Optional[GPStrategy] = None
        
    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        
        for i in range(self.population_size):
            tree = self.builder.create_random_tree(GPType.BOOLEAN, depth=2)
            
            strategy = GPStrategy(
                id=f"gp_{i}",
                tree=tree
            )
            self.population.append(strategy)
        
        logger.info(f"Initialized {len(self.population)} strategies")
    
    def evaluate_population(self, data: pd.DataFrame) -> List[GPStrategy]:
        """Evaluate fitness."""
        for strategy in self.population:
            try:
                signals = self.evaluator.evaluate(strategy.tree, data)
                fitness, ret, sharpe = self._backtest(signals, data['close'].values)
                
                strategy.fitness = fitness
                strategy.total_return = ret
                strategy.sharpe = sharpe
                
            except Exception as e:
                strategy.fitness = -1000
        
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        
        if self.best_strategy is None or self.population[0].fitness > self.best_strategy.fitness:
            self.best_strategy = copy.deepcopy(self.population[0])
        
        return self.population
    
    def _backtest(self, signals: np.ndarray, close: np.ndarray) -> Tuple:
        position = 0
        returns = []
        
        for i in range(1, len(close)):
            if signals[i] and position == 0:
                position = 1
                entry = close[i]
            elif not signals[i] and position == 1:
                ret = (close[i] - entry) / entry
                returns.append(ret)
                position = 0
        
        if position == 1:
            returns.append((close[-1] - entry) / entry)
        
        if not returns:
            return -100, 0, 0
        
        returns = np.array(returns)
        total_return = (1 + returns).prod() - 1
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        fitness = total_return * 100 + sharpe * 10
        
        return fitness, total_return * 100, sharpe
    
    def evolve_generation(self) -> List[GPStrategy]:
        """Create next generation."""
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        
        n_elite = max(1, int(self.population_size * self.elite_percent))
        new_pop = sorted_pop[:n_elite]
        
        while len(new_pop) < self.population_size:
            parent = random.choice(sorted_pop[:n_elite * 3])
            
            if random.random() < self.mutation_rate:
                new_tree = self.mutator.mutate(parent.tree)
                child = GPStrategy(id=f"gp_g{self.generation}_{len(new_pop)}", tree=new_tree)
            else:
                child = GPStrategy(id=f"gp_g{self.generation}_{len(new_pop)}", 
                                  tree=copy.deepcopy(parent.tree))
            
            new_pop.append(child)
        
        self.population = new_pop[:self.population_size]
        self.generation += 1
        
        return self.population
    
    def run(self, data: pd.DataFrame, n_generations: int = 10, verbose: bool = True) -> Dict:
        """Run GP evolution."""
        print("=" * 60)
        print("GENETIC PROGRAMMING EVOLUTION")
        print("=" * 60)
        
        self.initialize_population()
        
        for gen in range(n_generations):
            self.evaluate_population(data)
            
            best = self.population[0]
            
            if verbose:
                print(f"\n--- Generation {gen + 1}/{n_generations} ---")
                print(f"Best Tree: {best.tree}")
                print(f"Fitness: {best.fitness:.2f}")
                print(f"Return: {best.total_return:.2f}%")
                print(f"Sharpe: {best.sharpe:.2f}")
            
            self.evolve_generation()
        
        return {
            'best_strategy': self.best_strategy,
            'best_tree': str(self.best_strategy.tree),
            'fitness': self.best_strategy.fitness
        }


# ============================================================
# DEMO
# ============================================================
def demo():
    """Demo GP evolution."""
    print("=" * 60)
    print("GENETIC PROGRAMMING DEMO")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 500
    close = 45000 + np.cumsum(np.random.randn(n) * 100)
    
    data = pd.DataFrame({
        'close': close,
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'volume': np.random.randint(1000, 5000, n)
    })
    
    # Run GP evolution
    gp = GPEvolution(population_size=50, max_depth=4, mutation_rate=0.3)
    
    results = gp.run(data, n_generations=5, verbose=True)
    
    print(f"\n{'='*60}")
    print("BEST DISCOVERED STRATEGY")
    print(f"{'='*60}")
    print(f"Tree: {results['best_tree']}")
    print(f"Fitness: {results['fitness']:.2f}")
    
    return gp, results


if __name__ == "__main__":
    demo()
