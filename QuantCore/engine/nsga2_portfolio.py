"""
QuantCore - Multi-Asset Multi-Objective Genetic Evolution (NSGA-II)

This module implements a hedge-fund style multi-strategy portfolio optimizer:
- Runs evolution across 12 crypto assets simultaneously
- Uses NSGA-II for multi-objective optimization (Pareto frontier)
- Optimizes: Total Return, Sharpe Ratio, Max Drawdown
- Incorporates Kelly Criterion position sizing
- Mutates asset allocation weights
- Discovers correlation breakdown strategies

This mirrors how hedge funds build multi-strat funds:
1. Run many strategies on many assets
2. Optimize for risk-adjusted returns (not just returns)
3. Allocate capital based on performance + correlation
4. Diversify to reduce portfolio drawdown
5. Rebalance based on changing market regimes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================
# ASSET CONFIGURATION
# ============================================================
AVAILABLE_ASSETS = {
    'BTC-USD': {'name': 'Bitcoin', 'volatility': 0.70},
    'ETH-USD': {'name': 'Ethereum', 'volatility': 0.75},
    'SOL-USD': {'name': 'Solana', 'volatility': 0.90},
    'ADA-USD': {'name': 'Cardano', 'volatility': 0.85},
    'AVAX-USD': {'name': 'Avalanche', 'volatility': 0.95},
    'BNB-USD': {'name': 'BNB', 'volatility': 0.80},
    'DOGE-USD': {'name': 'Dogecoin', 'volatility': 1.00},
    'DOT-USD': {'name': 'Polkadot', 'volatility': 0.85},
    'LINK-USD': {'name': 'Chainlink', 'volatility': 0.80},
    'MATIC-USD': {'name': 'Polygon', 'volatility': 0.85},
    'XRP-USD': {'name': 'XRP', 'volatility': 0.75},
    'ATOM-USD': {'name': 'Cosmos', 'volatility': 0.80},
}

# Mapping to actual filenames
ASSET_FILENAME_MAP = {k: f"{k}_5y.csv" for k in AVAILABLE_ASSETS.keys()}


# ============================================================
# MULTI-OBJECTIVE OPTIMIZATION
# ============================================================
@dataclass
class PortfolioGenome:
    """
    A complete portfolio genome for NSGA-II.
    
    Contains:
    - Strategy parameters for each asset
    - Asset allocation weights
    - Position sizing parameters
    """
    # Asset allocation weights (must sum to 1.0)
    asset_weights: Dict[str, float] = field(default_factory=dict)
    
    # Strategy parameters per asset (asset -> params)
    strategy_params: Dict[str, Dict] = field(default_factory=dict)
    
    # Global position sizing
    use_kelly: bool = True
    kelly_fraction: float = 0.25  # Fraction of full Kelly to use
    
    # Risk management
    max_position_pct: float = 0.30  # Max weight per asset
    stop_loss_pct: float = 0.05    # 5% stop loss
    
    # Metadata
    fitness_return: float = 0.0
    fitness_sharpe: float = 0.0
    fitness_drawdown: float = 0.0
    fitness_kelly: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0
    
    def __post_init__(self):
        # Initialize equal weights if not set
        if not self.asset_weights:
            n_assets = len(AVAILABLE_ASSETS)
            equal_weight = 1.0 / n_assets
            self.asset_weights = {asset: equal_weight for asset in AVAILABLE_ASSETS.keys()}


@dataclass  
class PortfolioBacktestResult:
    """Result of portfolio backtest across all assets."""
    genome: PortfolioGenome
    
    # Individual asset results
    asset_returns: Dict[str, float] = field(default_factory=dict)
    asset_sharpes: Dict[str, float] = field(default_factory=dict)
    asset_drawdowns: Dict[str, float] = field(default_factory=dict)
    asset_trades: Dict[str, int] = field(default_factory=dict)
    
    # Portfolio-level metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # Correlation & diversification
    correlation_benefit: float = 0.0  # How much diversification helps
    kelly_optimal: float = 0.0       # Kelly-optimal position size
    
    # Combined fitness (for NSGA-II)
    objectives: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (return, sharpe, -drawdown)


# ============================================================
# KELLY CRITERION
# ============================================================
class KellyCalculator:
    """Calculate Kelly Criterion for position sizing."""
    
    @staticmethod
    def calculate_kelly(
        win_rate: float, 
        avg_win: float, 
        avg_loss: float,
        fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Formula: Kelly% = W - (1-W)/(R)
        Where W = win rate, R = avg_win/avg_loss
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            fraction: Kelly fraction to use (0.25 = Half-Kelly)
            
        Returns:
            Optimal position size as fraction of portfolio
        """
        if avg_loss == 0:
            return 0.0
            
        win_loss_ratio = avg_win / abs(avg_loss)
        
        # Full Kelly
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fraction (typically 0.25 for safety)
        kelly = kelly * fraction
        
        # Bound to reasonable range
        return max(0.0, min(kelly, 0.50))  # Max 50% position
    
    @staticmethod
    def calculate_portfolio_kelly(
        asset_returns: Dict[str, float],
        asset_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate Kelly for entire portfolio.
        
        Returns Kelly sizes for each asset based on their weight and returns.
        """
        kelly_sizes = {}
        
        for asset, weight in asset_weights.items():
            ret = asset_returns.get(asset, 0)
            # Simple Kelly based on return
            if abs(ret) > 0.001:
                # Assume positive return = good
                kelly = min(abs(ret) * 0.5, weight)
            else:
                kelly = weight
            kelly_sizes[asset] = kelly
        
        return kelly_sizes


# ============================================================
# CORRELATION ANALYSIS
# ============================================================
class CorrelationAnalyzer:
    """Analyze asset correlations for diversification."""
    
    @staticmethod
    def calculate_correlation_matrix(
        asset_returns: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Calculate return correlation matrix across assets."""
        # Align all returns to same length
        min_len = min(len(v) for v in asset_returns.values())
        if min_len < 2:
            return pd.DataFrame()  # Not enough data
        
        aligned = {k: v[:min_len] for k, v in asset_returns.items()}
        returns_df = pd.DataFrame(aligned)
        return returns_df.corr()
    
    @staticmethod
    def calculate_correlation_benefit(
        asset_returns: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate diversification benefit.
        
        Lower average correlation = better diversification.
        """
        if len(asset_returns) < 2:
            return 0.0
        
        corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(asset_returns)
        
        if corr_matrix.empty:
            return 0.0
        
        # Weighted average correlation
        weighted_corr = 0.0
        total_pairs = 0
        
        assets = list(weights.keys())
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j and asset1 in corr_matrix.columns and asset2 in corr_matrix.columns:
                    w1 = weights[asset1]
                    w2 = weights[asset2]
                    corr = corr_matrix.loc[asset1, asset2]
                    if not pd.isna(corr):
                        weighted_corr += corr * w1 * w2
                        total_pairs += 1
        
        # Diversification benefit (inverse of correlation)
        return 1.0 - abs(weighted_corr) if total_pairs > 0 else 0.0
    
    @staticmethod
    def find_correlation_breakouts(
        corr_matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[Tuple[str, str]]:
        """Find highly correlated pairs (for spread trading)."""
        breakouts = []
        
        for col in corr_matrix.columns:
            for idx in corr_matrix.index:
                if col < idx:  # Avoid duplicates
                    corr = corr_matrix.loc[idx, col]
                    if abs(corr) > threshold:
                        breakouts.append((idx, col, corr))
        
        return breakouts


# ============================================================
# MULTI-ASSET BACKTEST
# ============================================================
class MultiAssetBacktester:
    """Run backtests across multiple assets with portfolio-level metrics."""
    
    def __init__(self, data_dir: str = "data/csv"):
        self.data_dir = data_dir
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_asset_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for an asset."""
        if symbol in self.data_cache:
            return self.data_cache[symbol]
        
        # Try with _5y suffix
        filename = ASSET_FILENAME_MAP.get(symbol, f"{symbol}.csv")
        filepath = Path(self.data_dir) / filename
        
        if not filepath.exists():
            # Try without suffix
            filepath = Path(self.data_dir) / f"{symbol}.csv"
            
        if not filepath.exists():
            logger.warning(f"Data not found for {symbol}")
            return None
        
        try:
            df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            self.data_cache[symbol] = df
            return df
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            return None
    
    def run_portfolio_backtest(
        self,
        genome: PortfolioGenome,
        training_period_pct: float = 0.7
    ) -> PortfolioBacktestResult:
        """
        Run portfolio backtest across all assets.
        
        Args:
            genome: Portfolio genome with weights and strategy params
            training_period_pct: % of data for training (rest for validation)
        """
        asset_returns = {}
        asset_trades = {}
        asset_sharpes = {}
        asset_drawdowns = {}
        
        # Run strategy on each asset
        for symbol, weight in genome.asset_weights.items():
            if weight < 0.01:  # Skip negligible positions
                continue
            
            data = self.load_asset_data(symbol)
            if data is None:
                continue
            
            # Get strategy params for this asset
            params = genome.strategy_params.get(symbol, {
                'rsi_period': 14,
                'oversold': 30,
                'overbought': 70
            })
            
            # Run backtest
            result = self.run_single_asset_backtest(
                data, params, weight, training_period_pct
            )
            
            if result is not None:
                asset_returns[symbol] = result['returns']
                asset_trades[symbol] = result['trades']
                asset_sharpes[symbol] = result['sharpe']
                asset_drawdowns[symbol] = result['drawdown']
        
        if not asset_returns:
            return PortfolioBacktestResult(genome=genome)
        
        # Combine into portfolio returns
        portfolio_returns = self.combine_portfolio_returns(
            asset_returns, 
            genome.asset_weights
        )
        
        # Calculate portfolio metrics
        total_return = (1 + portfolio_returns).prod() - 1
        sharpe_ratio = self.calculate_sharpe(portfolio_returns)
        max_dd = self.calculate_max_drawdown(portfolio_returns)
        
        # Calculate correlation benefit
        corr_benefit = CorrelationAnalyzer.calculate_correlation_benefit(
            asset_returns, genome.asset_weights
        )
        
        # Calculate Kelly for portfolio
        kelly_optimal = KellyCalculator.calculate_portfolio_kelly(
            {k: v.mean() * 252 for k, v in asset_returns.items()},
            genome.asset_weights
        )
        
        # Create result
        result = PortfolioBacktestResult(
            genome=genome,
            asset_returns={k: v.sum() for k, v in asset_returns.items()},
            asset_sharpes=asset_sharpes,
            asset_drawdowns=asset_drawdowns,
            asset_trades=asset_trades,
            total_return=total_return * 100,  # As percentage
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd * 100,
            correlation_benefit=corr_benefit,
            kelly_optimal=sum(kelly_optimal.values()) if kelly_optimal else 0,
            objectives=(
                total_return * 100,  # Higher is better
                sharpe_ratio,         # Higher is better
                -max_dd * 100        # Lower drawdown (negative) is better
            )
        )
        
        # Store fitness values in genome
        genome.fitness_return = total_return * 100
        genome.fitness_sharpe = sharpe_ratio
        genome.fitness_drawdown = max_dd * 100
        
        return result
    
    def run_single_asset_backtest(
        self,
        data: pd.DataFrame,
        params: Dict,
        weight: float,
        training_pct: float
    ) -> Optional[Dict]:
        """Run backtest on single asset."""
        try:
            close = data['close'].values
            n = len(close)
            
            # Split data
            train_size = int(n * training_pct)
            train_data = close[:train_size]
            test_data = close[train_size:]
            
            # RSI strategy
            rsi_period = int(params.get('rsi_period', 14))
            oversold = params.get('oversold', 30)
            overbought = params.get('overbought', 70)
            
            # Calculate RSI on full dataset
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).fillna(50).values
            
            # Generate signals (use test period only)
            entries = rsi[train_size:] < oversold
            exits = rsi[train_size:] > overbought
            
            # Calculate returns
            asset_returns = []
            position = 0
            
            for i in range(len(test_data) - 1):
                if entries[i] and position == 0:
                    position = 1
                elif exits[i] and position == 1:
                    ret = (test_data[i + 1] - test_data[i]) / test_data[i]
                    asset_returns.append(ret * weight)  # Scale by allocation
                    position = 0
            
            if position == 1:
                ret = (test_data[-1] - test_data[-2]) / test_data[-2]
                asset_returns.append(ret * weight)
            
            if not asset_returns:
                return None
            
            returns = np.array(asset_returns)
            
            # Metrics
            total_ret = (1 + returns).prod() - 1
            sharpe = self.calculate_sharpe(returns) if len(returns) > 1 else 0
            dd = self.calculate_max_drawdown(returns)
            
            return {
                'returns': returns,
                'trades': len(returns),
                'sharpe': sharpe,
                'drawdown': dd
            }
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return None
    
    def combine_portfolio_returns(
        self,
        asset_returns: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Combine individual asset returns into portfolio returns."""
        if not asset_returns:
            return np.array([0.0])
        
        # Find common length
        min_len = min(len(v) for v in asset_returns.values())
        
        if min_len == 0:
            return np.array([0.0])
        
        # Trim and combine
        portfolio_returns = np.zeros(min_len)
        
        for asset, returns in asset_returns.items():
            trimmed = returns[:min_len]
            weight = weights.get(asset, 0)
            portfolio_returns += trimmed * weight
        
        return portfolio_returns
    
    def calculate_sharpe(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess = returns.mean() - risk_free
        return (excess / returns.std()) * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        equity = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return abs(np.min(drawdown))


# ============================================================
# NSGA-II IMPLEMENTATION
# ============================================================
class NSGA2:
    """
    Non-dominated Sorting Genetic Algorithm II.
    
    Optimizes multiple objectives simultaneously using Pareto dominance.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 10,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        elite_count: int = 5
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_count = elite_count
        
        self.population: List[PortfolioGenome] = []
        self.fronts: List[List[PortfolioGenome]] = []
        self.history: List[List[PortfolioGenome]] = []
    
    def initialize_population(self) -> List[PortfolioGenome]:
        """Create initial random population."""
        population = []
        
        for _ in range(self.population_size):
            genome = self.create_random_genome()
            population.append(genome)
        
        return population
    
    def create_random_genome(self) -> PortfolioGenome:
        """Create a random portfolio genome."""
        genome = PortfolioGenome()
        
        # Random asset weights (normalized)
        weights = {asset: random.random() for asset in AVAILABLE_ASSETS.keys()}
        total = sum(weights.values())
        genome.asset_weights = {k: v/total for k, v in weights.items()}
        
        # Ensure no asset exceeds max position
        for asset in genome.asset_weights:
            genome.asset_weights[asset] = min(
                genome.asset_weights[asset],
                genome.max_position_pct
            )
        
        # Re-normalize after clamping
        total = sum(genome.asset_weights.values())
        genome.asset_weights = {k: v/total for k, v in genome.asset_weights.items()}
        
        # Random strategy params per asset
        for asset in AVAILABLE_ASSETS.keys():
            genome.strategy_params[asset] = {
                'rsi_period': random.choice([7, 10, 14, 21]),
                'oversold': random.randint(20, 35),
                'overbought': random.randint(65, 80)
            }
        
        # Random Kelly settings
        genome.use_kelly = random.random() > 0.5
        genome.kelly_fraction = random.choice([0.25, 0.5, 0.75])
        
        return genome
    
    def evaluate_population(
        self, 
        population: List[PortfolioGenome],
        backtester: MultiAssetBacktester
    ) -> List[PortfolioBacktestResult]:
        """Evaluate all genomes in population."""
        results = []
        
        for i, genome in enumerate(population):
            result = backtester.run_portfolio_backtest(genome)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(population)}")
        
        return results
    
    def fast_non_dominated_sort(
        self, 
        results: List[PortfolioBacktestResult]
    ) -> List[List[PortfolioGenome]]:
        """
        Fast non-dominated sorting algorithm.
        
        Returns list of fronts, each front is a list of non-dominated genomes.
        """
        n = len(results)
        domination_count = [0] * n  # How many solutions dominate this one
        dominated_solutions = [[] for _ in range(n)]  # Solutions this one dominates
        
        fronts = [[]]
        
        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if self.dominates(results[i], results[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                # Check if j dominates i
                elif self.dominates(results[j], results[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        # First front
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(results[i].genome)
                results[i].genome.rank = 0
        
        # Subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for genome in fronts[current_front]:
                idx = population_index(results, genome)
                for dominated_idx in dominated_solutions[idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(results[dominated_idx].genome)
                        results[dominated_idx].genome.rank = current_front + 1
            current_front += 1
            fronts.append(next_front)
        
        # Remove empty front
        if not fronts[-1]:
            fronts = fronts[:-1]
        
        return fronts
    
    def dominates(
        self, 
        result1: PortfolioBacktestResult, 
        result2: PortfolioBacktestResult
    ) -> bool:
        """Check if result1 dominates result2 (Pareto dominance)."""
        obj1 = result1.objectives
        obj2 = result2.objectives
        
        # At least as good in all objectives
        better_or_equal = all(o1 >= o2 for o1, o2 in zip(obj1, obj2))
        
        # Strictly better in at least one
        strictly_better = any(o1 > o2 for o1, o2 in zip(obj1, obj2))
        
        return better_or_equal and strictly_better
    
    def calculate_crowding_distance(
        self, 
        front: List[PortfolioGenome],
        results: List[PortfolioBacktestResult]
    ) -> None:
        """Calculate crowding distance for diversity preservation."""
        if len(front) <= 2:
            for g in front:
                g.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for g in front:
            g.crowding_distance = 0.0
        
        # For each objective
        for obj_idx in range(3):
            # Sort by objective
            sorted_front = sorted(
                front, 
                key=lambda g: get_objective(g, results, obj_idx),
                reverse=(obj_idx != 2)  # Higher is better except for drawdown
            )
            
            # Boundary solutions get infinite distance
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_range = (
                get_objective(sorted_front[-1], results, obj_idx) -
                get_objective(sorted_front[0], results, obj_idx)
            )
            
            if obj_range == 0:
                continue
            
            # Interior solutions
            for i in range(1, len(sorted_front) - 1):
                distance = (
                    get_objective(sorted_front[i + 1], results, obj_idx) -
                    get_objective(sorted_front[i - 1], results, obj_idx)
                ) / obj_range
                sorted_front[i].crowding_distance += distance
    
    def select_parents(
        self, 
        results: List[PortfolioBacktestResult],
        population: List[PortfolioGenome]
    ) -> List[PortfolioGenome]:
        """Tournament selection based on rank and crowding distance."""
        selected = []
        
        for _ in range(len(population)):
            # Select two random individuals
            i1, i2 = random.sample(range(len(population)), 2)
            
            # Compare by rank (lower is better)
            if population[i1].rank < population[i2].rank:
                selected.append(population[i1])
            elif population[i2].rank < population[i1].rank:
                selected.append(population[i2])
            else:
                # Tie-break by crowding distance (higher is better)
                if population[i1].crowding_distance >= population[i2].crowding_distance:
                    selected.append(population[i1])
                else:
                    selected.append(population[i2])
        
        return selected
    
    def crossover(
        self, 
        parent1: PortfolioGenome, 
        parent2: PortfolioGenome
    ) -> Tuple[PortfolioGenome, PortfolioGenome]:
        """SBX-like crossover for portfolio genomes."""
        child1 = PortfolioGenome()
        child2 = PortfolioGenome()
        
        # Crossover weights
        if random.random() < self.crossover_prob:
            # Blend weights
            alpha = random.random()
            for asset in AVAILABLE_ASSETS.keys():
                w1 = parent1.asset_weights.get(asset, 0)
                w2 = parent2.asset_weights.get(asset, 0)
                child1.asset_weights[asset] = alpha * w1 + (1 - alpha) * w2
                child2.asset_weights[asset] = (1 - alpha) * w1 + alpha * w2
        else:
            child1.asset_weights = parent1.asset_weights.copy()
            child2.asset_weights = parent2.asset_weights.copy()
        
        # Normalize weights
        for child in [child1, child2]:
            total = sum(child.asset_weights.values())
            if total > 0:
                child.asset_weights = {k: v/total for k, v in child.asset_weights.items()}
        
        # Crossover strategy params
        child1.strategy_params = parent1.strategy_params.copy()
        child2.strategy_params = parent2.strategy_params.copy()
        
        return child1, child2
    
    def mutate(self, genome: PortfolioGenome) -> None:
        """Mutate a genome."""
        # Mutate weights
        if random.random() < self.mutation_prob:
            # Randomly adjust weights
            for asset in genome.asset_weights:
                if random.random() < 0.3:  # 30% chance per asset
                    change = random.uniform(-0.1, 0.1)
                    genome.asset_weights[asset] = max(0.01, 
                        genome.asset_weights.get(asset, 0.01) + change)
            
            # Re-normalize
            total = sum(genome.asset_weights.values())
            genome.asset_weights = {k: v/total for k, v in genome.asset_weights.items()}
        
        # Mutate strategy params
        if random.random() < self.mutation_prob:
            asset = random.choice(list(AVAILABLE_ASSETS.keys()))
            if asset not in genome.strategy_params:
                genome.strategy_params[asset] = {}
            
            param = random.choice(['rsi_period', 'oversold', 'overbought'])
            if param == 'rsi_period':
                genome.strategy_params[asset][param] = random.choice([7, 10, 14, 21])
            elif param == 'oversold':
                genome.strategy_params[asset][param] = random.randint(20, 35)
            else:
                genome.strategy_params[asset][param] = random.randint(65, 80)
        
        # Mutate Kelly
        if random.random() < self.mutation_prob:
            genome.use_kelly = random.random() > 0.5
            genome.kelly_fraction = random.choice([0.25, 0.5, 0.75])
    
    def evolve(
        self,
        backtester: MultiAssetBacktester,
        verbose: bool = True
    ) -> List[PortfolioGenome]:
        """Run NSGA-II evolution."""
        
        # Initialize
        if verbose:
            print("=" * 60)
            print("NSGA-II MULTI-OBJECTIVE PORTFOLIO OPTIMIZATION")
            print("=" * 60)
            print(f"Population: {self.population_size}")
            print(f"Generations: {self.generations}")
            print(f"Objectives: Return, Sharpe, Drawdown")
            print("=" * 60)
        
        self.population = self.initialize_population()
        
        # Main evolution loop
        for gen in range(self.generations):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Generation {gen + 1}/{self.generations}")
                print(f"{'='*50}")
            
            # Evaluate
            results = self.evaluate_population(self.population, backtester)
            
            # Non-dominated sorting
            self.fronts = self.fast_non_dominated_sort(results)
            
            # Calculate crowding distance for each front
            for front in self.fronts:
                self.calculate_crowding_distance(front, results)
            
            # Get all genomes
            all_genomes = [r.genome for r in results]
            
            # Print best
            if verbose and self.fronts:
                print(f"\n🏆 Best Portfolio (Rank 0):")
                best = self.fronts[0][0]
                best_result = results[population_index(results, best)]
                print(f"   Return: {best_result.total_return:.2f}%")
                print(f"   Sharpe: {best_result.sharpe_ratio:.3f}")
                print(f"   Max DD: {best_result.max_drawdown:.2f}%")
                print(f"   Top Assets: {get_top_assets(best, 3)}")
            
            # Create next generation
            if gen < self.generations - 1:
                # Selection
                parents = self.select_parents(results, self.population)
                
                # Create offspring
                offspring = []
                while len(offspring) < self.population_size:
                    p1, p2 = random.sample(parents, 2)
                    child1, child2 = self.crossover(p1, p2)
                    self.mutate(child1)
                    self.mutate(child2)
                    offspring.extend([child1, child2])
                
                # Combine and select
                combined = self.population[:self.elite_count] + offspring
                combined_results = self.evaluate_population(
                    combined[:self.population_size], backtester
                )
                
                # Sort by rank then crowding distance
                combined_results.sort(
                    key=lambda r: (r.genome.rank, -r.genome.crowding_distance)
                )
                
                self.population = [r.genome for r in combined_results[:self.population_size]]
        
        # Final evaluation
        results = self.evaluate_population(self.population, backtester)
        self.fronts = self.fast_non_dominated_sort(results)
        
        if verbose:
            print("\n" + "=" * 60)
            print("EVOLUTION COMPLETE - PARETO FRONTIER")
            print("=" * 60)
            
            # Get Pareto front
            pareto = self.fronts[0] if self.fronts else []
            print(f"\nPareto-optimal solutions: {len(pareto)}")
            
            print("\n🏆 TOP PORTFOLIOS:")
            for i, genome in enumerate(pareto[:5]):
                result = results[population_index(results, genome)]
                print(f"\n  Portfolio {i + 1}:")
                print(f"    Return: {result.total_return:.2f}%")
                print(f"    Sharpe: {result.sharpe_ratio:.3f}")
                print(f"    Max DD: {result.max_drawdown:.2f}%")
                print(f"    Assets: {get_top_assets(genome, 5)}")
        
        return self.fronts[0] if self.fronts else []


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_objective(
    genome: PortfolioGenome, 
    results: List[PortfolioBacktestResult], 
    obj_idx: int
) -> float:
    """Get objective value for a genome."""
    result = results[population_index(results, genome)]
    return result.objectives[obj_idx]


def population_index(results: List[PortfolioBacktestResult], genome: PortfolioGenome) -> int:
    """Find index of genome in results."""
    for i, r in enumerate(results):
        if r.genome is genome:
            return i
    return 0


def get_top_assets(genome: PortfolioGenome, n: int = 5) -> List[str]:
    """Get top N assets by weight."""
    sorted_assets = sorted(
        genome.asset_weights.items(), 
        key=lambda x: -x[1]
    )
    return [f"{a}:{w:.1%}" for a, w in sorted_assets[:n]]


# ============================================================
# MAIN
# ============================================================
def main():
    """Run NSGA-II portfolio optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NSGA-II Portfolio Optimization')
    parser.add_argument('--population', type=int, default=30, help='Population size')
    parser.add_argument('--generations', type=int, default=5, help='Generations')
    parser.add_argument('--output', type=str, default='pareto_results.json')
    args = parser.parse_args()
    
    # Create backtester
    backtester = MultiAssetBacktester("data/csv")
    
    # Run NSGA-II
    nsga2 = NSGA2(
        population_size=args.population,
        generations=args.generations
    )
    
    pareto_front = nsga2.evolve(backtester, verbose=True)
    
    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'pareto_size': len(pareto_front),
        'population_size': args.population,
        'generations': args.generations,
        'portfolios': [
            {
                'rank': g.rank,
                'return': g.fitness_return,
                'sharpe': g.fitness_sharpe,
                'drawdown': g.fitness_drawdown,
                'weights': g.asset_weights,
                'kelly': g.kelly_fraction if g.use_kelly else None
            }
            for g in pareto_front[:10]
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output}")
    
    return pareto_front


if __name__ == "__main__":
    main()
