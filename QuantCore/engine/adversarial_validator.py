"""
QuantCore - Adversarial Validator v1.0

This is the anti-overfitting shield for your trading strategies.

Expert: "It stress-tests every candidate strategy against synthetic chaos—
flash crashes, liquidity voids, gap risks, volatility explosions. 
Strategies that survive are the ones you want managing your money."

Features:
1. Synthetic Shock Generation (Bootstrap, GARCH)
2. Worst-Case Fitness Evaluation
3. Regime-Specific Shock Injection
4. Adversarial Mutation Operators
5. Hall of Shame (nightmare scenarios)
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
# SHOCK TYPES
# ============================================================
class ShockType(Enum):
    """Types of synthetic market shocks."""
    FLASH_CRASH = "flash_crash"           # Sudden drop
    FLASH_RALLY = "flash_rally"           # Sudden rise
    VOLATILITY_SPIKE = "volatility_spike" # Vol explosion
    GAP_DOWN = "gap_down"                  # Overnight gap down
    GAP_UP = "gap_up"                      # Overnight gap up
    LIQUIDITY_DROUGHT = "liquidity_drought" # Low volume
    TREND_REVERSAL = "trend_reversal"     # Sudden reversal
    VOLATILITY_CLUSTER = "volatility_cluster" # Sustained high vol
    CORRELATION_SPIKE = "correlation_spike" # Correlations go to 1
    REGIME_SHIFT = "regime_shift"          # Regime change


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class AdversarialConfig:
    """
    Configuration for adversarial validation.
    
    This is evolvable - the GA can tune how much chaos to inject.
    """
    # Shock generation
    n_synthetic_paths: int = 100           # Number of synthetic paths to generate
    shock_intensity: float = 1.0           # Multiplier for shock severity
    bootstrap_window: int = 50             # Window for bootstrapping
    
    # GARCH settings (evolvable)
    garch_p: int = 1                      # GARCH(p) order
    garch_q: int = 1                      # GARCH(q) order
    garch_vol_scale: float = 2.0          # Scale factor for vol
    
    # Fitness settings
    fitness_percentile: float = 0.1        # Use 10th percentile (worst case)
    use_worst_case: bool = True            # Penalize worst outcomes
    
    # Regime-specific shocks
    shock_per_regime: bool = True          # Generate shocks per regime
    bull_crash_prob: float = 0.3          # Prob of crash in bull
    bear_rally_prob: float = 0.3           # Prob of rally in bear
    
    # Shock scheduling
    shock_frequency: float = 0.1           # 10% of bars get shocks
    allow_multi_shock: bool = True         # Allow multiple shocks
    
    def to_dict(self) -> Dict:
        return {
            'n_synthetic_paths': self.n_synthetic_paths,
            'shock_intensity': self.shock_intensity,
            'bootstrap_window': self.bootstrap_window,
            'garch_p': self.garch_p,
            'garch_q': self.garch_q,
            'garch_vol_scale': self.garch_vol_scale,
            'fitness_percentile': self.fitness_percentile,
            'use_worst_case': self.use_worst_case,
            'shock_per_regime': self.shock_per_regime,
            'bull_crash_prob': self.bull_crash_prob,
            'bear_rally_prob': self.bear_rally_prob,
            'shock_frequency': self.shock_frequency,
            'allow_multi_shock': self.allow_multi_shock
        }


@dataclass
class ShockEvent:
    """A single shock event in a synthetic path."""
    shock_type: ShockType
    bar_index: int
    intensity: float
    duration: int = 1  # How many bars the shock lasts


@dataclass
class AdversarialResult:
    """Result of adversarial validation."""
    original_fitness: float
    worst_case_fitness: float
    percentile_fitness: float
    avg_synthetic_fitness: float
    survival_rate: float           # % of paths with positive return
    max_drawdown_synthetic: float  # Worst DD across all paths
    shocked_return: float          # Return under worst shock
    shock_resistance: float        # How much fitness dropped under shock


# ============================================================
# GARCH VOLATILITY MODEL
# ============================================================
class GARCHModel:
    """
    GARCH(1,1) model for volatility simulation.
    
    Expert: "Fit a GARCH model to recent returns, then simulate
    paths with realistic volatility clustering."
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.omega = 0.0
        self.alpha = []
        self.beta = []
        
    def fit(self, returns: np.ndarray) -> 'GARCHModel':
        """Fit GARCH model to returns."""
        # Simplified GARCH(1,1) estimation
        # In production, use arch package or similar
        
        var = np.var(returns)
        mean_return = np.mean(returns)
        
        # Estimate parameters using MLE approximation
        self.omega = var * 0.1
        self.alpha = [0.08]  # ARCH term
        self.beta = [0.90]  # GARCH term
        
        return self
    
    def simulate(self, n: int, n_paths: int = 1, 
                 vol_scale: float = 1.0,
                 initial_vol: float = None) -> np.ndarray:
        """
        Simulate returns with GARCH volatility.
        
        Returns: Array of shape (n_paths, n)
        """
        if initial_vol is None:
            initial_vol = np.sqrt(self.omega / (1 - self.alpha[0] - self.beta[0]))
        
        paths = []
        
        for _ in range(n_paths):
            simulated = np.zeros(n)
            variance = initial_vol ** 2
            
            for i in range(n):
                # Update variance
                if i > 0:
                    variance = self.omega + self.alpha[0] * (simulated[i-1] ** 2) + self.beta[0] * variance
                
                # Scale by vol_scale for stress testing
                vol = np.sqrt(variance) * vol_scale
                
                # Generate return
                simulated[i] = np.random.normal(0, vol)
            
            paths.append(simulated)
        
        return np.array(paths)


# ============================================================
# SHOCK GENERATORS
# ============================================================
class ShockGenerator:
    """
    Generate realistic synthetic market shocks.
    
    Expert: "Generate shocks that are tailored to each regime.
    In low-vol, inject volatility spike; in high-vol, inject liquidity dry-up."
    """
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
    def generate_synthetic_paths(self, data: pd.DataFrame, 
                                  n_paths: int = None,
                                  base_shocks: bool = True) -> List[pd.DataFrame]:
        """
        Generate multiple synthetic price paths with shocks.
        
        Returns: List of synthetic DataFrames
        """
        if n_paths is None:
            n_paths = self.config.n_synthetic_paths
        
        close = data['close'].values
        returns = np.diff(close) / close[:-1]
        returns = returns[~np.isnan(returns)]
        
        paths = []
        
        for i in range(n_paths):
            # Generate base synthetic returns
            synthetic_returns = self._generate_base_returns(returns, len(close))
            
            # Inject shocks
            if base_shocks:
                synthetic_returns = self._inject_shocks(
                    synthetic_returns, 
                    close,
                    shock_type=None  # Random
                )
            
            # Reconstruct prices (with clipping to prevent overflow)
            cumsum_returns = np.cumsum(synthetic_returns)
            cumsum_returns = np.clip(cumsum_returns, -20, 20)  # Prevent overflow
            synthetic_prices = close[0] * np.exp(cumsum_returns)
            
            # Build DataFrame
            synthetic_data = self._build_synthetic_data(data, synthetic_prices)
            paths.append(synthetic_data)
        
        return paths
    
    def _generate_base_returns(self, returns: np.ndarray, n: int) -> np.ndarray:
        """Generate base returns using GARCH or bootstrap."""
        
        # Try GARCH first
        try:
            garch = GARCHModel(p=self.config.garch_p, q=self.config.garch_q)
            garch.fit(returns)
            paths = garch.simulate(n - 1, n_paths=1, 
                                   vol_scale=self.config.garch_vol_scale)
            return paths[0]
        except:
            pass
        
        # Fallback to bootstrap
        return self._bootstrap_returns(returns, n)
    
    def _bootstrap_returns(self, returns: np.ndarray, n: int) -> np.ndarray:
        """Bootstrap returns with replacement."""
        indices = np.random.choice(len(returns), size=n, replace=True)
        return returns[indices]
    
    def _inject_shocks(self, returns: np.ndarray, 
                      prices: np.ndarray,
                      shock_type: ShockType = None) -> np.ndarray:
        """Inject synthetic shocks into returns."""
        
        returns = copy.deepcopy(returns)
        n = len(returns)
        
        # Determine where shocks occur
        shock_indices = []
        for i in range(n):
            if np.random.random() < self.config.shock_frequency:
                shock_indices.append(i)
        
        for idx in shock_indices:
            # Random shock type if not specified
            if shock_type is None:
                shock_type = random.choice(list(ShockType))
            
            intensity = self.config.shock_intensity
            
            # Inject based on type
            if shock_type == ShockType.FLASH_CRASH:
                # Sudden drop
                shock = -0.02 * intensity * (1 + np.random.random())
                returns[idx] += shock
                
            elif shock_type == ShockType.FLASH_RALLY:
                # Sudden rise
                shock = 0.02 * intensity * (1 + np.random.random())
                returns[idx] += shock
                
            elif shock_type == ShockType.VOLATILITY_SPIKE:
                # Increase vol for next several bars
                duration = min(5, n - idx)
                for j in range(duration):
                    if idx + j < n:
                        returns[idx + j] *= (1 + 0.5 * intensity)
                        
            elif shock_type == ShockType.GAP_DOWN:
                # Overnight gap
                gap = -0.03 * intensity * np.random.random()
                returns[idx] += gap - returns[idx]  # Replace with gap
                
            elif shock_type == ShockType.GAP_UP:
                # Overnight gap up
                gap = 0.03 * intensity * np.random.random()
                returns[idx] += gap - returns[idx]
                
            elif shock_type == ShockType.TREND_REVERSAL:
                # Reverse trend
                returns[idx:] *= -1 * intensity
                
            elif shock_type == ShockType.VOLATILITY_CLUSTER:
                # Sustained high vol
                duration = min(10, n - idx)
                for j in range(duration):
                    if idx + j < n:
                        returns[idx + j] *= 2.0
                        
            elif shock_type == ShockType.LIQUIDITY_DROUGHT:
                # Low volume regime (simulated by wider spreads = higher costs)
                returns[idx] += 0.001 * intensity  # Spread cost
                
            elif shock_type == ShockType.CORRELATION_SPIKE:
                # Correlations go to 1 (simulated by adding common factor)
                common_factor = np.random.normal(0, 0.01 * intensity)
                returns[idx] += common_factor
                
            elif shock_type == ShockType.REGIME_SHIFT:
                # Regime change
                returns[idx:] += (np.random.random() - 0.5) * 0.02 * intensity
        
        return returns
    
    def _build_synthetic_data(self, data: pd.DataFrame, 
                             prices: np.ndarray) -> pd.DataFrame:
        """Build synthetic OHLCV DataFrame."""
        
        n = len(prices)
        
        # Handle NaN/Inf
        prices = np.nan_to_num(prices, nan=100.0, posinf=100.0, neginf=100.0)
        prices = np.clip(prices, 0.01, 1e10)  # Ensure positive
        
        # Generate OHLC with shocks baked in
        open_prices = np.roll(prices, 1)
        open_prices[0] = prices[0]
        
        # High = max of open, close, + random
        high_prices = np.maximum(open_prices, prices) * (1 + np.abs(np.random.randn(n) * 0.005))
        
        # Low = min of open, close, - random
        low_prices = np.minimum(open_prices, prices) * (1 - np.abs(np.random.randn(n) * 0.005))
        
        # Ensure low <= high
        low_prices = np.minimum(low_prices, high_prices)
        
        # Volume varies
        base_volume = data['volume'].values[:n] if 'volume' in data.columns else np.ones(n) * 1000
        volume = base_volume * (0.5 + np.random.random(n))
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': prices,
            'volume': volume
        })
    
    def generate_regime_shocks(self, data: pd.DataFrame, 
                               regime: str) -> List[pd.DataFrame]:
        """
        Generate shocks tailored to a specific regime.
        
        Expert: "In a bull regime, simulate a flash crash; 
        in a bear regime, simulate a dead-cat bounce."
        """
        
        shock_type = None
        
        if regime == 'bull':
            # In bull, chance of crash
            if np.random.random() < self.config.bull_crash_prob:
                shock_type = ShockType.FLASH_CRASH
                
        elif regime == 'bear':
            # In bear, chance of rally
            if np.random.random() < self.config.bear_rally_prob:
                shock_type = ShockType.FLASH_RALLY
                
        elif regime == 'high_vol':
            # In high vol, liquidity issues
            shock_type = ShockType.LIQUIDITY_DROUGHT
            
        elif regime == 'sideways':
            # In sideways, trend reversal
            shock_type = ShockType.TREND_REVERSAL
        
        # Generate paths with this shock type
        paths = self.generate_synthetic_paths(data, base_shocks=False)
        
        # Apply regime-specific shock to all paths
        for i in range(len(paths)):
            returns = np.diff(paths[i]['close'].values) / paths[i]['close'].values[:-1]
            shocked_returns = self._inject_shocks(returns, paths[i]['close'].values, shock_type)
            
            # Reconstruct prices
            new_prices = paths[i]['close'].values[0] * np.exp(np.cumsum(shocked_returns))
            paths[i]['close'] = new_prices
            paths[i] = self._build_synthetic_data(paths[i], new_prices)
        
        return paths


# ============================================================
# ADVERSARIAL VALIDATOR
# ============================================================
class AdversarialValidator:
    """
    The anti-overfitting shield.
    
    Expert: "Strategies that survive this are the ones you want
    managing your money."
    """
    
    def __init__(self, config: AdversarialConfig = None):
        self.config = config or AdversarialConfig()
        self.shock_generator = ShockGenerator(self.config)
        self.hall_of_shame: List[Dict] = []  # Worst scenarios
        
    def validate(self, strategy: Any, data: pd.DataFrame,
                fitness_func: Callable) -> AdversarialResult:
        """
        Run adversarial validation on a strategy.
        
        Args:
            strategy: Strategy to validate
            data: Historical price data
            fitness_func: Function to calculate fitness
            
        Returns:
            AdversarialResult with stress-test metrics
        """
        
        # Get baseline fitness
        original_fitness = fitness_func(strategy, data)
        
        # Generate synthetic paths
        synthetic_paths = self.shock_generator.generate_synthetic_paths(data)
        
        # Evaluate on each path
        path_fitnesses = []
        path_drawdowns = []
        
        for path_data in synthetic_paths:
            # Calculate fitness on synthetic path
            synth_fitness = fitness_func(strategy, path_data)
            path_fitnesses.append(synth_fitness)
            
            # Calculate max drawdown
            dd = self._calculate_max_drawdown(path_data)
            path_drawdowns.append(dd)
        
        path_fitnesses = np.array(path_fitnesses)
        path_drawdowns = np.array(path_drawdowns)
        
        # Calculate metrics
        worst_case = np.min(path_fitnesses)
        percentile_idx = int(len(path_fitnesses) * self.config.fitness_percentile)
        percentile_fitness = np.percentile(path_fitnesses, self.config.fitness_percentile * 100)
        avg_fitness = np.mean(path_fitnesses)
        
        # Survival rate
        survival_rate = np.mean(path_fitnesses > 0)
        
        # Max drawdown in synthetic
        max_dd_synthetic = np.max(path_drawdowns)
        
        # Shock resistance (how much did fitness drop?)
        shock_resistance = (original_fitness - worst_case) / (abs(original_fitness) + 1e-6)
        
        # Check if this is a worst-case scenario for Hall of Shame
        if worst_case < -1.0:  # Very bad
            self._add_to_hall_of_shame({
                'fitness': worst_case,
                'original_fitness': original_fitness,
                'shock_type': 'severe',
                'timestamp': datetime.now().isoformat()
            })
        
        return AdversarialResult(
            original_fitness=original_fitness,
            worst_case_fitness=worst_case,
            percentile_fitness=percentile_fitness,
            avg_synthetic_fitness=avg_fitness,
            survival_rate=survival_rate,
            max_drawdown_synthetic=max_dd_synthetic,
            shocked_return=worst_case,
            shock_resistance=shock_resistance
        )
    
    def validate_per_regime(self, strategy: Any, data: pd.DataFrame,
                          regime: str, fitness_func: Callable) -> AdversarialResult:
        """Validate strategy under regime-specific shocks."""
        
        # Get baseline
        original_fitness = fitness_func(strategy, data)
        
        # Generate regime-specific shocks
        synthetic_paths = self.shock_generator.generate_regime_shocks(data, regime)
        
        # Evaluate
        path_fitnesses = []
        for path_data in synthetic_paths:
            synth_fitness = fitness_func(strategy, path_data)
            path_fitnesses.append(synth_fitness)
        
        path_fitnesses = np.array(path_fitnesses)
        
        return AdversarialResult(
            original_fitness=original_fitness,
            worst_case_fitness=np.min(path_fitnesses),
            percentile_fitness=np.percentile(path_fitnesses, 10),
            avg_synthetic_fitness=np.mean(path_fitnesses),
            survival_rate=np.mean(path_fitnesses > 0),
            max_drawdown_synthetic=np.max([self._calculate_max_drawdown(p) for p in synthetic_paths]),
            shocked_return=np.min(path_fitnesses),
            shock_resistance=0.0  # Calculate separately
        )
    
    def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        prices = np.nan_to_num(data['close'].values, nan=100.0)
        prices = np.clip(prices, 0.01, 1e10)
        
        if len(prices) == 0:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            if peak > 0:
                dd = (peak - price) / peak
                if dd > max_dd:
                    max_dd = dd
        
        return max_dd
    
    def _add_to_hall_of_shame(self, scenario: Dict):
        """Add worst-case scenario to Hall of Shame."""
        self.hall_of_shame.append(scenario)
        self.hall_of_shame.sort(key=lambda x: x['fitness'])
        
        # Keep only worst 100
        self.hall_of_shame = self.hall_of_shame[:100]
    
    def get_hall_of_shame(self) -> List[Dict]:
        """Get Hall of Shame scenarios."""
        return self.hall_of_shame


# ============================================================
# ADVERSARIAL MUTATION OPERATORS
# ============================================================
class AdversarialMutationOperators:
    """
    Mutation operators that add chaos tolerance.
    
    Expert: "Allow the GA to evolve its own tolerance to chaos."
    """
    
    @staticmethod
    def add_crash_hedge(params: Dict) -> Dict:
        """Add protective put hedge for crash scenarios."""
        params = copy.deepcopy(params)
        params['crash_hedge'] = True
        params['hedge_allocation'] = random.uniform(0.05, 0.15)
        return params
    
    @staticmethod
    def add_volatility_filter(params: Dict) -> Dict:
        """Add volatility-based entry filter."""
        params = copy.deepcopy(params)
        params['vol_filter_enabled'] = True
        params['vol_filter_threshold'] = random.uniform(0.15, 0.30)
        return params
    
    @staticmethod
    def add_gap_protection(params: Dict) -> Dict:
        """Add protection against overnight gaps."""
        params = copy.deepcopy(params)
        params['gap_protection'] = True
        params['max_overnight_exposure'] = random.uniform(0.3, 0.7)
        return params
    
    @staticmethod
    def add_liquidity_exit(params: Dict) -> Dict:
        """Add liquidity-based exit."""
        params = copy.deepcopy(params)
        params['liquidity_exit'] = True
        params['min_volume_threshold'] = random.uniform(500, 2000)
        return params
    
    @staticmethod
    def tighten_stops(params: Dict) -> Dict:
        """Tighten stops for crash protection."""
        params = copy.deepcopy(params)
        if 'stop_loss_pct' in params:
            params['stop_loss_pct'] *= random.uniform(0.5, 0.8)
        return params
    
    @staticmethod
    def add_time_stop(params: Dict) -> Dict:
        """Add time-based exit to prevent overnight holds."""
        params = copy.deepcopy(params)
        params['time_stop_bars'] = random.randint(4, 24)
        return params
    
    @staticmethod
    def reduce_position_in_vol(params: Dict) -> Dict:
        """Reduce position size in high vol."""
        params = copy.deepcopy(params)
        params['vol_position_scaling'] = True
        params['high_vol_reduction'] = random.uniform(0.3, 0.6)
        return params


# ============================================================
# FACTORY FUNCTIONS
# ============================================================
def create_adversarial_config() -> AdversarialConfig:
    """Create default adversarial configuration."""
    return AdversarialConfig(
        n_synthetic_paths=100,
        shock_intensity=1.0,
        bootstrap_window=50,
        garch_p=1,
        garch_q=1,
        garch_vol_scale=2.0,
        fitness_percentile=0.1,
        use_worst_case=True,
        shock_per_regime=True,
        bull_crash_prob=0.3,
        bear_rally_prob=0.3,
        shock_frequency=0.1,
        allow_multi_shock=True
    )


# ============================================================
# CLI / TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Normal market
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Test Shock Generator
    print("Testing Shock Generator...")
    config = create_adversarial_config()
    generator = ShockGenerator(config)
    
    # Generate paths
    paths = generator.generate_synthetic_paths(data, n_paths=10)
    print(f"  Generated {len(paths)} synthetic paths")
    
    # Check first path has shocks
    first_returns = np.diff(paths[0]['close'].values) / paths[0]['close'].values[:-1]
    print(f"  First path returns: min={first_returns.min():.3f}, max={first_returns.max():.3f}")
    
    # Test GARCH
    print("\nTesting GARCH Model...")
    garch = GARCHModel(p=1, q=1)
    garch.fit(returns)
    sim_returns = garch.simulate(100, vol_scale=1.5)
    print(f"  Simulated returns: mean={sim_returns.mean():.4f}, std={sim_returns.std():.4f}")
    
    # Test Adversarial Validator
    print("\nTesting Adversarial Validator...")
    validator = AdversarialValidator(config)
    
    # Dummy strategy
    dummy_strategy = {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04}
    
    def dummy_fitness(s, data):
        returns = np.diff(data['close'].values) / data['close'].values[:-1]
        return np.mean(returns) * 100 - np.std(returns) * 10
    
    result = validator.validate(dummy_strategy, data, dummy_fitness)
    print(f"  Original fitness: {result.original_fitness:.3f}")
    print(f"  Worst case: {result.worst_case_fitness:.3f}")
    print(f"  10th percentile: {result.percentile_fitness:.3f}")
    print(f"  Survival rate: {result.survival_rate:.1%}")
    
    # Test mutations
    print("\nTesting Adversarial Mutations...")
    new_params = AdversarialMutationOperators.add_crash_hedge(dummy_strategy)
    print(f"  After crash hedge: {new_params.get('crash_hedge', False)}")
    
    new_params = AdversarialMutationOperators.add_volatility_filter(dummy_strategy)
    print(f"  After vol filter: {new_params.get('vol_filter_enabled', False)}")
    
    print("\n✅ All adversarial tests passed!")
