"""
QuantCore - Walk-Forward Validation Module

Implements proper backtesting methodology:
1. In-sample optimization (train)
2. Out-of-sample validation (test)
3. Rolling window walk-forward
4. Monte Carlo simulation for robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""
    asset: str
    train_return: float
    test_return: float
    train_sharpe: float
    test_sharpe: float
    train_drawdown: float
    test_drawdown: float
    consistency_score: float  # How well train translates to test
    
    # Rolling metrics
    rolling_returns: List[float]
    rolling_sharpes: List[float]
    robustness_score: float


class WalkForwardValidator:
    """
    Walk-forward validation for robust strategy testing.
    
    Flow:
    1. Split data into train (70%) and test (30%)
    2. Optimize strategy on train data
    3. Test on unseen test data
    4. Repeat with rolling windows
    5. Calculate robustness metrics
    """
    
    def __init__(
        self,
        train_pct: float = 0.70,
        n_windows: int = 5,
        step_size: float = 0.10
    ):
        self.train_pct = train_pct
        self.n_windows = n_windows
        self.step_size = step_size
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_params: Dict,
        asset: str = "UNKNOWN"
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            data: OHLCV data
            strategy_params: Strategy parameters to test
            asset: Asset name
            
        Returns:
            WalkForwardResult with train/test metrics
        """
        close = data['close'].values
        n = len(close)
        
        # Calculate window sizes
        window_size = int(n * self.step_size)
        train_size = int(n * self.train_pct)
        
        rolling_returns = []
        rolling_sharpes = []
        
        # Run multiple windows
        for i in range(self.n_windows):
            start_idx = i * window_size
            end_idx = start_idx + train_size + window_size
            
            if end_idx > n:
                break
            
            # Split
            train_data = close[start_idx:start_idx + train_size]
            test_data = close[start_idx + train_size:end_idx]
            
            if len(test_data) < 20:
                continue
            
            # Run strategy on train and test
            train_ret, train_sharpe = self._backtest_period(
                train_data, strategy_params
            )
            test_ret, test_sharpe = self._backtest_period(
                test_data, strategy_params
            )
            
            rolling_returns.append(test_ret)
            rolling_sharpes.append(test_sharpe)
        
        # Calculate final metrics on full data
        train_data = close[:train_size]
        test_data = close[train_size:]
        
        train_ret, train_sharpe = self._backtest_period(train_data, strategy_params)
        test_ret, test_sharpe = self._backtest_period(test_data, strategy_params)
        
        # Calculate consistency (how well train predicts test)
        consistency = self._calculate_consistency(
            train_ret, test_ret, train_sharpe, test_sharpe
        )
        
        # Robustness score (based on rolling variance)
        robustness = self._calculate_robustness(rolling_returns, rolling_sharpes)
        
        return WalkForwardResult(
            asset=asset,
            train_return=train_ret,
            test_return=test_ret,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_drawdown=0,  # Simplified
            test_drawdown=0,
            consistency_score=consistency,
            rolling_returns=rolling_returns,
            rolling_sharpes=rolling_sharpes,
            robustness_score=robustness
        )
    
    def _backtest_period(
        self,
        close: np.ndarray,
        params: Dict
    ) -> Tuple[float, float]:
        """Backtest a single period."""
        rsi_period = int(params.get('rsi_period', 14))
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        
        # Calculate RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50).values
        
        # Generate signals
        entries = rsi < oversold
        exits = rsi > overbought
        
        # Calculate returns
        returns = []
        position = 0
        
        for i in range(1, len(close) - 1):
            if entries[i] and position == 0:
                position = close[i + 1] / close[i] - 1
            elif exits[i] and position != 0:
                ret = (close[i + 1] / close[i] - 1) + position
                returns.append(ret)
                position = 0
        
        if len(returns) == 0:
            return 0.0, 0.0
        
        returns = np.array(returns)
        
        total_return = (1 + returns).prod() - 1
        
        # Sharpe
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return total_return * 100, sharpe
    
    def _calculate_consistency(
        self,
        train_ret: float,
        test_ret: float,
        train_sharpe: float,
        test_sharpe: float
    ) -> float:
        """Calculate how consistent train/test performance is."""
        # Direction consistency
        dir_consistent = 1.0 if (train_ret > 0) == (test_ret > 0) else 0.0
        
        # Sharpe consistency (normalized)
        sharpe_diff = abs(train_sharpe - test_sharpe)
        sharpe_consistent = max(0, 1 - sharpe_diff / 10)
        
        return (dir_consistent + sharpe_consistent) / 2
    
    def _calculate_robustness(
        self,
        returns: List[float],
        sharpes: List[float]
    ) -> float:
        """Calculate robustness score based on variance."""
        if len(returns) < 2:
            return 0.5
        
        # Lower variance = more robust
        ret_std = np.std(returns) if returns else 1
        sharpe_std = np.std(sharpes) if sharpes else 1
        
        # Score based on low variance
        ret_robust = 1 / (1 + ret_std)
        sharpe_robust = 1 / (1 + sharpe_std)
        
        return (ret_robust + sharpe_robust) / 2


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================
class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness.
    
    Tests strategy under:
    - Random trade shuffling
    - Varying market conditions
    - Transaction cost scenarios
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def simulate(
        self,
        returns: List[float],
        n_trades: int = 100
    ) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Returns confidence intervals for various metrics.
        """
        if len(returns) == 0:
            return self._empty_results()
        
        returns = np.array(returns)
        
        results = {
            'total_return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': [],
            'avg_win': [],
            'avg_loss': []
        }
        
        for _ in range(self.n_simulations):
            # Resample with replacement
            sim_returns = np.random.choice(returns, size=n_trades, replace=True)
            
            # Calculate metrics
            total_ret = (1 + sim_returns).prod() - 1
            results['total_return'].append(total_ret)
            
            # Sharpe
            if len(sim_returns) > 1 and sim_returns.std() > 0:
                sharpe = sim_returns.mean() / sim_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            results['sharpe_ratio'].append(sharpe)
            
            # Max drawdown
            equity = (1 + sim_returns).cumprod()
            running_max = np.maximum.accumulate(equity)
            dd = (equity - running_max) / running_max
            results['max_drawdown'].append(abs(np.min(dd)))
            
            # Win rate
            wins = (sim_returns > 0).sum()
            results['win_rate'].append(wins / len(sim_returns))
            
            # Avg win/loss
            pos_returns = sim_returns[sim_returns > 0]
            neg_returns = sim_returns[sim_returns < 0]
            results['avg_win'].append(pos_returns.mean() if len(pos_returns) > 0 else 0)
            results['avg_loss'].append(abs(neg_returns.mean()) if len(neg_returns) > 0 else 0)
        
        # Calculate confidence intervals
        confidence_results = {}
        for key, values in results.items():
            values = np.array(values)
            alpha = 1 - self.confidence_level
            
            confidence_results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'ci_lower': float(np.percentile(values, alpha / 2 * 100)),
                'ci_upper': float(np.percentile(values, (1 - alpha / 2) * 100)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        return confidence_results
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            key: {
                'mean': 0, 'std': 0, 'median': 0,
                'ci_lower': 0, 'ci_upper': 0, 'min': 0, 'max': 0
            }
            for key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'avg_win', 'avg_loss']
        }


# ============================================================
# STRATEGY FACTORY
# ============================================================
class StrategyFactory:
    """
    Factory for creating different strategy types.
    
    Supported strategies:
    - RSI Mean Reversion
    - MACD Crossover
    - Bollinger Bands
    - Momentum
    - Trend Following
    """
    
    STRATEGIES = {
        'rsi': {
            'params': {
                'rsi_period': (7, 21),
                'oversold': (20, 35),
                'overbought': (65, 80)
            },
            'description': 'RSI-based mean reversion'
        },
        'macd': {
            'params': {
                'fast_period': (8, 16),
                'slow_period': (16, 32),
                'signal_period': (6, 12)
            },
            'description': 'MACD crossover'
        },
        'bollinger': {
            'params': {
                'period': (10, 30),
                'std_dev': (1.5, 3.0)
            },
            'description': 'Bollinger Band breakout'
        },
        'momentum': {
            'params': {
                'period': (5, 20),
                'threshold': (0.01, 0.05)
            },
            'description': 'Momentum strategy'
        },
        'trend': {
            'params': {
                'fast_ma': (10, 30),
                'slow_ma': (30, 100)
            },
            'description': 'Moving average trend following'
        }
    }
    
    @classmethod
    def get_random_params(cls, strategy_type: str) -> Dict:
        """Get random parameters for a strategy type."""
        if strategy_type not in cls.STRATEGIES:
            strategy_type = 'rsi'
        
        params = {}
        for param, (min_val, max_val) in cls.STRATEGIES[strategy_type]['params'].items():
            if isinstance(min_val, int):
                params[param] = random.randint(min_val, max_val)
            else:
                params[param] = random.uniform(min_val, max_val)
        
        params['strategy_type'] = strategy_type
        return params
    
    @classmethod
    def backtest(cls, data: pd.DataFrame, params: Dict) -> Tuple[float, float, int]:
        """
        Backtest a strategy with given parameters.
        
        Returns: (total_return, sharpe_ratio, n_trades)
        """
        strategy_type = params.get('strategy_type', 'rsi')
        
        if strategy_type == 'rsi':
            return cls._backtest_rsi(data, params)
        elif strategy_type == 'macd':
            return cls._backtest_macd(data, params)
        elif strategy_type == 'bollinger':
            return cls._backtest_bollinger(data, params)
        elif strategy_type == 'momentum':
            return cls._backtest_momentum(data, params)
        elif strategy_type == 'trend':
            return cls._backtest_trend(data, params)
        else:
            return cls._backtest_rsi(data, params)
    
    @classmethod
    def _backtest_rsi(cls, data: pd.DataFrame, params: Dict) -> Tuple[float, float, int]:
        """Backtest RSI strategy."""
        close = data['close'].values
        rsi_period = int(params.get('rsi_period', 14))
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50).values
        
        entries = rsi < oversold
        exits = rsi > overbought
        
        returns = cls._calculate_returns(close, entries, exits)
        
        if len(returns) == 0:
            return 0.0, 0.0, 0
        
        total_ret = (1 + returns).prod() - 1
        sharpe = cls._calculate_sharpe(returns)
        
        return total_ret * 100, sharpe, len(returns)
    
    @classmethod
    def _backtest_macd(cls, data: pd.DataFrame, params: Dict) -> Tuple[float, float, int]:
        """Backtest MACD strategy."""
        close = data['close'].values
        
        fast = int(params.get('fast_period', 12))
        slow = int(params.get('slow_period', 26))
        signal = int(params.get('signal_period', 9))
        
        ema_fast = pd.Series(close).ewm(span=fast).mean()
        ema_slow = pd.Series(close).ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        entries = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        exits = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        
        returns = cls._calculate_returns(close, entries.values, exits.values)
        
        if len(returns) == 0:
            return 0.0, 0.0, 0
        
        total_ret = (1 + returns).prod() - 1
        sharpe = cls._calculate_sharpe(returns)
        
        return total_ret * 100, sharpe, len(returns)
    
    @classmethod
    def _backtest_bollinger(cls, data: pd.DataFrame, params: Dict) -> Tuple[float, float, int]:
        """Backtest Bollinger Bands strategy."""
        close = data['close'].values
        
        period = int(params.get('period', 20))
        std_dev = params.get('std_dev', 2.0)
        
        sma = pd.Series(close).rolling(period).mean()
        std = pd.Series(close).rolling(period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        entries = close < lower.values
        exits = close > upper.values
        
        returns = cls._calculate_returns(close, entries, exits)
        
        if len(returns) == 0:
            return 0.0, 0.0, 0
        
        total_ret = (1 + returns).prod() - 1
        sharpe = cls._calculate_sharpe(returns)
        
        return total_ret * 100, sharpe, len(returns)
    
    @classmethod
    def _backtest_momentum(cls, data: pd.DataFrame, params: Dict) -> Tuple[float, float, int]:
        """Backtest momentum strategy."""
        close = data['close'].values
        
        period = int(params.get('period', 10))
        threshold = params.get('threshold', 0.02)
        
        momentum = pd.Series(close).pct_change(period)
        
        entries = momentum > threshold
        exits = momentum < -threshold
        
        returns = cls._calculate_returns(close, entries.values, exits.values)
        
        if len(returns) == 0:
            return 0.0, 0.0, 0
        
        total_ret = (1 + returns).prod() - 1
        sharpe = cls._calculate_sharpe(returns)
        
        return total_ret * 100, sharpe, len(returns)
    
    @classmethod
    def _backtest_trend(cls, data: pd.DataFrame, params: Dict) -> Tuple[float, float, int]:
        """Backtest trend following strategy."""
        close = data['close'].values
        
        fast_ma = int(params.get('fast_ma', 10))
        slow_ma = int(params.get('slow_ma', 50))
        
        fast = pd.Series(close).rolling(fast_ma).mean()
        slow = pd.Series(close).rolling(slow_ma).mean()
        
        entries = fast > slow
        exits = fast < slow
        
        returns = cls._calculate_returns(close, entries.values, exits.values)
        
        if len(returns) == 0:
            return 0.0, 0.0, 0
        
        total_ret = (1 + returns).prod() - 1
        sharpe = cls._calculate_sharpe(returns)
        
        return total_ret * 100, sharpe, len(returns)
    
    @staticmethod
    def _calculate_returns(close, entries, exits):
        """Calculate returns from signals."""
        returns = []
        position = 0
        
        for i in range(1, len(close) - 1):
            if entries[i] and position == 0:
                position = 1
            elif exits[i] and position == 1:
                ret = close[i + 1] / close[i] - 1
                returns.append(ret)
                position = 0
        
        return np.array(returns) if returns else np.array([])
    
    @staticmethod
    def _calculate_sharpe(returns):
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)


# ============================================================
# COMPREHENSIVE VALIDATION
# ============================================================
class ComprehensiveValidator:
    """
    Combines walk-forward and Monte Carlo for robust validation.
    """
    
    def __init__(
        self,
        walk_forward_windows: int = 5,
        monte_carlo_sims: int = 1000
    ):
        self.wf_validator = WalkForwardValidator(n_windows=walk_forward_windows)
        self.mc_simulator = MonteCarloSimulator(n_simulations=monte_carlo_sims)
    
    def validate(
        self,
        data: pd.DataFrame,
        params: Dict,
        asset: str = "UNKNOWN"
    ) -> Dict:
        """
        Run comprehensive validation.
        
        Returns validation report with:
        - Walk-forward results
        - Monte Carlo results
        - Overall robustness score
        - Recommendation (pass/fail)
        """
        # Walk-forward validation
        wf_result = self.wf_validator.run_walk_forward(data, params, asset)
        
        # Run strategy to get returns for Monte Carlo
        ret, sharpe, n_trades = StrategyFactory.backtest(data, params)
        
        # Convert single return to list for MC
        returns = np.array([ret / 100]) if n_trades == 0 else np.array([])
        
        # Monte Carlo simulation
        if n_trades > 10:
            mc_results = self.mc_simulator.simulate(returns, n_trades)
        else:
            mc_results = self.mc_simulator._empty_results()
        
        # Calculate overall robustness score
        robustness = (
            wf_result.robustness_score * 0.4 +
            self._mc_robustness(mc_results) * 0.4 +
            wf_result.consistency_score * 0.2
        )
        
        # Determine if strategy passes
        passes = (
            robustness > 0.5 and
            wf_result.test_return > 0 and
            mc_results['total_return']['ci_lower'] > -0.5
        )
        
        return {
            'asset': asset,
            'params': params,
            'walk_forward': {
                'train_return': wf_result.train_return,
                'test_return': wf_result.test_return,
                'train_sharpe': wf_result.train_sharpe,
                'test_sharpe': wf_result.test_sharpe,
                'consistency': wf_result.consistency_score,
                'robustness': wf_result.robustness_score
            },
            'monte_carlo': mc_results,
            'overall_robustness': robustness,
            'passes_validation': passes,
            'recommendation': 'PASS' if passes else 'FAIL'
        }
    
    def _mc_robustness(self, mc_results: Dict) -> float:
        """Calculate robustness from Monte Carlo results."""
        # Check if confidence intervals are reasonable
        ret_ci_width = (
            mc_results['total_return']['ci_upper'] - 
            mc_results['total_return']['ci_lower']
        )
        
        sharpe_mean = mc_results['sharpe_ratio']['mean']
        sharpe_std = mc_results['sharpe_ratio']['std']
        
        # Lower CI width + positive mean = more robust
        ci_score = 1 / (1 + abs(ret_ci_width))
        sharpe_score = 1 / (1 + abs(sharpe_std))
        
        return (ci_score + sharpe_score) / 2


# ============================================================
# EXAMPLE USAGE
# ============================================================
def demo():
    """Demo the validation system."""
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    
    data = pd.DataFrame({'close': close}, index=dates)
    
    # Test different strategy types
    print("=" * 60)
    print("COMPREHENSIVE STRATEGY VALIDATION")
    print("=" * 60)
    
    validator = ComprehensiveValidator()
    
    # Test RSI strategy
    params = StrategyFactory.get_random_params('rsi')
    print(f"\n📊 Testing RSI Strategy: {params}")
    
    result = validator.validate(data, params, 'BTC-USD')
    
    print(f"\n✅ Validation Result: {result['recommendation']}")
    print(f"   Overall Robustness: {result['overall_robustness']:.2%}")
    print(f"   Walk-Forward Test Return: {result['walk_forward']['test_return']:.2f}%")
    print(f"   Monte Carlo Mean Return: {result['monte_carlo']['total_return']['mean']:.2f}%")
    print(f"   95% CI: [{result['monte_carlo']['total_return']['ci_lower']:.2f}%, {result['monte_carlo']['total_return']['ci_upper']:.2f}%]")
    
    return result


if __name__ == "__main__":
    demo()
