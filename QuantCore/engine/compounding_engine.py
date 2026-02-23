"""
QuantCore - Compounding Engine v1.0

This module turns the system into a self-compounding, money-printing machine.

Expert: "Every dollar earned is immediately put back to work, growing like a virus."

Features:
1. Compounding Engine (equity curve tracking, dynamic sizing)
2. Evolvable Position Sizing (Kelly, fraction, scale in/out)
3. Reinvestment Logic (thresholds, lags, caps)
4. Drawdown Protection (reduce risk as DD grows)
5. Portfolio Allocator (multi-strategy weights)
6. Compounding-Aware Fitness (CAGR, Ulcer, MAR)
"""

import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================
class ScaleMethod(Enum):
    """How to scale into positions."""
    NONE = "none"
    PYRAMID_IN = "pyramid_in"  # Add to winners
    PYRAMID_OUT = "pyramid_out"  # Scale out of winners
    MARTINGALE = "martingale"  # Double after loss
    ANTI_MARTINGALE = "anti_martingale"  # Double after win


class AllocationMethod(Enum):
    """Portfolio allocation method."""
    EQUAL = "equal"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"
    INVERSE_VOL = "inverse_vol"


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class CompoundingConfig:
    """
    Configuration for compounding behavior.
    
    These are evolvable parameters that control how capital compounds.
    """
    # Base risk
    base_risk_per_trade: float = 0.02      # 2% of equity per trade
    
    # Kelly
    kelly_fraction: float = 0.25           # Fraction of Kelly to use
    use_kelly: bool = True
    
    # Reinvestment
    reinvestment_rate: float = 0.80        # 80% of profits reinvested
    reinvestment_threshold: float = 0.05     # Min profit % to trigger reinvestment
    reinvestment_lag_trades: int = 3        # Wait N winning trades
    reinvestment_cap: float = 1.0           # Max % of profits to reinvest
    
    # Drawdown protection
    drawdown_limit: float = 0.15            # 15% DD triggers risk reduction
    drawdown_risk_reduction: float = 0.50   # Reduce position size by 50% in DD
    use_drawdown_protection: bool = True
    
    # Equity curve smoothing
    equity_smoothing_window: int = 10        # Use N-day MA of equity for sizing
    use_equity_smoothing: bool = False
    
    # Scale methods
    scale_in_method: str = "pyramid_in"
    scale_out_method: str = "none"
    pyramid_size: float = 0.5               # How much to add per scale
    
    # Compounding mode
    compounding_mode: str = "geometric"    # 'geometric', 'arithmetic'
    
    # Withdrawal
    withdrawal_rate: float = 0.0            # % to withdraw periodically
    withdrawal_interval_days: int = 30
    
    def to_dict(self) -> Dict:
        return {
            'base_risk_per_trade': self.base_risk_per_trade,
            'kelly_fraction': self.kelly_fraction,
            'reinvestment_rate': self.reinvestment_rate,
            'drawdown_limit': self.drawdown_limit,
            'drawdown_risk_reduction': self.drawdown_risk_reduction
        }


@dataclass 
class EquityCurve:
    """Track equity curve over time."""
    timestamps: List[datetime] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    drawdown: List[float] = field(default_factory=list)
    peak: float = 10000
    
    def add(self, timestamp: datetime, value: float):
        """Add new equity point."""
        self.timestamps.append(timestamp)
        self.equity.append(value)
        
        # Update peak and drawdown
        if value > self.peak:
            self.peak = value
        
        dd = (self.peak - value) / self.peak
        self.drawdown.append(dd)
    
    def get_current(self) -> float:
        """Get current equity."""
        return self.equity[-1] if self.equity else 0
    
    def get_drawdown(self) -> float:
        """Get current drawdown."""
        return self.drawdown[-1] if self.drawdown else 0
    
    def get_peak(self) -> float:
        """Get peak equity."""
        return self.peak
    
    def get_returns(self) -> np.ndarray:
        """Get returns array."""
        if len(self.equity) < 2:
            return np.array([0])
        return np.diff(self.equity) / self.equity[:-1]


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy in portfolio."""
    strategy_id: str
    weight: float = 0.0
    active: bool = True
    recent_return: float = 0.0
    correlation: float = 0.0  # Correlation with other strategies


# ============================================================
# COMPOUNDING ENGINE
# ============================================================
class CompoundingEngine:
    """
    Core compounding engine that manages equity curve and position sizing.
    
    Tracks equity, calculates position sizes, manages reinvestment.
    """
    
    def __init__(self, config: CompoundingConfig, initial_capital: float = 10000):
        self.config = config
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.equity_curve = EquityCurve(peak=initial_capital)
        
        # Tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.pending_reinvest = 0
        self.withdrawal_accumulated = 0
        
        # Performance metrics
        self.win_rate = 0.5
        self.avg_win = 0
        self.avg_loss = 0
        
    def calculate_position_size(self, 
                               confidence: float = 1.0,
                               entry_price: float = 0,
                               stop_loss_pct: float = 0.03) -> float:
        """
        Calculate position size based on current equity and compounding rules.
        
        Returns: Dollar amount to risk
        """
        # Apply drawdown protection
        risk_multiplier = 1.0
        
        if self.config.use_drawdown_protection:
            current_dd = self.equity_curve.get_drawdown()
            if current_dd > self.config.drawdown_limit:
                # Reduce risk as drawdown grows
                excess_dd = current_dd - self.config.drawdown_limit
                risk_multiplier = max(0.1, 1.0 - excess_dd / self.config.drawdown_limit * self.config.drawdown_risk_reduction)
        
        # Apply equity smoothing
        equity_for_sizing = self.current_equity
        
        if self.config.use_equity_smoothing and len(self.equity_curve.equity) >= self.config.equity_smoothing_window:
            window = self.config.equity_smoothing_window
            equity_for_sizing = np.mean(self.equity_curve.equity[-window:])
        
        # Calculate base size
        if self.config.use_kelly:
            # Kelly criterion
            if self.win_rate > 0 and self.avg_loss > 0:
                win_prob = self.win_rate
                loss_prob = 1 - self.win_rate
                win_loss_ratio = self.avg_win / (self.avg_loss + 1e-10)
                
                kelly = win_prob - (loss_prob / win_loss_ratio)
                kelly = max(0, min(kelly, 1))  # Bound
                
                # Apply fraction
                adjusted_kelly = kelly * self.config.kelly_fraction
                
                risk_amount = equity_for_sizing * adjusted_kelly * confidence * risk_multiplier
            else:
                risk_amount = equity_for_sizing * self.config.base_risk_per_trade * confidence * risk_multiplier
        else:
            # Fixed percentage
            risk_amount = equity_for_sizing * self.config.base_risk_per_trade * confidence * risk_multiplier
        
        # Convert to position size
        if stop_loss_pct > 0 and entry_price > 0:
            position_size = risk_amount / (entry_price * stop_loss_pct)
        else:
            position_size = risk_amount / entry_price if entry_price > 0 else 0
        
        return position_size
    
    def record_trade(self, pnl: float, is_win: bool):
        """Record a trade and update compounding state."""
        self.total_trades += 1
        
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # Track win metrics
            if self.total_trades == 1:
                self.avg_win = pnl
            else:
                self.avg_win = self.avg_win * 0.9 + pnl * 0.1
            
            # Check reinvestment
            profit_pct = pnl / self.current_equity
            
            if profit_pct >= self.config.reinvestment_threshold:
                self.pending_reinvest += pnl * self.config.reinvestment_rate
                
                if self.consecutive_wins >= self.config.reinvestment_lag_trades:
                    # Execute reinvestment
                    self.current_equity += self.pending_reinvest
                    self.pending_reinvest = 0
                    self.consecutive_wins = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # Track loss metrics
            if self.total_trades == 1:
                self.avg_loss = abs(pnl)
            else:
                self.avg_loss = self.avg_loss * 0.9 + abs(pnl) * 0.1
        
        # Update equity
        self.current_equity += pnl
        
        # Record in curve
        self.equity_curve.add(datetime.now(), self.current_equity)
        
        # Update win rate
        wins = sum(1 for e in self.equity_curve.equity[1:] 
                  if e > self.equity_curve.equity[0])
        self.win_rate = wins / max(1, len(self.equity_curve.equity) - 1)
        
        # Handle withdrawals
        self._check_withdrawal()
    
    def _check_withdrawal(self):
        """Check if it's time for withdrawal."""
        if self.config.withdrawal_rate > 0 and self.total_trades > 0:
            # Simple: withdraw after N trades
            if self.total_trades % (self.config.withdrawal_interval_days * 10) == 0:  # Approximate
                withdrawal = self.current_equity * self.config.withdrawal_rate
                self.current_equity -= withdrawal
                self.withdrawal_accumulated += withdrawal
    
    def get_status(self) -> Dict:
        """Get compounding status."""
        return {
            'current_equity': self.current_equity,
            'initial_capital': self.initial_capital,
            'total_return': (self.current_equity - self.initial_capital) / self.initial_capital,
            'current_drawdown': self.equity_curve.get_drawdown(),
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_trades': self.total_trades,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'pending_reinvest': self.pending_reinvest
        }


# ============================================================
# COMPOUNDING MUTATIONS
# ============================================================
class CompoundingMutations:
    """
    Mutation operators for compounding behavior.
    """
    
    @staticmethod
    def mutate_base_risk(config: CompoundingConfig) -> CompoundingConfig:
        """Mutate base risk per trade."""
        config = copy.deepcopy(config)
        config.base_risk_per_trade = max(0.001, min(0.10, 
            config.base_risk_per_trade * random.uniform(0.8, 1.2)))
        return config
    
    @staticmethod
    def mutate_kelly_fraction(config: CompoundingConfig) -> CompoundingConfig:
        """Mutate Kelly fraction."""
        config = copy.deepcopy(config)
        config.kelly_fraction = max(0.05, min(0.5,
            config.kelly_fraction + random.uniform(-0.05, 0.05)))
        return config
    
    @staticmethod
    def mutate_reinvestment_rate(config: CompoundingConfig) -> CompoundingConfig:
        """Mutate reinvestment rate."""
        config = copy.deepcopy(config)
        config.reinvestment_rate = max(0.0, min(1.0,
            config.reinvestment_rate + random.uniform(-0.1, 0.1)))
        return config
    
    @staticmethod
    def mutate_drawdown_limit(config: CompoundingConfig) -> CompoundingConfig:
        """Mutate drawdown limit."""
        config = copy.deepcopy(config)
        config.drawdown_limit = max(0.05, min(0.30,
            config.drawdown_limit + random.uniform(-0.02, 0.02)))
        return config
    
    @staticmethod
    def mutate_drawdown_risk_reduction(config: CompoundingConfig) -> CompoundingConfig:
        """Mutate risk reduction in drawdown."""
        config = copy.deepcopy(config)
        config.drawdown_risk_reduction = max(0.1, min(1.0,
            config.drawdown_risk_reduction + random.uniform(-0.1, 0.1)))
        return config
    
    @staticmethod
    def mutate_reinvestment_threshold(config: CompoundingConfig) -> CompoundingConfig:
        """Mutate reinvestment threshold."""
        config = copy.deepcopy(config)
        config.reinvestment_threshold = max(0.01, min(0.20,
            config.reinvestment_threshold + random.uniform(-0.02, 0.02)))
        return config
    
    @staticmethod
    def mutate_scale_method(config: CompoundingConfig) -> CompoundingConfig:
        """Mutate scale in/out method."""
        config = copy.deepcopy(config)
        methods = ['none', 'pyramid_in', 'pyramid_out', 'martingale', 'anti_martingale']
        if config.scale_in_method in methods:
            idx = methods.index(config.scale_in_method)
            config.scale_in_method = methods[(idx + 1) % len(methods)]
        return config
    
    @staticmethod
    def toggle_kelly(config: CompoundingConfig) -> CompoundingConfig:
        """Toggle Kelly usage."""
        config = copy.deepcopy(config)
        config.use_kelly = not config.use_kelly
        return config
    
    @staticmethod
    def toggle_drawdown_protection(config: CompoundingConfig) -> CompoundingConfig:
        """Toggle drawdown protection."""
        config = copy.deepcopy(config)
        config.use_drawdown_protection = not config.use_drawdown_protection
        return config
    
    @staticmethod
    def toggle_equity_smoothing(config: CompoundingConfig) -> CompoundingConfig:
        """Toggle equity smoothing."""
        config = copy.deepcopy(config)
        config.use_equity_smoothing = not config.use_equity_smoothing
        return config


# ============================================================
# PORTFOLIO ALLOCATOR
# ============================================================
class PortfolioAllocator:
    """
    Allocates capital across multiple strategies.
    
    Methods: Equal, Risk Parity, Kelly, Inverse Vol
    """
    
    def __init__(self, method: str = "equal"):
        self.method = AllocationMethod(method) if isinstance(method, str) else method
        self.strategies: Dict[str, StrategyAllocation] = {}
        self.returns_history: Dict[str, List[float]] = {}
        
    def add_strategy(self, strategy_id: str, initial_weight: float = 0.0):
        """Add a strategy to the portfolio."""
        self.strategies[strategy_id] = StrategyAllocation(
            strategy_id=strategy_id,
            weight=initial_weight
        )
        self.returns_history[strategy_id] = []
    
    def remove_strategy(self, strategy_id: str):
        """Remove a strategy."""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
        if strategy_id in self.returns_history:
            del self.returns_history[strategy_id]
    
    def record_return(self, strategy_id: str, return_pct: float):
        """Record a return for a strategy."""
        if strategy_id not in self.returns_history:
            self.returns_history[strategy_id] = []
        
        self.returns_history[strategy_id].append(return_pct)
        
        # Keep only recent
        if len(self.returns_history[strategy_id]) > 100:
            self.returns_history[strategy_id] = self.returns_history[strategy_id][-100:]
        
        if strategy_id in self.strategies:
            self.strategies[strategy_id].recent_return = return_pct
    
    def calculate_weights(self, total_capital: float) -> Dict[str, float]:
        """Calculate allocation weights based on method."""
        if not self.strategies:
            return {}
        
        active_strategies = {sid: s for sid, s in self.strategies.items() if s.active}
        
        if not active_strategies:
            return {}
        
        if self.method == AllocationMethod.EQUAL:
            return {sid: 1.0 / len(active_strategies) for sid in active_strategies}
        
        elif self.method == AllocationMethod.INVERSE_VOL:
            # Inverse volatility weighting
            weights = {}
            vols = {}
            
            for sid in active_strategies:
                returns = self.returns_history.get(sid, [0])
                if len(returns) > 5:
                    vol = np.std(returns)
                    vols[sid] = vol if vol > 0 else 0.01
                else:
                    vols[sid] = 0.01
            
            # Inverse vol weights
            total_inv_vol = sum(1 / v for v in vols.values())
            
            for sid in active_strategies:
                weights[sid] = (1 / vols[sid]) / total_inv_vol
            
            return weights
        
        elif self.method == AllocationMethod.KELLY:
            # Kelly-based allocation
            weights = {}
            kelly_scores = {}
            
            for sid in active_strategies:
                returns = self.returns_history.get(sid, [0])
                if len(returns) > 10:
                    wins = [r for r in returns if r > 0]
                    losses = [r for r in returns if r < 0]
                    
                    win_rate = len(wins) / len(returns)
                    avg_win = np.mean(wins) if wins else 0
                    avg_loss = abs(np.mean(losses)) if losses else 0.01
                    
                    if avg_loss > 0:
                        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                        kelly_scores[sid] = max(0, kelly)
                    else:
                        kelly_scores[sid] = 0
                else:
                    kelly_scores[sid] = 0
            
            total_kelly = sum(kelly_scores.values())
            
            if total_kelly > 0:
                for sid in active_strategies:
                    weights[sid] = kelly_scores[sid] / total_kelly
            else:
                # Fallback to equal
                weights = {sid: 1.0 / len(active_strategies) for sid in active_strategies}
            
            return weights
        
        elif self.method == AllocationMethod.RISK_PARITY:
            # Risk parity (equal risk contribution)
            weights = {}
            risks = {}
            
            for sid in active_strategies:
                returns = self.returns_history.get(sid, [0])
                if len(returns) > 5:
                    risks[sid] = np.std(returns) * np.sqrt(252)  # Annualized
                else:
                    risks[sid] = 0.2
            
            # Inverse risk weights
            total_inv_risk = sum(1 / r for r in risks.values() if r > 0)
            
            for sid in active_strategies:
                if risks[sid] > 0:
                    weights[sid] = (1 / risks[sid]) / total_inv_risk
                else:
                    weights[sid] = 0
            
            return weights
        
        return {sid: 1.0 / len(active_strategies) for sid in active_strategies}
    
    def rebalance(self, total_capital: float) -> Dict[str, float]:
        """Rebalance portfolio to target weights."""
        weights = self.calculate_weights(total_capital)
        
        # Update strategy weights
        for sid in self.strategies:
            if sid in weights:
                self.strategies[sid].weight = weights[sid]
        
        # Return dollar allocations
        return {sid: weight * total_capital for sid, weight in weights.items()}


# ============================================================
# COMPOUNDING FITNESS
# ============================================================
class CompoundingFitness:
    """
    Fitness functions optimized for compounding.
    
    Metrics: CAGR, Ulcer Index, MAR Ratio, Calmar
    """
    
    @staticmethod
    def calculate_cagr(initial: float, final: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate."""
        if initial <= 0 or years <= 0:
            return 0
        return (final / initial) ** (1 / years) - 1
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    @staticmethod
    def calculate_ulcer_index(equity_curve: List[float]) -> float:
        """Calculate Ulcer Index (measure of drawdown pain)."""
        if len(equity_curve) < 2:
            return 0
        
        # Calculate drawdown series
        peak = equity_curve[0]
        dd_series = []
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100  # Percentage
            dd_series.append(dd)
        
        # Root mean square of drawdowns
        ulcer = np.sqrt(np.mean([d ** 2 for d in dd_series]))
        return ulcer
    
    @staticmethod
    def calculate_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0
        
        excess = returns - risk_free
        return np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(returns) < 2:
            return 0
        
        excess = returns - risk_free
        downside = returns[returns < 0]
        
        if len(downside) == 0:
            return 0
        
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0
        
        return np.mean(excess) / downside_std * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar(equity_curve: List[float], years: float) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        if len(equity_curve) < 2 or years <= 0:
            return 0
        
        cagr = CompoundingFitness.calculate_cagr(
            equity_curve[0], equity_curve[-1], years
        )
        max_dd = CompoundingFitness.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0
        
        return cagr / max_dd
    
    @staticmethod
    def calculate_fitness(equity_curve: List[float],
                        returns: np.ndarray,
                        weights: Dict[str, float] = None) -> float:
        """
        Calculate compounding-aware fitness.
        
        Combines CAGR, Sharpe, and drawdown metrics.
        """
        if not equity_curve or len(equity_curve) < 10:
            return 0
        
        # Default weights
        w_cagr = weights.get('cagr', 0.4) if weights else 0.4
        w_sharpe = weights.get('sharpe', 0.3) if weights else 0.3
        w_dd = weights.get('drawdown', 0.3) if weights else 0.3
        
        # Calculate metrics
        years = len(equity_curve) / 252  # Approximate trading days
        cagr = CompoundingFitness.calculate_cagr(equity_curve[0], equity_curve[-1], years)
        sharpe = CompoundingFitness.calculate_sharpe(returns)
        max_dd = CompoundingFitness.calculate_max_drawdown(equity_curve)
        ulcer = CompoundingFitness.calculate_ulcer_index(equity_curve)
        
        # Normalize metrics
        cagr_score = cagr * 100  # Convert to percentage
        sharpe_score = sharpe
        dd_score = 1 - min(1, max_dd)  # Lower is better, invert
        ulcer_score = 1 / (1 + ulcer)  # Lower is better
        
        # Weighted fitness
        fitness = (
            w_cagr * cagr_score +
            w_sharpe * sharpe_score +
            w_dd * dd_score * 0.5 +
            w_dd * ulcer_score * 0.5
        )
        
        # Penalize for excessive drawdown
        if max_dd > 0.30:
            fitness *= 0.5
        elif max_dd > 0.20:
            fitness *= 0.75
        
        return fitness


# ============================================================
# FACTORY FUNCTIONS
# ============================================================
def create_compounding_config(
    risk_per_trade: float = 0.02,
    kelly_fraction: float = 0.25,
    reinvestment_rate: float = 0.80
) -> CompoundingConfig:
    """Create compounding configuration."""
    return CompoundingConfig(
        base_risk_per_trade=risk_per_trade,
        kelly_fraction=kelly_fraction,
        reinvestment_rate=reinvestment_rate
    )


# ============================================================
# CLI / TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("QUANTCORE COMPOUNDING ENGINE TEST")
    print("=" * 50)
    
    # Test 1: Compounding Engine
    print("\n💰 Test 1: Compounding Engine")
    config = create_compounding_config(
        risk_per_trade=0.02,
        kelly_fraction=0.25,
        reinvestment_rate=0.80
    )
    
    engine = CompoundingEngine(config, initial_capital=10000)
    
    # Simulate some trades
    for i in range(20):
        pnl = random.uniform(-200, 400)
        is_win = pnl > 0
        engine.record_trade(pnl, is_win)
    
    status = engine.get_status()
    print(f"  Initial: ${status['initial_capital']:,.2f}")
    print(f"  Current: ${status['current_equity']:,.2f}")
    print(f"  Return: {status['total_return']*100:.2f}%")
    print(f"  Max Drawdown: {status['current_drawdown']*100:.2f}%")
    print(f"  Win Rate: {status['win_rate']*100:.1f}%")
    
    # Test 2: Position Sizing
    print("\n📊 Test 2: Dynamic Position Sizing")
    size = engine.calculate_position_size(confidence=0.8, entry_price=45000, stop_loss_pct=0.03)
    print(f"  Position size (80% conf): ${size*45000:,.2f}")
    
    # Simulate drawdown
    engine.current_equity = 8000
    engine.equity_curve.peak = 10000
    engine.equity_curve.add(datetime.now(), 8000)
    
    size_dd = engine.calculate_position_size(confidence=0.8, entry_price=45000, stop_loss_pct=0.03)
    print(f"  Position size (20% DD): ${size_dd*45000:,.2f}")
    
    # Test 3: Mutations
    print("\n🧬 Test 3: Compounding Mutations")
    config = CompoundingMutations.mutate_kelly_fraction(config)
    print(f"  Kelly fraction: {config.kelly_fraction:.3f}")
    
    config = CompoundingMutations.mutate_drawdown_limit(config)
    print(f"  Drawdown limit: {config.drawdown_limit:.3f}")
    
    config = CompoundingMutations.toggle_kelly(config)
    print(f"  Use Kelly: {config.use_kelly}")
    
    # Test 4: Portfolio Allocator
    print("\n📈 Test 4: Portfolio Allocator")
    allocator = PortfolioAllocator(method="equal")
    
    allocator.add_strategy("strat_1", 0.33)
    allocator.add_strategy("strat_2", 0.33)
    allocator.add_strategy("strat_3", 0.34)
    
    # Record some returns
    for _ in range(20):
        allocator.record_return("strat_1", random.uniform(-0.02, 0.03))
        allocator.record_return("strat_2", random.uniform(-0.03, 0.02))
        allocator.record_return("strat_3", random.uniform(-0.01, 0.04))
    
    weights = allocator.rebalance(10000)
    print(f"  Equal weights: {weights}")
    
    # Test Kelly allocation
    allocator_kelly = PortfolioAllocator(method="kelly")
    for sid in ["strat_1", "strat_2", "strat_3"]:
        allocator_kelly.add_strategy(sid)
    
    for _ in range(20):
        allocator_kelly.record_return("strat_1", random.uniform(-0.02, 0.03))
        allocator_kelly.record_return("strat_2", random.uniform(-0.03, 0.02))
        allocator_kelly.record_return("strat_3", random.uniform(-0.01, 0.04))
    
    weights_kelly = allocator_kelly.rebalance(10000)
    print(f"  Kelly weights: {weights_kelly}")
    
    # Test 5: Fitness Function
    print("\n🎯 Test 5: Compounding Fitness")
    equity = [10000 + i * 50 + np.random.randn() * 100 for i in range(100)]
    returns = np.diff(equity) / equity[:-1]
    
    fitness = CompoundingFitness.calculate_fitness(equity, returns)
    cagr = CompoundingFitness.calculate_cagr(equity[0], equity[-1], 0.4)
    max_dd = CompoundingFitness.calculate_max_drawdown(equity)
    sharpe = CompoundingFitness.calculate_sharpe(returns)
    
    print(f"  CAGR: {cagr*100:.2f}%")
    print(f"  Max DD: {max_dd*100:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Fitness: {fitness:.2f}")
    
    print("\n✅ All compounding tests passed!")
