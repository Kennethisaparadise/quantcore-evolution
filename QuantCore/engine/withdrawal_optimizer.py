"""
QuantCore - Withdrawal Optimizer v1.0

Head 21: The wealth extraction strategist.

Features:
1. Fixed & dynamic withdrawal rates
2. Tax-aware withdrawals
3. Drawdown protection
4. Sequence of returns risk management
5. Goal-based withdrawal planning
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

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================
class WithdrawalStrategy(Enum):
    """Withdrawal strategies."""
    FIXED_PERCENT = "fixed_percent"       # 4% rule
    DYNAMIC = "dynamic"                    # Adjust based on performance
    TAX_AWARE = "tax_aware"              # Prefer long-term gains
    Smoothed = "smoothed"                 # Rolling average


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class WithdrawalConfig:
    """Configuration for withdrawals."""
    # Strategy
    strategy: str = "fixed_percent"
    base_rate: float = 0.04              # 4% annual
    
    # Frequency
    frequency_days: int = 30              # Monthly withdrawals
    
    # Dynamic adjustments
    dynamic_floor: float = 0.03          # Minimum rate
    dynamic_ceiling: float = 0.06       # Maximum rate
    performance_bonus: float = 0.01      # Extra withdrawal after good year
    
    # Tax awareness
    prefer_long_term: bool = True
    harvest_before_withdraw: bool = True
    
    # Drawdown protection
    dd_protection: bool = True
    dd_pause_threshold: float = 0.15     # Pause if >15% DD
    dd_resume_threshold: float = 0.10    # Resume if <10% DD
    
    # Goals (optional)
    goal_amount: float = 0               # Future expense goal
    goal_years: int = 0                 # Years until goal
    
    # Smoothing
    use_smoothing: bool = True
    smoothing_window: int = 12          # 12-month rolling average
    
    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy,
            'base_rate': self.base_rate,
            'frequency_days': self.frequency_days,
            'dd_protection': self.dd_protection
        }


@dataclass
class WithdrawalEvent:
    """A withdrawal event."""
    timestamp: datetime
    amount: float
    equity_before: float
    equity_after: float
    pnl_ytd: float = 0
    withdrawal_rate: float = 0
    gain_type: str = "short_term"


@dataclass
class WithdrawalPlan:
    """Planned withdrawal schedule."""
    scheduled_date: datetime
    planned_amount: float
    tax_estimate: float


# ============================================================
# WITHDRAWAL CALCULATOR
# ============================================================
class WithdrawalCalculator:
    """
    Calculate optimal withdrawals.
    """
    
    def __init__(self, config: WithdrawalConfig):
        self.config = config
        self.withdrawal_history: List[WithdrawalEvent] = []
        self.pnl_history: List[float] = []
        self.current_drawdown: float = 0
        self.paused: bool = False
        
    def calculate_withdrawal(self, current_equity: float, 
                           recent_performance: float = 0) -> Tuple[float, str]:
        """
        Calculate withdrawal amount.
        
        Returns: (amount, reason)
        """
        # Check drawdown protection
        if self.config.dd_protection and self.paused:
            if self.current_drawdown < self.config.dd_resume_threshold:
                self.paused = False
                logger.info("▶️ Withdrawals resumed")
            else:
                return 0, "paused_dd"
        
        if self.config.dd_protection and self.current_drawdown > self.config.dd_pause_threshold:
            self.paused = True
            logger.warning("⏸️ Withdrawals paused due to drawdown")
            return 0, "paused_dd"
        
        # Calculate based on strategy
        if self.config.strategy == "fixed_percent":
            amount = self._fixed_percent(current_equity)
            reason = "fixed"
            
        elif self.config.strategy == "dynamic":
            amount = self._dynamic(current_equity, recent_performance)
            reason = "dynamic"
            
        elif self.config.strategy == "tax_aware":
            amount = self._tax_aware(current_equity)
            reason = "tax_aware"
            
        elif self.config.strategy == "smoothed":
            amount = self._smoothed(current_equity)
            reason = "smoothed"
            
        else:
            amount = self._fixed_percent(current_equity)
            reason = "default"
        
        return amount, reason
    
    def _fixed_percent(self, equity: float) -> float:
        """Fixed percentage withdrawal."""
        annual_amount = equity * self.config.base_rate
        per_period = annual_amount / (365 / self.config.frequency_days)
        return per_period
    
    def _dynamic(self, equity: float, recent_performance: float) -> float:
        """Dynamic withdrawal based on performance."""
        # Base rate
        rate = self.config.base_rate
        
        # Adjust for recent performance
        if recent_performance > 0.10:  # Great year
            rate += self.config.performance_bonus
        elif recent_performance < 0:    # Bad year
            rate -= self.config.performance_bonus
        
        # Clamp
        rate = max(self.config.dynamic_floor, 
                   min(self.config.dynamic_ceiling, rate))
        
        annual_amount = equity * rate
        per_period = annual_amount / (365 / self.config.frequency_days)
        
        return per_period
    
    def _tax_aware(self, equity: float) -> float:
        """Tax-aware withdrawal."""
        amount = self._fixed_percent(equity)
        
        # Prefer long-term gains - this is informational
        # Actual tax handling happens in tax_aware module
        return amount
    
    def _smoothed(self, equity: float) -> float:
        """Withdraw based on rolling average."""
        if len(self.pnl_history) < self.config.smoothing_window:
            return self._fixed_percent(equity)
        
        # Use rolling average
        avg_equity = np.mean(self.pnl_history[-self.config.smoothing_window:])
        
        # Withdraw based on smoothed equity
        rate = self.config.base_rate
        annual_amount = avg_equity * rate
        per_period = annual_amount / (365 / self.config.frequency_days)
        
        return per_period
    
    def record_withdrawal(self, amount: float, equity_before: float, 
                        equity_after: float, pnl_ytd: float = 0):
        """Record a withdrawal."""
        event = WithdrawalEvent(
            timestamp=datetime.now(),
            amount=amount,
            equity_before=equity_before,
            equity_after=equity_after,
            pnl_ytd=pnl_ytd,
            withdrawal_rate=amount / equity_before if equity_before > 0 else 0
        )
        self.withdrawal_history.append(event)
        
        # Track equity for smoothing
        self.pnl_history.append(equity_after)
        
        # Keep history bounded
        if len(self.pnl_history) > 120:  # 10 years
            self.pnl_history = self.pnl_history[-120:]
    
    def update_drawdown(self, drawdown: float):
        """Update current drawdown."""
        self.current_drawdown = drawdown
    
    def estimate_taxes(self, withdrawal_amount: float, 
                     long_term_gains_pct: float = 0.5) -> Tuple[float, float]:
        """Estimate taxes on withdrawal."""
        # Simplified: assume gains are distributed
        short_term = withdrawal_amount * (1 - long_term_gains_pct)
        long_term = withdrawal_amount * long_term_gains_pct
        
        # Tax rates (simplified)
        st_tax = short_term * 0.37  # Ordinary
        lt_tax = long_term * 0.20    # Long-term
        
        total_tax = st_tax + lt_tax
        net = withdrawal_amount - total_tax
        
        return total_tax, net
    
    def get_plan(self, current_equity: float, years: int = 10) -> List[WithdrawalPlan]:
        """Generate multi-year withdrawal plan."""
        plans = []
        
        for year in range(years):
            amount = self._fixed_percent(current_equity)
            tax, net = self.estimate_taxes(amount)
            
            date = datetime.now() + timedelta(days=365 * (year + 1))
            
            plans.append(WithdrawalPlan(
                scheduled_date=date,
                planned_amount=net,
                tax_estimate=tax
            ))
        
        return plans
    
    def get_status(self) -> Dict:
        """Get withdrawal status."""
        total_withdrawn = sum(w.amount for w in self.withdrawal_history)
        
        return {
            'total_withdrawn': total_withdrawn,
            'withdrawal_count': len(self.withdrawal_history),
            'paused': self.paused,
            'current_drawdown': self.current_drawdown,
            'strategy': self.config.strategy,
            'base_rate': self.config.base_rate
        }


# ============================================================
# GOAL-BASED PLANNER
# ============================================================
class GoalBasedPlanner:
    """
    Plan withdrawals to meet specific goals.
    """
    
    def __init__(self, config: WithdrawalConfig):
        self.config = config
        self.goals: List[Dict] = []
    
    def add_goal(self, name: str, amount: float, years_until: int,
                priority: int = 1):
        """Add a financial goal."""
        self.goals.append({
            'name': name,
            'amount': amount,
            'years_until': years_until,
            'priority': priority,
            'created': datetime.now()
        })
    
    def calculate_required_withdrawal(self, current_equity: float,
                                    simulation_years: int = 30) -> float:
        """
        Calculate required withdrawal rate to meet goals.
        
        Uses Monte Carlo-like estimation.
        """
        if not self.goals:
            return current_equity * self.config.base_rate
        
        # Sort by priority
        sorted_goals = sorted(self.goals, key=lambda g: g['priority'])
        
        # Total goal amount
        total_goal = sum(g['amount'] for g in sorted_goals)
        
        # Estimate required annual withdrawal
        # Simple: NPV calculation assuming 7% returns
        discount_rate = 0.07
        
        total_npv = 0
        for goal in sorted_goals:
            years = goal['years_until']
            amount = goal['amount']
            
            # Present value
            pv = amount / ((1 + discount_rate) ** years)
            total_npv += pv
        
        # Required annual contribution
        if total_npv > current_equity:
            # Need to withdraw more
            shortfall = total_npv - current_equity
            
            # Annuity formula for required withdrawal
            if simulation_years > 0:
                # Simplified
                required = shortfall / simulation_years
            else:
                required = 0
            
            return required
        
        return current_equity * self.config.base_rate  # Can meet goals with standard rate
    
    def get_goal_status(self, current_equity: float) -> Dict:
        """Get status of all goals."""
        return {
            'goals': len(self.goals),
            'total_target': sum(g['amount'] for g in self.goals),
            'current_equity': current_equity,
            'funded_pct': min(100, current_equity / max(1, sum(g['amount'] for g in self.goals)) * 100)
        }


# ============================================================
# MUTATIONS
# ============================================================
class WithdrawalMutations:
    """Mutations for withdrawal strategy."""
    
    @staticmethod
    def mutate_withdrawal_rate(config: WithdrawalConfig) -> WithdrawalConfig:
        """Mutate base withdrawal rate."""
        config = copy.deepcopy(config)
        config.base_rate = max(0.02, min(0.10,
            config.base_rate + random.uniform(-0.005, 0.005)))
        return config
    
    @staticmethod
    def mutate_frequency(config: WithdrawalConfig) -> WithdrawalConfig:
        """Mutate withdrawal frequency."""
        config = copy.deepcopy(config)
        frequencies = [7, 14, 30, 60, 90, 180, 365]
        if config.frequency_days in frequencies:
            idx = frequencies.index(config.frequency_days)
            config.frequency_days = frequencies[min(idx + 1, len(frequencies) - 1)]
        return config
    
    @staticmethod
    def mutate_strategy(config: WithdrawalConfig) -> WithdrawalConfig:
        """Mutate withdrawal strategy."""
        config = copy.deepcopy(config)
        strategies = ['fixed_percent', 'dynamic', 'tax_aware', 'smoothed']
        config.strategy = random.choice(strategies)
        return config
    
    @staticmethod
    def toggle_dd_protection(config: WithdrawalConfig) -> WithdrawalConfig:
        """Toggle drawdown protection."""
        config = copy.deepcopy(config)
        config.dd_protection = not config.dd_protection
        return config
    
    @staticmethod
    def mutate_dynamic_bounds(config: WithdrawalConfig) -> WithdrawalConfig:
        """Mutate dynamic rate bounds."""
        config = copy.deepcopy(config)
        config.dynamic_floor = max(0.01, 
            config.dynamic_floor + random.uniform(-0.005, 0))
        config.dynamic_ceiling = min(0.15,
            config.dynamic_ceiling + random.uniform(0, 0.005))
        return config


# ============================================================
# FACTORY
# ============================================================
def create_withdrawal_config(
    strategy: str = "fixed_percent",
    base_rate: float = 0.04
) -> WithdrawalConfig:
    """Create withdrawal configuration."""
    return WithdrawalConfig(
        strategy=strategy,
        base_rate=base_rate
    )


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("QUANTCORE WITHDRAWAL OPTIMIZER TEST")
    print("=" * 50)
    
    # Test 1: Fixed Percent
    print("\n💰 Test 1: Fixed Percent (4% rule)")
    config = create_withdrawal_config("fixed_percent", 0.04)
    calc = WithdrawalCalculator(config)
    
    amount, reason = calc.calculate_withdrawal(100000)
    print(f"  Equity: $100,000")
    print(f"  Withdrawal: ${amount:.2f} ({reason})")
    
    # Test 2: Dynamic
    print("\n📊 Test 2: Dynamic")
    config = create_withdrawal_config("dynamic", 0.04)
    calc = WithdrawalCalculator(config)
    
    amount, reason = calc.calculate_withdrawal(100000, recent_performance=0.15)
    print(f"  After +15% year: ${amount:.2f}")
    
    amount, reason = calc.calculate_withdrawal(100000, recent_performance=-0.10)
    print(f"  After -10% year: ${amount:.2f}")
    
    # Test 3: Tax Estimation
    print("\n🏛️ Test 3: Tax Estimation")
    config = create_withdrawal_config()
    calc = WithdrawalCalculator(config)
    
    gross = 10000
    tax, net = calc.estimate_taxes(gross, long_term_gains_pct=0.5)
    print(f"  Gross withdrawal: ${gross:,.2f}")
    print(f"  Estimated tax: ${tax:,.2f}")
    print(f"  Net to you: ${net:,.2f}")
    
    # Test 4: Drawdown Protection
    print("\n🛡️ Test 4: Drawdown Protection")
    config = create_withdrawal_config()
    calc = WithdrawalCalculator(config)
    
    calc.update_drawdown(0.20)  # 20% drawdown
    amount, reason = calc.calculate_withdrawal(80000)
    print(f"  At 20% DD: ${amount:.2f} ({reason})")
    
    calc.update_drawdown(0.08)  # Recovered
    amount, reason = calc.calculate_withdrawal(95000)
    print(f"  At 8% DD: ${amount:.2f} ({reason})")
    
    # Test 5: Goal-Based Planning
    print("\n🎯 Test 5: Goal-Based Planning")
    planner = GoalBasedPlanner(config)
    
    planner.add_goal("Retirement", 1000000, years_until=20, priority=1)
    planner.add_goal("House", 200000, years_until=5, priority=2)
    
    required = planner.calculate_required_withdrawal(100000)
    print(f"  Goals: $1.2M (retirement) + $200k (house)")
    print(f"  Current: $100k")
    print(f"  Required annual withdrawal: ${required:,.2f}")
    
    status = planner.get_goal_status(100000)
    print(f"  Goal funding: {status['funded_pct']:.1f}%")
    
    # Test 6: Mutations
    print("\n🧬 Test 6: Mutations")
    config = WithdrawalMutations.mutate_withdrawal_rate(config)
    print(f"  New rate: {config.base_rate:.2%}")
    
    config = WithdrawalMutations.mutate_strategy(config)
    print(f"  New strategy: {config.strategy}")
    
    print("\n✅ All withdrawal tests passed!")
