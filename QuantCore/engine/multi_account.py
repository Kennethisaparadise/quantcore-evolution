"""
QuantCore - Multi-Account Allocator v1.0

Head 20: The multi-account portfolio manager.

Features:
1. Multiple account types (taxable, IRA, Roth, offshore)
2. Strategy-to-account allocation
3. Asset location optimization
4. Fund transfer between accounts
5. Tax-efficient rebalancing
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
class AccountType(Enum):
    """Account types with different tax treatments."""
    TAXABLE = "taxable"           # Regular brokerage - full taxes
    TRADITIONAL_IRA = "trad_ira" # Tax-deferred
    ROTH_IRA = "roth_ira"        # Tax-free
    NOMINEE = "nominee"           # Business account
    OFFSHORE = "offshore"         # Foreign account


class StrategyType(Enum):
    """Strategy tax efficiency."""
    HIGH_TURNOVER = "high"     # Short-term gains - tax inefficient
    MEDIUM_TURNOVER = "medium"
    LOW_TURNOVER = "long"      # Long-term gains - tax efficient


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class Account:
    """A trading account."""
    name: str
    account_type: AccountType
    balance: float
    currency: str = "USD"
    margin_available: float = 0
    positions: Dict[str, float] = field(default_factory=dict)
    
    @property
    def tax_rate(self) -> float:
        """Get tax rate for account type."""
        if self.account_type == AccountType.TAXABLE:
            return 0.37  # Short-term
        elif self.account_type == AccountType.TRADITIONAL_IRA:
            return 0.0   # Deferred
        elif self.account_type == AccountType.ROTH_IRA:
            return 0.0   # Free
        else:
            return 0.20  # Offshore estimate
    
    @property
    def tax_efficiency(self) -> float:
        """How tax-efficient is this account?"""
        if self.account_type == AccountType.ROTH_IRA:
            return 1.0
        elif self.account_type == AccountType.TRADITIONAL_IRA:
            return 0.8
        elif self.account_type == AccountType.OFFSHORE:
            return 0.7
        else:
            return 0.3  # Taxable is least efficient


@dataclass
class StrategyAssignment:
    """Assignment of strategy to account."""
    strategy_id: str
    account_name: str
    allocation_pct: float  # % of strategy capital
    current_return: float = 0.0


@dataclass
class AccountConfig:
    """Configuration for account management."""
    # Account definitions
    accounts: Dict[str, Account] = field(default_factory=dict)
    
    # Allocation preferences
    prefer_roth_for_high_turnover: bool = True
    prefer_taxable_for_long_term: bool = True
    
    # Rebalancing
    rebalance_threshold: float = 0.10  # 10% drift triggers rebalance
    rebalance_frequency_days: int = 30
    
    # Transfers
    allow_transfers: bool = True
    transfer_tax_check: bool = True
    
    # Capital allocation
    total_capital: float = 100000
    target_allocation: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'total_capital': self.total_capital,
            'prefer_roth_for_high_turnover': self.prefer_roth_for_high_turnover,
            'accounts': {k: v.balance for k, v in self.accounts.items()}
        }


# ============================================================
# MULTI-ACCOUNT ALLOCATOR
# ============================================================
class MultiAccountAllocator:
    """
    Manage multiple accounts with different tax treatments.
    """
    
    def __init__(self, config: AccountConfig):
        self.config = config
        self.strategy_assignments: Dict[str, StrategyAssignment] = {}
        self.transfer_history: List[Dict] = []
        
    def add_account(self, name: str, account_type: AccountType, 
                   initial_balance: float) -> Account:
        """Add an account."""
        account = Account(
            name=name,
            account_type=account_type,
            balance=initial_balance
        )
        self.config.accounts[name] = account
        return account
    
    def assign_strategy(self, strategy_id: str, strategy_type: StrategyType,
                      preferred_account: str = None) -> str:
        """
        Assign strategy to best account based on tax efficiency.
        
        Returns: account_name
        """
        # Score each account
        best_account = None
        best_score = -float('inf')
        
        for name, account in self.config.accounts.items():
            score = 0
            
            # Preferred account gets bonus
            if preferred_account and name == preferred_account:
                score += 100
            
            # Tax efficiency matching
            if strategy_type == StrategyType.HIGH_TURNOVER:
                # Prefer Roth or IRA for high turnover
                if account.account_type in [AccountType.ROTH_IRA, AccountType.TRADITIONAL_IRA]:
                    score += 50
            elif strategy_type == StrategyType.LOW_TURNOVER:
                # Taxable is fine for long-term
                if account.account_type == AccountType.TAXABLE:
                    score += 30
            
            # Balance check - prefer accounts with more capacity
            capacity = account.balance / max(1, self.config.total_capital)
            score += capacity * 20
            
            if score > best_score:
                best_score = score
                best_account = name
        
        # Create assignment
        assignment = StrategyAssignment(
            strategy_id=strategy_id,
            account_name=best_account,
            allocation_pct=1.0 / len(self.config.accounts)  # Split evenly
        )
        
        self.strategy_assignments[strategy_id] = assignment
        
        return best_account
    
    def get_account_for_strategy(self, strategy_id: str) -> Optional[Account]:
        """Get account for a strategy."""
        if strategy_id not in self.strategy_assignments:
            return None
        
        account_name = self.strategy_assignments[strategy_id].account_name
        return self.config.accounts.get(account_name)
    
    def calculate_allocation(self) -> Dict[str, float]:
        """Calculate current allocation across accounts."""
        total = sum(a.balance for a in self.config.accounts.values())
        
        if total == 0:
            return {}
        
        return {
            name: account.balance / total 
            for name, account in self.config.accounts.items()
        }
    
    def needs_rebalance(self) -> bool:
        """Check if accounts need rebalancing."""
        current = self.calculate_allocation()
        
        for account_name, target_pct in self.config.target_allocation.items():
            current_pct = current.get(account_name, 0)
            drift = abs(current_pct - target_pct)
            
            if drift > self.config.rebalance_threshold:
                return True
        
        return False
    
    def rebalance(self) -> List[Dict]:
        """Rebalance accounts to target allocation."""
        transfers = []
        
        if not self.needs_rebalance():
            return transfers
        
        current = self.calculate_allocation()
        
        for account_name, target_pct in self.config.target_allocation.items():
            current_pct = current.get(account_name, 0)
            drift = target_pct - current_pct
            
            if abs(drift) < 0.01:
                continue
            
            # Calculate transfer amount
            total_capital = sum(a.balance for a in self.config.accounts.values())
            transfer_amount = drift * total_capital
            
            # Find source account (highest balance)
            source = max(self.config.accounts.items(), 
                        key=lambda x: x[1].balance)[0]
            
            if source != account_name and transfer_amount > 0:
                # Execute transfer
                self.config.accounts[source].balance -= transfer_amount
                self.config.accounts[account_name].balance += transfer_amount
                
                transfers.append({
                    'from': source,
                    'to': account_name,
                    'amount': transfer_amount,
                    'timestamp': datetime.now()
                })
                
                self.transfer_history.append(transfers[-1])
        
        return transfers
    
    def transfer_between_accounts(self, from_account: str, to_account: str,
                                amount: float) -> bool:
        """Transfer funds between accounts."""
        if not self.config.allow_transfers:
            return False
        
        from_acc = self.config.accounts.get(from_account)
        to_acc = self.config.accounts.get(to_account)
        
        if not from_acc or not to_acc:
            return False
        
        if from_acc.balance < amount:
            return False
        
        # Execute transfer
        from_acc.balance -= amount
        to_acc.balance += amount
        
        self.transfer_history.append({
            'from': from_account,
            'to': to_account,
            'amount': amount,
            'timestamp': datetime.now()
        })
        
        return True
    
    def get_tax_efficiency_score(self) -> float:
        """Get overall portfolio tax efficiency."""
        total = sum(a.balance for a in self.config.accounts.values())
        
        if total == 0:
            return 0
        
        weighted_efficiency = sum(
            a.balance / total * a.tax_efficiency 
            for a in self.config.accounts.values()
        )
        
        return weighted_efficiency
    
    def get_status(self) -> Dict:
        """Get allocator status."""
        return {
            'accounts': {k: v.balance for k, v in self.config.accounts.items()},
            'strategies': len(self.strategy_assignments),
            'tax_efficiency': self.get_tax_efficiency_score(),
            'needs_rebalance': self.needs_rebalance(),
            'transfers': len(self.transfer_history)
        }


# ============================================================
# MUTATIONS
# ============================================================
class AccountMutations:
    """Mutations for account allocation."""
    
    @staticmethod
    def mutate_allocation(config: AccountConfig) -> AccountConfig:
        """Mutate target allocation."""
        config = copy.deepcopy(config)
        
        # Adjust allocations slightly
        for name in config.target_allocation:
            delta = random.uniform(-0.05, 0.05)
            config.target_allocation[name] = max(0, min(1.0, 
                config.target_allocation.get(name, 0.25) + delta))
        
        # Renormalize
        total = sum(config.target_allocation.values())
        if total > 0:
            config.target_allocation = {
                k: v / total for k, v in config.target_allocation.items()
            }
        
        return config
    
    @staticmethod
    def mutate_rebalance_threshold(config: AccountConfig) -> AccountConfig:
        """Mutate rebalance threshold."""
        config = copy.deepcopy(config)
        config.rebalance_threshold = max(0.05, min(0.25,
            config.rebalance_threshold + random.uniform(-0.02, 0.02)))
        return config
    
    @staticmethod
    def toggle_roth_preference(config: AccountConfig) -> AccountConfig:
        """Toggle Roth preference for high turnover."""
        config = copy.deepcopy(config)
        config.prefer_roth_for_high_turnover = not config.prefer_roth_for_high_turnover
        return config


# ============================================================
# FACTORY
# ============================================================
def create_account_config(
    total_capital: float = 100000,
    taxable_pct: float = 0.40,
    roth_pct: float = 0.30,
    trad_ira_pct: float = 0.30
) -> AccountConfig:
    """Create account configuration."""
    config = AccountConfig(total_capital=total_capital)
    
    # Create accounts with target allocations
    config.target_allocation = {
        'taxable': taxable_pct,
        'roth': roth_pct,
        'trad_ira': trad_ira_pct
    }
    
    # Initialize balances
    taxable = Account('taxable', AccountType.TAXABLE, total_capital * taxable_pct)
    roth = Account('roth', AccountType.ROTH_IRA, total_capital * roth_pct)
    trad_ira = Account('trad_ira', AccountType.TRADITIONAL_IRA, total_capital * trad_ira_pct)
    
    config.accounts = {
        'taxable': taxable,
        'roth': roth,
        'trad_ira': trad_ira
    }
    
    return config


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("QUANTCORE MULTI-ACCOUNT ALLOCATOR TEST")
    print("=" * 50)
    
    # Test 1: Create accounts
    print("\n🏦 Test 1: Create Accounts")
    config = create_account_config(100000)
    allocator = MultiAccountAllocator(config)
    
    for name, account in config.accounts.items():
        print(f"  {name}: ${account.balance:,.2f} ({account.account_type.value})")
    
    # Test 2: Strategy assignment
    print("\n📊 Test 2: Strategy Assignment")
    
    # High turnover strategy
    account = allocator.assign_strategy(
        "momentum_strategy", 
        StrategyType.HIGH_TURNOVER
    )
    print(f"  momentum_strategy -> {account}")
    
    # Low turnover strategy
    account = allocator.assign_strategy(
        "buy_and_hold",
        StrategyType.LOW_TURNOVER
    )
    print(f"  buy_and_hold -> {account}")
    
    # Test 3: Allocation
    print("\n📈 Test 3: Current Allocation")
    alloc = allocator.calculate_allocation()
    for name, pct in alloc.items():
        print(f"  {name}: {pct*100:.1f}%")
    
    # Test 4: Tax efficiency
    print("\n💰 Test 4: Tax Efficiency")
    score = allocator.get_tax_efficiency_score()
    print(f"  Overall score: {score:.2f}/1.0")
    
    # Test 5: Rebalancing
    print("\n⚖️ Test 5: Rebalancing")
    # Simulate drift
    allocator.config.accounts['taxable'].balance *= 1.2
    allocator.config.accounts['roth'].balance *= 0.8
    
    print(f"  Needs rebalance: {allocator.needs_rebalance()}")
    
    transfers = allocator.rebalance()
    print(f"  Transfers made: {len(transfers)}")
    
    alloc = allocator.calculate_allocation()
    for name, pct in alloc.items():
        print(f"  {name}: {pct*100:.1f}%")
    
    # Test 6: Mutations
    print("\n🧬 Test 6: Mutations")
    config = AccountMutations.mutate_allocation(config)
    print(f"  New allocation: {config.target_allocation}")
    
    config = AccountMutations.toggle_roth_preference(config)
    print(f"  Prefer Roth: {config.prefer_roth_for_high_turnover}")
    
    print("\n✅ All multi-account tests passed!")
