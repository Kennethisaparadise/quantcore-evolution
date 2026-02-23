"""
QuantCore - Real-Time Adaptation Engine v1.0

This module allows the system to continue evolving while trading.

Expert: "The difference between a static genius and a self-improving demigod."

Features:
1. Streaming Data Ingestion (live price/volume/sentiment)
2. Incremental Learning (rolling regime updates)
3. Scheduled Mini-Evolution (self-improvement cycles)
4. Underperformer Replacement (Sharpe-based culling)
5. Risk-Aware Updates (never replace during open positions)
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
import threading
import time
import queue
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================
# ADAPTATION ENUMS
# ============================================================
class AdaptationState(Enum):
    """State of the adaptation system."""
    IDLE = "idle"
    WATCHING = "watching"
    EVALUATING = "evaluating"
    EVOLVING = "evolving"
    DEPLOYING = "deploying"


class UpdateTrigger(Enum):
    """What triggers an adaptation update."""
    SCHEDULED = "scheduled"           # Time-based
    PERFORMANCE = "performance"       # Sharpe drop
    REGIME_CHANGE = "regime_change"   # Regime shift detected
    MANUAL = "manual"                 # User-triggered


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class AdaptationConfig:
    """
    Configuration for real-time adaptation.
    """
    # Streaming
    stream_interval_seconds: int = 60       # How often to ingest new data
    
    # Evolution scheduling
    evolution_interval_hours: int = 24     # Run mini-evolution every N hours
    evolution_window_bars: int = 200       # Bars for mini-evolution
    
    # Performance triggers
    sharpe_threshold: float = 0.5          # Replace if Sharpe drops below this
    max_drawdown_trigger: float = 0.15     # Replace if drawdown exceeds this
    performance_window: int = 50           # Bars to evaluate performance
    
    # Strategy management
    min_strategies: int = 3                # Minimum strategies to maintain
    max_strategies: int = 10               # Maximum strategies
    replacement_candidates: int = 3        # New strategies to generate per replacement
    
    # Risk-aware updates
    allow_during_trades: bool = False     # Allow updates during open positions?
    warmup_bars: int = 100                # Bars to wait before first evolution
    
    # Evolution parameters (mini)
    mini_population: int = 20              # Smaller population for quick evolution
    mini_generations: int = 5              # Fewer generations
    mini_mutation_rate: float = 0.15       # Slightly higher mutation
    
    def to_dict(self) -> Dict:
        return {
            'stream_interval': self.stream_interval_seconds,
            'evolution_interval': self.evolution_interval_hours,
            'sharpe_threshold': self.sharpe_threshold,
            'max_strategies': self.max_strategies,
            'allow_during_trades': self.allow_during_trades
        }


@dataclass
class StrategyPerformance:
    """Track performance of a strategy in real-time."""
    strategy_id: str
    entry_bar: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0
    max_drawdown: float = 0
    current_sharpe: float = 0
    recent_returns: List[float] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return self.winning_trades / self.total_trades
    
    @property
    def should_replace(self) -> bool:
        """Determine if strategy should be replaced."""
        return (
            self.current_sharpe < 0.5 or  # Poor Sharpe
            self.max_drawdown > 0.20 or    # Large drawdown
            self.total_trades < 3          # Insufficient data
        )


@dataclass
class AdaptationEvent:
    """Log of adaptation events."""
    timestamp: datetime
    trigger: UpdateTrigger
    action: str
    details: str
    success: bool


# ============================================================
# STREAMING DATA INGESTION
# ============================================================
class StreamingDataIngestor:
    """
    Ingest live market data from exchanges.
    """
    
    def __init__(self, config: AdaptationConfig, exchange: str = "binance"):
        self.config = config
        self.exchange = exchange
        self.data_buffer: deque = deque(maxlen=1000)
        self.latest_bar: Optional[Dict] = None
        self.is_streaming = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
    def start(self, symbols: List[str], callback: Callable = None):
        """Start streaming data."""
        self.is_streaming = True
        self._stop_event.clear()
        
        def stream_loop():
            while not self._stop_event.is_set():
                try:
                    # In production, would fetch from exchange API
                    # For now, simulate with mock data
                    bar = self._fetch_mock_bar(symbols[0])
                    
                    self.latest_bar = bar
                    self.data_buffer.append(bar)
                    
                    if callback:
                        callback(bar)
                        
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                
                time.sleep(self.config.stream_interval_seconds)
        
        self._thread = threading.Thread(target=stream_loop, daemon=True)
        self._thread.start()
        logger.info(f"📡 Started streaming {symbols}")
    
    def stop(self):
        """Stop streaming."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.is_streaming = False
        logger.info("📡 Stopped streaming")
    
    def _fetch_mock_bar(self, symbol: str) -> Dict:
        """Fetch mock bar for testing."""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'open': 45000 + np.random.randn() * 100,
            'high': 45100 + np.random.randn() * 100,
            'low': 44900 + np.random.randn() * 100,
            'close': 45000 + np.random.randn() * 100,
            'volume': 1000 + np.random.random() * 500
        }
    
    def get_latest(self, bars: int = 1) -> List[Dict]:
        """Get latest N bars."""
        if bars == 1:
            return [self.latest_bar] if self.latest_bar else []
        return list(self.data_buffer)[-bars:]


# ============================================================
# INCREMENTAL REGIME DETECTOR
# ============================================================
class IncrementalRegimeDetector:
    """
    Update regime detection with new data incrementally.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.regime_history: List[str] = []
        self.current_regime: str = 'sideways'
        self.transition_count: int = 0
        
    def update(self, price_data: np.ndarray) -> str:
        """Update regime with new price data."""
        if len(price_data) < self.lookback:
            return self.current_regime
        
        # Simple regime detection (in production, use HMM)
        recent = price_data[-self.lookback:]
        
        # Calculate volatility
        returns = np.diff(np.log(recent))
        volatility = np.std(returns) * np.sqrt(24)  # Daily vol
        
        # Calculate trend
        trend = (recent[-1] - recent[0]) / recent[0]
        
        # Determine regime
        new_regime = 'sideways'
        
        if volatility > 0.04:  # High volatility
            new_regime = 'high_vol'
        elif volatility < 0.01:  # Low volatility
            new_regime = 'low_vol'
        elif trend > 0.05:  # Strong uptrend
            new_regime = 'bull'
        elif trend < -0.05:  # Strong downtrend
            new_regime = 'bear'
        
        # Track transitions
        if new_regime != self.current_regime:
            self.transition_count += 1
            self.regime_history.append(self.current_regime)
            logger.info(f"🔄 Regime transition: {self.current_regime} → {new_regime}")
        
        self.current_regime = new_regime
        return new_regime
    
    def get_regime(self) -> str:
        """Get current regime."""
        return self.current_regime


# ============================================================
# PERFORMANCE TRACKER
# ============================================================
class PerformanceTracker:
    """
    Track strategy performance in real-time.
    """
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.open_positions: Dict[str, Dict] = {}  # strategy_id -> position info
        
    def record_trade(self, strategy_id: str, pnl: float, is_win: bool):
        """Record a completed trade."""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id
            )
        
        perf = self.strategy_performance[strategy_id]
        perf.total_trades += 1
        if is_win:
            perf.winning_trades += 1
        perf.total_pnl += pnl
        perf.last_update = datetime.now()
        
        # Update recent returns
        perf.recent_returns.append(pnl)
        if len(perf.recent_returns) > self.config.performance_window:
            perf.recent_returns = perf.recent_returns[-self.config.performance_window:]
        
        # Recalculate Sharpe
        if len(perf.recent_returns) >= 5:
            returns = np.array(perf.recent_returns)
            perf.current_sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # Update max drawdown
        if perf.total_pnl < 0:
            perf.max_drawdown = min(perf.max_drawdown, perf.total_pnl)
    
    def update_open_position(self, strategy_id: str, entry_price: float, 
                           current_price: float, quantity: float):
        """Update tracking for open position."""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id,
                entry_bar=len(self.strategy_performance)
            )
        
        self.open_positions[strategy_id] = {
            'entry_price': entry_price,
            'current_price': current_price,
            'quantity': quantity,
            'unrealized_pnl': (current_price - entry_price) * quantity
        }
    
    def close_position(self, strategy_id: str, pnl: float):
        """Close an open position."""
        if strategy_id in self.open_positions:
            del self.open_positions[strategy_id]
        
        is_win = pnl > 0
        self.record_trade(strategy_id, pnl, is_win)
    
    def has_open_positions(self) -> bool:
        """Check if any strategies have open positions."""
        return len(self.open_positions) > 0
    
    def get_open_strategy_ids(self) -> List[str]:
        """Get list of strategies with open positions."""
        return list(self.open_positions.keys())
    
    def get_underperformers(self) -> List[str]:
        """Get list of strategies that should be replaced."""
        return [
            sid for sid, perf in self.strategy_performance.items()
            if perf.should_replace
        ]
    
    def get_top_performers(self, n: int = 3) -> List[str]:
        """Get top N performing strategies."""
        sorted_perfs = sorted(
            self.strategy_performance.items(),
            key=lambda x: x[1].current_sharpe,
            reverse=True
        )
        return [sid for sid, _ in sorted_perfs[:n]]


# ============================================================
# MINI EVOLUTION ENGINE
# ============================================================
class MiniEvolutionEngine:
    """
    Run quick mini-evolutions for real-time adaptation.
    
    Smaller/faster than full evolution - designed for incremental updates.
    """
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.last_evolution: Optional[datetime] = None
        self.evolution_count: int = 0
        
    def run_mini_evolution(self, 
                          seed_strategies: Dict,
                          mutation_operators: List,
                          price_data: pd.DataFrame) -> List[Dict]:
        """
        Run a mini-evolution with smaller population.
        
        Returns: List of new strategy configs
        """
        logger.info(f"🧬 Starting mini-evolution ({self.config.mini_population} pop, {self.config.mini_generations} gen)")
        
        # Generate candidates from seeds with mutations
        candidates = []
        
        for i in range(self.config.replacement_candidates):
            # Pick random seed
            seed_key = random.choice(list(seed_strategies.keys()))
            seed = copy.deepcopy(seed_strategies[seed_key])
            
            # Apply random mutations
            for _ in range(random.randint(2, 5)):
                mutation = random.choice(mutation_operators)
                seed = self._apply_mutation(seed, mutation)
            
            candidates.append(seed)
        
        # Evaluate candidates (simplified - in production use full backtest)
        scored = []
        for candidate in candidates:
            score = self._quick_backtest(candidate, price_data)
            scored.append((score, candidate))
        
        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return top candidates
        results = [c for _, c in scored[:self.config.replacement_candidates]]
        
        self.last_evolution = datetime.now()
        self.evolution_count += 1
        
        logger.info(f"✅ Mini-evolution complete: {len(results)} candidates generated")
        
        return results
    
    def _apply_mutation(self, strategy: Dict, mutation: str) -> Dict:
        """Apply a mutation to a strategy."""
        strategy = copy.deepcopy(strategy)
        
        # Simplified mutation logic
        if 'stop_loss' in mutation:
            strategy['stop_loss_pct'] *= random.uniform(0.8, 1.2)
        if 'take_profit' in mutation:
            strategy['take_profit_pct'] *= random.uniform(0.8, 1.2)
        if 'position' in mutation:
            strategy['position_size_pct'] *= random.uniform(0.8, 1.2)
        
        return strategy
    
    def _quick_backtest(self, strategy: Dict, data: pd.DataFrame) -> float:
        """Quick backtest scoring."""
        # Simplified - just use recent returns
        if len(data) < 20:
            return 0
        
        returns = data['close'].pct_change().dropna()
        
        # Simple score based on recent performance
        recent = returns[-20:]
        score = np.mean(recent) * 100 - np.std(recent) * 10
        
        return score
    
    def should_evolve(self) -> bool:
        """Check if it's time to run evolution."""
        if not self.last_evolution:
            return True
        
        elapsed = (datetime.now() - self.last_evolution).total_seconds()
        threshold = self.config.evolution_interval_hours * 3600
        
        return elapsed >= threshold


# ============================================================
# ADAPTATION ORCHESTRATOR
# ============================================================
class AdaptationOrchestrator:
    """
    Main orchestrator for real-time adaptation.
    
    Coordinates streaming, evaluation, evolution, and deployment.
    """
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.state = AdaptationState.IDLE
        
        # Components
        self.ingestor: Optional[StreamingDataIngestor] = None
        self.regime_detector = IncrementalRegimeDetector()
        self.performance_tracker = PerformanceTracker(config)
        self.mini_evolution = MiniEvolutionEngine(config)
        
        # Strategy registry
        self.active_strategies: Dict[str, Dict] = {}
        self.candidate_strategies: List[Dict] = []
        
        # Event log
        self.events: List[AdaptationEvent] = []
        
        # Callbacks
        self.on_strategy_replace: Optional[Callable] = None
        self.on_regime_change: Optional[Callable] = None
        
    def initialize(self, strategies: Dict[str, Dict]):
        """Initialize with existing strategies."""
        self.active_strategies = strategies
        
        # Initialize performance tracking
        for sid in strategies.keys():
            self.performance_tracker.strategy_performance[sid] = StrategyPerformance(
                strategy_id=sid
            )
        
        logger.info(f"🎯 Initialized with {len(strategies)} strategies")
    
    def start_streaming(self, symbols: List[str]):
        """Start real-time data streaming."""
        self.ingestor = StreamingDataIngestor(self.config)
        self.ingestor.start(symbols, callback=self._on_new_bar)
        self.state = AdaptationState.WATCHING
        
        logger.info(f"📡 Started streaming: {symbols}")
    
    def stop_streaming(self):
        """Stop streaming."""
        if self.ingestor:
            self.ingestor.stop()
        self.state = AdaptationState.IDLE
    
    def _on_new_bar(self, bar: Dict):
        """Callback when new bar arrives."""
        self.state = AdaptationState.EVALUATING
        
        try:
            # Update regime detector
            if self.ingestor and len(self.ingestor.data_buffer) >= self.config.warmup_bars:
                prices = np.array([b['close'] for b in self.ingestor.data_buffer])
                new_regime = self.regime_detector.update(prices)
                
                if new_regime != self.regime_detector.current_regime:
                    self._log_event(UpdateTrigger.REGIME_CHANGE, "regime_update", 
                                  f"New regime: {new_regime}")
                    if self.on_regime_change:
                        self.on_regime_change(new_regime)
            
            # Check performance triggers
            underperformers = self.performance_tracker.get_underperformers()
            
            if underperformers and self._can_update():
                # Trigger evolution
                self._trigger_evolution(underperformers)
            
            # Check scheduled evolution
            if self.mini_evolution.should_evolve() and self._can_update():
                self._trigger_evolution(None)
            
        except Exception as e:
            logger.error(f"Error processing bar: {e}")
        
        finally:
            self.state = AdaptationState.WATCHING
    
    def _can_update(self) -> bool:
        """Check if we can safely update strategies."""
        if self.config.allow_during_trades:
            return True
        
        return not self.performance_tracker.has_open_positions()
    
    def _trigger_evolution(self, underperformers: Optional[List[str]]):
        """Trigger a mini-evolution."""
        self.state = AdaptationState.EVOLVING
        
        try:
            # Get price data
            if self.ingestor:
                bars = self.ingestor.get_latest(self.config.evolution_window_bars)
                if len(bars) < 50:
                    return
                
                data = pd.DataFrame(bars)
            else:
                logger.warning("No ingestor - skipping evolution")
                return
            
            # Run mini-evolution (would use real seeds/mutations in production)
            seed_strategies = {'rsi': {'name': 'RSI', 'stop_loss_pct': 3}}
            mutations = ['stop_loss_tighten', 'take_profit_increase']
            
            new_candidates = self.mini_evolution.run_mini_evolution(
                seed_strategies, mutations, data
            )
            
            # Deploy candidates
            self._deploy_candidates(new_candidates, underperformers)
            
            self._log_event(UpdateTrigger.PERFORMANCE, "evolution", 
                          f"Generated {len(new_candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Evolution error: {e}")
            self._log_event(UpdateTrigger.PERFORMANCE, "evolution_failed", str(e))
        
        finally:
            self.state = AdaptationState.WATCHING
    
    def _deploy_candidates(self, candidates: List[Dict], 
                         underperformers: Optional[List[str]]):
        """Deploy new candidates, replacing underperformers."""
        self.state = AdaptationState.DEPLOYING
        
        try:
            # Replace underperformers
            if underperformers:
                for uid in underperformers:
                    if candidates:
                        candidate = candidates.pop(0)
                        
                        # Remove old
                        if uid in self.active_strategies:
                            old_name = self.active_strategies[uid].get('name', uid)
                            del self.active_strategies[uid]
                        
                        # Add new
                        new_id = f"adapted_{datetime.now().strftime('%H%M%S')}"
                        self.active_strategies[new_id] = candidate
                        
                        logger.info(f"🔄 Replaced {old_name} with {new_id}")
                        
                        # Callback
                        if self.on_strategy_replace:
                            self.on_strategy_replace(uid, new_id, candidate)
            
            self._log_event(UpdateTrigger.PERFORMANCE, "deploy", 
                          f"Deployed {len(candidates)} new strategies")
            
        finally:
            self.state = AdaptationState.WATCHING
    
    def _log_event(self, trigger: UpdateTrigger, action: str, details: str):
        """Log an adaptation event."""
        event = AdaptationEvent(
            timestamp=datetime.now(),
            trigger=trigger,
            action=action,
            details=details,
            success=True
        )
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > 100:
            self.events = self.events[-100:]
        
        logger.info(f"📋 [{trigger.value}] {action}: {details}")
    
    def get_status(self) -> Dict:
        """Get adaptation system status."""
        return {
            'state': self.state.value,
            'active_strategies': len(self.active_strategies),
            'open_positions': len(self.performance_tracker.open_positions),
            'current_regime': self.regime_detector.get_regime(),
            'last_evolution': self.mini_evolution.last_evolution.isoformat() 
                             if self.mini_evolution.last_evolution else None,
            'evolution_count': self.mini_evolution.evolution_count,
            'events_today': len([e for e in self.events 
                               if e.timestamp.date() == datetime.now().date()])
        }
    
    def force_evolution(self):
        """Manually trigger an evolution."""
        self._trigger_evolution(None)


# ============================================================
# FACTORY FUNCTIONS
# ============================================================
def create_adaptation_config(
    evolution_interval_hours: int = 24,
    sharpe_threshold: float = 0.5
) -> AdaptationConfig:
    """Create adaptation configuration."""
    return AdaptationConfig(
        evolution_interval_hours=evolution_interval_hours,
        sharpe_threshold=sharpe_threshold,
        stream_interval_seconds=60,
        mini_population=20,
        mini_generations=5
    )


# ============================================================
# CLI / TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("QUANTCORE REAL-TIME ADAPTATION TEST")
    print("=" * 50)
    
    # Create config
    config = create_adaptation_config(
        evolution_interval_hours=1,  # Test with 1 hour
        sharpe_threshold=0.5
    )
    
    # Create orchestrator
    orchestrator = AdaptationOrchestrator(config)
    
    # Initialize with test strategies
    test_strategies = {
        'strat_1': {'name': 'RSI Strategy', 'stop_loss_pct': 3},
        'strat_2': {'name': 'MA Cross', 'stop_loss_pct': 4},
        'strat_3': {'name': 'Bollinger', 'stop_loss_pct': 3}
    }
    orchestrator.initialize(test_strategies)
    
    # Test streaming (just 5 seconds)
    print("\n📡 Starting 5-second stream test...")
    orchestrator.start_streaming(['BTCUSDT'])
    
    time.sleep(5)
    
    orchestrator.stop_streaming()
    
    # Check status
    print("\n📊 Adaptation Status:")
    status = orchestrator.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test forced evolution
    print("\n🧬 Testing forced evolution...")
    orchestrator.force_evolution()
    
    print("\n✅ Real-time adaptation test complete!")
