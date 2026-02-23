"""
QuantCore - Paper Trading Engine

Simulates live trading with strategies:
- Executes signals from evolved strategies
- Tracks positions, P&L, drawdown
- Risk management (position sizing, stops)
- Performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Order:
    """A trading order."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float = 0.0
    status: str = "pending"  # pending, filled, cancelled
    filled_price: float = 0.0
    filled_time: Optional[datetime] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status,
            'filled_price': self.filled_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


@dataclass
class Position:
    """A trading position."""
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    def update(self, current_price: float):
        """Update position with current price."""
        self.current_price = current_price
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        else:
            self.unrealized_pnl = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


@dataclass
class Trade:
    """A completed trade."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    pnl: float = 0.0
    commission: float = 0.0
    duration_bars: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'pnl': self.pnl,
            'commission': self.commission
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_duration_bars: int = 0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio
        }


# ============================================================
# PAPER TRADING ENGINE
# ============================================================
class PaperTradingEngine:
    """
    Paper trading engine that simulates live trading.
    
    Features:
    - Position management (long/short)
    - Order execution (market/limit/stop)
    - Risk management (stops, position sizing)
    - Performance tracking
    - Equity curve
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,    # 0.05%
        max_position_pct: float = 0.25,  # Max 25% per position
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        
        # State
        self.cash = initial_capital
        self.equity = initial_capital
        self.position: Optional[Position] = None
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        
        # Performance
        self.metrics = PerformanceMetrics()
        self.equity_history = [initial_capital]
        self.entry_bar = 0
        
        # Trade counter
        self.trade_counter = 0
        
    def reset(self):
        """Reset the engine to initial state."""
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.position = None
        self.orders = []
        self.trades = []
        self.metrics = PerformanceMetrics()
        self.equity_history = [self.initial_capital]
        self.trade_counter = 0
    
    # ==================== ORDER MANAGEMENT ====================
    
    def can_open_position(self, price: float, quantity: float) -> bool:
        """Check if we can open a position."""
        cost = price * quantity * (1 + self.commission + self.slippage)
        
        # Check cash
        if cost > self.cash:
            return False
        
        # Check max position size
        position_value = price * quantity
        if position_value > self.equity * self.max_position_pct:
            return False
        
        return True
    
    def open_long(
        self,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0
    ) -> Optional[Order]:
        """Open a long position."""
        if self.position is not None:
            logger.warning(f"Position already open: {self.position.side}")
            return None
        
        if not self.can_open_position(price, quantity):
            logger.warning(f"Insufficient capital for long")
            return None
        
        # Calculate actual cost with slippage/commission
        fill_price = price * (1 + self.slippage)
        cost = fill_price * quantity * (1 + self.commission)
        
        self.cash -= cost
        
        # Create position
        self.position = Position(
            id=f"pos_{self.trade_counter}",
            symbol=symbol,
            side=PositionSide.LONG,
            entry_price=fill_price,
            quantity=quantity,
            entry_time=datetime.now(),
            current_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.entry_bar = len(self.equity_history)
        
        # Create order
        order = Order(
            id=f"ord_{len(self.orders)}",
            timestamp=datetime.now(),
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            status="filled",
            filled_price=fill_price,
            filled_time=datetime.now()
        )
        
        self.orders.append(order)
        
        return order
    
    def open_short(
        self,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0
    ) -> Optional[Order]:
        """Open a short position."""
        if self.position is not None:
            logger.warning(f"Position already open: {self.position.side}")
            return None
        
        # For short, we "borrow" and sell high
        fill_price = price * (1 - self.slippage)
        proceeds = fill_price * quantity * (1 - self.commission)
        
        self.cash += proceeds
        
        # Create position (value based on entry, pnl calculated as diff)
        self.position = Position(
            id=f"pos_{self.trade_counter}",
            symbol=symbol,
            side=PositionSide.SHORT,
            entry_price=fill_price,
            quantity=quantity,
            entry_time=datetime.now(),
            current_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.entry_bar = len(self.equity_history)
        
        order = Order(
            id=f"ord_{len(self.orders)}",
            timestamp=datetime.now(),
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            status="filled",
            filled_price=fill_price,
            filled_time=datetime.now()
        )
        
        self.orders.append(order)
        
        return order
    
    def close_position(self, price: float, reason: str = "signal") -> Optional[Trade]:
        """Close current position."""
        if self.position is None:
            return None
        
        self.trade_counter += 1
        
        # Calculate fill price
        if self.position.side == PositionSide.LONG:
            fill_price = price * (1 - self.slippage)
            pnl = (fill_price - self.position.entry_price) * self.position.quantity
            commission = fill_price * self.position.quantity * self.commission
        else:
            fill_price = price * (1 + self.slippage)
            pnl = (self.position.entry_price - fill_price) * self.position.quantity
            commission = fill_price * self.position.quantity * self.commission
        
        pnl -= commission
        
        # Update cash
        if self.position.side == PositionSide.LONG:
            proceeds = fill_price * self.position.quantity * (1 - self.commission)
            self.cash += proceeds
        else:
            # Short: we return the borrowed, keep the rest
            self.cash -= self.position.entry_price * self.position.quantity
        
        # Calculate duration
        duration = len(self.equity_history) - self.entry_bar
        
        # Create trade
        trade = Trade(
            id=f"trade_{self.trade_counter}",
            timestamp=datetime.now(),
            symbol=self.position.symbol,
            side=OrderSide.SELL if self.position.side == PositionSide.LONG else OrderSide.BUY,
            quantity=self.position.quantity,
            price=fill_price,
            pnl=pnl,
            commission=commission,
            duration_bars=duration
        )
        
        self.trades.append(trade)
        
        # Update metrics
        self._update_metrics(trade)
        
        # Clear position
        self.position = None
        
        return trade
    
    def _update_metrics(self, trade: Trade):
        """Update performance metrics."""
        self.metrics.total_trades += 1
        
        if trade.pnl > 0:
            self.metrics.winning_trades += 1
            self.metrics.gross_profit += trade.pnl
        else:
            self.metrics.losing_trades += 1
            self.metrics.gross_loss += abs(trade.pnl)
        
        self.metrics.total_pnl += trade.pnl
        
        # Win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        # Avg win/loss
        if self.metrics.winning_trades > 0:
            self.metrics.avg_win = self.metrics.gross_profit / self.metrics.winning_trades
        if self.metrics.losing_trades > 0:
            self.metrics.avg_loss = self.metrics.gross_loss / self.metrics.losing_trades
        
        # Max drawdown
        peak = max(self.equity_history)
        if peak > 0:
            dd = (peak - self.equity) / peak
            if dd > self.metrics.max_drawdown_pct:
                self.metrics.max_drawdown_pct = dd
                self.metrics.max_drawdown = peak - self.equity
    
    # ==================== MARKET UPDATE ====================
    
    def update(self, symbol: str, current_price: float, bar_num: int = 0) -> Dict:
        """
        Update engine with current market price.
        
        Returns:
            Dict with signals: { 'action': 'hold' | 'buy' | 'sell', 'reason': str }
        """
        signal = {'action': 'hold', 'reason': 'no_position'}
        
        # Update position if exists
        if self.position is not None:
            self.position.update(current_price)
            
            # Check stops
            if self.position.side == PositionSide.LONG:
                if self.position.stop_loss > 0 and current_price <= self.position.stop_loss:
                    self.close_position(current_price, "stop_loss")
                    signal = {'action': 'sell', 'reason': 'stop_loss'}
                elif self.position.take_profit > 0 and current_price >= self.position.take_profit:
                    self.close_position(current_price, "take_profit")
                    signal = {'action': 'sell', 'reason': 'take_profit'}
            elif self.position.side == PositionSide.SHORT:
                if self.position.stop_loss > 0 and current_price >= self.position.stop_loss:
                    self.close_position(current_price, "stop_loss")
                    signal = {'action': 'buy', 'reason': 'stop_loss'}
                elif self.position.take_profit > 0 and current_price <= self.position.take_profit:
                    self.close_position(current_price, "take_profit")
                    signal = {'action': 'buy', 'reason': 'take_profit'}
        
        # Update equity
        if self.position is not None:
            self.equity = self.cash + self.position.unrealized_pnl
        else:
            self.equity = self.cash
        
        self.equity_history.append(self.equity)
        
        # Calculate Sharpe (if we have enough data)
        if len(self.equity_history) > 30:
            returns = np.diff(self.equity_history) / self.equity_history[:-1]
            if returns.std() > 0:
                self.metrics.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        return signal
    
    # ==================== SIGNALS ====================
    
    def process_signal(
        self,
        signal: str,  # 'buy', 'sell', 'hold'
        price: float,
        params: Dict = None
    ) -> Optional[Trade]:
        """
        Process a trading signal.
        
        Args:
            signal: 'buy', 'sell', 'hold'
            price: Current market price
            params: Strategy parameters (position_size, stop_loss, etc.)
        
        Returns:
            Trade if position was closed, None otherwise
        """
        params = params or {}
        
        trade = None
        
        if signal == 'buy' and self.position is None:
            # Open long
            position_size = params.get('position_size', 0.1)
            quantity = (self.equity * position_size) / price
            
            stop_loss = params.get('stop_loss', 0.0)
            if stop_loss > 0:
                stop_loss = price * (1 - stop_loss)
            
            take_profit = params.get('take_profit', 0.0)
            if take_profit > 0:
                take_profit = price * (1 + take_profit)
            
            self.open_long("BTCUSDT", quantity, price, stop_loss, take_profit)
            
        elif signal == 'sell' and self.position is not None:
            # Close position
            trade = self.close_position(price, "signal")
        
        return trade
    
    # ==================== UTILITY ====================
    
    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            'cash': self.cash,
            'equity': self.equity,
            'position': self.position.to_dict() if self.position else None,
            'open_position': self.position is not None,
            'total_pnl': self.metrics.total_pnl,
            'open_pnl': self.position.unrealized_pnl if self.position else 0,
            'win_rate': self.metrics.win_rate,
            'total_trades': self.metrics.total_trades
        }
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self.metrics
    
    def to_dict(self) -> Dict:
        """Serialize engine state."""
        return {
            'initial_capital': self.initial_capital,
            'current_equity': self.equity,
            'cash': self.cash,
            'position': self.position.to_dict() if self.position else None,
            'metrics': self.metrics.to_dict(),
            'trades': [t.to_dict() for t in self.trades],
            'equity_curve': self.equity_history
        }
    
    def save(self, filepath: str):
        """Save engine state to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# ============================================================
# BACKTEST WRAPPER
# ============================================================
class Backtester:
    """Backtest a strategy using the paper trading engine."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.engine = PaperTradingEngine(initial_capital)
    
    def run(
        self,
        data: pd.DataFrame,
        params: Dict,
        strategy_fn=None
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV columns
            params: Strategy parameters
            strategy_fn: Function that generates signals
        
        Returns:
            Backtest results
        """
        self.engine.reset()
        
        # Default RSI strategy
        if strategy_fn is None:
            strategy_fn = self._rsi_strategy
        
        # Run through data
        for i, (idx, row) in enumerate(data.iterrows()):
            # Get current price
            current_price = row['close']
            
            # Update engine (check stops)
            self.engine.update("BTCUSDT", current_price, i)
            
            # Get signal from strategy
            signal = strategy_fn(data, i, params)
            
            # Process signal
            self.engine.process_signal(signal, current_price, params)
        
        # Close any open position
        if self.engine.position is not None:
            final_price = data['close'].iloc[-1]
            self.engine.close_position(final_price, "end_of_data")
        
        # Return results
        return self.engine.to_dict()
    
    def _rsi_strategy(
        self,
        data: pd.DataFrame,
        i: int,
        params: Dict
    ) -> str:
        """Default RSI strategy."""
        if i < 14:
            return 'hold'
        
        # Calculate RSI
        period = int(params.get('rsi_period', 14))
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        
        if i < period:
            return 'hold'
        
        # RSI calculation
        close = data['close'].values
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains[i-period:i])
        avg_loss = np.mean(losses[i-period:i])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Signal
        invert = params.get('invert', False)
        
        if invert:
            if rsi > overbought:
                return 'buy'
            elif rsi < oversold:
                return 'sell'
        else:
            if rsi < oversold:
                return 'buy'
            elif rsi > overbought:
                return 'sell'
        
        return 'hold'


# ============================================================
# DEMO
# ============================================================
def demo():
    """Demo the paper trading engine."""
    print("=" * 50)
    print("Paper Trading Engine Demo")
    print("=" * 50)
    
    # Load data
    data = pd.read_csv('data/csv/BTC-USD_5y.csv', parse_dates=['date'], index_col='date')
    data = data.tail(500)
    
    print(f"\nLoaded {len(data)} candles")
    
    # Create backtester
    backtester = Backtester(initial_capital=10000)
    
    # Strategy params
    params = {
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
        'position_size': 0.1,
        'stop_loss': 0.05,
        'take_profit': 0.10,
        'invert': False
    }
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtester.run(data, params)
    
    # Print results
    metrics = results['metrics']
    
    print(f"\n{'='*50}")
    print("BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Equity: ${results['current_equity']:,.2f}")
    print(f"Total Return: {((results['current_equity']/results['initial_capital'])-1)*100:.2f}%")
    print(f"\nTrades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print("\n✅ Paper trading engine working!")


if __name__ == "__main__":
    demo()
