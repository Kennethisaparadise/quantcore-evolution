"""
QuantCore - Strategy Engine Base

Abstract base classes and core components for strategy development.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import logging
import inspect
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"
    HOLD = "hold"
    CLOSE = "close"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class PositionSizeType(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    KELLY = "kelly"
    FIXED_FRACTION = "fixed_fraction"
    VOLATILITY = "volatility"
    MAX_RISK = "max_risk"


@dataclass
class Signal:
    """Trading signal representation."""
    symbol: str
    signal_type: SignalType
    strength: float = 1.0
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Signal strength must be between 0.0 and 1.0")


@dataclass
class Order:
    """Order representation."""
    symbol: str
    order_type: OrderType
    side: str
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_percent: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    signal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        return self.quantity * (self.price or 0)


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float = 0.0
    entry_date: datetime = field(default_factory=datetime.now)
    side: str = "long"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0
        return self.unrealized_pnl / (self.entry_price * self.quantity) * 100


@dataclass
class StrategyParams:
    """Strategy parameters container."""
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __getattr__(self, name: str) -> Any:
        return self.params.get(name)
    
    def __setattr__(self, name: str, value: Any):
        if name == 'params':
            super().__setattr__(name, value)
        else:
            self.params[name] = value


class StrategyBase(ABC):
    """
    Abstract base class for all strategies.
    
    Strategies must implement:
    - init(): Setup indicators and data
    - next(): Called on each bar with logic
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        cash: float = 100000,
        commission: float = 0.001,
        **kwargs
    ):
        self.data = data.copy()
        self.original_data = data.copy()
        self.params = StrategyParams(params or {})
        self.cash = cash
        self.commission = commission
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.signals: List[Signal] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [cash]
        self.current_bar = 0
        self.current_date = None
        self.current_price = {}
        self.indicators: Dict[str, pd.Series] = {}
        self.logger = logging.getLogger(f"Strategy.{self.__class__.__name__}")
        
        self._validate_data()
        self._setup()
    
    def _validate_data(self):
        """Validate input data has required columns."""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'timestamp' in self.data.columns:
                self.data.set_index('timestamp', inplace=True)
        
        self.data = self.data.sort_index()
        self.data = self.data[~self.data.index.duplicated(keep='first')]
    
    def _setup(self):
        """Initialize strategy."""
        self.init()
    
    @abstractmethod
    def init(self):
        """Initialize indicators and data."""
        pass
    
    @abstractmethod
    def next(self):
        """Main strategy logic for each bar."""
        pass
    
    def on_buy_signal(self, symbol: str, price: float, strength: float = 1.0):
        self.logger.info(f"BUY signal for {symbol} at {price:.2f}")
    
    def on_sell_signal(self, symbol: str, price: float, strength: float = 1.0):
        self.logger.info(f"SELL signal for {symbol} at {price:.2f}")
    
    def indicator(self, name: str, **kwargs) -> pd.Series:
        """Create and cache an indicator."""
        cache_key = f"{name}_{'_'.join(f'{k}{v}' for k, v in sorted(kwargs.items()))}"
        
        if cache_key in self.indicators:
            return self.indicators[cache_key]
        
        indicator = self._calculate_indicator(name, **kwargs)
        self.indicators[cache_key] = indicator
        return indicator
    
    def _calculate_indicator(self, name: str, **kwargs) -> pd.Series:
        """Calculate an indicator based on name.
        
        Supported indicators:
        - sma: Simple Moving Average
        - ema: Exponential Moving Average
        - rsi: Relative Strength Index
        - macd: Moving Average Convergence Divergence
        - bollinger: Bollinger Bands
        - atr: Average True Range
        - vwap: Volume Weighted Average Price
        - momentum: Momentum
        - returns: Returns
        - stochastic: Stochastic Oscillator
        - adx: Average Directional Index
        - obv: On Balance Volume
        """
        close = self.data['close']
        
        if name == 'sma':
            period = kwargs.get('period', 20)
            return pd.Series(
                close.rolling(period).mean(), 
                name=f"sma_{period}"
            ).dropna()
        
        elif name == 'ema':
            period = kwargs.get('period', 20)
            return pd.Series(
                close.ewm(span=period, adjust=False).mean(), 
                name=f"ema_{period}"
            )
        
        elif name == 'rsi':
            period = kwargs.get('period', 14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = pd.Series(100 - (100 / (1 + rs)), name=f"rsi_{period}")
            return rsi.dropna()
        
        elif name == 'macd':
            fast = kwargs.get('fast', 12)
            slow = kwargs.get('slow', 26)
            signal_period = kwargs.get('signal', 9)
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            macd_df = pd.DataFrame({
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            })
            return macd_df['histogram'].rename('macd_hist')
        
        elif name == 'bollinger':
            period = kwargs.get('period', 20)
            std = close.rolling(period).std()
            sma = close.rolling(period).mean()
            bb_upper = sma + (std * 2)
            bb_lower = sma - (std * 2)
            
            bb_df = pd.DataFrame({
                'upper': bb_upper,
                'middle': sma,
                'lower': bb_lower
            })
            return bb_df['upper'].rename('bb_upper')
        
        elif name == 'atr':
            period = kwargs.get('period', 14)
            high = self.data['high']
            low = self.data['low']
            prev_close = self.data['close'].shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            atr = pd.Series(tr.rolling(period).mean(), name=f"atr_{period}")
            return atr.dropna()
        
        elif name == 'vwap':
            typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
            cum_vol = self.data['volume'].cumsum()
            cum_tp_vol = (typical_price * self.data['volume']).cumsum()
            vwap = pd.Series(cum_tp_vol / cum_vol, name="vwap")
            return vwap
        
        elif name == 'momentum':
            period = kwargs.get('period', 10)
            mom = pd.Series(close / close.shift(period) - 1, name=f"mom_{period}")
            return mom.dropna()
        
        elif name == 'returns':
            return close.pct_change().rename('returns')
        
        elif name == 'stochastic':
            period = kwargs.get('period', 14)
            smooth_k = kwargs.get('smooth_k', 3)
            smooth_d = kwargs.get('smooth_d', 3)
            
            lowest_low = self.data['low'].rolling(period).min()
            highest_high = self.data['high'].rolling(period).max()
            
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_d = stoch_k.rolling(smooth_d).mean()
            
            stoch_df = pd.DataFrame({
                'k': stoch_k,
                'd': stoch_d
            })
            return stoch_df['k'].rename(f"stoch_k_{period}")
        
        elif name == 'adx':
            period = kwargs.get('period', 14)
            
            # +DM and -DM
            high = self.data['high']
            low = self.data['low']
            prev_close = self.data['close'].shift(1)
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # True Range
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            
            # Smoothed
            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            adx_series = pd.Series(adx, name=f"adx_{period}")
            return adx_series.dropna()
        
        elif name == 'obv':
            close = self.data['close']
            volume = self.data['volume']
            
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv.rename('obv')
        
        else:
            raise ValueError(f"Unknown indicator: {name}. Supported: sma, ema, rsi, macd, bollinger, atr, vwap, momentum, returns, stochastic, adx, obv")
    
    def crossover(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Check if series1 crossed above series2."""
        if len(series1) < 2 or len(series2) < 2:
            return False
        return (series1.iloc[-2] < series2.iloc[-2] and series1.iloc[-1] > series2.iloc[-1])
    
    def crossunder(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Check if series1 crossed below series2."""
        if len(series1) < 2 or len(series2) < 2:
            return False
        return (series1.iloc[-2] > series2.iloc[-2] and series1.iloc[-1] < series2.iloc[-1])
    
    def current_bar_data(self) -> Dict[str, Any]:
        """Get current bar data."""
        if self.current_bar >= len(self.data):
            return {}
        return {
            'timestamp': self.data.index[self.current_bar],
            'open': self.data['open'].iloc[self.current_bar],
            'high': self.data['high'].iloc[self.current_bar],
            'low': self.data['low'].iloc[self.current_bar],
            'close': self.data['close'].iloc[self.current_bar],
            'volume': self.data['volume'].iloc[self.current_bar]
        }
    
    def current_close(self, symbol: str = None) -> float:
        """Get current close price."""
        if self.current_bar >= len(self.data):
            return 0
        if symbol:
            return self.current_price.get(symbol, self.data['close'].iloc[self.current_bar])
        return self.data['close'].iloc[self.current_bar]
    
    def position_size(
        self,
        symbol: str,
        method: PositionSizeType = PositionSizeType.FIXED_FRACTION,
        risk_per_trade: float = 0.02,
        volatility_period: int = 14
    ) -> int:
        """Calculate position size."""
        if symbol in self.positions:
            return 0
        
        if method == PositionSizeType.FIXED:
            return int(self.cash * 0.1 / self.current_close(symbol))
        
        elif method == PositionSizeType.FIXED_FRACTION:
            risk_amount = self.cash * risk_per_trade
            atr = self.indicator('atr', period=volatility_period).iloc[-1]
            if atr == 0:
                atr = self.current_close(symbol) * 0.02
            return int(risk_amount / atr)
        
        elif method == PositionSizeType.VOLATILITY:
            returns = self.indicator('returns')
            vol = returns.rolling(volatility_period).std() * np.sqrt(252)
            target_vol = 0.20
            position_size = target_vol / (vol.iloc[-1] if vol.iloc[-1] > 0 else 1)
            return int(position_size * self.cash / self.current_close(symbol))
        
        return int(self.cash * 0.1 / self.current_close(symbol))
    
    def buy(
        self,
        symbol: str = None,
        quantity: int = None,
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        size_method: PositionSizeType = PositionSizeType.FIXED_FRACTION
    ) -> Order:
        """Place a buy order."""
        symbol = symbol or self.get_symbol()
        price = price or self.current_close(symbol)
        quantity = quantity or self.position_size(symbol, method=size_method)
        
        if quantity <= 0:
            self.logger.warning(f"Invalid position size for {symbol}")
            return None
        
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side='buy',
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.orders.append(order)
        self.on_buy_signal(symbol, price)
        return order
    
    def sell(self, symbol: str = None, quantity: int = None, price: float = None) -> Order:
        """Place a sell order."""
        symbol = symbol or self.get_symbol()
        price = price or self.current_close(symbol)
        
        if symbol not in self.positions:
            self.logger.warning(f"No position to sell for {symbol}")
            return None
        
        pos = self.positions[symbol]
        quantity = quantity or pos.quantity
        
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side='sell',
            quantity=quantity,
            price=price
        )
        
        self.orders.append(order)
        self.on_sell_signal(symbol, price)
        return order
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close all positions for a symbol."""
        if symbol in self.positions:
            return self.sell(symbol=symbol)
        return None
    
    def get_symbol(self) -> str:
        """Get current symbol being processed."""
        return self.data.attrs.get('symbol', 'UNKNOWN')
    
    def run(self) -> Dict[str, Any]:
        """Run the strategy over all bars."""
        self.logger.info(f"Running strategy: {self.__class__.__name__}")
        
        for i in range(len(self.data)):
            self.current_bar = i
            self.current_date = self.data.index[i]
            self.current_price = {
                'open': self.data['open'].iloc[i],
                'high': self.data['high'].iloc[i],
                'low': self.data['low'].iloc[i],
                'close': self.data['close'].iloc[i],
                'volume': self.data['volume'].iloc[i]
            }
            
            for symbol, pos in self.positions.items():
                pos.current_price = self.current_close(symbol)
            
            self.next()
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Calculate comprehensive strategy performance metrics.
        
        Returns:
            Dict with all performance metrics
        """
        # Basic metrics
        equity = np.array(self.equity_curve)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        initial_equity = self.equity_curve[0]
        final_equity = self.equity_curve[-1]
        total_pnl = final_equity - initial_equity
        total_return = (final_equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0
        
        # Trade statistics
        closed_trades = [t for t in self.trades if t.get('status') == 'closed']
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # P&L metrics
        gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * abs(avg_loss)) if total_trades > 0 else 0
        
        # Risk metrics
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        max_drawdown_idx = drawdown.idxmin()
        recovery_factor = abs(total_pnl / max_drawdown) if max_drawdown > 0 else float('inf')
        
        # Calmar ratio (annualized return / max drawdown)
        years = len(self.data) / 252 if len(self.data) > 0 else 1
        annualized_return = ((final_equity / initial_equity) ** (1 / years) - 1) * 100 if years > 0 else 0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Streaks
        winning_streak, losing_streak = self._calculate_streaks(closed_trades)
        
        # System Quality Index (SQI)
        sqi = sharpe * win_rate / 100 if sharpe > 0 else 0
        
        # Ulcer Index (measure of drawdown severity)
        rolling_max = equity_series.cummax()
        drawdown_pct = (equity_series - rolling_max) / rolling_max * 100
        ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2)) if len(drawdown_pct) > 0 else 0
        
        # Trade duration
        trade_durations = []
        entry_idx = None
        for i, trade in enumerate(closed_trades):
            if trade.get('side') == 'buy' and entry_idx is None:
                entry_idx = i
            elif trade.get('side') == 'sell' and entry_idx is not None:
                trade_durations.append(i - entry_idx)
                entry_idx = None
        
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        return {
            'strategy_name': self.__class__.__name__,
            'initial_cash': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'expectancy': expectancy,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown': max_drawdown,
            'recovery_factor': recovery_factor,
            'calmar_ratio': calmar_ratio,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'var_95': var_95,
            'var_99': var_99,
            'system_quality_index': sqi,
            'ulcer_index': ulcer_index,
            'longest_winning_streak': winning_streak,
            'longest_losing_streak': losing_streak,
            'avg_trade_duration': avg_trade_duration,
            'equity_curve': self.equity_curve,
            'drawdown_curve': drawdown.tolist(),
            'signals': self.signals,
            'trades': self.trades
        }
    
    def _calculate_streaks(self, trades: List[Dict]) -> Tuple[int, int]:
        """Calculate longest winning/losing streaks."""
        winning_streak = 0
        losing_streak = 0
        current_win = 0
        current_loss = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                current_win += 1
                current_loss = 0
                winning_streak = max(winning_streak, current_win)
            elif pnl < 0:
                current_loss += 1
                current_win = 0
                losing_streak = max(losing_streak, current_loss)
            else:
                current_win = 0
                current_loss = 0
        
        return winning_streak, losing_streak
