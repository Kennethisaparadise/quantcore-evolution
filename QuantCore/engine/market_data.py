"""
QuantCore - Live Market Data Connector

Real-time market data from Binance:
- REST API for historical data (synchronous, uses requests)
- Optional WebSocket for live streaming
- Auto-caching
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Candle:
    """A single candlestick."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class MarketDataBuffer:
    """Buffer for storing market data."""
    symbol: str
    interval: str
    candles: deque = None
    
    def __post_init__(self):
        self.candles = deque(maxlen=2000)
    
    def add(self, candle: Candle):
        self.candles.append(candle)
    
    def to_dataframe(self) -> pd.DataFrame:
        if not self.candles:
            return pd.DataFrame()
        
        data = [c.to_dict() for c in self.candles]
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_latest(self, n: int = 100) -> pd.DataFrame:
        """Get last n candles."""
        return self.to_dataframe().tail(n)


# ============================================================
# BINANCE API CLIENT
# ============================================================
class BinanceClient:
    """
    Binance REST API client for market data.
    """
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.session = requests.Session()
        
    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch historical klines (candles) from Binance."""
        url = f"{self.BASE_URL}/api/v3/klines"
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        response = self.session.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        
        # Parse klines
        candles = []
        for k in data:
            candle = Candle(
                timestamp=datetime.fromtimestamp(k[0] / 1000),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5])
            )
            candles.append(candle)
        
        # Create DataFrame
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame([c.to_dict() for c in candles])
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        url = f"{self.BASE_URL}/api/v3/exchangeInfo"
        
        response = self.session.get(url, timeout=10)
        data = response.json()
        
        symbols = [s['symbol'] for s in data['symbols'] 
                  if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT']
        
        return symbols
    
    def get_ticker(self, symbol: str = "BTCUSDT") -> Dict:
        """Get 24h ticker data."""
        url = f"{self.BASE_URL}/api/v3/ticker/24hr"
        
        params = {'symbol': symbol.upper()}
        
        response = self.session.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        return response.json()


# ============================================================
# MAIN MARKET DATA CLASS
# ============================================================
class MarketData:
    """
    Simple wrapper for market data.
    
    Usage:
        data = MarketData()
        df = data.get_bars("BTCUSDT", "1h", 500)
        print(df.tail())
    """
    
    # Supported intervals
    INTERVALS = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d'
    }
    
    def __init__(self):
        self.client = BinanceClient()
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_time: Dict[str, datetime] = {}
        self.cache_ttl = 60  # seconds
    
    def get_bars(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get OHLCV bars (synchronous)."""
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            cache_age = (datetime.now() - self.cache_time.get(cache_key, datetime.min)).total_seconds()
            if cache_age < self.cache_ttl:
                return self.cache[cache_key]
        
        # Fetch fresh data
        df = self.client.fetch_klines(symbol, interval, limit)
        
        if not df.empty:
            self.cache[cache_key] = df
            self.cache_time[cache_key] = datetime.now()
        
        return df
    
    def get_multiple(
        self,
        symbols: List[str],
        interval: str = "1h",
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols."""
        result = {}
        
        for symbol in symbols:
            try:
                df = self.get_bars(symbol, interval, limit)
                result[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        
        return result
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available USDT pairs."""
        return self.client.get_symbols()
    
    def get_price(self, symbol: str = "BTCUSDT") -> float:
        """Get current price."""
        ticker = self.client.get_ticker(symbol)
        return float(ticker['lastPrice'])
    
    def get_24h_stats(self, symbol: str = "BTCUSDT") -> Dict:
        """Get 24h trading stats."""
        return self.client.get_ticker(symbol)
    
    def clear_cache(self):
        """Clear data cache."""
        self.cache.clear()
        self.cache_time.clear()


# ============================================================
# CRYPTO LIST FOR COMMON PAIRS
# ============================================================
TOP_CRYPTOS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'XLMUSDT'
]


# ============================================================
# DEMO
# ============================================================
def demo():
    """Demo the market data client."""
    print("=" * 50)
    print("Binance Market Data Demo")
    print("=" * 50)
    
    # Create client
    data = MarketData()
    
    # Get BTC data
    print("\nFetching BTC-USDT data...")
    btc = data.get_bars("BTCUSDT", "1h", 20)
    print(f"Got {len(btc)} candles")
    if not btc.empty:
        print(btc[['close', 'volume']].tail(3))
    
    # Get current price
    price = data.get_price("BTCUSDT")
    print(f"\nBTC current price: ${price:,.2f}")
    
    # Get 24h stats
    stats = data.get_24h_stats("BTCUSDT")
    print(f"24h change: {stats['priceChangePercent']}%")
    print(f"24h high: ${float(stats['highPrice']):,.2f}")
    print(f"24h low: ${float(stats['lowPrice']):,.2f}")
    
    # Get multiple symbols
    print("\nFetching multiple symbols...")
    symbols = ["ETHUSDT", "SOLUSDT", "XRPUSDT"]
    multi = data.get_multiple(symbols, "1h", 10)
    
    for sym, df in multi.items():
        if not df.empty:
            print(f"  {sym}: ${df['close'].iloc[-1]:.4f}")
    
    print("\n✅ Market data working!")


if __name__ == "__main__":
    demo()
