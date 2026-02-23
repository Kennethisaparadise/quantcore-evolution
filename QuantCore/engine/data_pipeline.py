"""
QuantCore - Market Data Pipeline

Centralized market data ingestion, storage, and retrieval.
Supports multiple adapters: OpenAlgo, Polymarket, Generic (CCXT).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketType(Enum):
    """Supported market types."""
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"
    PREDICTION = "prediction"
    COMMODITY = "commodity"
    INDEX = "index"


class TimeFrame(Enum):
    """Supported timeframes."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    HOUR = "1h"
    FOUR_HOUR = "4h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"


@dataclass
class OHLCV:
    """OHLCV data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
    vwap: Optional[float] = None
    trades: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adjusted_close': self.adjusted_close,
            'vwap': self.vwap,
            'trades': self.trades
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OHLCV':
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data['volume']),
            adjusted_close=float(data.get('adjusted_close')) if data.get('adjusted_close') else None,
            vwap=float(data.get('vwap')) if data.get('vwap') else None,
            trades=int(data['trades']) if data.get('trades') else None
        )


@dataclass
class MarketDataRequest:
    """Request structure for market data."""
    symbol: str
    market_type: MarketType
    exchange: str
    start_date: datetime
    end_date: datetime
    timeframe: TimeFrame = TimeFrame.DAY
    adjusted: bool = False
    extended_hours: bool = False
    source: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class MarketDataResult:
    """Result structure for market data."""
    data: pd.DataFrame
    symbol: str
    exchange: str
    timeframe: TimeFrame
    start_date: datetime
    end_date: datetime
    source: str
    cached: bool = False
    download_time: Optional[float] = None
    rows: int = 0
    columns: List[str] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = list(self.data.columns) if hasattr(self.data, 'columns') else []
        self.rows = len(self.data)
    
    @property
    def is_empty(self) -> bool:
        return self.data.empty
    
    @property
    def date_range(self) -> tuple:
        if self.is_empty:
            return (None, None)
        return (self.data.index.min(), self.data.index.max())
    
    def summary(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'rows': self.rows,
            'date_range': self.date_range,
            'cached': self.cached,
            'download_time_ms': round(self.download_time * 1000, 2) if self.download_time else None
        }


class DataAdapter(ABC):
    """Abstract base class for market data adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    def fetch_data(self, request: MarketDataRequest) -> MarketDataResult:
        """Fetch market data according to the request."""
        pass
    
    @abstractmethod
    def get_supported_exchanges(self) -> List[str]:
        """Return list of supported exchanges."""
        pass
    
    @abstractmethod
    def get_supported_symbols(self, exchange: str) -> List[str]:
        """Return list of available symbols for an exchange."""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Return health status of the adapter."""
        return {
            'name': self.name,
            'connected': self.connected,
            'timestamp': datetime.now().isoformat()
        }


class DataPipeline:
    """
    Centralized market data pipeline.
    
    Responsibilities:
    - Manage data adapters
    - Route data requests to appropriate adapters
    - Cache data (CSV/Parquet)
    - Data validation and normalization
    - Historical data downloads
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.adapters: Dict[str, DataAdapter] = {}
        self.cache_dir = Path(self.config.get('data_storage', {}).get('raw_path', 'data/csv'))
        self.processed_dir = Path(self.config.get('data_storage', {}).get('processed_path', 'data/parquet'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.json'
        
        import json
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataPipeline')
    
    def register_adapter(self, name: str, adapter: DataAdapter) -> None:
        """Register a data adapter."""
        self.adapters[name] = adapter
        self.logger.info(f"Registered adapter: {name}")
    
    def get_adapter(self, source: str) -> Optional[DataAdapter]:
        """Get a registered adapter by name."""
        return self.adapters.get(source)
    
    def download_data(
        self,
        symbol: str,
        exchange: str,
        market_type: MarketType,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        timeframe: TimeFrame = TimeFrame.DAY,
        source: str = 'default',
        force_download: bool = False,
        save_csv: bool = True
    ) -> MarketDataResult:
        """
        Download market data.
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE', 'BTC-USD')
            exchange: Exchange code (e.g., 'NSE', 'BINANCE')
            market_type: Type of market
            start_date: Start date (inclusive)
            end_date: End date (inclusive), defaults to now
            timeframe: Data timeframe
            source: Data source adapter to use
            force_download: Skip cache and download fresh
            save_csv: Save raw data to CSV
        
        Returns:
            MarketDataResult with DataFrame and metadata
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        if end_date is None:
            end_date = datetime.now()
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, exchange, start_date, end_date, timeframe)
        cache_path = self.cache_dir / f"{cache_key}.csv"
        
        if cache_path.exists() and not force_download:
            self.logger.info(f"Loading from cache: {cache_key}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return MarketDataResult(
                data=df,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                source=source,
                cached=True
            )
        
        # Get adapter and fetch data
        adapter = self.get_adapter(source)
        if adapter is None:
            raise ValueError(f"No adapter registered for source: {source}")
        
        if not adapter.connected:
            adapter.connect()
        
        request = MarketDataRequest(
            symbol=symbol,
            market_type=market_type,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        import time
        start_time = time.time()
        result = adapter.fetch_data(request)
        download_time = time.time() - start_time
        
        result.download_time = download_time
        result.cached = False
        
        # Save to cache
        if save_csv and not result.is_empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.data.to_csv(cache_path)
            self.logger.info(f"Cached data to: {cache_path}")
        
        return result
    
    def _get_cache_key(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame
    ) -> str:
        """Generate cache key for data request."""
        return f"{exchange}_{symbol}_{timeframe.value}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    
    def load_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded CSV: {filepath} ({len(df)} rows)")
        return df
    
    def load_parquet(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from Parquet file."""
        df = pd.read_parquet(filepath)
        self.logger.info(f"Loaded Parquet: {filepath} ({len(df)} rows)")
        return df
    
    def save_csv(self, df: pd.DataFrame, filepath: Union[str, Path]) -> None:
        """Save data to CSV file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)
        self.logger.info(f"Saved CSV: {filepath} ({len(df)} rows)")
    
    def save_parquet(self, df: pd.DataFrame, filepath: Union[str, Path]) -> None:
        """Save data to Parquet file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, engine='pyarrow')
        self.logger.info(f"Saved Parquet: {filepath} ({len(df)} rows)")
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to standard OHLCV format."""
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure numeric types
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure timestamp is index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Sort by index
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Forward fill missing values (small gaps only)
        df = df.ffill(limit=5)
        
        return df
    
    def resample_data(
        self,
        df: pd.DataFrame,
        timeframe: TimeFrame,
        aggregation: str = 'ohlc'
    ) -> pd.DataFrame:
        """Resample data to different timeframe."""
        rule = self._timeframe_to_pandas_rule(timeframe)
        
        if aggregation == 'ohlc':
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif aggregation == 'last':
            resampled = df.resample(rule).last()
        else:
            resampled = df.resample(rule).agg(aggregation)
        
        return resampled.dropna()
    
    def _timeframe_to_pandas_rule(self, timeframe: TimeFrame) -> str:
        """Convert TimeFrame enum to pandas offset alias."""
        mapping = {
            TimeFrame.TICK: 'ms',
            TimeFrame.SECOND: '1s',
            TimeFrame.MINUTE: '1min',
            TimeFrame.FIVE_MIN: '5min',
            TimeFrame.FIFTEEN_MIN: '15min',
            TimeFrame.THIRTY_MIN: '30min',
            TimeFrame.HOUR: '1h',
            TimeFrame.FOUR_HOUR: '4h',
            TimeFrame.DAY: '1d',
            TimeFrame.WEEK: '1W',
            TimeFrame.MONTH: '1ME'
        }
        return mapping.get(timeframe, '1d')
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all adapters and pipeline."""
        adapter_health = {}
        for name, adapter in self.adapters.items():
            adapter_health[name] = adapter.health_check()
        
        return {
            'pipeline': 'healthy',
            'adapters': adapter_health,
            'cache_dir': str(self.cache_dir),
            'processed_dir': str(self.processed_dir),
            'timestamp': datetime.now().isoformat()
        }
