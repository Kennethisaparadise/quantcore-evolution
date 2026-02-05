"""
Polymarket Data Connector
Real-time data from Polymarket Gamma API.

Provides:
- Market data fetching
- Price streaming
- Historical data for backtesting
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import time

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class Market:
    """A single Polymarket market."""
    slug: str
    question: str
    outcome_prices: List[float]
    volume_24h: float
    volume: float
    clob_token_ids: List[str]
    active: bool
    closed: bool
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None
    
    @property
    def yes_price(self) -> float:
        return self.outcome_prices[0] if self.outcome_prices else 0.5
    
    @property
    def no_price(self) -> float:
        return self.outcome_prices[1] if len(self.outcome_prices) > 1 else 0.5


@dataclass
class PricePoint:
    """A single price observation."""
    price: float
    timestamp: datetime
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class MarketHistory:
    """Historical price data for a market."""
    slug: str
    question: str
    prices: List[PricePoint] = field(default_factory=list)
    
    @property
    def current_price(self) -> float:
        return self.prices[-1].price if self.prices else 0.0
    
    @property
    def price_change_24h(self) -> float:
        if len(self.prices) < 2:
            return 0.0
        price_24h_ago = self.prices[0].price
        return (self.current_price - price_24h_ago) / price_24h_ago * 100


class PolymarketConnector:
    """
    Connector for Polymarket Gamma API.
    
    API Endpoints:
    - /events - List active events
    - /markets - Market details
    - /tags - Market categories
    - /sports - Sports leagues
    """
    
    GAMMA_BASE = "https://gamma-api.polymarket.com"
    CLOB_BASE = "https://clob.polymarket.com"
    
    def __init__(self, cache_dir: str = "data"):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        
        # Cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _request(self, url: str, params: dict = None, 
                cache_seconds: int = 300) -> Optional[Any]:
        """
        Make API request with caching.
        
        Args:
            url: API endpoint
            params: Query parameters
            cache_seconds: How long to cache responses (default 5 min)
        """
        # Check cache first
        cache_key = f"{url}_{params}".replace(" ", "").replace("'", "")
        cache_file = self.cache_dir / f"{hash(cache_key)}.json"
        
        # Return cached if valid
        if cache_file.exists():
            age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds()
            if age < cache_seconds:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Make request
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
        
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def get_events(self, active: bool = True, limit: int = 50) -> List[Dict]:
        """Fetch active events."""
        url = f"{self.GAMMA_BASE}/events"
        params = {
            "active": str(active).lower(),
            "closed": "false",
            "limit": str(limit)
        }
        return self._request(url, params) or []
    
    def get_markets(self, slug: str = None, 
                   tags: List[str] = None,
                   limit: int = 100) -> List[Market]:
        """Fetch markets."""
        url = f"{self.GAMMA_BASE}/markets"
        params = {"limit": str(limit)}
        
        if slug:
            params["slug"] = slug
        if tags:
            params["tags"] = ",".join(tags)
        
        data = self._request(url, params)
        if not data:
            return []
        
        markets = []
        for m in data:
            market = Market(
                slug=m.get('slug', ''),
                question=m.get('question', m.get('title', '')),
                outcome_prices=m.get('outcomePrices', ['0.5', '0.5']),
                volume_24h=float(m.get('volume24Hr', 0)),
                volume=float(m.get('volume', 0)),
                clob_token_ids=m.get('clobTokenIds', []),
                active=m.get('active', True),
                closed=m.get('closed', False),
                created_at=m.get('creationTransaction'),
                resolved_at=m.get('resolutionTransaction')
            )
            markets.append(market)
        
        return markets
    
    def get_price(self, token_id: str, side: str = "buy") -> Optional[float]:
        """Get current price from CLOB."""
        url = f"{self.CLOB_BASE}/price"
        params = {"token_id": token_id, "side": side}
        
        data = self._request(url, params, cache_seconds=30)
        return float(data['price']) if data and 'price' in data else None
    
    def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Get order book for a token."""
        url = f"{self.CLOB_BASE}/book"
        params = {"token_id": token_id}
        
        return self._request(url, params, cache_seconds=30)
    
    def get_ticker(self, token_id: str) -> Optional[Dict]:
        """Get 24h ticker data."""
        url = f"{self.CLOB_BASE}/ticker"
        params = {"token_id": token_id}
        
        return self._request(url, params, cache_seconds=60)
    
    def get_categories(self, limit: int = 50) -> List[Dict]:
        """Get market categories."""
        url = f"{self.GAMMA_BASE}/tags"
        params = {"limit": str(limit)}
        
        return self._request(url, params) or []
    
    def get_sports_leagues(self) -> List[Dict]:
        """Get available sports leagues."""
        url = f"{self.GAMMA_BASE}/sports"
        
        return self._request(url) or []
    
    def fetch_price_history(self, slug: str, 
                          days: int = 7,
                          interval_minutes: int = 60) -> MarketHistory:
        """
        Fetch price history for backtesting.
        
        Note: Gamma API doesn't provide historical prices directly.
        This simulates by fetching current price and tracking.
        
        For real backtesting, you'd need:
        - Polymarket subgraph (historical)
        - Third-party data providers
        - Your own price tracking database
        """
        markets = self.get_markets(slug=slug)
        
        if not markets:
            return MarketHistory(slug=slug, question="")
        
        market = markets[0]
        history = MarketHistory(
            slug=slug,
            question=market.question
        )
        
        # For now, just record current price
        # Real history requires external data source
        token_id = market.clob_token_ids[0] if market.clob_token_ids else None
        
        if token_id:
            price = self.get_price(token_id)
            if price:
                history.prices.append(PricePoint(
                    price=price,
                    timestamp=datetime.now(),
                    volume=market.volume_24h
                ))
        
        return history
    
    def list_market_slugs(self, categories: List[str] = None,
                         min_volume: float = 1000) -> List[str]:
        """
        List available market slugs for trading.
        
        Filters by volume to avoid illiquid markets.
        """
        markets = self.get_markets(tags=categories)
        
        slugs = []
        for market in markets:
            if market.volume_24h >= min_volume and market.active and not market.closed:
                slugs.append(market.slug)
        
        return slugs


class PolymarketBacktester:
    """
    Backtesting framework for trading strategies.
    
    Loads historical data and simulates trades.
    """
    
    def __init__(self, connector: PolymarketConnector):
        self.connector = connector
        self.trades: List[Dict] = []
        self.price_history: Dict[str, MarketHistory] = {}
    
    def load_history(self, slug: str, days: int = 30) -> MarketHistory:
        """Load price history for a market."""
        history = self.connector.fetch_price_history(slug, days)
        self.price_history[slug] = history
        return history
    
    def simulate_trades(self, slug: str, strategy_func) -> List[Dict]:
        """
        Simulate trades using a strategy function.
        
        Args:
            slug: Market to trade
            strategy_func: Function that takes price and returns signal
            
        Returns:
            List of simulated trades
        """
        if slug not in self.price_history:
            self.load_history(slug)
        
        history = self.price_history[slug]
        trades = []
        
        for i, point in enumerate(history.prices[1:], 1):
            prev = history.prices[i-1]
            
            # Get signal from strategy
            signal = strategy_func(prev.price, point.price, history.prices)
            
            if signal in ['buy', 'sell']:
                trade = {
                    'slug': slug,
                    'signal': signal,
                    'entry_price': point.price,
                    'entry_time': point.timestamp,
                    'direction': signal,
                    'result': 'open'
                }
                trades.append(trade)
        
        self.trades.extend(trades)
        return trades
    
    def calculate_metrics(self) -> Dict:
        """Calculate backtesting metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'profit_factor': 0
            }
        
        wins = [t for t in self.trades if t.get('result') == 'win']
        losses = [t for t in self.trades if t.get('result') == 'loss']
        
        total = len(self.trades)
        win_rate = len(wins) / total * 100 if total > 0 else 0
        
        avg_win = sum(t.get('profit', 0) for t in wins) / len(wins) if wins else 0
        avg_loss = sum(abs(t.get('profit', 0) for t in losses) / len(losses) if losses else 0
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'total_trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'profit_factor': round(profit_factor, 2)
        }
    
    def run_backtest(self, slug: str, strategy_func, 
                    initial_capital: float = 10000) -> Dict:
        """
        Run complete backtest.
        
        Args:
            slug: Market to trade
            strategy_func: Trading strategy function
            initial_capital: Starting capital
            
        Returns:
            Complete backtest results
        """
        trades = self.simulate_trades(slug, strategy_func)
        metrics = self.calculate_metrics()
        
        return {
            'slug': slug,
            'initial_capital': initial_capital,
            'metrics': metrics,
            'trades': trades[:100],  # Limit stored trades
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demo of Polymarket connector."""
    
    print("\n" + "=" * 60)
    print("POLYMARKET DATA CONNECTOR")
    print("=" * 60)
    
    connector = PolymarketConnector()
    
    # Get events
    print("\n📥 Fetching active events...")
    events = connector.get_events(limit=10)
    print(f"Found {len(events)} active events")
    
    # Get markets
    print("\n📊 Fetching markets...")
    markets = connector.get_markets(limit=10)
    print(f"Found {len(markets)} markets")
    
    # Display markets
    print("\n" + "-" * 60)
    for market in markets[:5]:
        print(f"\n📌 {market.question[:50]}...")
        print(f"   YES: ${market.yes_price:.4f} | NO: ${market.no_price:.4f}")
        print(f"   Volume 24h: ${market.volume_24h:,.0f}")
    
    # Get categories
    print("\n📂 Categories:")
    categories = connector.get_categories(limit=10)
    for cat in categories[:5]:
        print(f"  • {cat.get('label', 'Unknown')} ({cat.get('id', 'N/A')})")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
