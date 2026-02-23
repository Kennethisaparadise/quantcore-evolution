#!/usr/bin/env python3
"""
QuantCore - Quick Start Paper Trading Script
=============================================
Run this to start paper trading immediately.

Usage:
    python run_paper_trading.py
    python run_paper_trading.py --symbol BTC --capital 10000
    python run_paper_trading.py --mode live  # Go live!
"""

import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime

# Add engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engine'))

from live_trading import TradingOrchestrator, create_trading_config
from compounding_engine import CompoundingEngine, create_compounding_config
from sentiment_divergence import SentimentSimulator, SentimentFlowRegimeHybridizer, create_sentiment_config
from correlation_pairs import PairSignalGenerator, create_pair_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'paper_trading_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)


class PaperTrader:
    """Easy paper trading runner."""
    
    def __init__(self, symbol: str = "BTCUSDT", capital: float = 10000):
        self.symbol = symbol
        
        # Trading config
        self.trading_config = create_trading_config(
            mode="paper",
            exchange="binance",
            initial_capital=capital
        )
        
        # Compounding config
        self.compounding_config = create_compounding_config(
            risk_per_trade=0.02,
            kelly_fraction=0.25,
            reinvestment_rate=0.80
        )
        
        # Sentiment config
        self.sentiment_config = create_sentiment_config()
        
        # Initialize
        self.orchestrator = TradingOrchestrator(self.trading_config)
        self.compounding = CompoundingEngine(self.compounding_config, capital)
        self.sentiment = SentimentFlowRegimeHybridizer(self.sentiment_config)
        
        logger.info(f"🚀 Initialized Paper Trading: {symbol}, ${capital:,.2f}")
    
    async def generate_signal(self, price_data: dict) -> tuple:
        """Generate trading signal using all modules."""
        # This would use real market data in production
        # For now, generate based on simple logic
        
        import numpy as np
        
        # Simulate price analysis
        close = price_data.get('close', 45000)
        volume = price_data.get('volume', 1000)
        
        # Simple RSI-like signal (replace with real strategy)
        returns = np.random.randn() * 0.02
        
        direction = 0
        confidence = 0.5
        
        if returns > 0.01:
            direction = 1  # Buy
            confidence = min(abs(returns) * 20, 1.0)
        elif returns < -0.01:
            direction = -1  # Sell
            confidence = min(abs(returns) * 20, 1.0)
        
        return direction, confidence
    
    async def run_cycle(self, iteration: int):
        """Run one trading cycle."""
        import numpy as np
        
        # Simulate market data (replace with real API in production)
        price_data = {
            'close': 45000 + np.random.randn() * 500,
            'volume': 1000 + np.random.random() * 500,
            'open': 45000,
            'high': 45100,
            'low': 44900
        }
        
        # Get signal
        direction, confidence = await self.generate_signal(price_data)
        
        if direction != 0:
            # Calculate position size from compounding engine
            position_size = self.compounding.calculate_position_size(
                confidence=confidence,
                entry_price=price_data['close'],
                stop_loss_pct=0.03
            )
            
            # Execute trade
            try:
                order = await self.orchestrator.execute_signal(
                    symbol=self.symbol,
                    direction=direction,
                    quantity=position_size,
                    strategy_id="auto_signal"
                )
                
                if order:
                    logger.info(f"📝 Trade #{iteration}: {direction > 0 and 'BUY' or 'SELL'} "
                              f"{position_size:.4f} @ ${price_data['close']:.2f}")
                    
                    # Simulate PnL (in production, wait for real close)
                    pnl = np.random.randn() * price_data['close'] * 0.01
                    is_win = pnl > 0
                    
                    self.compounding.record_trade(pnl, is_win)
                    
            except Exception as e:
                logger.warning(f"Trade failed: {e}")
        
        # Log status
        status = self.compounding.get_status()
        logger.info(f"💰 Equity: ${status['current_equity']:,.2f} | "
                   f"PnL: {status['total_return']*100:+.2f}% | "
                   f"DD: {status['current_drawdown']*100:.1f}% | "
                    f"Trades: {status['total_trades']}")
        
        return status
    
    async def run(self, cycles: int = 100, delay_seconds: int = 60):
        """Run paper trading for N cycles."""
        logger.info(f"🟢 Starting Paper Trading: {cycles} cycles, {delay_seconds}s interval")
        logger.info(f"💵 Initial Capital: ${self.trading_config.initial_capital:,.2f}")
        logger.info("=" * 60)
        
        for i in range(cycles):
            try:
                status = await self.run_cycle(i + 1)
                
                # Check risk limits
                if status['current_drawdown'] > 0.20:
                    logger.error("🛑 STOP: Max drawdown exceeded!")
                    break
                    
                if status['total_return'] < -0.10:
                    logger.error("🛑 STOP: Max daily loss exceeded!")
                    break
                
                # Wait for next cycle
                if i < cycles - 1:
                    await asyncio.sleep(delay_seconds)
                    
            except KeyboardInterrupt:
                logger.info("⏹️ Stopped by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(delay_seconds)
        
        # Final report
        logger.info("=" * 60)
        logger.info("📊 FINAL REPORT")
        status = self.compounding.get_status()
        logger.info(f"  Initial: ${self.trading_config.initial_capital:,.2f}")
        logger.info(f"  Final:   ${status['current_equity']:,.2f}")
        logger.info(f"  Return:  {status['total_return']*100:+.2f}%")
        logger.info(f"  Trades:  {status['total_trades']}")
        logger.info(f"  Win Rate: {status['win_rate']*100:.1f}%")
        logger.info(f"  Max DD:  {status['current_drawdown']*100:.1f}%")
        
        # Trading orchestrator status
        orch_status = self.orchestrator.get_status()
        logger.info(f"  Total PnL: ${orch_status['total_pnl']:,.2f}")
        
        logger.info("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='QuantCore Paper Trading')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--cycles', type=int, default=100, help='Number of cycles')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between cycles')
    parser.add_argument('--mode', default='paper', choices=['paper', 'live'], help='Trading mode')
    
    args = parser.parse_args()
    
    # Update mode if live
    if args.mode == 'live':
        logger.warning("⚠️ LIVE TRADING - REAL MONEY AT RISK!")
        input("Press Enter to continue or Ctrl+C to cancel...")
    
    trader = PaperTrader(args.symbol, args.capital)
    await trader.run(args.cycles, args.interval)


if __name__ == "__main__":
    asyncio.run(main())
