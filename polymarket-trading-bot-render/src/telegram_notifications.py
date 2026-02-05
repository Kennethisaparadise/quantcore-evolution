"""
Telegram Notifications
Sends trading signals and alerts to Telegram.

Usage:
1. Create bot via @BotFather on Telegram
2. Get bot token
3. Get chat ID (via @userinfobot or group)
4. Configure in config.json
"""

import requests
import json
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger('telegram')


class MessageType(Enum):
    """Types of messages we can send."""
    SIGNAL = "signal"
    ENTRY = "entry"
    EXIT = "exit"
    ALERT = "alert"
    SUMMARY = "summary"
    ERROR = "error"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


@dataclass
class TelegramConfig:
    """Telegram configuration."""
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False
    notify_on_signal: bool = True
    notify_on_trade: bool = True
    notify_on_summary: bool = True
    notify_on_error: bool = True


class TelegramNotifier:
    """
    Sends notifications to Telegram.
    
    Usage:
    notifier = TelegramNotifier(config)
    notifier.send_signal(market, signal, confidence)
    """
    
    BASE_URL = "https://api.telegram.org/bot"
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.session = requests.Session()
        
        if not self.config.enabled:
            logger.warning("Telegram notifications disabled")
            return
        
        if not self.config.bot_token or not self.config.chat_id:
            logger.error("Telegram token or chat_id not configured")
            self.config.enabled = False
            return
        
        logger.info("Telegram notifier initialized")
    
    def _send(self, payload: Dict) -> bool:
        """Send a message to Telegram."""
        if not self.config.enabled:
            return False
        
        url = f"{self.BASE_URL}{self.config.bot_token}/sendMessage"
        
        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def _format_price(self, price: float) -> str:
        """Format price for display."""
        if price >= 0.1:
            return f"${price:.4f}"
        elif price >= 0.01:
            return f"${price:.4f}"
        else:
            return f"${price:.4f}"
    
    def send_message(self, text: str, 
                    parse_mode: str = "Markdown",
                    disable_preview: bool = False) -> bool:
        """Send a raw message."""
        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview
        }
        
        return self._send(payload)
    
    def send_signal(self, market_slug: str, 
                   direction: str,
                   confidence: float,
                   target: float,
                   stop_loss: float) -> bool:
        """
        Send trading signal notification.
        
        Example:
        🟢 SIGNAL: BTC-100K
        
        Direction: LONG
        Confidence: 75%
        Target: $0.78
        Stop: $0.65
        """
        emoji = "🟢" if direction == "buy" else "🔴"
        
        text = f"""
{emoji} SIGNAL: {market_slug}

📊 Direction: {direction.upper()}
🎯 Confidence: {confidence:.0%}
🎯 Target: {self._format_price(target)}
🛑 Stop: {self._format_price(stop_loss)}

🤖 Master Trading System
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        return self.send_message(text, disable_preview=True)
    
    def send_entry(self, market_slug: str,
                  direction: str,
                  entry_price: float,
                  quantity: float,
                  stop_loss: float,
                  take_profit: float) -> bool:
        """Send entry notification."""
        emoji = "🟢" if direction == "buy" else "🔴"
        
        text = f"""
{emoji} ENTRY EXECUTED

📌 {market_slug}
💰 {direction.upper()} {quantity:.4f} @ {self._format_price(entry_price)}
🛑 SL: {self._format_price(stop_loss)}
🎯 TP: {self._format_price(take_profit)}

💼 Master Trading System
"""
        
        return self.send_message(text, disable_preview=True)
    
    def send_exit(self, market_slug: str,
                  direction: str,
                  entry_price: float,
                  exit_price: float,
                  pnl: float,
                  pnl_percent: float,
                  reason: str) -> bool:
        """Send exit notification."""
        emoji = "💰" if pnl >= 0 else "💸"
        result = "WIN" if pnl >= 0 else "LOSS"
        
        text = f"""
{emoji} EXIT: {market_slug}

📌 Entry: {self._format_price(entry_price)}
📌 Exit: {self._format_price(exit_price)}
💵 P&L: ${pnl:.2f} ({pnl_percent:+.1f}%)
🏆 {result}
📋 Reason: {reason}

💼 Master Trading System
"""
        
        return self.send_message(text, disable_preview=True)
    
    def send_summary(self, stats: Dict) -> bool:
        """Send daily/weekly summary."""
        win_rate = stats.get('win_rate', 0)
        emoji = "🟢" if win_rate >= 60 else "🟡" if win_rate >= 50 else "🔴"
        
        text = f"""
📊 TRADING SUMMARY

{emoji} Win Rate: {win_rate:.1f}%
💰 P&L: ${stats.get('total_pnl', 0):.2f}
📈 Trades: {stats.get('total_trades', 0)}
🎯 Wins: {stats.get('wins', 0)}
❌ Losses: {stats.get('losses', 0)}

💼 Master Trading System
📅 {datetime.now().strftime('%Y-%m-%d')}
"""
        
        return self.send_message(text)
    
    def send_alert(self, alert_type: str, message: str) -> bool:
        """Send generic alert."""
        emoji_map = {
            'warning': "⚠️",
            'error': "🚨",
            'info': "ℹ️",
            'success': "✅",
            'danger': "🚨"
        }
        
        emoji = emoji_map.get(alert_type, "📢")
        
        text = f"""
{emoji} ALERT: {alert_type.upper()}

{message}

💼 Master Trading System
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        return self.send_message(text)
    
    def send_error(self, error: str, context: str = "") -> bool:
        """Send error notification."""
        text = f"""
🚨 SYSTEM ERROR

❌ {error}

📋 Context: {context if context else 'No context'}

💼 Master Trading System
"""
        
        return self.send_message(text)
    
    def send_startup(self, config: Dict) -> bool:
        """Send startup notification."""
        text = f"""
🚀 TRADING SYSTEM STARTED

📊 Initial Capital: ${config.get('initial_capital', 0):,.2f}
📈 Scan Interval: {config.get('scan_interval_seconds', 300)}s
🎯 Confidence Threshold: {config.get('confidence_threshold', 0.65):.0%}
🛡️ Max Daily Loss: {config.get('max_daily_loss_pct', 0.05):.0%}

💼 Master Trading System
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        return self.send_message(text)
    
    def send_shutdown(self, stats: Dict) -> bool:
        """Send shutdown notification."""
        text = f"""
🛑 TRADING SYSTEM STOPPED

📊 Final Capital: ${stats.get('final_capital', 0):,.2f}
💰 Session P&L: ${stats.get('session_pnl', 0):.2f}
📈 Total Trades: {stats.get('total_trades', 0)}
🎯 Win Rate: {stats.get('win_rate', 0):.1f}%

💼 Master Trading System
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        return self.send_message(text)


def create_telegram_config(bot_token: str, chat_id: str) -> TelegramConfig:
    """Factory function to create config."""
    return TelegramConfig(
        bot_token=bot_token,
        chat_id=chat_id,
        enabled=True
    )


def demo():
    """Demo (won't actually send without real credentials)."""
    
    print("\n" + "=" * 60)
    print("TELEGRAM NOTIFICATIONS")
    print("=" * 60)
    
    print("\nTo enable Telegram notifications:")
    print("1. Message @BotFather on Telegram")
    print("2. Create a new bot: /newbot")
    print("3. Copy the bot token")
    print("4. Message @userinfobot to get your chat ID")
    print("5. Add to config.json:")
    print("""
{
  "telegram": {
    "bot_token": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
    "chat_id": "123456789",
    "enabled": true
  }
}
""")
    
    print("=" * 60)


if __name__ == "__main__":
    demo()
