"""
Telegram module for Enhanced Bybit Trading System.
Note: This module is named to avoid conflicts with python-telegram-bot package.
"""

from .telegram import (
    TelegramBootstrapManager,
    run_bootstrap_mode, 
    check_bootstrap_needed,
    send_trading_notification
)

__all__ = [
    'TelegramBootstrapManager',
    'run_bootstrap_mode',
    'check_bootstrap_needed', 
    'send_trading_notification'
]