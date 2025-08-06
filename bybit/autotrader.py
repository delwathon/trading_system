"""
Auto-Trading System for Enhanced Bybit Trading System.
Handles scheduled scanning, position management, and automated trading with leverage support.
UPDATED: Enhanced with critical strategy revisions for improved performance and risk management.
"""

import asyncio
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import uuid
import ccxt

from config.config import EnhancedSystemConfig
from core.system import CompleteEnhancedBybitSystem
from database.models import DatabaseManager, TradingPosition, AutoTradingSession, TradingSignal, TradingOpportunity
from utils.database_manager import EnhancedDatabaseManager
from telegram_bot_and_notification.bootstrap_manager import send_trading_notification


@dataclass
class PositionData:
    """Data class for position tracking"""
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: float
    risk_amount: float
    stop_loss: float
    take_profit: float
    position_id: str
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    partial_1_taken: bool = False
    partial_2_taken: bool = False
    partial_3_taken: bool = False
    original_size: float = 0.0
    trailing_stop_active: bool = False
    highest_price: float = 0.0  # For trailing stops
    lowest_price: float = 0.0   # For trailing stops


class LeverageManager:
    """Handle leverage validation and conversion"""
    
    ACCEPTABLE_LEVERAGE = ['10', '12.5', '25', '50', 'max']
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
    
    def validate_leverage(self, leverage_str: str) -> bool:
        """Validate leverage is acceptable"""
        return leverage_str in self.ACCEPTABLE_LEVERAGE
    
    def convert_leverage_to_float(self, leverage_str: str, symbol: str) -> float:
        """Convert leverage string to float, handling 'max' case"""
        try:
            if leverage_str == 'max':
                # Get maximum leverage for symbol from exchange
                market_info = self.exchange.market(symbol)
                max_leverage = market_info.get('limits', {}).get('leverage', {}).get('max', 100)
                self.logger.debug(f"Max leverage for {symbol}: {max_leverage}")
                return float(max_leverage)
            else:
                return float(leverage_str)
        except Exception as e:
            self.logger.error(f"Error converting leverage {leverage_str} for {symbol}: {e}")
            return 10.0  # Fallback to safe leverage
    
    def set_symbol_leverage(self, symbol: str, leverage: float) -> bool:
        """Set leverage for symbol on exchange"""
        try:
            # Get current leverage first
            try:
                positions = self.exchange.fetch_positions([symbol])
                current_leverage = None
                for pos in positions:
                    if pos['symbol'] == symbol:
                        current_leverage = pos.get('leverage', 1)
                        break
                
                # If leverage is already set correctly, skip
                if current_leverage and abs(float(current_leverage) - leverage) < 0.1:
                    self.logger.debug(f"✅ Leverage already set to {current_leverage}x for {symbol}")
                    return True
                    
            except Exception as e:
                self.logger.debug(f"Could not check current leverage for {symbol}: {e}")
            
            # Set leverage on Bybit
            result = self.exchange.set_leverage(leverage, symbol)
            self.logger.debug(f"✅ Set leverage {leverage}x for {symbol}")
            return True
        except Exception as e:
            error_msg = str(e)
            # If leverage not modified error, it might already be set correctly
            if "110043" in error_msg or "leverage not modified" in error_msg:
                self.logger.warning(f"⚠️ Leverage not modified for {symbol} - may already be set correctly")
                return True  # Continue with trade execution
            else:
                self.logger.error(f"❌ Failed to set leverage {leverage}x for {symbol}: {e}")
                return False
    
    def get_max_leverage_for_symbol(self, symbol: str) -> float:
        """Get maximum available leverage for symbol"""
        try:
            market_info = self.exchange.market(symbol)
            return float(market_info.get('limits', {}).get('leverage', {}).get('max', 100))
        except Exception as e:
            self.logger.error(f"Error getting max leverage for {symbol}: {e}")
            return 50.0  # Conservative fallback


class PositionSizer:
    """Calculate position sizes with leverage - ENHANCED with adaptive sizing"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, risk_amount_pct: float, leverage: float, entry_price: float) -> float:
        """
        Calculate position size using: (account_balance × risk_percentage × leverage) / entry_price
        
        Args:
            risk_amount_pct: Risk percentage of account balance (e.g., 5.0 for 5%)
            leverage: Trading leverage (e.g., 25)
            entry_price: Entry price for the position
            
        Returns:
            Position size in base currency units
        """
        try:
            if entry_price <= 0 or leverage <= 0:
                raise ValueError(f"Invalid entry_price ({entry_price}) or leverage ({leverage})")
            
            # Get current account balance
            account_balance = self.get_available_balance()
            if account_balance <= 0:
                raise ValueError(f"Invalid account balance: {account_balance}")
            
            # Calculate risk amount in USDT
            risk_amount_usdt = account_balance * (risk_amount_pct / 100)
            
            # Calculate position size: (risk_amount_usdt × leverage) / entry_price
            position_size = (risk_amount_usdt * leverage) / entry_price
            
            self.logger.debug(f"Position size calculation:")
            self.logger.debug(f"  Account Balance: {account_balance} USDT")
            self.logger.debug(f"  Risk Percentage: {risk_amount_pct}%")
            self.logger.debug(f"  Risk Amount: {risk_amount_usdt} USDT")
            self.logger.debug(f"  Leverage: {leverage}x")
            self.logger.debug(f"  Entry Price: {entry_price}")
            self.logger.debug(f"  Position Size: ({risk_amount_usdt} × {leverage}) / {entry_price} = {position_size}")
            
            return position_size
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_adaptive_position_size(self, risk_amount_pct: float, leverage: float, 
                                       entry_price: float, signal: Dict, df=None) -> Tuple[float, float]:
        """
        ENHANCED: Adaptive position sizing based on market conditions and signal quality
        
        Returns:
            Tuple of (position_size, adjusted_risk_pct)
        """
        try:
            base_risk_pct = risk_amount_pct
            
            # 1. MTF Quality Adjustment
            mtf_validated = signal.get('mtf_validated', False)
            analysis_details = signal.get('analysis_details', {})
            signal_strength = analysis_details.get('signal_strength', 'moderate')
            confidence = signal.get('confidence', 50)
            
            if mtf_validated and signal_strength == 'strong' and confidence > 75:
                risk_multiplier = 1.3  # Increase risk for premium signals
            elif mtf_validated and confidence > 65:
                risk_multiplier = 1.1  # Slightly increase for good MTF signals
            elif mtf_validated:
                risk_multiplier = 1.0  # Normal risk for MTF validated
            elif signal_strength == 'strong' and confidence > 70:
                risk_multiplier = 0.9  # Slightly reduce for non-MTF strong signals
            else:
                risk_multiplier = 0.7  # Reduce risk for weaker signals
            
            # 2. Volatility Adjustment (if dataframe available)
            volatility_multiplier = 1.0
            if df is not None and len(df) > 20:
                try:
                    # Calculate ATR-based volatility
                    high_low = df['high'] - df['low']
                    high_close = np.abs(df['high'] - df['close'].shift())
                    low_close = np.abs(df['low'] - df['close'].shift())
                    import pandas as pd
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = true_range.rolling(window=14).mean().iloc[-1]
                    volatility_pct = (atr / entry_price) * 100
                    
                    if volatility_pct > 8:  # High volatility
                        volatility_multiplier = 0.6  # Reduce position size significantly
                    elif volatility_pct > 5:  # Medium volatility
                        volatility_multiplier = 0.8  # Reduce moderately
                    else:  # Low volatility
                        volatility_multiplier = 1.0
                        
                    self.logger.debug(f"📊 Volatility adjustment: {volatility_pct:.1f}% → {volatility_multiplier:.1f}x")
                except Exception as e:
                    self.logger.debug(f"Could not calculate volatility: {e}")
            
            # 3. Risk/Reward Adjustment
            rr_ratio = signal.get('risk_reward_ratio', 2.0)
            if rr_ratio >= 3.5:
                rr_multiplier = 1.2  # Increase for excellent R/R
            elif rr_ratio >= 2.8:
                rr_multiplier = 1.1
            elif rr_ratio >= 2.2:
                rr_multiplier = 1.0
            else:
                rr_multiplier = 0.8  # Reduce for poor R/R
            
            # 4. Market Conditions Adjustment
            price_change_24h = abs(signal.get('price_change_24h', 0))
            if price_change_24h > 15:  # Extreme movement
                market_multiplier = 0.5  # Very conservative
            elif price_change_24h > 8:  # High movement
                market_multiplier = 0.7
            else:
                market_multiplier = 1.0
            
            # 5. Volume Quality Adjustment
            volume_24h = signal.get('volume_24h', 0)
            if volume_24h < 500_000:  # Low liquidity
                volume_multiplier = 0.6
            elif volume_24h < 1_000_000:  # Medium liquidity
                volume_multiplier = 0.8
            else:  # Good liquidity
                volume_multiplier = 1.0
            
            # Calculate final risk percentage
            adjusted_risk_pct = (base_risk_pct * risk_multiplier * volatility_multiplier * 
                                rr_multiplier * market_multiplier * volume_multiplier)
            
            # Apply reasonable limits
            adjusted_risk_pct = max(1.0, min(10.0, adjusted_risk_pct))
            
            # Calculate position size with adjusted risk
            account_balance = self.get_available_balance()
            risk_amount_usdt = account_balance * (adjusted_risk_pct / 100)
            position_size = (risk_amount_usdt * leverage) / entry_price
            
            self.logger.info(f"📊 Adaptive Position Sizing:")
            self.logger.info(f"   Base Risk: {base_risk_pct}% → Adjusted: {adjusted_risk_pct:.1f}%")
            self.logger.info(f"   Quality: {risk_multiplier:.1f}x | Volatility: {volatility_multiplier:.1f}x")
            self.logger.info(f"   R/R: {rr_multiplier:.1f}x | Market: {market_multiplier:.1f}x | Volume: {volume_multiplier:.1f}x")
            self.logger.info(f"   Final Risk: {risk_amount_usdt:.2f} USDT ({adjusted_risk_pct:.1f}%)")
            
            return position_size, adjusted_risk_pct
            
        except Exception as e:
            self.logger.error(f"Adaptive position sizing error: {e}")
            # Fallback to regular calculation
            position_size = self.calculate_position_size(risk_amount_pct, leverage, entry_price)
            return position_size, risk_amount_pct
    
    def validate_position_size(self, symbol: str, size: float) -> Tuple[bool, float]:
        """Validate position size against exchange limits"""
        try:
            market_info = self.exchange.market(symbol)
            limits = market_info.get('limits', {})
            amount_limits = limits.get('amount', {})
            
            min_size = amount_limits.get('min', 0.001)
            max_size = amount_limits.get('max', 1000000)
            
            if size < min_size:
                self.logger.warning(f"Position size {size} below minimum {min_size} for {symbol}")
                return False, min_size
            
            if size > max_size:
                self.logger.warning(f"Position size {size} above maximum {max_size} for {symbol}")
                return False, max_size
            
            return True, size
        except Exception as e:
            self.logger.error(f"Error validating position size for {symbol}: {e}")
            return False, 0.0
               
    def get_available_balance(self) -> float:
        """Get available USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0


class ScheduleManager:
    """Handle scan timing and scheduling logic - ENHANCED with market hours awareness"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def parse_start_hour(self, start_hour_str: str) -> Tuple[int, int]:
        """Parse start hour string like '01:00' to (hour, minute)"""
        try:
            hour, minute = start_hour_str.split(':')
            return int(hour), int(minute)
        except Exception as e:
            self.logger.error(f"Error parsing start hour {start_hour_str}: {e}")
            return 1, 0  # Default to 01:00
    
    def is_optimal_trading_time(self) -> Tuple[bool, str]:
        """
        ENHANCED: Check if current time is optimal for trading
        
        Returns:
            Tuple of (is_optimal, session_name)
        """
        try:
            now_utc = datetime.now(timezone.utc)
            hour_utc = now_utc.hour
            
            # Market session analysis
            if 0 <= hour_utc <= 6:  # Asian session
                return True, "asian_session"
            elif 6 <= hour_utc <= 14:  # European session
                return True, "european_session" 
            elif 14 <= hour_utc <= 22:  # US session
                return True, "us_session"
            else:  # Low liquidity hours (22-24 UTC)
                # Allow trading but note low liquidity
                return True, "low_liquidity"
            
        except Exception:
            return True, "unknown"
    
    def calculate_next_scan_time(self) -> datetime:
        """Calculate next scheduled scan time"""
        try:
            now = datetime.now()
            start_hour, start_minute = self.parse_start_hour(self.config.day_trade_start_hour)
            scan_interval_hours = self.config.scan_interval / 3600  # Convert seconds to hours
            
            # Create today's start time
            today_start = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            
            # Find next scan time
            if now < today_start:
                # If current time is before today's start, next scan is today's start
                next_scan = today_start
            else:
                # Calculate how many intervals have passed since start
                elapsed = now - today_start
                elapsed_hours = elapsed.total_seconds() / 3600
                intervals_passed = int(elapsed_hours / scan_interval_hours)
                
                # Next scan is the next interval
                next_scan = today_start + timedelta(hours=(intervals_passed + 1) * scan_interval_hours)
                
                # If next scan is tomorrow, move to next day's start time
                if next_scan.date() > now.date():
                    tomorrow_start = (now + timedelta(days=1)).replace(
                        hour=start_hour, minute=start_minute, second=0, microsecond=0
                    )
                    next_scan = tomorrow_start
            
            self.logger.debug(f"Next scan calculated: {next_scan}")
            return next_scan
        except Exception as e:
            self.logger.error(f"Error calculating next scan time: {e}")
            # Fallback: scan in 1 hour
            return datetime.now() + timedelta(hours=1)
    
    def is_scan_time(self) -> bool:
        """Check if it's time to scan (within 1 minute of scheduled time)"""
        try:
            next_scan = self.calculate_next_scan_time()
            now = datetime.now()
            
            # Check if we're within 5 minute of scan time
            time_diff = abs((next_scan - now).total_seconds())
            return time_diff <= 300
        except Exception as e:
            self.logger.error(f"Error checking scan time: {e}")
            return False

    def wait_for_next_scan(self) -> datetime:
        """Wait until next scheduled scan time"""
        try:
            next_scan = self.calculate_next_scan_time()
            now = datetime.now()

            if next_scan > now:
                wait_seconds = (next_scan - now).total_seconds()
                
                # Check if it's optimal trading time
                is_optimal, session = self.is_optimal_trading_time()
                session_info = f" ({session})" if session != "unknown" else ""
                
                self.logger.info(f"⏰ Next scan scheduled for: {next_scan}{session_info}")
                self.logger.info(f"⏳ Waiting {wait_seconds / 60:.1f} minutes...")

                time.sleep(wait_seconds)

            return next_scan
        except Exception as e:
            self.logger.error(f"Error waiting for next scan: {e}")
            time.sleep(3600)  # Wait 1 hour on error
            return datetime.now()


class LeveragedProfitMonitor:
    """Monitor leveraged profits/losses and auto-close positions - ENHANCED with trailing stops"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.position_tracker = {}  # Track position states
    
    def calculate_leveraged_profit_pct(self, position: PositionData, current_price: float) -> float:
        """Calculate profit percentage considering leverage"""
        try:
            if position.entry_price <= 0:
                return 0.0
            
            # Calculate price change percentage
            price_change_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # Apply leverage multiplier
            leveraged_profit_pct = price_change_pct * position.leverage
            
            # Consider position side
            if position.side.lower() == 'sell':
                leveraged_profit_pct = -leveraged_profit_pct
            
            return leveraged_profit_pct
        except Exception as e:
            self.logger.error(f"Error calculating leveraged profit for {position.symbol}: {e}")
            return 0.0
    
    def should_auto_close(self, position: PositionData, current_price: float) -> tuple[bool, str]:
        """Check if position should be auto-closed based on profit/loss targets"""
        try:
            leveraged_profit_pct = self.calculate_leveraged_profit_pct(position, current_price)
            
            # Check profit target
            if leveraged_profit_pct >= self.config.auto_close_profit_at:
                return True, 'profit_target'
            
            # Check loss limit
            if leveraged_profit_pct <= -abs(self.config.auto_close_loss_at):
                return True, 'loss_limit'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error checking auto-close for {position.symbol}: {e}")
            return False, None
    
    def update_trailing_stop(self, position: PositionData, current_price: float) -> Optional[float]:
        """
        ENHANCED: Implement trailing stop logic
        
        Returns new stop loss price if trailing should be activated, None otherwise
        """
        try:
            entry_price = position.entry_price
            side = position.side.lower()
            symbol = position.symbol
            
            # Initialize tracking if not exists
            if symbol not in self.position_tracker:
                self.position_tracker[symbol] = {
                    'highest_price': current_price if side == 'buy' else entry_price,
                    'lowest_price': current_price if side == 'sell' else entry_price,
                    'trailing_active': False,
                    'original_stop': position.stop_loss
                }
            
            tracker = self.position_tracker[symbol]
            trail_distance_pct = 0.025  # 2.5% trailing distance
            profit_threshold = 0.03  # Start trailing after 3% profit
            
            if side == 'buy':  # LONG position
                # Update highest price seen
                if current_price > tracker['highest_price']:
                    tracker['highest_price'] = current_price
                
                # Calculate profit percentage
                profit_pct = (current_price - entry_price) / entry_price
                
                # Start trailing after profit threshold
                if profit_pct >= profit_threshold:
                    if not tracker['trailing_active']:
                        tracker['trailing_active'] = True
                        self.logger.info(f"📈 Trailing stop activated for {symbol} at {profit_pct*100:.1f}% profit")
                    
                    # Calculate new trailing stop
                    new_stop = tracker['highest_price'] * (1 - trail_distance_pct)
                    
                    # Only move stop up, never down
                    if new_stop > position.stop_loss:
                        self.logger.info(f"📈 Trailing stop update for {symbol}: {position.stop_loss:.6f} → {new_stop:.6f}")
                        return new_stop
            
            else:  # SHORT position
                # Update lowest price seen
                if current_price < tracker['lowest_price']:
                    tracker['lowest_price'] = current_price
                
                # Calculate profit percentage
                profit_pct = (entry_price - current_price) / entry_price
                
                # Start trailing after profit threshold
                if profit_pct >= profit_threshold:
                    if not tracker['trailing_active']:
                        tracker['trailing_active'] = True
                        self.logger.info(f"📉 Trailing stop activated for {symbol} at {profit_pct*100:.1f}% profit")
                    
                    # Calculate new trailing stop
                    new_stop = tracker['lowest_price'] * (1 + trail_distance_pct)
                    
                    # Only move stop down, never up
                    if new_stop < position.stop_loss:
                        self.logger.info(f"📉 Trailing stop update for {symbol}: {position.stop_loss:.6f} → {new_stop:.6f}")
                        return new_stop
            
            return None  # No trailing update needed
            
        except Exception as e:
            self.logger.error(f"Trailing stop error for {position.symbol}: {e}")
            return None
    
    def check_partial_profit_taking(self, position: PositionData, current_price: float) -> Optional[Dict]:
        """
        ENHANCED: Check for partial profit taking opportunities
        
        Returns partial close instruction if conditions are met
        """
        try:
            entry_price = position.entry_price
            side = position.side.lower()
            symbol = position.symbol
            
            if side == 'buy':
                profit_pct = (current_price - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - current_price) / entry_price * 100
            
            # Partial profit levels - only if not already taken
            if profit_pct >= 6.0 and not position.partial_3_taken:
                return {'action': 'close_partial', 'percentage': 30, 'level': 'partial_3', 'profit_pct': profit_pct}
            elif profit_pct >= 4.0 and not position.partial_2_taken:
                return {'action': 'close_partial', 'percentage': 25, 'level': 'partial_2', 'profit_pct': profit_pct}
            elif profit_pct >= 2.0 and not position.partial_1_taken:
                return {'action': 'close_partial', 'percentage': 25, 'level': 'partial_1', 'profit_pct': profit_pct}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Partial profit check error for {position.symbol}: {e}")
            return None
    
    def execute_partial_close(self, position: PositionData, partial_info: Dict) -> bool:
        """Execute partial position close"""
        try:
            symbol = position.symbol
            percentage = partial_info['percentage']
            level = partial_info['level']
            profit_pct = partial_info['profit_pct']
            
            # Get current position size
            positions = self.exchange.fetch_positions([symbol])
            current_position = None
            
            for pos in positions:
                if pos['symbol'] == symbol and pos['contracts'] > 0:
                    current_position = pos
                    break
            
            if not current_position:
                self.logger.warning(f"No active position found for partial close: {symbol}")
                return False
            
            # Calculate partial close size
            current_size = current_position['contracts']
            close_size = current_size * (percentage / 100)
            
            # Determine close side
            current_side = current_position['side']
            if current_side.lower() == 'long':
                close_side = 'Sell'
            elif current_side.lower() == 'short':
                close_side = 'Buy'
            else:
                self.logger.error(f"Unknown position side for partial close: {current_side}")
                return False
            
            # Execute partial close
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=close_size,
                params={'reduceOnly': True}
            )
            
            self.logger.info(f"💰 Partial profit taken: {symbol}")
            self.logger.info(f"   Level: {level} ({percentage}% of position)")
            self.logger.info(f"   Profit: {profit_pct:.1f}%")
            self.logger.info(f"   Size: {close_size:.4f} units")
            
            # Mark this partial as taken
            setattr(position, level, True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Partial close execution error for {position.symbol}: {e}")
            return False
    
    def close_position(self, position: PositionData) -> bool:
        """Close a position on the exchange"""
        try:
            # Clean up position tracker
            if position.symbol in self.position_tracker:
                del self.position_tracker[position.symbol]
            
            # Get current position details from exchange
            positions = self.exchange.fetch_positions([position.symbol])
            current_position = None
            
            for pos in positions:
                if pos['symbol'] == position.symbol and pos['contracts'] > 0:
                    current_position = pos
                    break
            
            if not current_position:
                self.logger.warning(f"No active position found for {position.symbol}")
                return False
            
            # Determine correct close side based on current position
            current_side = current_position['side']
            position_size = current_position['contracts']
            
            # FIXED: Use proper capitalized sides for Bybit API
            if current_side.lower() == 'long':
                close_side = 'Sell'  # Capitalized
            elif current_side.lower() == 'short':
                close_side = 'Buy'   # Capitalized
            else:
                self.logger.error(f"Unknown position side: {current_side}")
                return False
            
            self.logger.debug(f"Closing {current_side} position of {position_size} {position.symbol} with {close_side} order")
            
            # Place market order to close position
            order = self.exchange.create_order(
                symbol=position.symbol,
                type='market',
                side=close_side,  # Use capitalized side
                amount=position_size,
                params={'reduceOnly': True}  # Ensure this closes the position
            )
            
            self.logger.debug(f"✅ Closed position {position.symbol}")
            self.logger.debug(f"Close order: {order}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "110017" in error_msg or "reduce-only order has same side" in error_msg:
                self.logger.error(f"❌ Close order side error for {position.symbol}. Trying alternative method...")
                # Try closing with position close API instead
                try:
                    close_result = self.exchange.close_position(position.symbol)
                    self.logger.info(f"✅ Closed position {position.symbol} using close_position API")
                    return True
                except Exception as e2:
                    self.logger.error(f"❌ Alternative close method also failed for {position.symbol}: {e2}")
            else:
                self.logger.error(f"❌ Failed to close position {position.symbol}: {e}")
            return False
    
    def get_current_positions(self) -> List[PositionData]:
        """Get current positions from exchange"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = []
            
            for pos in positions:
                if pos['contracts'] > 0:  # Active position
                    # Map exchange position side to our format
                    side = 'buy' if pos['side'].lower() == 'long' else 'sell'
                    
                    position_data = PositionData(
                        symbol=pos['symbol'],
                        side=side,
                        size=pos['contracts'],
                        entry_price=pos['entryPrice'],
                        leverage=pos.get('leverage', 1.0),
                        risk_amount=0.0,  # Will be updated from database
                        stop_loss=0.0,
                        take_profit=0.0,
                        position_id=pos.get('id', ''),
                        unrealized_pnl=pos.get('unrealizedPnl', 0.0),
                        unrealized_pnl_pct=pos.get('percentage', 0.0),
                        original_size=pos['contracts']  # Store original size
                    )
                    active_positions.append(position_data)
            
            return active_positions
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return []
    
    def monitor_positions(self):
        """Main monitoring loop - ENHANCED with trailing stops and partial profits"""
        try:
            while self.monitoring:
                positions = self.get_current_positions()
                
                for position in positions:
                    try:
                        # Get current price
                        ticker = self.exchange.fetch_ticker(position.symbol)
                        current_price = ticker['last']
                        
                        # 1. Check for partial profit taking
                        # partial_info = self.check_partial_profit_taking(position, current_price)
                        # if partial_info:
                        #     if self.execute_partial_close(position, partial_info):
                        #         self.logger.info(f"✅ Partial profit taken for {position.symbol}")
                        #         continue  # Skip other checks this cycle
                        
                        # 2. Update trailing stops
                        new_stop = self.update_trailing_stop(position, current_price)
                        if new_stop:
                            # Update stop loss in the system (this would need database update)
                            position.stop_loss = new_stop
                            # TODO: Update stop loss order on exchange if needed
                        
                        # 3. Check if auto-close is enabled and should auto-close (profit or loss)
                        if self.config.auto_close_enabled:
                            should_close, close_reason = self.should_auto_close(position, current_price)
                            
                            if should_close:
                                leveraged_profit = self.calculate_leveraged_profit_pct(position, current_price)
                                
                                # Log appropriate message based on close reason
                                if close_reason == 'profit_target':
                                    self.logger.info(
                                        f"💰 Profit target reached for {position.symbol}: "
                                        f"{leveraged_profit:.2f}% profit (target: {self.config.auto_close_profit_at}%)"
                                    )
                                elif close_reason == 'loss_limit':
                                    self.logger.warning(
                                        f"🛑 Stop loss triggered for {position.symbol}: "
                                        f"{leveraged_profit:.2f}% loss (limit: -{abs(self.config.auto_close_loss_at)}%)"
                                    )
                                
                                # Attempt to close position
                                if self.close_position(position):
                                    # Update database record
                                    self.update_position_in_database(position, 'closed', close_reason)
                                    
                                    # Log final close message
                                    if close_reason == 'profit_target':
                                        self.logger.info(f"✅ Successfully closed profitable position {position.symbol}")
                                    else:
                                        self.logger.info(f"✅ Successfully closed losing position {position.symbol}")
                        else:
                            # Optional: Log that monitoring is happening but auto-close is disabled
                            self.logger.debug(f"Monitoring {position.symbol} but auto-close is disabled")
                        
                    except Exception as e:
                        self.logger.error(f"Error monitoring position {position.symbol}: {e}")
                
                # Sleep between monitoring cycles
                time.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
    
    def start_monitoring(self):
        """Start position monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_positions, daemon=True)
            self.monitor_thread.start()
            self.logger.info("📊 Started enhanced position monitoring (trailing stops + partial profits)")
    
    def stop_monitoring(self):
        """Stop position monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("🛑 Stopped position monitoring")
    
    def update_position_in_database(self, position: PositionData, status: str, close_reason: str = None):
        """Update position status in database"""
        # This would be implemented to update the TradingPosition table
        # close_reason can be: 'profit_target', 'loss_limit', 'partial_profit', etc.
        pass


class OrderExecutor:
    """Execute leveraged orders with proper sizing - ENHANCED with market regime awareness"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig, leverage_manager: LeverageManager, position_sizer: PositionSizer):
        self.exchange = exchange
        self.config = config
        self.leverage_manager = leverage_manager
        self.position_sizer = position_sizer
        self.logger = logging.getLogger(__name__)
    
    def check_market_regime_compatibility(self, signal: Dict) -> Tuple[bool, str]:
        """
        ENHANCED: Check if signal is compatible with current market regime
        
        Returns:
            Tuple of (is_compatible, regime_info)
        """
        try:
            # Simple market regime detection based on signal data
            price_change_24h = signal.get('price_change_24h', 0)
            side = signal['side']
            confidence = signal.get('confidence', 50)
            
            # Determine regime
            if price_change_24h > 5:
                regime = 'strong_bullish'
            elif price_change_24h > 2:
                regime = 'bullish'
            elif price_change_24h < -5:
                regime = 'strong_bearish'
            elif price_change_24h < -2:
                regime = 'bearish'
            else:
                regime = 'ranging'
            
            # Check compatibility
            if regime == 'strong_bullish' and side == 'sell':
                # Only allow very strong SHORT signals in bull market
                if confidence < 75:
                    return False, f"SHORT signal too weak for {regime} market"
            elif regime == 'strong_bearish' and side == 'buy':
                # Only allow very strong LONG signals in bear market
                if confidence < 75:
                    return False, f"LONG signal too weak for {regime} market"
            
            return True, f"Compatible with {regime} regime"
            
        except Exception as e:
            self.logger.error(f"Market regime check error: {e}")
            return True, "regime_check_failed"
    
    def check_correlation_risk(self, signal: Dict, existing_positions: List = None) -> Tuple[bool, str]:
        """
        ENHANCED: Check correlation risk before opening new position
        
        Returns:
            Tuple of (is_safe, risk_info)
        """
        try:
            if existing_positions is None:
                # Get current positions from exchange
                positions = self.exchange.fetch_positions()
                existing_positions = [pos for pos in positions if pos.get('contracts', 0) > 0]
            
            new_symbol = signal['symbol']
            new_side = signal['side']
            
            # Define correlation groups
            correlation_groups = {
                'BTC': ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'ADA/USDT:USDT'],
                'ETH': ['ETH/USDT:USDT', 'BTC/USDT:USDT', 'MATIC/USDT:USDT', 'LINK/USDT:USDT'],
                'MAJOR_ALTS': ['ADA/USDT:USDT', 'DOT/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT'],
                'DEFI': ['UNI/USDT:USDT', 'SUSHI/USDT:USDT', 'AAVE/USDT:USDT', 'COMP/USDT:USDT']
            }
            
            # Find which group the new symbol belongs to
            new_group = None
            for group_name, symbols in correlation_groups.items():
                if new_symbol in symbols:
                    new_group = group_name
                    break
            
            if not new_group:
                return True, "No correlation group found"
            
            # Count same-direction positions in the correlation group
            same_direction_count = 0
            correlated_symbols = correlation_groups[new_group]
            
            for position in existing_positions:
                pos_symbol = position['symbol']
                pos_side = 'buy' if position['side'].lower() == 'long' else 'sell'
                
                if pos_symbol in correlated_symbols and pos_side == new_side:
                    same_direction_count += 1
            
            # Risk limits
            max_correlated_positions = 2
            
            if same_direction_count >= max_correlated_positions:
                return False, f"Too many {new_side} positions in {new_group} group ({same_direction_count}/{max_correlated_positions})"
            
            return True, f"Correlation risk acceptable ({same_direction_count}/{max_correlated_positions} in {new_group})"
            
        except Exception as e:
            self.logger.error(f"Correlation risk check error: {e}")
            return True, "correlation_check_failed"
    
    def place_leveraged_order(self, signal: Dict, risk_amount_pct: float, leverage_str: str) -> Tuple[bool, str, Dict]:
        """Place a leveraged order based on signal with percentage-based risk - ENHANCED"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = 0

            if self.config.auto_close_profit_at == 1000:
                take_profit = signal.get('take_profit_1', signal.get('take_profit', 0))
            else:
                take_profit = signal.get('take_profit_1', signal.get('take_profit', 0)) if self.config.default_tp_level == 'take_profit_1' else signal.get('take_profit_2', signal.get('take_profit', 0))
            
            self.logger.info(f"🚀 Placing {side.upper()} order for {symbol}")
            
            # ENHANCED: Market regime compatibility check
            regime_compatible, regime_info = self.check_market_regime_compatibility(signal)
            if not regime_compatible:
                return False, f"Market regime filter: {regime_info}", {}
            else:
                self.logger.debug(f"📊 Market regime: {regime_info}")
            
            # ENHANCED: Correlation risk check
            correlation_safe, correlation_info = self.check_correlation_risk(signal)
            if not correlation_safe:
                return False, f"Correlation risk: {correlation_info}", {}
            else:
                self.logger.debug(f"🔗 Correlation: {correlation_info}")
            
            if self.config.sandbox_mode:
                risk_amount_pct = 0.001  # Use a smaller investment capital in sandbox mode to avoid max qty limit

            # Convert leverage to float
            leverage = self.leverage_manager.convert_leverage_to_float(leverage_str, symbol)
            
            # Set leverage on exchange
            if not self.leverage_manager.set_symbol_leverage(symbol, leverage):
                return False, "Failed to set leverage", {}
            
            # ENHANCED: Use adaptive position sizing
            position_size, adjusted_risk_pct = self.position_sizer.calculate_adaptive_position_size(
                risk_amount_pct, leverage, entry_price, signal
            )
            
            # Validate position size
            is_valid, adjusted_size = self.position_sizer.validate_position_size(symbol, position_size)
            if not is_valid:
                return False, f"Invalid position size: {position_size}", {}
            
            position_size = adjusted_size
            
            # Check available balance and calculate required margin
            available_balance = self.position_sizer.get_available_balance()
            required_margin = (position_size * entry_price) / leverage
            risk_amount_usdt = available_balance * (adjusted_risk_pct / 100)
            
            if required_margin > available_balance:
                return False, f"Insufficient balance: need {required_margin}, have {available_balance}", {}
            
            self.logger.info(f"💰 Enhanced risk calculation:")
            self.logger.info(f"   Account Balance: {available_balance:.2f} USDT")
            self.logger.info(f"   Risk Percentage: {risk_amount_pct:.0f}% → {adjusted_risk_pct:.1f}% (adaptive)")
            self.logger.info(f"   Risk Amount: {risk_amount_usdt:.2f} USDT")
            self.logger.info(f"   Position Size: {position_size:.2f} units")
            self.logger.info(f"   Required Margin: {required_margin:.2f} USDT")
            
            # FIXED: Capitalize the side parameter for Bybit API
            bybit_side = side.capitalize()  # 'buy' -> 'Buy', 'sell' -> 'Sell'
            
            # Place entry order with integrated SL/TP
            order_type = 'market' if signal.get('order_type') == 'market' else 'limit'
            
            # Prepare order parameters with integrated SL/TP
            order_params = {}
            if stop_loss > 0:
                order_params['stopLoss'] = stop_loss
            if take_profit > 0:
                order_params['takeProfit'] = take_profit
            
            if order_type == 'market':
                entry_order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=bybit_side,  # Use capitalized side
                    amount=position_size,
                    params=order_params  # Include SL/TP in main order
                )
            else:
                entry_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=bybit_side,  # Use capitalized side
                    amount=position_size,
                    price=entry_price,
                    params=order_params  # Include SL/TP in main order
                )
            
            self.logger.info(f"✅ Entry order placed: {entry_order['id']}")
            if stop_loss > 0:
                self.logger.debug(f"✅ Stop loss integrated: ${stop_loss:.6f}")
            if take_profit > 0:
                self.logger.debug(f"✅ Take profit integrated: ${take_profit:.6f}")
            
            # SL/TP are now integrated - no separate orders needed
            sl_order = None
            tp_order = None
            
            # Create position tracking data
            position_data = {
                'symbol': symbol,
                'side': side,  # Keep original lowercase for internal tracking
                'position_size': position_size,
                'entry_price': entry_price,
                'leverage': leverage,
                'risk_amount': risk_amount_usdt,  # Store actual USDT amount
                'risk_percentage': adjusted_risk_pct,  # Store adaptive percentage
                'original_risk_percentage': risk_amount_pct,  # Store original percentage
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_order_id': entry_order['id'],
                'stop_loss_order_id': sl_order['id'] if sl_order else None,
                'take_profit_order_id': tp_order['id'] if tp_order else None,
                'signal_confidence': signal.get('confidence', 0),
                'mtf_status': signal.get('mtf_status', ''),
                'mtf_validated': signal.get('mtf_validated', False),
                'entry_strategy': signal.get('entry_strategy', 'immediate'),
                'market_regime': regime_info,
                'correlation_info': correlation_info,
                'auto_close_profit_target': self.config.auto_close_profit_at,
                'auto_close_loss_target': self.config.auto_close_loss_at
            }
            self.logger.info("=" * 100 + "\n")

            return True, "Order placed successfully", position_data
            
        except Exception as e:
            error_msg = f"Failed to place order for {signal.get('symbol', 'unknown')}: {e}"
            self.logger.error(error_msg)
            return False, error_msg, {}


class PositionManager:
    """Track and manage concurrent positions - ENHANCED with portfolio heat tracking"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig, db_manager: DatabaseManager):
        self.exchange = exchange
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def get_current_positions_count(self) -> int:
        """Get count of current active positions"""
        try:
            positions = self.exchange.fetch_positions()
            active_count = sum(1 for pos in positions if pos.get('contracts', 0) > 0 or pos.get('size', 0) > 0)
            return active_count
        except Exception as e:
            self.logger.error(f"Error getting positions count: {e}")
            return 0
    
    def calculate_portfolio_heat(self) -> float:
        """
        ENHANCED: Calculate total portfolio risk exposure (portfolio heat)
        
        Returns:
            Portfolio heat as percentage of account balance
        """
        try:
            positions = self.exchange.fetch_positions()
            total_risk = 0
            
            for position in positions:
                if position.get('contracts', 0) > 0:
                    # Estimate risk based on position value and typical stop distance
                    position_value = position['contracts'] * position['entryPrice']
                    estimated_risk = position_value * 0.05  # Assume 5% risk per position
                    total_risk += estimated_risk
            
            # Get account balance
            balance = self.exchange.fetch_balance()
            account_balance = balance['USDT']['total']
            
            portfolio_heat = (total_risk / account_balance) * 100 if account_balance > 0 else 0
            
            self.logger.debug(f"🔥 Portfolio Heat: {portfolio_heat:.1f}% (${total_risk:.2f} risk on ${account_balance:.2f})")
            
            return portfolio_heat
            
        except Exception as e:
            self.logger.error(f"Portfolio heat calculation error: {e}")
            return 0
    
    def get_symbols_with_positions(self) -> Set[str]:
        """Get set of symbols that have active positions"""
        try:
            positions = self.exchange.fetch_positions()
            symbols_with_positions = set()
            
            for pos in positions:
                if pos.get('contracts', 0) > 0 or pos.get('size', 0) > 0:
                    symbols_with_positions.add(pos['symbol'])
            
            return symbols_with_positions
            
        except Exception as e:
            self.logger.error(f"Error getting symbols with positions: {e}")
            return set()
    
    def get_symbols_with_orders(self) -> Set[str]:
        """Get set of symbols that have pending orders"""
        try:
            open_orders = self.exchange.fetch_open_orders()
            symbols_with_orders = set()
            
            for order in open_orders:
                symbols_with_orders.add(order['symbol'])
            
            return symbols_with_orders
            
        except Exception as e:
            self.logger.error(f"Error getting symbols with orders: {e}")
            return set()
    
    def filter_signals_by_existing_symbols(self, signals: List[Dict]) -> List[Dict]:
        """Filter out signals for symbols that already have positions or orders"""
        try:
            symbols_with_positions = self.get_symbols_with_positions()
            symbols_with_orders = self.get_symbols_with_orders()
            excluded_symbols = symbols_with_positions.union(symbols_with_orders)
            
            filtered_signals = []
            skipped_count = 0
            
            for signal in signals:
                symbol = signal['symbol']
                if symbol in excluded_symbols:
                    reason = ""
                    if symbol in symbols_with_positions:
                        reason = "existing position"
                    elif symbol in symbols_with_orders:
                        reason = "pending orders"
                    
                    self.logger.info(f"⏭️ Skipping {symbol} - {reason}")
                    skipped_count += 1
                else:
                    filtered_signals.append(signal)
            
            if skipped_count > 0:
                self.logger.info(f"🔍 Filtered out {skipped_count} symbols with existing exposure")
                
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error filtering signals by existing symbols: {e}")
            return signals
    
    def can_open_new_positions(self, requested_count: int) -> Tuple[bool, int]:
        """Check if we can open new positions - ENHANCED with portfolio heat check"""
        try:
            current_count = self.get_current_positions_count()
            max_positions = self.config.max_concurrent_positions
            available_slots = max_positions - current_count
            
            # Check portfolio heat limit
            portfolio_heat = self.calculate_portfolio_heat()
            max_portfolio_heat = 25.0  # Max 25% of account at risk
            
            if portfolio_heat > max_portfolio_heat:
                self.logger.warning(f"🔥 Portfolio heat too high: {portfolio_heat:.1f}% > {max_portfolio_heat}%")
                return False, 0
            
            if available_slots <= 0:
                return False, 0
            
            can_open = min(requested_count, available_slots)
            
            self.logger.debug(f"📊 Position capacity: {current_count}/{max_positions} positions, {portfolio_heat:.1f}% heat")
            
            return True, can_open
        except Exception as e:
            self.logger.error(f"Error checking position availability: {e}")
            return False, 0
    
    def save_position_to_database(self, position_data: Dict, scan_session_id: int = None) -> str:
        """Save position to database and return position ID"""
        try:
            session = self.db_manager.get_session()
            
            position_id = str(uuid.uuid4())
            
            position = TradingPosition(
                position_id=position_id,
                scan_session_id=scan_session_id,
                symbol=position_data['symbol'],
                side=position_data['side'],
                entry_price=position_data['entry_price'],
                position_size=position_data['position_size'],
                leverage=str(position_data['leverage']),
                risk_amount=position_data['risk_amount'],
                entry_order_id=position_data.get('entry_order_id'),
                stop_loss_order_id=position_data.get('stop_loss_order_id'),
                take_profit_order_id=position_data.get('take_profit_order_id'),
                stop_loss_price=position_data.get('stop_loss', 0),
                take_profit_price=position_data.get('take_profit', 0),
                auto_close_profit_target=position_data.get('auto_close_profit_target', 10.0),
                signal_confidence=position_data.get('signal_confidence'),
                mtf_status=position_data.get('mtf_status'),
                status='open'
            )
            
            session.add(position)
            session.commit()
            session.close()
            
            self.logger.debug(f"Position saved to database: {position_id}")
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error saving position to database: {e}")
            return ""
    
    def update_position_status(self, position_id: str, status: str, close_reason: str = None):
        """Update position status in database"""
        try:
            session = self.db_manager.get_session()
            
            position = session.query(TradingPosition).filter(
                TradingPosition.position_id == position_id
            ).first()
            
            if position:
                position.status = status
                if close_reason:
                    position.close_reason = close_reason
                if status == 'closed':
                    position.closed_at = datetime.utcnow()
                
                session.commit()
                self.logger.debug(f"Updated position {position_id} status to {status}")
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error updating position status: {e}")


def escape_markdown(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2
    FIXED: Prevents Telegram parsing errors
    """
    if not text:
        return ""
    
    # Convert to string if it's a number
    text = str(text)
    
    # Characters that need escaping in MarkdownV2
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    
    return text

def format_price(price: float) -> str:
    """Format price with proper escaping"""
    if price == 0:
        return "0"
    
    formatted = f"{price:.6f}"
    # Remove trailing zeros after decimal
    formatted = formatted.rstrip('0').rstrip('.')
    return escape_markdown(formatted)

def format_percentage(value: float) -> str:
    """Format percentage with proper escaping"""
    formatted = f"{value:.1f}%"
    return escape_markdown(formatted)


class AutoTrader:
    """Main auto-trading orchestration class - ENHANCED with all strategy improvements"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize database components
        self.db_manager = DatabaseManager(config.db_config.get_database_url())
        self.enhanced_db_manager = EnhancedDatabaseManager(self.db_manager)
        
        # Initialize trading system
        self.trading_system = CompleteEnhancedBybitSystem(config)
        self.exchange = self.trading_system.exchange_manager.exchange
        
        if not self.exchange:
            raise Exception("Failed to initialize exchange connection")
        
        # Initialize auto-trading components
        self.leverage_manager = LeverageManager(self.exchange)
        self.position_sizer = PositionSizer(self.exchange)
        self.schedule_manager = ScheduleManager(config)
        self.position_manager = PositionManager(self.exchange, config, self.db_manager)
        self.profit_monitor = LeveragedProfitMonitor(self.exchange, config)
        self.order_executor = OrderExecutor(
            self.exchange, config, self.leverage_manager, self.position_sizer
        )
        
        # Session tracking
        self.trading_session_id = None
        self.is_running = False
    
    def start_trading_session(self) -> str:
        """Start new auto-trading session"""
        try:
            session_id = f"autotrader_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trading_session_id = session_id
            
            # Save session to database
            session = self.db_manager.get_session()
            
            auto_session = AutoTradingSession(
                session_id=session_id,
                config_snapshot=self.config.to_dict()
            )
            
            session.add(auto_session)
            session.commit()
            session.close()
            
            self.logger.debug(f"🚀 Started enhanced auto-trading session: {session_id}")
            self.logger.debug(f"⚙️ Enhanced Configuration:")
            self.logger.debug(f"   Max concurrent positions: {self.config.max_concurrent_positions}")
            self.logger.debug(f"   Max executions per scan: {self.config.max_execution_per_trade}")
            self.logger.debug(f"   Base risk per trade: {self.config.risk_amount}% (adaptive)")
            self.logger.debug(f"   Leverage: {self.config.leverage}")
            self.logger.debug(f"   Auto-close profit target: {self.config.auto_close_profit_at}%")
            self.logger.debug(f"   Auto-close loss target: {self.config.auto_close_loss_at}%")
            self.logger.debug(f"   Scan interval: {self.config.scan_interval / 3600:.1f} hours")
            self.logger.debug(f"   🔥 Portfolio heat monitoring: Active")
            self.logger.debug(f"   📈 Trailing stops: Active")
            self.logger.debug(f"   💰 Partial profit taking: Active")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start trading session: {e}")
            raise
    
    async def run_scan_and_execute(self) -> Tuple[int, int]:
        """
        Enhanced scan and execute method integrating with the new MTF-aware system
        
        Maintains compatibility with existing autotrader.py architecture while
        leveraging the improved signal generation from the enhanced system.py
        """
        try:
            # Check if it's optimal trading time
            is_optimal, session = self.schedule_manager.is_optimal_trading_time()
            if session == "low_liquidity":
                self.logger.info("⏰ Low liquidity hours - proceeding with caution")
            
            self.logger.info("📊 Running enhanced MTF signal analysis...")
            
            # ===== PHASE 1: ENHANCED SIGNAL ANALYSIS =====
            # The trading_system now uses the enhanced MTF generator internally
            results = self.trading_system.run_complete_analysis_parallel_mtf()
            
            if not results or not results.get('signals'):
                self.logger.warning("No signals generated in this scan")
                return 0, 0
            
            signals_count = len(results['signals'])
            self.logger.debug(f"📈 Generated {signals_count} signals")
            
            # ===== ENHANCED MTF ANALYSIS LOGGING =====
            # Log MTF validation results from the enhanced system
            self._log_mtf_analysis_results(results)
            
            # ===== PHASE 2: FILTER EXISTING POSITIONS & ORDERS =====
            print("\n🔍 FILTERING OUT EXISTING POSITIONS & ORDERS...")
            
            already_ranked_signals = results.get('signals', [])
            signals_left_after_filter_out_existing_orders = self.position_manager.filter_signals_by_existing_symbols(already_ranked_signals)
            
            if not signals_left_after_filter_out_existing_orders:
                # Save original results to database first
                save_result = self.enhanced_db_manager.save_all_results(results)
                
                self.logger.warning("⚠️ No signals available for execution after filtering existing symbols")
                print("⚠️  ALL SIGNALS FILTERED OUT:")
                print("   Either symbols already have positions/orders")
                print("   or no valid signals generated")
                print("")
                return signals_count, 0
            
            # ===== PHASE 3: UPDATE RESULTS WITH FILTERED DATA =====
            filtered_results = self._update_results_with_filtering(
                results, already_ranked_signals, signals_left_after_filter_out_existing_orders
            )
            
            # ===== PHASE 4: SAVE FILTERED RESULTS TO DATABASE =====
            self.logger.debug("💾 Saving filtered results to MySQL database...")
            save_result = self.enhanced_db_manager.save_all_results(filtered_results)
            if save_result.get('error'):
                self.logger.error(f"Failed to save results: {save_result['error']}")
            else:
                self.logger.debug(f"✅ Filtered results saved - Scan ID: {save_result.get('scan_id', 'Unknown')}")
            
            # ===== PHASE 5: DISPLAY RESULTS TABLE =====
            self.trading_system.print_comprehensive_results_with_mtf(filtered_results)
            print("=" * 150 + "\n")
            
            # ===== PHASE 6: SEND SIGNAL NOTIFICATIONS =====
            self.logger.info("📢 Sending signal notifications for filtered opportunities...")
            await self._send_enhanced_signal_notifications(signals_left_after_filter_out_existing_orders)
            
            # ===== PHASE 7: CHECK POSITION AVAILABILITY =====
            can_trade, available_slots = self.position_manager.can_open_new_positions(
                self.config.max_execution_per_trade
            )
            
            if not can_trade:
                portfolio_heat = self.position_manager.calculate_portfolio_heat()
                self.logger.warning(
                    f"⚠️ Cannot open new positions - at max capacity or portfolio heat too high "
                    f"({self.config.max_concurrent_positions} positions, {portfolio_heat:.1f}% heat)"
                )
                print(f"⚠️  POSITION LIMIT REACHED:")
                print(f"   Max positions: {self.config.max_concurrent_positions}")
                print(f"   Portfolio heat: {portfolio_heat:.1f}%")
                print(f"   No new trades will be executed until positions are closed")
                return signals_count, 0
            
            # ===== PHASE 8: EXECUTE TRADES =====
            execution_count = min(available_slots, self.config.max_execution_per_trade, len(signals_left_after_filter_out_existing_orders))
            selected_opportunities = signals_left_after_filter_out_existing_orders[:execution_count]
            
            # Enhanced execution logging
            self._log_enhanced_execution_plan(
                signals_count, 
                len(signals_left_after_filter_out_existing_orders), 
                available_slots, 
                execution_count
            )
            
            executed_count = 0
            
            # Execute selected trades
            if self.config.auto_execute_trades:
                executed_count = await self._execute_enhanced_trades(
                    selected_opportunities, save_result.get('scan_session_id')
                )
            else:
                self.logger.info("🔄 Auto-execution disabled - signals ready for manual review")
            
            # ===== PHASE 9: FINAL SUMMARY =====
            self._log_enhanced_execution_summary(signals_count, len(signals_left_after_filter_out_existing_orders), executed_count)
            
            return signals_count, executed_count
            
        except Exception as e:
            self.logger.error(f"Error in enhanced scan and execute: {e}")
            raise

    def _log_mtf_analysis_results(self, results: Dict):
        """Log enhanced MTF analysis results"""
        try:
            signals = results.get('signals', [])
            system_performance = results.get('system_performance', {})
            scan_info = results.get('scan_info', {})
            
            # MTF validation metrics
            mtf_validated_signals = system_performance.get('mtf_validated_signals', 0)
            traditional_signals = system_performance.get('traditional_signals', 0)
            mtf_validation_rate = system_performance.get('mtf_validation_rate', 0)
            structure_filtered_rate = system_performance.get('structure_filtered_rate', 0)
            
            # Analysis method
            method = scan_info.get('method', 'unknown')
            execution_time = scan_info.get('execution_time_seconds', 0)
            
            self.logger.info(f"🎯 Enhanced MTF Analysis Complete:")
            self.logger.info(f"   Method: {method.upper()}")
            self.logger.info(f"   Execution Time: {execution_time:.1f}s")
            self.logger.info(f"   Total Signals: {len(signals)}")
            self.logger.info(f"   MTF Validated: {mtf_validated_signals}")
            self.logger.info(f"   Traditional: {traditional_signals}")
            self.logger.info(f"   MTF Validation Rate: {mtf_validation_rate:.1f}%")
            self.logger.info(f"   Structure Filtering: {structure_filtered_rate:.1f}% symbols filtered")
            
            # Sample signal analysis for debugging
            if signals and len(signals) >= 3:
                self._debug_enhanced_signals(signals[:3])
                
        except Exception as e:
            self.logger.error(f"Error logging MTF analysis results: {e}")

    def _debug_enhanced_signals(self, sample_signals: List[Dict]):
        """Debug logging for enhanced signals with MTF details"""
        try:
            print("\n🔍 ENHANCED SIGNAL ANALYSIS:")
            for i, signal in enumerate(sample_signals, 1):
                symbol = signal['symbol']
                mtf_validated = signal.get('mtf_validated', False)
                mtf_status = signal.get('mtf_status', 'UNKNOWN')
                entry_strategy = signal.get('entry_strategy', 'immediate')
                analysis_method = signal.get('analysis_method', 'unknown')
                
                # Risk/reward analysis
                original_rr = signal.get('original_risk_reward_ratio', signal.get('risk_reward_ratio', 0))
                final_rr = signal.get('risk_reward_ratio', 0)
                
                # MTF details
                analysis_details = signal.get('analysis_details', {})
                mtf_trend = analysis_details.get('mtf_trend', 'unknown')
                structure_timeframe = analysis_details.get('structure_timeframe', 'unknown')
                
                print(f"  {i}. {symbol}:")
                print(f"     🎯 MTF Status: {'✅ VALIDATED' if mtf_validated else '❌ TRADITIONAL'} ({mtf_status})")
                print(f"     📊 Analysis: {analysis_method}")
                print(f"     📈 Trend Context: {mtf_trend} ({structure_timeframe})")
                print(f"     🎪 Entry Strategy: {entry_strategy}")
                print(f"     💰 R/R Ratio: {original_rr:.2f} → {final_rr:.2f}")
                print(f"     🎯 Confidence: {signal['confidence']:.1f}%")
                print(f"     📋 TP Level: {self.config.default_tp_level}")
                print()
                
        except Exception as e:
            self.logger.error(f"Error in enhanced signal debugging: {e}")

    def _update_results_with_filtering(self, results: Dict, original_signals: List[Dict], 
                                    filtered_signals: List[Dict]) -> Dict:
        """Update results structure with filtering information"""
        try:
            # Get filtered symbols for chart generation
            filtered_symbol_set = set(opp['symbol'] for opp in filtered_signals)
            top_filtered_signals = [signal for signal in original_signals if signal['symbol'] in filtered_symbol_set]
            
            # Update results structure
            updated_results = results.copy()
            updated_results['top_opportunities'] = filtered_signals  # For display
            updated_results['signals'] = top_filtered_signals  # For chart generation
            
            # Update scan info to reflect filtering
            if 'scan_info' not in updated_results:
                updated_results['scan_info'] = {}
                
            scan_info = updated_results['scan_info']
            scan_info['original_signals_count'] = len(original_signals)
            scan_info['total_signals_left_after_filter_out_existing_orders'] = len(filtered_signals)
            scan_info['displayed_opportunities'] = len(filtered_signals)
            scan_info['filtered_signals_count'] = len(top_filtered_signals)
            scan_info['already_ranked_signals_count'] = len(original_signals)
            scan_info['symbols_filtered_out'] = len(original_signals) - len(filtered_signals)
            
            return updated_results
            
        except Exception as e:
            self.logger.error(f"Error updating results with filtering: {e}")
            return results

    async def _send_enhanced_signal_notifications(self, opportunities: List[Dict]):
        """Send enhanced signal notifications with MTF context"""
        try:
            notification_count = 0
            mtf_validated_count = 0
            
            for opportunity in opportunities:
                try:
                    # Track MTF validation for reporting
                    if opportunity.get('mtf_validated', False):
                        mtf_validated_count += 1
                    
                    # Send signal notification (enhanced method)
                    await self.send_signal_notification(opportunity)
                    notification_count += 1
                    self.logger.debug(f"✅ Notification sent for {opportunity['symbol']}")
                    
                except Exception as e:
                    self.logger.error(f"Error sending notification for {opportunity['symbol']}: {e}")
            
            self.logger.info(f"📱 Notifications: {notification_count}/{len(opportunities)} sent")
            self.logger.info(f"   🎯 MTF Validated: {mtf_validated_count}/{len(opportunities)}")
            
        except Exception as e:
            self.logger.error(f"Error sending enhanced signal notifications: {e}")

    def _log_enhanced_execution_plan(self, original_signals: int, available_signals: int, 
                                available_slots: int, execution_count: int):
        """Enhanced execution plan logging"""
        try:
            portfolio_heat = self.position_manager.calculate_portfolio_heat()
            
            print("")
            print("=" * 150 + "\n")
            print(f"🎯 ENHANCED EXECUTION PLAN:")
            print(f"   Original Signals: {original_signals}")
            print(f"   Total After Position/Order Filter: {available_signals}")
            print(f"   Available Position Slots: {available_slots}")
            print(f"   Portfolio Heat: {portfolio_heat:.1f}%")
            print(f"   Selected for Execution: {execution_count}")
            print(f"   Auto-Execute: {'✅ Enabled' if self.config.auto_execute_trades else '❌ Disabled'}")
            print("")
            
            self.logger.debug(
                f"🎯 Enhanced execution: {execution_count} trades from {available_signals} available "
                f"(slots: {available_slots}, heat: {portfolio_heat:.1f}%, original: {original_signals})"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging enhanced execution plan: {e}")

    async def _execute_enhanced_trades(self, selected_opportunities: List[Dict], scan_session_id: str) -> int:
        """Execute trades with enhanced MTF context logging and risk management"""
        try:
            self.logger.info("🚀 Starting enhanced trade execution...")
            executed_count = 0
            
            for i, opportunity in enumerate(selected_opportunities):
                try:
                    symbol = opportunity['symbol']
                    mtf_status = opportunity.get('mtf_status', 'UNKNOWN')
                    entry_strategy = opportunity.get('entry_strategy', 'immediate')
                    mtf_validated = opportunity.get('mtf_validated', False)
                    analysis_method = opportunity.get('analysis_method', 'unknown')
                    
                    # Enhanced execution logging
                    validation_emoji = "🎯" if mtf_validated else "⚠️"
                    self.logger.info(f"📝 {validation_emoji} Executing trade {i+1}/{len(selected_opportunities)}: {symbol}")
                    self.logger.info(f"     MTF Status: {mtf_status}")
                    self.logger.info(f"     Entry Strategy: {entry_strategy}")
                    self.logger.info(f"     Analysis Method: {analysis_method}")
                    
                    # Execute trade using enhanced order executor
                    success, message, position_data = self.order_executor.place_leveraged_order(
                        opportunity, self.config.risk_amount, self.config.leverage
                    )
                    
                    if success:
                        # Save position to database
                        position_id = self.position_manager.save_position_to_database(position_data, scan_session_id)
                        
                        if not position_id:
                            self.logger.error(f"Failed to save position data for {symbol}")
                            continue

                        executed_count += 1
                        
                        # Enhanced success logging
                        self.logger.info(f"✅ Trade {i+1} executed successfully")
                        self.logger.info(f"   Position ID: {position_id}")
                        self.logger.info(f"   Entry: ${opportunity['entry_price']:.6f}")
                        self.logger.info(f"   Risk: {position_data.get('risk_percentage', self.config.risk_amount):.1f}% (adaptive)")
                        self.logger.info(f"   MTF Validated: {mtf_validated}")
                        
                        # Send trade notification using enhanced method
                        await self.send_trade_notification(opportunity, position_data, success=True)
                    else:
                        print(f"   ❌ TRADE FAILED!")
                        print(f"   Error: {message}")
                        self.logger.error(f"Trade execution failed: {message}")
                        
                        # Send failure notification
                        await self.send_trade_notification(opportunity, {}, success=False, error_message=message)
                        
                except Exception as e:
                    error_msg = f"Error executing trade for {opportunity['symbol']}: {e}"
                    print(f"   ❌ {error_msg}")
                    self.logger.error(error_msg)
            
            self.logger.info(f"🏁 Enhanced execution completed: {executed_count}/{len(selected_opportunities)} successful")
            return executed_count
            
        except Exception as e:
            self.logger.error(f"Error in enhanced trade execution: {e}")
            return 0

    def _log_enhanced_execution_summary(self, original_signals: int, available_signals: int, executed_count: int):
        """Enhanced execution summary with MTF metrics"""
        try:
            filter_rate = ((original_signals - available_signals) / original_signals * 100) if original_signals > 0 else 0
            execution_rate = (executed_count / available_signals * 100) if available_signals > 0 else 0
            portfolio_heat = self.position_manager.calculate_portfolio_heat()
            
            print(f"\n📊 ENHANCED EXECUTION SUMMARY:")
            print(f"   Original Signals: {original_signals}")
            print(f"   Available After Filtering: {available_signals} ({100-filter_rate:.1f}%)")
            print(f"   Successfully Executed: {executed_count} ({execution_rate:.1f}%)")
            print(f"   Portfolio Heat: {portfolio_heat:.1f}%")
            print(f"   Filter Efficiency: {filter_rate:.1f}% (prevents overexposure)")
            print(f"   Risk Management: Enhanced (adaptive sizing + correlation filters)")
            print("=" * 150)
            
            self.logger.info(f"🏁 Enhanced scan summary: {original_signals} → {available_signals} → {executed_count} trades")
            self.logger.info(f"   📈 Filter rate: {filter_rate:.1f}% (risk management)")
            self.logger.info(f"   🎯 Execution rate: {execution_rate:.1f}% (of available)")
            self.logger.info(f"   🔥 Portfolio heat: {portfolio_heat:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error logging enhanced execution summary: {e}")

    # ===== ENHANCED NOTIFICATION METHODS WITH FIXED MARKDOWN =====

    async def send_signal_notification(self, opportunity: Dict):
        """Enhanced signal notification with FIXED Markdown formatting"""
        try:
            symbol = escape_markdown(opportunity['symbol'])
            chart_file = opportunity.get('chart_file', '')
            order_type = escape_markdown(opportunity['order_type'].upper())
            side = opportunity['side'].upper()
            entry_price = format_price(opportunity['entry_price'])
            take_profit_1 = format_price(opportunity.get('take_profit_1', 0))
            take_profit_2 = format_price(opportunity.get('take_profit_2', 0))
            stop_loss = format_price(opportunity.get('stop_loss', 0))
            risk_reward = opportunity.get('risk_reward_ratio', 0)
            confidence = opportunity.get('confidence', 0)
            mtf_status = escape_markdown(opportunity.get('mtf_status', 'N/A'))
            
            # Enhanced MTF context
            mtf_validated = opportunity.get('mtf_validated', False)
            entry_strategy = escape_markdown(opportunity.get('entry_strategy', 'immediate'))
            analysis_details = opportunity.get('analysis_details', {})
            mtf_trend = escape_markdown(analysis_details.get('mtf_trend', 'unknown'))
            structure_tf = escape_markdown(analysis_details.get('structure_timeframe', 'unknown'))

            # Enhanced message with FIXED Markdown formatting
            validation_emoji = "🎯" if mtf_validated else "⚠️"
            
            # Build message with proper escaping
            message_parts = []
            message_parts.append(f"📊 *{validation_emoji} ENHANCED SIGNAL*")
            message_parts.append("")
            
            if side == 'BUY':
                message_parts.append(f"🟢 *LONG {symbol}*")
            else:
                message_parts.append(f"🔴 *SHORT {symbol}*")
                
            message_parts.append(f"📈 *{order_type} ORDER*")
            message_parts.append(f"    💰 Entry Price: ${entry_price}")
            message_parts.append(f"    🎯 Take Profit 1: ${take_profit_1}")
            message_parts.append(f"    🎯 Take Profit 2: ${take_profit_2}")
            message_parts.append(f"    🚫 Stop Loss: ${stop_loss}")
            message_parts.append("")
            
            message_parts.append(f"📊 *ENHANCED ANALYSIS*")
            leverage_escaped = escape_markdown(str(self.config.leverage))
            risk_amount_escaped = format_percentage(self.config.risk_amount)
            risk_reward_escaped = escape_markdown(f"{risk_reward:.2f}")
            entry_strategy_display = entry_strategy.replace('_', ' ').title()
            
            message_parts.append(f"    🔍 Leverage: {leverage_escaped}")
            message_parts.append(f"    🔍 Base Risk: {risk_amount_escaped} \\(adaptive\\)")
            message_parts.append(f"    📊 Risk/Reward: {risk_reward_escaped}:1")
            message_parts.append(f"    🎯 Entry Strategy: {entry_strategy_display}")
            message_parts.append("")
            
            message_parts.append(f"🎯 *MTF VALIDATION:*")
            validation_status = "✅" if mtf_validated else "❌"
            confidence_escaped = format_percentage(confidence)
            
            message_parts.append(f"    {validation_status} MTF Validated: {mtf_status}")
            message_parts.append(f"    📈 Structure Trend: {mtf_trend} \\({structure_tf}\\)")
            message_parts.append(f"    🎯 Confidence: {confidence_escaped}")
            message_parts.append("")
            
            # Add timestamp
            timestamp = escape_markdown(datetime.now().strftime('%H:%M:%S'))
            message_parts.append(f"🕒 {timestamp}")
            message_parts.append("")
            message_parts.append("Enhanced with multi\\-timeframe structure analysis and adaptive risk management\\!")

            # Join message parts
            message = "\n".join(message_parts)
            
            keyboard = []

            # Send notification with chart if available
            if chart_file and chart_file.endswith('.png') and chart_file.startswith('charts/'):
                await send_trading_notification(self.config, message, keyboard, image_path=chart_file)
            else:
                await send_trading_notification(self.config, message, keyboard)

        except Exception as e:
            self.logger.error(f"Failed to send enhanced signal notification: {e}")

    async def send_trade_notification(self, opportunity: Dict, position_data: Dict, success: bool, error_message: str = None):
        """Enhanced trade notification with FIXED Markdown formatting"""
        try:
            symbol = escape_markdown(opportunity['symbol'])
            side = opportunity['side'].upper()
            confidence = opportunity.get('confidence', 0)
            mtf_status = escape_markdown(opportunity.get('mtf_status', 'N/A'))
            mtf_validated = opportunity.get('mtf_validated', False)
            entry_strategy = escape_markdown(opportunity.get('entry_strategy', 'immediate'))
            
            message_parts = []
            
            if success:
                # Enhanced success notification
                risk_usdt = position_data.get('risk_amount', 0)
                leverage = position_data.get('leverage', 0)
                position_size = position_data.get('position_size', 0)
                entry_price = position_data.get('entry_price', 0)
                adaptive_risk = position_data.get('risk_percentage', self.config.risk_amount)
                original_risk = position_data.get('original_risk_percentage', self.config.risk_amount)
                
                validation_emoji = "🎯" if mtf_validated else "⚠️"
                
                message_parts.append(f"🚀 *{validation_emoji} ENHANCED TRADE EXECUTED*")
                message_parts.append("")
                message_parts.append(f"📊 *{symbol}* {side}")
                
                entry_price_escaped = format_price(entry_price)
                position_size_escaped = escape_markdown(f"{position_size:.4f}")
                leverage_escaped = escape_markdown(str(leverage))
                risk_usdt_escaped = escape_markdown(f"{risk_usdt:.2f}")
                
                message_parts.append(f"💰 Entry Price: ${entry_price_escaped}")
                message_parts.append(f"📈 Position Size: {position_size_escaped} units")
                message_parts.append(f"⚡ Leverage: {leverage_escaped}x")
                message_parts.append(f"💵 Risk: {risk_usdt_escaped} USDT")
                
                # Show adaptive risk adjustment
                if abs(adaptive_risk - original_risk) > 0.1:
                    original_risk_escaped = format_percentage(original_risk)
                    adaptive_risk_escaped = format_percentage(adaptive_risk)
                    message_parts.append(f"📊 Risk Adjustment: {original_risk_escaped} → {adaptive_risk_escaped}")
                else:
                    risk_pct_escaped = format_percentage(adaptive_risk)
                    message_parts.append(f"📊 Risk: {risk_pct_escaped}")
                
                message_parts.append("")
                
                message_parts.append(f"🎯 *ENHANCED SIGNAL QUALITY:*")
                validation_status = "✅" if mtf_validated else "❌"
                entry_strategy_display = entry_strategy.replace('_', ' ').title()
                confidence_escaped = format_percentage(confidence)
                
                message_parts.append(f"   {validation_status} MTF Validated: {mtf_status}")
                message_parts.append(f"   📈 Entry Strategy: {entry_strategy_display}")
                message_parts.append(f"   🎯 Confidence: {confidence_escaped}")
                message_parts.append("")
                
                profit_target_escaped = format_percentage(self.config.auto_close_profit_at)
                loss_target_escaped = format_percentage(self.config.auto_close_loss_at)
                timestamp_escaped = escape_markdown(datetime.now().strftime('%H:%M:%S'))
                
                message_parts.append(f"🎯 Profit Target: {profit_target_escaped}")
                message_parts.append(f"🛑 Loss Target: {loss_target_escaped}")
                message_parts.append(f"📈 Trailing Stops: Active")
                message_parts.append(f"💰 Partial Profits: Active")
                message_parts.append(f"🕒 {timestamp_escaped}")
                
                # Add position management keyboard
                keyboard = [
                    [{"text": "📊 Check Position", "callback_data": f"check_pos_{symbol}"}],
                    [{"text": "🔴 Close Position", "callback_data": f"close_pos_{symbol}"}]
                ]
                
            else:
                # Enhanced failure notification
                validation_emoji = "🎯" if mtf_validated else "⚠️"
                
                message_parts.append(f"❌ *{validation_emoji} ENHANCED TRADE FAILED*")
                message_parts.append("")
                message_parts.append(f"📊 *{symbol}* {side}")
                message_parts.append(f"🎯 MTF Status: {mtf_status}")
                
                entry_strategy_display = entry_strategy.replace('_', ' ').title()
                confidence_escaped = format_percentage(confidence)
                error_escaped = escape_markdown(str(error_message)) if error_message else "Unknown error"
                timestamp_escaped = escape_markdown(datetime.now().strftime('%H:%M:%S'))
                
                message_parts.append(f"📈 Entry Strategy: {entry_strategy_display}")
                message_parts.append(f"🎯 Confidence: {confidence_escaped}")
                message_parts.append("")
                message_parts.append(f"💥 *Error:* {error_escaped}")
                message_parts.append("")
                message_parts.append(f"🕒 {timestamp_escaped}")
                
                keyboard = None
            
            # Join message parts
            message = "\n".join(message_parts)
            
            await send_trading_notification(self.config, message, keyboard)
            
        except Exception as e:
            self.logger.error(f"Failed to send enhanced trade notification: {e}")

    async def send_scan_notification(self, total_signals: int, executed_count: int):
        """Send scan completion notification with FIXED formatting"""
        try:
            next_scan = self.schedule_manager.calculate_next_scan_time()
            portfolio_heat = self.position_manager.calculate_portfolio_heat()
            
            message_parts = []
            message_parts.append("📊 *ENHANCED SCAN COMPLETE*")
            message_parts.append("")
            
            total_escaped = escape_markdown(str(total_signals))
            executed_escaped = escape_markdown(str(executed_count))
            success_rate = (executed_count/total_signals*100) if total_signals > 0 else 0
            success_escaped = format_percentage(success_rate)
            heat_escaped = format_percentage(portfolio_heat)
            next_time_escaped = escape_markdown(next_scan.strftime('%H:%M'))
            timestamp_escaped = escape_markdown(datetime.now().strftime('%H:%M:%S'))
            
            message_parts.append(f"🔍 Signals Found: {total_escaped}")
            message_parts.append(f"⚡ Trades Executed: {executed_escaped}")
            message_parts.append(f"📈 Success Rate: {success_escaped}")
            message_parts.append(f"🔥 Portfolio Heat: {heat_escaped}")
            message_parts.append("")
            message_parts.append(f"⏰ Next Scan: {next_time_escaped}")
            message_parts.append(f"🕒 {timestamp_escaped}")
            message_parts.append("")
            message_parts.append("Enhanced with adaptive risk management and correlation filtering\\!")
            
            message = "\n".join(message_parts)
            
            await send_trading_notification(self.config, message)
            
        except Exception as e:
            self.logger.error(f"Failed to send scan notification: {e}")

    def main_trading_loop(self):
        """Main auto-trading loop - ENHANCED"""
        try:
            self.is_running = True
            session_id = self.start_trading_session()
            
            # Start enhanced profit monitoring
            if self.config.auto_close_enabled:
                self.logger.debug("📈 Starting enhanced profit monitoring (trailing stops + partial profits)")
                self.profit_monitor.start_monitoring()
            
            self.logger.info("🤖 Enhanced auto-trading loop started")
            
            while self.is_running:
                try:
                    # Wait for next scheduled scan
                    next_scan_time = self.schedule_manager.wait_for_next_scan()
                    
                    if not self.is_running:
                        break
                    
                    # Check trading time optimality
                    is_optimal, session = self.schedule_manager.is_optimal_trading_time()
                    session_info = f" ({session})" if session != "unknown" else ""
                    
                    self.logger.info(f"⏰ Scan time reached: {next_scan_time}{session_info}")
                    
                    # Run enhanced scan and execute trades with proper async handling
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        total_signals, executed_count = loop.run_until_complete(
                            self.run_scan_and_execute()
                        )
                    finally:
                        loop.close()
                    
                    self.logger.info(
                        f"📊 Enhanced scan complete - Signals: {total_signals}, "
                        f"Executed: {executed_count}"
                    )
                    
                    # Update session statistics
                    self.update_session_stats(total_signals, executed_count)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Received interrupt signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in enhanced trading loop: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
            
        except Exception as e:
            self.logger.error(f"Critical error in enhanced trading loop: {e}")
        finally:
            self.stop_trading()
    
    def stop_trading(self):
        """Stop auto-trading"""
        try:
            self.is_running = False
            self.profit_monitor.stop_monitoring()
            
            if self.trading_session_id:
                # Update session end time in database
                session = self.db_manager.get_session()
                auto_session = session.query(AutoTradingSession).filter(
                    AutoTradingSession.session_id == self.trading_session_id
                ).first()
                
                if auto_session:
                    auto_session.ended_at = datetime.utcnow()
                    auto_session.status = 'stopped'
                    session.commit()
                
                session.close()
            
            self.logger.info("🛑 Enhanced auto-trading stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
        
    def update_session_stats(self, total_signals: int, executed_count: int):
        """Update session statistics in database"""
        try:
            if not self.trading_session_id:
                return
            
            session = self.db_manager.get_session()
            auto_session = session.query(AutoTradingSession).filter(
                AutoTradingSession.session_id == self.trading_session_id
            ).first()
            
            if auto_session:
                auto_session.total_scans += 1
                auto_session.total_trades_placed += executed_count
                auto_session.last_scan_at = datetime.utcnow()
                auto_session.next_scan_at = self.schedule_manager.calculate_next_scan_time()
                
                session.commit()
            
            session.close()
            
            # Send enhanced scan notification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.send_scan_notification(total_signals, executed_count))
            finally:
                loop.close()
            
        except Exception as e:
            self.logger.error(f"Error updating session stats: {e}")


# Main execution function for standalone usage
def main():
    """Main function for running the enhanced auto-trader"""
    try:
        from config.config import DatabaseConfig, EnhancedSystemConfig
        from utils.logging import setup_logging
        
        # Setup logging
        logger = setup_logging("INFO")
        
        # Load configuration
        db_config = DatabaseConfig()  # Load from environment or defaults
        config = EnhancedSystemConfig.from_database(db_config, 'default')
        
        # Validate auto-trading configuration
        if not config.leverage or config.leverage not in LeverageManager.ACCEPTABLE_LEVERAGE:
            logger.error(f"Invalid leverage configuration: {config.leverage}")
            return
        
        if config.risk_amount <= 0:
            logger.error(f"Invalid risk amount: {config.risk_amount}")
            return
        
        # Start enhanced auto-trader
        logger.info("🚀 Starting Enhanced Auto-Trader with:")
        logger.info("   📊 Adaptive Position Sizing")
        logger.info("   🛑 Dynamic Stop Losses + Trailing")
        logger.info("   💰 Partial Profit Taking")
        logger.info("   🔗 Correlation Risk Management")
        logger.info("   📈 Market Regime Awareness")
        logger.info("   🔥 Portfolio Heat Monitoring")
        
        auto_trader = AutoTrader(config)
        auto_trader.main_trading_loop()
        
    except KeyboardInterrupt:
        logger.info("Enhanced auto-trader stopped by user")
    except Exception as e:
        logger.error(f"Enhanced auto-trader failed: {e}")
        raise

if __name__ == "__main__":
    main()